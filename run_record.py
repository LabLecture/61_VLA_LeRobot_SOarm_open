"""Record SO-101 demonstrations through a local, hardware-safe web dashboard."""

from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from lerobot.cameras import ColorMode, Cv2Backends
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.common.control_utils import sanity_check_dataset_robot_compatibility
from lerobot.configs.video import RGBEncoderConfig
from lerobot.datasets import LeRobotDataset, VideoEncodingManager
from lerobot.motors import MotorCalibration, MotorNormMode
from lerobot.motors.feetech import OperatingMode
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame, hw_to_dataset_features
from lerobot.utils.keyboard_input import init_keyboard_listener
from lerobot.utils.robot_utils import precise_sleep

from recording_web_dashboard import DEFAULT_WEB_PORT, RecordingWebDashboard
from run_teleoperate import (
    CAMERA_DARK_MEAN_THRESHOLD,
    CAMERA_FOURCC,
    CAMERA_FPS,
    CAMERA_HEIGHT,
    CAMERA_INDEX,
    CAMERA_WIDTH,
    FOLLOWER_USB_SERIAL,
    FPS,
    LEADER_USB_SERIAL,
    MAX_START_ANGLE_DELTA_DEG,
    MAX_START_GRIPPER_DELTA,
    MIN_CALIBRATION_SPAN_TICKS,
    MOTOR_IDS,
    NUM_READ_RETRIES,
    connect_follower_cameras,
    find_port,
    joint_position_delta,
    limit_action_step,
    safely_enable_follower,
    validate_calibration_spans,
)


DEFAULT_REPO_ID = "local/so101_dataset"
DEFAULT_DATASET_ROOT = Path("datasets/so101_dataset")
DEFAULT_EPISODES = 200
DEFAULT_EPISODE_SECONDS = 60.0
DEFAULT_RESET_SECONDS = 20.0
DEFAULT_PREVIEW_HZ = 10.0
DEFAULT_MOTION_TRIGGERED = True
DEFAULT_MOTION_BODY_THRESHOLD_DEG_S = 4.0
DEFAULT_MOTION_GRIPPER_THRESHOLD_PCT_S = 5.0
DEFAULT_MOTION_PRE_ROLL_S = 0.5
DEFAULT_MOTION_STOP_SECONDS = 1.0
CAMERA_PREFLIGHT_FRAMES = 10
MIN_RECORDING_RATE_RATIO = 0.90
MIN_CONTROL_FPS = 5
MAX_CONTROL_FPS = 20
MAX_EPISODE_SECONDS = 300.0
MAX_RESET_SECONDS = 600.0
CALIBRATION_READ_PERIOD_S = 0.02
CALIBRATION_HEARTBEAT_TIMEOUT_S = 5.0
PLAYBACK_HEARTBEAT_TIMEOUT_S = 2.5
MOTION_START_DEBOUNCE_FRAMES = 2
MOTION_STOP_THRESHOLD_RATIO = 0.5
MOTION_STOP_MAX_BODY_ERROR_DEG = 3.0
MAX_MOTION_PRE_ROLL_S = 1.0
MAX_RECORDING_FRAME_GAP_PERIODS = 2.0
JOINT_NAMES = tuple(MOTOR_IDS)


class UserCancelled(RuntimeError):
    """Raised when the web operator requests a safe shutdown."""


@dataclass(frozen=True)
class MotionCaptureSettings:
    enabled: bool
    body_threshold_deg_s: float
    gripper_threshold_pct_s: float
    pre_roll_s: float
    stop_after_s: float


@dataclass(frozen=True)
class RuntimeSettings:
    control_fps: int
    episodes: int
    episode_seconds: float
    reset_seconds: float
    preview_hz: float
    calibration: str
    motion_capture: MotionCaptureSettings


@dataclass(frozen=True)
class SegmentResult:
    saved_frames: int
    control_frames: int
    elapsed_s: float
    rate_hz: float
    saved_rate_hz: float
    max_saved_interval_s: float
    states: tuple[tuple[float, ...], ...]
    actions: tuple[tuple[float, ...], ...]
    leaders: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class BufferedMotionSample:
    dataset_frame: dict[str, object]
    state_values: tuple[float, ...]
    action_values: tuple[float, ...]
    leader_values: tuple[float, ...]
    sample_time_s: float


@dataclass(frozen=True)
class MotionCaptureUpdate:
    samples: tuple[BufferedMotionSample, ...]
    just_triggered: bool
    should_finish: bool
    status: str
    body_speed_deg_s: float
    gripper_speed_pct_s: float


def calculate_sample_rate_hz(sample_times: list[float]) -> float:
    if len(sample_times) < 2:
        return 0.0
    elapsed_s = sample_times[-1] - sample_times[0]
    if not math.isfinite(elapsed_s) or elapsed_s <= 0:
        return 0.0
    return (len(sample_times) - 1) / elapsed_s


def calculate_max_sample_interval_s(sample_times: list[float]) -> float:
    if len(sample_times) < 2:
        return 0.0
    return max(current - previous for previous, current in zip(sample_times, sample_times[1:]))


def measure_motion_speeds(
    previous: dict[str, float] | None,
    current: dict[str, float],
    *,
    dt_s: float,
) -> tuple[float, float]:
    """Measure wrap-aware joint motion using the actual time between samples."""
    if previous is None:
        return 0.0, 0.0
    if not math.isfinite(dt_s) or dt_s <= 0:
        raise ValueError("Motion sample interval must be a positive finite value.")
    body_speed = max(
        abs(joint_position_delta(key, current[key], previous[key])) / dt_s
        for key in JOINT_NAMES
        if key != "gripper.pos"
    )
    gripper_speed = (
        abs(joint_position_delta("gripper.pos", current["gripper.pos"], previous["gripper.pos"]))
        / dt_s
    )
    return body_speed, gripper_speed


class MotionTriggeredCapture:
    """Trim leading/trailing idle while preserving one continuous fixed-FPS motion segment."""

    def __init__(self, settings: MotionCaptureSettings, *, fps: int) -> None:
        if not settings.enabled:
            raise ValueError("MotionTriggeredCapture requires enabled settings.")
        self.settings = settings
        self.fps = fps
        pre_roll_frames = math.ceil(settings.pre_roll_s * fps)
        self._pre_roll: deque[BufferedMotionSample] = deque(
            maxlen=max(MOTION_START_DEBOUNCE_FRAMES, pre_roll_frames + MOTION_START_DEBOUNCE_FRAMES)
        )
        self._previous: dict[str, float] | None = None
        self._previous_leader: dict[str, float] | None = None
        self._moving_frames = 0
        self._stationary_frames = 0
        self._stop_frames = max(1, math.ceil(settings.stop_after_s * fps))
        self.active = False

    def update(
        self,
        follower_values: dict[str, float],
        sample: BufferedMotionSample,
        *,
        leader_values: dict[str, float] | None = None,
        dt_s: float | None = None,
    ) -> MotionCaptureUpdate:
        sample_interval = 1.0 / self.fps if dt_s is None else dt_s
        leader_values = follower_values if leader_values is None else leader_values
        follower_body_speed, follower_gripper_speed = measure_motion_speeds(
            self._previous,
            follower_values,
            dt_s=sample_interval,
        )
        leader_body_speed, leader_gripper_speed = measure_motion_speeds(
            self._previous_leader,
            leader_values,
            dt_s=sample_interval,
        )
        self._previous = dict(follower_values)
        self._previous_leader = dict(leader_values)
        body_speed = max(follower_body_speed, leader_body_speed)
        gripper_speed = max(follower_gripper_speed, leader_gripper_speed)

        body_tracking_error = max(
            abs(joint_position_delta(key, leader_values[key], follower_values[key]))
            for key in JOINT_NAMES
            if key != "gripper.pos"
        )
        above_start = (
            body_speed >= self.settings.body_threshold_deg_s
            or gripper_speed >= self.settings.gripper_threshold_pct_s
        )
        # A grasped object can intentionally leave a large leader/follower gripper position error.
        # Require both grippers to be stationary, but apply the tracking-error guard only to body joints.
        below_stop = (
            body_speed <= self.settings.body_threshold_deg_s * MOTION_STOP_THRESHOLD_RATIO
            and gripper_speed
            <= self.settings.gripper_threshold_pct_s * MOTION_STOP_THRESHOLD_RATIO
            and body_tracking_error <= MOTION_STOP_MAX_BODY_ERROR_DEG
        )

        if not self.active:
            self._pre_roll.append(sample)
            self._moving_frames = self._moving_frames + 1 if above_start else 0
            if self._moving_frames < MOTION_START_DEBOUNCE_FRAMES:
                return MotionCaptureUpdate(
                    samples=(),
                    just_triggered=False,
                    should_finish=False,
                    status="ARMED",
                    body_speed_deg_s=body_speed,
                    gripper_speed_pct_s=gripper_speed,
                )
            self.active = True
            samples = tuple(self._pre_roll)
            self._pre_roll.clear()
            return MotionCaptureUpdate(
                samples=samples,
                just_triggered=True,
                should_finish=False,
                status="RECORDING",
                body_speed_deg_s=body_speed,
                gripper_speed_pct_s=gripper_speed,
            )

        self._stationary_frames = self._stationary_frames + 1 if below_stop else 0
        should_finish = self._stationary_frames >= self._stop_frames
        return MotionCaptureUpdate(
            samples=(sample,),
            just_triggered=False,
            should_finish=should_finish,
            status="COMPLETE" if should_finish else ("POST-ROLL" if self._stationary_frames else "RECORDING"),
            body_speed_deg_s=body_speed,
            gripper_speed_pct_s=gripper_speed,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Record SO-101 demonstrations with a localhost web dashboard. "
            "Control rate, durations, and calibration are selected in the browser before connection."
        )
    )
    parser.add_argument(
        "--task",
        required=True,
        help="One concrete task instruction used for every frame in this dataset.",
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--episodes",
        type=int,
        default=DEFAULT_EPISODES,
        help="Initial value shown in the web settings dialog.",
    )
    parser.add_argument(
        "--episode-seconds",
        type=float,
        default=DEFAULT_EPISODE_SECONDS,
        help="Initial episode duration shown in the web settings dialog.",
    )
    parser.add_argument(
        "--reset-seconds",
        type=float,
        default=DEFAULT_RESET_SECONDS,
        help="Initial reset duration shown in the web settings dialog.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to an existing compatible dataset instead of creating a new one.",
    )
    parser.add_argument(
        "--allow-dark-camera",
        action="store_true",
        help="Bypass the almost-black camera safety gate after visually checking the feed.",
    )
    parser.add_argument("--dashboard-port", type=int, default=DEFAULT_WEB_PORT)
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Start the localhost dashboard without opening the default browser automatically.",
    )
    args = parser.parse_args()
    args.task = args.task.strip()
    if not args.task:
        parser.error("--task must not be empty")
    if args.episodes <= 0:
        parser.error("--episodes must be greater than zero")
    if args.episode_seconds <= 0:
        parser.error("--episode-seconds must be greater than zero")
    if args.reset_seconds < 0:
        parser.error("--reset-seconds must be zero or greater")
    if not 1024 <= args.dashboard_port <= 65535:
        parser.error("--dashboard-port must be between 1024 and 65535")
    if "/" not in args.repo_id or args.repo_id.startswith("/") or args.repo_id.endswith("/"):
        parser.error("--repo-id must have the form namespace/dataset_name")
    if args.repo_id.split("/", 1)[1].startswith("eval_"):
        parser.error("dataset names beginning with 'eval_' are reserved for evaluation")
    return args


def validate_dataset_target(root: Path, resume: bool) -> Path:
    root = root.expanduser().resolve()
    metadata_file = root / "meta" / "info.json"
    if resume and not metadata_file.is_file():
        raise FileNotFoundError(f"Cannot resume: LeRobot metadata was not found at {metadata_file}.")
    if not resume and root.exists():
        raise FileExistsError(
            f"Dataset directory already exists: {root}\n"
            "Use a new --root, or pass --resume to append to this dataset."
        )
    return root


def existing_dataset_info(root: Path) -> dict | None:
    path = root / "meta" / "info.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def make_hardware(follower_port: str, leader_port: str) -> tuple[SO101Follower, SO101Leader]:
    follower = SO101Follower(
        SO101FollowerConfig(
            port=follower_port,
            id="so101_follower_main",
            num_read_retries=NUM_READ_RETRIES,
            cameras={
                "front": OpenCVCameraConfig(
                    index_or_path=CAMERA_INDEX,
                    width=CAMERA_WIDTH,
                    height=CAMERA_HEIGHT,
                    fps=CAMERA_FPS,
                    color_mode=ColorMode.RGB,
                    fourcc=CAMERA_FOURCC,
                    backend=Cv2Backends.DSHOW,
                )
            },
        )
    )
    leader = SO101Leader(
        SO101LeaderConfig(
            port=leader_port,
            id="so101_leader_main",
            num_read_retries=NUM_READ_RETRIES,
        )
    )
    return follower, leader


def make_dataset_features(follower: SO101Follower) -> dict[str, dict]:
    return {
        **hw_to_dataset_features(follower.action_features, ACTION, use_video=True),
        **hw_to_dataset_features(follower.observation_features, OBS_STR, use_video=True),
    }


def open_dataset(
    *,
    repo_id: str,
    root: Path,
    resume: bool,
    follower: SO101Follower,
    features: dict[str, dict],
    fps: int,
) -> LeRobotDataset:
    encoder = RGBEncoderConfig(vcodec="h264", preset="fast")
    common = {
        "root": root,
        "video_backend": "pyav",
        "rgb_encoder": encoder,
        "batch_encoding_size": 1,
        "image_writer_processes": 0,
        "image_writer_threads": 4,
        "streaming_encoding": False,
        "encoder_threads": 2,
    }
    if resume:
        dataset = LeRobotDataset.resume(repo_id, **common)
        sanity_check_dataset_robot_compatibility(dataset, follower, fps, features)
        return dataset
    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        robot_type=follower.name,
        features=features,
        use_videos=True,
        **common,
    )


def check_camera_brightness(follower: SO101Follower, allow_dark_camera: bool) -> float:
    means = [
        float(follower.cameras["front"].read_latest().mean())
        for _ in range(CAMERA_PREFLIGHT_FRAMES)
    ]
    mean_brightness = sum(means) / len(means)
    if mean_brightness < CAMERA_DARK_MEAN_THRESHOLD and not allow_dark_camera:
        raise RuntimeError(
            f"Camera image is almost black ({mean_brightness:.1f}/255). Remove the lens cover, "
            "add lighting/check exposure, and run again. Use --allow-dark-camera only after "
            "visually confirming that a dark scene is intentional."
        )
    return mean_brightness


def validate_runtime_settings(params: dict, root: Path) -> RuntimeSettings:
    def number(name: str) -> float:
        value = params[name]
        if isinstance(value, bool):
            raise ValueError
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError
        return parsed

    try:
        control_fps_value = number("control_fps")
        episodes_value = number("episodes")
        if not control_fps_value.is_integer() or not episodes_value.is_integer():
            raise ValueError
        control_fps = int(control_fps_value)
        episodes = int(episodes_value)
        episode_seconds = number("episode_seconds")
        reset_seconds = number("reset_seconds")
        preview_hz = number("preview_hz")
        calibration = str(params["calibration"])
        motion_triggered = params["motion_triggered"]
        if not isinstance(motion_triggered, bool):
            raise ValueError
        motion_body_threshold_deg_s = number("motion_body_threshold_deg_s")
        motion_gripper_threshold_pct_s = number("motion_gripper_threshold_pct_s")
        motion_pre_roll_s = number("motion_pre_roll_s")
        motion_stop_seconds = number("motion_stop_seconds")
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Invalid web settings payload.") from exc
    if not MIN_CONTROL_FPS <= control_fps <= MAX_CONTROL_FPS:
        raise ValueError(
            f"Control FPS must be {MIN_CONTROL_FPS}-{MAX_CONTROL_FPS}; 20 Hz is recommended."
        )
    if episodes <= 0 or episodes > 1000:
        raise ValueError("Episode count must be between 1 and 1000.")
    if not 0 < episode_seconds <= MAX_EPISODE_SECONDS:
        raise ValueError(f"Episode duration must be 0-{MAX_EPISODE_SECONDS:.0f} seconds.")
    if not 0 <= reset_seconds <= MAX_RESET_SECONDS:
        raise ValueError(f"Reset duration must be 0-{MAX_RESET_SECONDS:.0f} seconds.")
    if not 1 <= preview_hz <= min(10, control_fps):
        raise ValueError("Preview rate must be 1-10 Hz and no greater than control FPS.")
    if calibration not in {"none", "leader", "follower", "both"}:
        raise ValueError("Unknown calibration selection.")
    if not 0.1 <= motion_body_threshold_deg_s <= 180.0:
        raise ValueError("Body motion threshold must be between 0.1 and 180 deg/s.")
    if not 0.1 <= motion_gripper_threshold_pct_s <= 200.0:
        raise ValueError("Gripper motion threshold must be between 0.1 and 200 %/s.")
    if not 0.0 <= motion_pre_roll_s <= MAX_MOTION_PRE_ROLL_S:
        raise ValueError(
            f"Motion pre-roll must be between 0 and {MAX_MOTION_PRE_ROLL_S:g} seconds."
        )
    if not 0.1 <= motion_stop_seconds <= 10.0:
        raise ValueError("Motion stop time must be between 0.1 and 10 seconds.")
    info = existing_dataset_info(root)
    if info is not None and int(info["fps"]) != control_fps:
        raise ValueError(
            f"This dataset is fixed at {info['fps']} FPS. Select {info['fps']} FPS or use a new root."
        )
    if info is not None and int(info.get("total_episodes", 0)) > 0 and calibration != "none":
        raise ValueError(
            "Calibration changes the joint coordinate system. Use a new dataset root before "
            "recalibrating an arm that already has saved episodes."
        )
    return RuntimeSettings(
        control_fps=control_fps,
        episodes=episodes,
        episode_seconds=episode_seconds,
        reset_seconds=reset_seconds,
        preview_hz=preview_hz,
        calibration=calibration,
        motion_capture=MotionCaptureSettings(
            enabled=motion_triggered,
            body_threshold_deg_s=motion_body_threshold_deg_s,
            gripper_threshold_pct_s=motion_gripper_threshold_pct_s,
            pre_roll_s=motion_pre_roll_s,
            stop_after_s=motion_stop_seconds,
        ),
    )


def wait_for_web_settings(
    dashboard: RecordingWebDashboard,
    *,
    root: Path,
    follower: SO101Follower,
    leader: SO101Leader,
) -> RuntimeSettings:
    dashboard.set_phase("BOOT", "웹 설정을 적용하면 포트 연결을 시작합니다.")
    while True:
        commands = dashboard.drain_commands()
        if any(command.action == "stop_collection" for command in commands):
            raise UserCancelled("Cancelled from the web dashboard before hardware connection.")
        for command in commands:
            if command.action != "start_setup":
                dashboard.log(f"BOOT 단계에서는 {command.action} 명령을 사용할 수 없습니다.", "warn")
                continue
            try:
                settings = validate_runtime_settings(command.params or {}, root)
                if settings.calibration == "none" and (not follower.calibration or not leader.calibration):
                    raise ValueError(
                        "Saved calibration is missing. Select the missing arm or both arms for calibration."
                    )
            except (OSError, RuntimeError, ValueError) as exc:
                dashboard.log(str(exc), "error")
                dashboard.set_phase("BOOT", str(exc))
                continue
            dashboard.set_camera_preview_hz(settings.preview_hz)
            dashboard.lock_settings(asdict(settings))
            dashboard.log(
                f"설정 확정: {settings.control_fps} FPS, {settings.episodes} episodes, "
                f"record {settings.episode_seconds:.0f}s, reset {settings.reset_seconds:.0f}s, "
                f"preview {settings.preview_hz:.0f} Hz."
            )
            if settings.motion_capture.enabled:
                dashboard.log(
                    "움직임 감지 녹화 사용: "
                    f"body {settings.motion_capture.body_threshold_deg_s:.1f} deg/s, "
                    f"gripper {settings.motion_capture.gripper_threshold_pct_s:.1f} %/s, "
                    f"pre-roll {settings.motion_capture.pre_roll_s:.2f}s, "
                    f"stop {settings.motion_capture.stop_after_s:.2f}s."
                )
            else:
                dashboard.log("움직임 감지 녹화 사용 안 함: RECORD 전체 시간을 연속 저장합니다.")
            return settings
        time.sleep(0.05)


def calibration_plot_ranges(follower: SO101Follower) -> dict[str, dict[str, float | str]]:
    ranges: dict[str, dict[str, float | str]] = {}
    for motor_name, motor in follower.bus.motors.items():
        calibration = follower.calibration[motor_name]
        if motor.norm_mode is MotorNormMode.RANGE_0_100:
            low, high, unit = 0.0, 100.0, "%"
        elif motor.norm_mode is MotorNormMode.RANGE_M100_100:
            low, high, unit = -100.0, 100.0, "%"
        else:
            resolution = follower.bus.model_resolution_table[motor.model] - 1
            midpoint = (calibration.range_min + calibration.range_max) / 2
            low = (calibration.range_min - midpoint) * 360 / resolution
            high = (calibration.range_max - midpoint) * 360 / resolution
            unit = "deg"
        ranges[f"{motor_name}.pos"] = {
            "min": float(min(low, high)),
            "max": float(max(low, high)),
            "unit": unit,
            "raw_min": float(calibration.range_min),
            "raw_max": float(calibration.range_max),
        }
    return ranges


def calibration_backup_path(device) -> Path | None:
    calibration_path = Path(device.calibration_fpath)
    if not calibration_path.is_file():
        return None
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    backup = calibration_path.with_name(f"{calibration_path.stem}.backup-{timestamp}.json")
    shutil.copy2(calibration_path, backup)
    return backup


def wait_calibration_button(
    dashboard: RecordingWebDashboard,
    *,
    accepted_action: str,
    target: str,
) -> None:
    while True:
        if not dashboard.client_is_alive(CALIBRATION_HEARTBEAT_TIMEOUT_S):
            raise UserCancelled(f"Browser heartbeat was lost during {target} calibration.")
        for command in dashboard.drain_commands():
            if command.action == "stop_collection":
                raise UserCancelled(f"{target} calibration cancelled from dashboard.")
            if command.action == accepted_action:
                return
            dashboard.log(f"현재 calibration 단계에서는 {command.action} 명령을 사용할 수 없습니다.", "warn")
        time.sleep(0.05)


def run_web_calibration(device, *, target: str, port: str, dashboard: RecordingWebDashboard) -> None:
    bus = device.bus
    original_file_calibration = copy.deepcopy(device.calibration)
    original_motor_calibration: dict[str, MotorCalibration] | None = None
    homing_modified = False
    backup_path: Path | None = None
    calibration_path = Path(device.calibration_fpath)
    calibration_temp = calibration_path.with_suffix(".json.tmp")
    calibration_file_replaced = False
    dashboard.set_phase("CALIBRATION", f"{target} calibration 준비")
    dashboard.publish_joint_rows([])
    dashboard.log(f"{target} calibration: {port} 단독 연결을 시작합니다.")
    try:
        bus.connect()
        bus.disable_torque(num_retry=NUM_READ_RETRIES)
        original_motor_calibration = copy.deepcopy(bus.read_calibration())
        for motor in bus.motors:
            torque = int(bus.read("Torque_Enable", motor, normalize=False, num_retry=NUM_READ_RETRIES))
            if torque != 0:
                raise RuntimeError(f"Torque disable was not confirmed for {target} motor {motor}.")
            bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
        dashboard.set_calibration_stage(
            target,
            "center",
            "토크가 해제되었습니다. 암을 손으로 지지한 뒤 모든 관절을 가동 범위 중앙에 놓고 "
            "'중앙 자세 설정 완료'를 누르세요.",
        )
        wait_calibration_button(
            dashboard,
            accepted_action="calibration_continue",
            target=target,
        )
        dashboard.set_calibration_stage(target, "working", "Homing offset을 기록하고 있습니다.")
        homing_modified = True
        homing_offsets = bus.set_half_turn_homings()
        measured_motors = [motor for motor in bus.motors if motor != "wrist_roll"]
        positions = bus.sync_read(
            "Present_Position",
            measured_motors,
            normalize=False,
            num_retry=NUM_READ_RETRIES,
        )
        minimums = positions.copy()
        maximums = positions.copy()
        dashboard.set_calibration_stage(
            target,
            "range",
            "wrist_roll을 제외한 관절을 하나씩 전체 가동 범위로 천천히 움직이세요. "
            "모든 SPAN이 100 tick 이상이고 전체 범위를 확인한 뒤 '범위 기록 완료'를 누르세요.",
        )
        while True:
            if not dashboard.client_is_alive(CALIBRATION_HEARTBEAT_TIMEOUT_S):
                raise UserCancelled(f"Browser heartbeat was lost during {target} range capture.")
            positions = bus.sync_read(
                "Present_Position",
                measured_motors,
                normalize=False,
                num_retry=NUM_READ_RETRIES,
            )
            minimums = {name: min(minimums[name], positions[name]) for name in measured_motors}
            maximums = {name: max(maximums[name], positions[name]) for name in measured_motors}
            rows = []
            for motor in bus.motors:
                if motor == "wrist_roll":
                    pos, low, high = 2047.0, 0.0, 4095.0
                else:
                    pos, low, high = positions[motor], minimums[motor], maximums[motor]
                rows.append(
                    {
                        "joint": motor,
                        "follower": float(pos),
                        "leader": float(low),
                        "command": float(high),
                        "error": float(high - low),
                    }
                )
            dashboard.publish_joint_rows(rows)
            finish_requested = False
            for command in dashboard.drain_commands():
                if command.action == "stop_collection":
                    raise UserCancelled(f"{target} calibration cancelled from dashboard.")
                if command.action == "calibration_finish":
                    finish_requested = True
                else:
                    dashboard.log(
                        f"범위 기록 중에는 {command.action} 명령을 사용할 수 없습니다.", "warn"
                    )
            if finish_requested:
                spans = {name: maximums[name] - minimums[name] for name in measured_motors}
                too_small = {
                    name: span for name, span in spans.items() if span < MIN_CALIBRATION_SPAN_TICKS
                }
                if too_small:
                    dashboard.log(
                        "Calibration 범위가 너무 작습니다: "
                        + ", ".join(f"{name}={span}" for name, span in too_small.items()),
                        "error",
                    )
                else:
                    break
            time.sleep(CALIBRATION_READ_PERIOD_S)
        minimums["wrist_roll"] = 0
        maximums["wrist_roll"] = 4095
        new_calibration = {
            motor_name: MotorCalibration(
                id=motor.id,
                drive_mode=0,
                homing_offset=int(homing_offsets[motor_name]),
                range_min=int(minimums[motor_name]),
                range_max=int(maximums[motor_name]),
            )
            for motor_name, motor in bus.motors.items()
        }
        validate_calibration_spans(target, new_calibration)
        dashboard.set_calibration_stage(target, "working", "모터 EEPROM 기록과 readback 검증 중입니다.")
        backup_path = calibration_backup_path(device)
        bus.write_calibration(new_calibration)
        readback = bus.read_calibration()
        if readback != new_calibration:
            raise RuntimeError(f"{target} calibration readback did not match the values written.")
        device.calibration = new_calibration
        device._save_calibration(calibration_temp)
        calibration_temp.replace(calibration_path)
        calibration_file_replaced = True
        dashboard.log(
            f"{target} calibration 저장 및 검증 완료"
            + (f" (이전 파일: {backup_path.name})" if backup_path else "")
        )
        dashboard.set_calibration_stage(target, "done", "Calibration 저장과 검증이 완료되었습니다.")
    except BaseException:
        rollback_errors: list[str] = []
        if homing_modified and original_motor_calibration:
            try:
                bus.disable_torque(num_retry=NUM_READ_RETRIES)
                bus.write_calibration(original_motor_calibration)
                if bus.read_calibration() != original_motor_calibration:
                    raise RuntimeError("motor calibration rollback verification failed")
                dashboard.log(f"{target} calibration 실패 후 기존 모터 보정값을 복구했습니다.", "warn")
            except Exception as restore_exc:
                rollback_errors.append(f"motor rollback: {restore_exc}")
        device.calibration = original_file_calibration
        if calibration_file_replaced:
            try:
                if backup_path is not None:
                    shutil.copy2(backup_path, calibration_path)
                else:
                    calibration_path.unlink(missing_ok=True)
                dashboard.log(f"{target} calibration 실패 후 기존 보정 파일을 복구했습니다.", "warn")
            except Exception as restore_exc:
                rollback_errors.append(f"file rollback: {restore_exc}")
        try:
            calibration_temp.unlink(missing_ok=True)
        except Exception as cleanup_exc:
            rollback_errors.append(f"temporary file cleanup: {cleanup_exc}")
        if rollback_errors:
            dashboard.log(
                f"{target} calibration 복구 실패 ({'; '.join(rollback_errors)}). "
                "모터 전원을 끄고 다시 보정하세요.",
                "error",
            )
        raise
    finally:
        dashboard.set_calibration_stage(None, None, "")
        dashboard.publish_joint_rows([])
        if bus.is_connected:
            try:
                bus.disconnect(disable_torque=True)
            except Exception as exc:
                dashboard.log(
                    f"{target} calibration port disconnect failed: {exc}. 모터 전원을 끄세요.",
                    "error",
                )


def apply_saved_calibration(device, *, target: str, dashboard: RecordingWebDashboard) -> None:
    if not device.calibration:
        raise RuntimeError(f"No saved calibration is available for {target}.")
    if not device.bus.is_calibrated:
        dashboard.log(f"{target} 모터와 저장 파일이 달라 토크 해제 후 저장값을 적용합니다.", "warn")
        device.bus.disable_torque(num_retry=NUM_READ_RETRIES)
        device.bus.write_calibration(device.calibration)
        if not device.bus.is_calibrated:
            raise RuntimeError(f"{target} saved calibration write/readback verification failed.")


def build_joint_rows(
    observation: dict[str, object],
    leader_action: dict[str, float] | None,
    sent_action: dict[str, float],
) -> list[dict]:
    rows = []
    for key in JOINT_NAMES:
        follower_value = float(observation[key])
        leader_value = float(leader_action[key]) if leader_action is not None else follower_value
        command_value = float(sent_action[key])
        rows.append(
            {
                "joint": key.removesuffix(".pos"),
                "follower": follower_value,
                "leader": leader_value,
                "command": command_value,
                "error": joint_position_delta(key, leader_value, follower_value),
            }
        )
    return rows


def validated_joint_values(values: dict, *, source: str) -> dict[str, float]:
    validated: dict[str, float] = {}
    for key in JOINT_NAMES:
        try:
            raw_value = values[key]
            if isinstance(raw_value, bool):
                raise ValueError
            value = float(raw_value)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError(f"{source} has an invalid value for {key}.") from exc
        if not math.isfinite(value):
            raise RuntimeError(f"{source} has a non-finite value for {key}.")
        validated[key] = value
    return validated


def publish_dashboard_sample(
    dashboard: RecordingWebDashboard,
    *,
    observation: dict[str, object],
    leader_action: dict[str, float] | None,
    sent_action: dict[str, float],
    remaining_s: float,
    control_hz: float,
    buffered_frames: int,
    append_trace: bool = True,
) -> None:
    dashboard.publish_sample(
        camera_rgb=np.asarray(observation["front"]),
        joint_rows=build_joint_rows(observation, leader_action, sent_action),
        state_values=[float(observation[key]) for key in JOINT_NAMES],
        action_values=[float(sent_action[key]) for key in JOINT_NAMES],
        leader_values=(
            [float(leader_action[key]) for key in JOINT_NAMES]
            if leader_action is not None
            else None
        ),
        remaining_s=remaining_s,
        control_hz=control_hz,
        buffered_frames=buffered_frames,
        append_trace=append_trace,
    )


def action_from_observation(observation: dict[str, object]) -> dict[str, float]:
    return validated_joint_values(observation, source="Follower observation")


def wait_for_safe_alignment_web(
    follower: SO101Follower,
    leader: SO101Leader,
    *,
    dashboard: RecordingWebDashboard,
    ranges: dict[str, dict[str, float | str]],
) -> bool:
    dashboard.begin_trace(label="Startup alignment", joint_names=list(JOINT_NAMES), ranges=ranges)
    dashboard.set_phase("ALIGNMENT", "Leader를 follower 현재 자세와 맞추세요.")
    while True:
        observation = follower.get_observation()
        leader_action = leader.get_action()
        follower_values = validated_joint_values(observation, source="Follower alignment observation")
        leader_values = validated_joint_values(leader_action, source="Leader alignment action")
        hold_action = dict(follower_values)
        deltas = {
            key: joint_position_delta(key, leader_values[key], follower_values[key])
            for key in JOINT_NAMES
        }
        angle_safe = all(
            abs(delta) <= MAX_START_ANGLE_DELTA_DEG
            for key, delta in deltas.items()
            if key != "gripper.pos"
        )
        gripper_safe = abs(deltas["gripper.pos"]) <= MAX_START_GRIPPER_DELTA
        safe = angle_safe and gripper_safe
        worst = max(deltas, key=lambda key: abs(deltas[key]))
        message = (
            f"{'안전 범위입니다' if safe else '아직 정렬이 필요합니다'}. "
            f"최대 오차: {worst} {deltas[worst]:+.1f}; "
            f"허용값 body {MAX_START_ANGLE_DELTA_DEG:.0f} deg / gripper {MAX_START_GRIPPER_DELTA:.0f}%."
        )
        dashboard.set_alignment(safe, message)
        publish_dashboard_sample(
            dashboard,
            observation=observation,
            leader_action=leader_values,
            sent_action=hold_action,
            remaining_s=0,
            control_hz=0,
            buffered_frames=0,
        )
        for command in dashboard.drain_commands():
            if command.action == "stop_collection":
                return False
            if command.action == "confirm_alignment":
                if safe:
                    dashboard.log("시작 자세 정렬 확인 완료. follower를 안전하게 활성화합니다.")
                    return True
                dashboard.log("허용 오차 밖이므로 follower를 활성화하지 않았습니다.", "error")
            elif command.action != "recheck_alignment":
                dashboard.log(f"ALIGNMENT 단계에서는 {command.action} 명령을 사용할 수 없습니다.", "warn")
        time.sleep(0.1)


def reset_event_flags(events: dict[str, bool]) -> None:
    events["exit_early"] = False
    events["rerecord_episode"] = False


def handle_episode_target_increase(
    command,
    *,
    dashboard: RecordingWebDashboard,
) -> bool:
    if command.action != "increase_episode_target":
        return False
    try:
        params = command.params or {}
        additional = params["additional_episodes"]
        if isinstance(additional, bool) or not isinstance(additional, int):
            raise ValueError("추가 episode 수는 정수여야 합니다.")
        new_target = dashboard.increase_episode_target(additional)
        dashboard.log(f"수집 목표를 {additional}개 늘렸습니다. 새 목표: {new_target} episodes.")
    except (KeyError, TypeError, ValueError) as exc:
        dashboard.log(f"수집 목표 변경 실패: {exc}", "error")
    return True


def process_segment_commands(
    dashboard: RecordingWebDashboard,
    *,
    events: dict[str, bool],
    recording: bool,
) -> None:
    for command in dashboard.drain_commands():
        if command.action == "stop_collection":
            events["stop_recording"] = True
            events["exit_early"] = True
        elif command.action == "finish_segment":
            events["exit_early"] = True
        elif command.action == "discard_segment":
            if recording:
                events["rerecord_episode"] = True
            events["exit_early"] = True
        elif handle_episode_target_increase(command, dashboard=dashboard):
            continue
        else:
            dashboard.log(f"현재 제어 구간에서는 {command.action} 명령을 사용할 수 없습니다.", "warn")


def run_control_segment(
    *,
    follower: SO101Follower,
    leader: SO101Leader,
    follower_port: str,
    leader_port: str,
    duration_s: float,
    fps: int,
    events: dict[str, bool],
    label: str,
    dashboard: RecordingWebDashboard,
    dataset: LeRobotDataset | None = None,
    task: str | None = None,
    motion_capture: MotionCaptureSettings | None = None,
) -> SegmentResult:
    frame_period = 1.0 / fps
    segment_started = time.perf_counter()
    control_frames = 0
    saved_frames = 0
    states: list[tuple[float, ...]] = []
    actions: list[tuple[float, ...]] = []
    leaders: list[tuple[float, ...]] = []
    saved_sample_times: list[float] = []
    motion_gate = (
        MotionTriggeredCapture(motion_capture, fps=fps)
        if dataset is not None and motion_capture is not None and motion_capture.enabled
        else None
    )
    motion_status = "ARMED" if motion_gate is not None else ("CONTINUOUS" if dataset is not None else "OFF")
    body_speed_deg_s = 0.0
    gripper_speed_pct_s = 0.0
    last_motion_sample_started: float | None = None
    motion_auto_finished = False
    dashboard.set_motion_capture(
        motion_status,
        body_speed_deg_s=body_speed_deg_s,
        gripper_speed_pct_s=gripper_speed_pct_s,
    )
    capture_started_at: float | None = None
    while True:
        timing_started = capture_started_at if capture_started_at is not None else segment_started
        if time.perf_counter() - timing_started >= duration_s:
            break
        process_segment_commands(dashboard, events=events, recording=dataset is not None)
        if events["stop_recording"] or events["exit_early"] or events["rerecord_episode"]:
            break
        frame_started = time.perf_counter()
        motion_sample_interval = (
            frame_period
            if last_motion_sample_started is None
            else max(1e-6, frame_started - last_motion_sample_started)
        )
        last_motion_sample_started = frame_started
        try:
            observation = follower.get_observation()
        except Exception as exc:
            raise ConnectionError(
                f"FOLLOWER READ failed on {follower_port} while reading an observation."
            ) from exc
        try:
            leader_action = leader.get_action()
        except Exception as exc:
            raise ConnectionError(
                f"LEADER READ failed on {leader_port} while reading an action."
            ) from exc
        follower_values = validated_joint_values(observation, source="Follower observation")
        leader_values = validated_joint_values(leader_action, source="Leader action")
        command = validated_joint_values(
            limit_action_step(leader_values, follower_values),
            source="Limited follower command",
        )
        try:
            sent_action = follower.send_action(command)
        except Exception as exc:
            raise ConnectionError(
                f"FOLLOWER WRITE failed on {follower_port} while sending Goal_Position."
            ) from exc
        sent_action = validated_joint_values(sent_action, source="Follower accepted action")
        append_trace = dataset is None or motion_gate is None
        should_finish = False
        if dataset is not None:
            observation_frame = build_dataset_frame(dataset.features, observation, prefix=OBS_STR)
            action_frame = build_dataset_frame(dataset.features, sent_action, prefix=ACTION)
            dataset_frame = {**observation_frame, **action_frame, "task": task}
            if motion_gate is not None and not motion_gate.active:
                dataset_frame = copy.deepcopy(dataset_frame)
            sample = BufferedMotionSample(
                dataset_frame=dataset_frame,
                state_values=tuple(float(observation[key]) for key in JOINT_NAMES),
                action_values=tuple(float(sent_action[key]) for key in JOINT_NAMES),
                leader_values=tuple(float(leader_values[key]) for key in JOINT_NAMES),
                sample_time_s=frame_started,
            )
            if motion_gate is None:
                samples_to_save = (sample,)
            else:
                motion_update = motion_gate.update(
                    follower_values,
                    sample,
                    leader_values=leader_values,
                    dt_s=motion_sample_interval,
                )
                samples_to_save = motion_update.samples
                motion_status = motion_update.status
                body_speed_deg_s = motion_update.body_speed_deg_s
                gripper_speed_pct_s = motion_update.gripper_speed_pct_s
                should_finish = motion_update.should_finish
                append_trace = motion_gate.active and not motion_update.just_triggered
                if motion_update.just_triggered:
                    capture_started_at = samples_to_save[0].sample_time_s
                    dashboard.append_trace_samples(
                        states=[list(item.state_values) for item in samples_to_save],
                        actions=[list(item.action_values) for item in samples_to_save],
                        leaders=[list(item.leader_values) for item in samples_to_save],
                    )
                    dashboard.log(
                        f"움직임 감지: pre-roll을 포함한 {len(samples_to_save)} frames부터 저장합니다."
                    )
            for buffered_sample in samples_to_save:
                dataset.add_frame(buffered_sample.dataset_frame)
                states.append(buffered_sample.state_values)
                actions.append(buffered_sample.action_values)
                leaders.append(buffered_sample.leader_values)
                saved_sample_times.append(buffered_sample.sample_time_s)
            saved_frames += len(samples_to_save)
        control_frames += 1
        now = time.perf_counter()
        elapsed_s = now - segment_started
        timing_started = capture_started_at if capture_started_at is not None else segment_started
        remaining_s = max(0.0, duration_s - (now - timing_started))
        live_rate_hz = control_frames / elapsed_s if elapsed_s > 0 else 0.0
        dashboard.set_motion_capture(
            motion_status,
            body_speed_deg_s=body_speed_deg_s,
            gripper_speed_pct_s=gripper_speed_pct_s,
        )
        publish_dashboard_sample(
            dashboard,
            observation=observation,
            leader_action=leader_values,
            sent_action=sent_action,
            remaining_s=remaining_s,
            control_hz=live_rate_hz,
            buffered_frames=saved_frames,
            append_trace=append_trace,
        )
        if should_finish:
            motion_auto_finished = True
            dashboard.log(
                f"Follower 정지가 {motion_capture.stop_after_s:.2f}s 유지되어 녹화를 자동 완료합니다."
            )
            break
        precise_sleep(max(0.0, frame_period - (time.perf_counter() - frame_started)))
    elapsed_s = time.perf_counter() - segment_started
    if motion_gate is not None:
        if events["stop_recording"] or events["rerecord_episode"]:
            final_status = "ABORTED"
        elif not motion_gate.active:
            final_status = "NO MOTION"
        elif motion_auto_finished or events["exit_early"]:
            final_status = "COMPLETE"
        else:
            final_status = "TIMEOUT"
        dashboard.set_motion_capture(
            final_status,
            body_speed_deg_s=body_speed_deg_s,
            gripper_speed_pct_s=gripper_speed_pct_s,
        )
        if final_status == "TIMEOUT":
            dashboard.log(
                "움직임 구간이 episode 최대 시간에 도달했습니다. 완료 동작인지 확인 후 저장하세요.",
                "warn",
            )
    return SegmentResult(
        saved_frames=saved_frames,
        control_frames=control_frames,
        elapsed_s=elapsed_s,
        rate_hz=control_frames / elapsed_s if elapsed_s > 0 else 0.0,
        saved_rate_hz=calculate_sample_rate_hz(saved_sample_times),
        max_saved_interval_s=calculate_max_sample_interval_s(saved_sample_times),
        states=tuple(states),
        actions=tuple(actions),
        leaders=tuple(leaders),
    )


def wait_for_episode_review(
    *,
    follower: SO101Follower,
    leader: SO101Leader,
    dashboard: RecordingWebDashboard,
    events: dict[str, bool],
    fps: int,
    saved_frames: int,
) -> str:
    dashboard.set_phase("REVIEW", "최종 작업 상태를 확인하고 성공본 저장 또는 폐기를 선택하세요.")
    dashboard.log("REVIEW: 카메라와 최종 상태를 확인한 뒤 저장 여부를 선택하세요.")
    reset_event_flags(events)
    frame_period = 1.0 / fps
    while True:
        if events["stop_recording"]:
            return "stop"
        if events["rerecord_episode"]:
            return "discard"
        if events["exit_early"]:
            return "save"
        for command in dashboard.drain_commands(max_items=1):
            if command.action == "save_episode":
                return "save"
            if command.action == "discard_episode":
                return "discard"
            if command.action == "stop_collection":
                events["stop_recording"] = True
                return "stop"
            if handle_episode_target_increase(command, dashboard=dashboard):
                continue
            dashboard.log(f"REVIEW 단계에서는 {command.action} 명령을 사용할 수 없습니다.", "warn")
        started = time.perf_counter()
        observation = follower.get_observation()
        leader_action = leader.get_action()
        hold_action = action_from_observation(observation)
        leader_action = validated_joint_values(leader_action, source="Leader review action")
        publish_dashboard_sample(
            dashboard,
            observation=observation,
            leader_action=leader_action,
            sent_action=hold_action,
            remaining_s=0,
            control_hz=0,
            buffered_frames=saved_frames,
            append_trace=False,
        )
        precise_sleep(max(0.0, frame_period - (time.perf_counter() - started)))


def hold_follower(
    follower: SO101Follower,
    *,
    follower_port: str,
    dashboard: RecordingWebDashboard,
) -> None:
    try:
        observation = follower.get_observation()
        hold_action = action_from_observation(observation)
        follower.send_action(hold_action)
        dashboard.log("새 명령 전송을 중단하고 follower 현재 자세를 유지합니다.")
    except Exception as exc:
        dashboard.log(
            f"Follower hold failed on {follower_port}: {exc}. 모터 전원을 끄세요.",
            "error",
        )


def run_scenario_playback(
    *,
    scenario_id: str,
    follower: SO101Follower,
    follower_port: str,
    dashboard: RecordingWebDashboard,
    ranges: dict[str, dict[str, float | str]],
    events: dict[str, bool],
) -> None:
    try:
        scenario = dashboard.library.get(scenario_id)
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        dashboard.log(str(exc), "error")
        dashboard.refresh_scenarios()
        return
    if tuple(scenario["joint_names"]) != JOINT_NAMES:
        dashboard.log(f"{scenario_id}: 관절 schema가 현재 follower와 다릅니다.", "error")
        return
    if not scenario["actions"]:
        dashboard.log(f"{scenario_id}: 재생할 action이 없습니다.", "error")
        return
    fps = int(scenario["fps"])
    if not MIN_CONTROL_FPS <= fps <= MAX_CONTROL_FPS:
        dashboard.log(
            f"{scenario_id}: 재생 FPS {fps}가 안전 범위 {MIN_CONTROL_FPS}-{MAX_CONTROL_FPS} 밖입니다.",
            "error",
        )
        return
    observation = follower.get_observation()
    try:
        current_pose = validated_joint_values(observation, source="Follower playback start observation")
    except RuntimeError as exc:
        dashboard.log(f"{scenario_id}: {exc}", "error")
        return
    first_action = {
        key: float(value) for key, value in zip(JOINT_NAMES, scenario["actions"][0], strict=True)
    }
    deltas = {
        key: joint_position_delta(key, first_action[key], current_pose[key])
        for key in JOINT_NAMES
    }
    unsafe = {
        key: delta
        for key, delta in deltas.items()
        if abs(delta)
        > (MAX_START_GRIPPER_DELTA if key == "gripper.pos" else MAX_START_ANGLE_DELTA_DEG)
    }
    if unsafe:
        dashboard.log(
            f"{scenario_id} 실행 거부: 첫 자세 오차가 큽니다 ("
            + ", ".join(f"{key}={value:+.1f}" for key, value in unsafe.items())
            + "). Leader로 follower를 첫 자세 근처에 맞추세요.",
            "error",
        )
        return
    for step_index, row in enumerate(scenario["actions"]):
        for key, value in zip(JOINT_NAMES, row, strict=True):
            joint_range = ranges[key]
            numeric_value = float(value)
            if (
                not math.isfinite(numeric_value)
                or not float(joint_range["min"]) <= numeric_value <= float(joint_range["max"])
            ):
                dashboard.log(
                    f"{scenario_id} 실행 거부: step {step_index} {key}={value}가 calibration 범위 밖입니다.",
                    "error",
                )
                return
    total_steps = len(scenario["actions"])
    dashboard.begin_trace(
        label=f"Playback {scenario_id}",
        joint_names=list(JOINT_NAMES),
        ranges=ranges,
    )
    dashboard.set_playback_progress(0, total_steps)
    dashboard.set_phase(
        "PLAYBACK",
        f"{scenario_id} 저장 action 재생 중 — 정책 추론이 아닙니다.",
        scenario_running=scenario_id,
    )
    dashboard.log(f"{scenario_id}: {total_steps} steps @ {fps} FPS 재생 시작", "warn")
    stopped = False
    playback_started = time.perf_counter()
    playback_frames = 0
    playback_outcome = "오류로 중단"
    try:
        for step_index, action_values in enumerate(scenario["actions"]):
            frame_started = time.perf_counter()
            if not dashboard.client_is_alive(PLAYBACK_HEARTBEAT_TIMEOUT_S):
                dashboard.log("브라우저 heartbeat가 끊겨 시나리오 재생을 정지합니다.", "error")
                stopped = True
                break
            for command in dashboard.drain_commands():
                if command.action == "stop_scenario":
                    stopped = True
                elif command.action == "stop_collection":
                    stopped = True
                    events["stop_recording"] = True
                elif handle_episode_target_increase(command, dashboard=dashboard):
                    continue
                else:
                    dashboard.log(f"PLAYBACK 중에는 {command.action} 명령을 사용할 수 없습니다.", "warn")
            if stopped:
                break
            observation = follower.get_observation()
            follower_values = validated_joint_values(
                observation,
                source=f"Follower playback observation at step {step_index}",
            )
            target = {
                key: float(value)
                for key, value in zip(JOINT_NAMES, action_values, strict=True)
            }
            command = validated_joint_values(
                limit_action_step(target, follower_values),
                source=f"Limited playback command at step {step_index}",
            )
            sent_action = follower.send_action(command)
            sent_action = validated_joint_values(
                sent_action,
                source=f"Follower accepted playback action at step {step_index}",
            )
            playback_frames += 1
            dashboard.set_playback_progress(playback_frames, total_steps)
            playback_elapsed = time.perf_counter() - playback_started
            publish_dashboard_sample(
                dashboard,
                observation=observation,
                leader_action=target,
                sent_action=sent_action,
                remaining_s=(len(scenario["actions"]) - step_index - 1) / fps,
                control_hz=(playback_frames / playback_elapsed if playback_elapsed > 0 else 0.0),
                buffered_frames=step_index + 1,
            )
            precise_sleep(max(0.0, 1.0 / fps - (time.perf_counter() - frame_started)))
        playback_outcome = "정지" if stopped else "재생 완료"
    finally:
        hold_follower(follower, follower_port=follower_port, dashboard=dashboard)
        terminal_detail = f"{scenario_id} {playback_outcome}"
        dashboard.finish_playback(terminal_detail)
        dashboard.log(
            terminal_detail,
            "error" if playback_outcome == "오류로 중단" else "info",
        )


def service_idle_mode(
    *,
    follower: SO101Follower,
    leader: SO101Leader,
    follower_port: str,
    leader_port: str,
    dashboard: RecordingWebDashboard,
    ranges: dict[str, dict[str, float | str]],
    events: dict[str, bool],
    fps: int,
    saved_in_run: int,
    allow_record: bool,
) -> str:
    phase = "READY" if allow_record else "LIBRARY"
    detail = (
        "작업물을 배치하고 Leader로 시작 자세를 맞춘 뒤 녹화 시작을 누르세요."
        if allow_record
        else "목표 episode 저장 완료. 더 수집하려면 '목표 늘리기', 끝내려면 '프로그램 종료'를 누르세요."
    )
    dashboard.set_phase(
        phase,
        detail,
        accepted_progress=f"{saved_in_run}/{dashboard.collection_target()}",
    )
    dashboard.set_motion_capture("OFF", body_speed_deg_s=0.0, gripper_speed_pct_s=0.0)
    dashboard.begin_trace(label=f"{phase} manual control", joint_names=list(JOINT_NAMES), ranges=ranges)
    dashboard.log(f"{phase}: {detail}")
    reset_event_flags(events)
    frame_period = 1.0 / fps
    report_started = time.perf_counter()
    report_frames = 0
    browser_was_alive = True
    while not events["stop_recording"]:
        for command in dashboard.drain_commands(max_items=1):
            if command.action == "stop_collection":
                events["stop_recording"] = True
                return "stop"
            if command.action == "start_record":
                if allow_record:
                    return "record"
                dashboard.log("이번 실행의 목표 episode 수를 이미 채웠습니다.", "warn")
            elif handle_episode_target_increase(command, dashboard=dashboard):
                return "goal_changed"
            elif command.action == "run_scenario" and command.scenario_id:
                run_scenario_playback(
                    scenario_id=command.scenario_id,
                    follower=follower,
                    follower_port=follower_port,
                    dashboard=dashboard,
                    ranges=ranges,
                    events=events,
                )
                if events["stop_recording"]:
                    return "stop"
                if (saved_in_run < dashboard.collection_target()) != allow_record:
                    return "goal_changed"
                dashboard.set_collection_progress(saved_in_run)
                dashboard.set_phase(
                    phase,
                    detail,
                    accepted_progress=f"{saved_in_run}/{dashboard.collection_target()}",
                )
                dashboard.begin_trace(
                    label=f"{phase} manual control",
                    joint_names=list(JOINT_NAMES),
                    ranges=ranges,
                )
            elif command.action == "delete_scenario" and command.scenario_id:
                try:
                    removed = dashboard.library.delete(command.scenario_id)
                    dashboard.refresh_scenarios()
                    dashboard.log(
                        f"{removed['id']}를 재생 목록에서 제거했습니다. LeRobot episode는 유지됩니다.",
                        "warn",
                    )
                except (FileNotFoundError, ValueError) as exc:
                    dashboard.log(str(exc), "error")
            elif command.action != "stop_scenario":
                dashboard.log(f"{phase} 단계에서는 {command.action} 명령을 사용할 수 없습니다.", "warn")
        if events["stop_recording"]:
            return "stop"
        if events["exit_early"] and not events["rerecord_episode"] and allow_record:
            return "record"
        frame_started = time.perf_counter()
        try:
            observation = follower.get_observation()
        except Exception as exc:
            raise ConnectionError(f"FOLLOWER READ failed on {follower_port} in {phase}.") from exc
        try:
            leader_action = leader.get_action()
        except Exception as exc:
            raise ConnectionError(f"LEADER READ failed on {leader_port} in {phase}.") from exc
        follower_values = validated_joint_values(observation, source=f"Follower {phase} observation")
        leader_values = validated_joint_values(leader_action, source=f"Leader {phase} action")
        browser_alive = dashboard.client_is_alive(PLAYBACK_HEARTBEAT_TIMEOUT_S)
        if browser_alive:
            command = limit_action_step(leader_values, follower_values)
        else:
            command = dict(follower_values)
            if browser_was_alive:
                dashboard.log("브라우저 연결이 끊겨 manual follower 명령을 hold로 전환했습니다.", "warn")
        browser_was_alive = browser_alive
        command = validated_joint_values(command, source=f"Limited follower command in {phase}")
        try:
            sent_action = follower.send_action(command)
        except Exception as exc:
            raise ConnectionError(f"FOLLOWER WRITE failed on {follower_port} in {phase}.") from exc
        sent_action = validated_joint_values(sent_action, source=f"Follower accepted action in {phase}")
        report_frames += 1
        now = time.perf_counter()
        elapsed_report = now - report_started
        rate_hz = report_frames / elapsed_report if elapsed_report > 0 else 0.0
        publish_dashboard_sample(
            dashboard,
            observation=observation,
            leader_action=leader_values,
            sent_action=sent_action,
            remaining_s=0,
            control_hz=rate_hz,
            buffered_frames=0,
        )
        precise_sleep(max(0.0, frame_period - (time.perf_counter() - frame_started)))
    return "stop"


def record_episodes_web(
    *,
    dataset: LeRobotDataset,
    follower: SO101Follower,
    leader: SO101Leader,
    follower_port: str,
    leader_port: str,
    task: str,
    settings: RuntimeSettings,
    events: dict[str, bool],
    dashboard: RecordingWebDashboard,
    ranges: dict[str, dict[str, float | str]],
) -> int:
    saved_in_this_run = 0
    dashboard.set_collection_progress(saved_in_this_run)
    dashboard.refresh_scenarios()
    while not events["stop_recording"]:
        target_episodes = dashboard.collection_target()
        allow_record = saved_in_this_run < target_episodes
        idle_result = service_idle_mode(
            follower=follower,
            leader=leader,
            follower_port=follower_port,
            leader_port=leader_port,
            dashboard=dashboard,
            ranges=ranges,
            events=events,
            fps=settings.control_fps,
            saved_in_run=saved_in_this_run,
            allow_record=allow_record,
        )
        if idle_result == "stop":
            break
        if idle_result == "goal_changed":
            continue
        next_index = dataset.num_episodes
        reset_event_flags(events)
        dashboard.begin_trace(
            label=f"RECORD episode {next_index}",
            joint_names=list(JOINT_NAMES),
            ranges=ranges,
        )
        dashboard.set_phase(
            "RECORD",
            f"Episode {next_index}: {task}",
            is_recording=True,
            accepted_progress=f"{saved_in_this_run}/{dashboard.collection_target()}",
        )
        dashboard.log(
            f"Episode {next_index} 녹화 시작: 최대 {settings.episode_seconds:.0f}s @ "
            f"{settings.control_fps} FPS"
            + (
                "; follower 움직임을 기다리는 중"
                if settings.motion_capture.enabled
                else "; 전체 구간 연속 저장"
            )
        )
        result = run_control_segment(
            follower=follower,
            leader=leader,
            follower_port=follower_port,
            leader_port=leader_port,
            duration_s=settings.episode_seconds,
            fps=settings.control_fps,
            events=events,
            label=f"RECORD {next_index}",
            dashboard=dashboard,
            dataset=dataset,
            task=task,
            motion_capture=settings.motion_capture,
        )
        if events["stop_recording"]:
            dataset.clear_episode_buffer()
            dashboard.log(f"Episode {next_index} 미저장 buffer를 폐기하고 종료합니다.", "warn")
            break
        minimum_rate = settings.control_fps * MIN_RECORDING_RATE_RATIO
        save_candidate = True
        if events["rerecord_episode"]:
            save_candidate = False
            dashboard.log(f"Episode {next_index}를 폐기했습니다.", "warn")
        elif result.saved_frames <= 0:
            save_candidate = False
            dashboard.log(f"Episode {next_index}에 frame이 없어 폐기했습니다.", "warn")
        elif result.max_saved_interval_s > MAX_RECORDING_FRAME_GAP_PERIODS / settings.control_fps:
            save_candidate = False
            dashboard.log(
                f"Episode {next_index} 폐기: 저장 구간 최대 frame 간격 "
                f"{result.max_saved_interval_s * 1000:.1f} ms가 허용값 "
                f"{MAX_RECORDING_FRAME_GAP_PERIODS / settings.control_fps * 1000:.1f} ms를 넘었습니다.",
                "error",
            )
        elif result.saved_rate_hz < minimum_rate:
            save_candidate = False
            dashboard.log(
                f"Episode {next_index} 폐기: 저장 구간 {result.saved_rate_hz:.1f} Hz "
                f"< 품질 기준 {minimum_rate:.1f} Hz.",
                "error",
            )
        if not save_candidate:
            dataset.clear_episode_buffer()
        else:
            decision = wait_for_episode_review(
                follower=follower,
                leader=leader,
                dashboard=dashboard,
                events=events,
                fps=settings.control_fps,
                saved_frames=result.saved_frames,
            )
            if decision == "save":
                dataset.save_episode()
                saved_in_this_run += 1
                dashboard.set_collection_progress(saved_in_this_run)
                try:
                    dashboard.library.save(
                        episode_index=next_index,
                        task=task,
                        fps=settings.control_fps,
                        joint_names=list(JOINT_NAMES),
                        ranges=ranges,
                        states=[list(row) for row in result.states],
                        actions=[list(row) for row in result.actions],
                        leaders=[list(row) for row in result.leaders],
                        capture_settings=asdict(settings.motion_capture),
                    )
                    dashboard.refresh_scenarios()
                except Exception as exc:
                    dashboard.log(
                        f"Episode은 학습 데이터에 저장됐지만 재생 시나리오 생성에 실패했습니다: {exc}",
                        "error",
                    )
                dashboard.log(
                    f"Episode {next_index} 저장 완료: {result.saved_frames} frames, "
                    f"저장 {result.saved_rate_hz:.1f} Hz / 제어 {result.rate_hz:.1f} Hz, run "
                    f"{saved_in_this_run}/{dashboard.collection_target()}."
                )
            else:
                dataset.clear_episode_buffer()
                dashboard.log(f"Episode {next_index}를 저장하지 않았습니다.", "warn")
                if decision == "stop":
                    break
        reset_event_flags(events)
        if events["stop_recording"]:
            break
        if settings.reset_seconds > 0:
            dashboard.begin_trace(
                label="RESET (not recorded)",
                joint_names=list(JOINT_NAMES),
                ranges=ranges,
            )
            dashboard.set_phase(
                "RESET",
                "Leader로 follower와 작업물을 시작 상태로 되돌리세요. RESET은 저장되지 않습니다.",
                accepted_progress=f"{saved_in_this_run}/{dashboard.collection_target()}",
            )
            run_control_segment(
                follower=follower,
                leader=leader,
                follower_port=follower_port,
                leader_port=leader_port,
                duration_s=settings.reset_seconds,
                fps=settings.control_fps,
                events=events,
                label="RESET",
                dashboard=dashboard,
            )
            reset_event_flags(events)
    return saved_in_this_run


def write_hardware_manifest(
    root: Path,
    *,
    follower: SO101Follower,
    leader: SO101Leader,
    settings: RuntimeSettings,
) -> None:
    path = root / "meta" / "so101_hardware.json"
    payload = {
        "schema_version": 1,
        "follower_usb_serial": FOLLOWER_USB_SERIAL,
        "leader_usb_serial": LEADER_USB_SERIAL,
        "control_fps": settings.control_fps,
        "follower_calibration": {name: asdict(value) for name, value in follower.calibration.items()},
        "leader_calibration": {name: asdict(value) for name, value in leader.calibration.items()},
    }
    serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            info = existing_dataset_info(root) or {}
            if int(info.get("total_frames", 0)) > 0:
                raise RuntimeError(
                    "Hardware/calibration manifest differs from this dataset. Use the original arms and "
                    "calibration, or create a new dataset root."
                )
        else:
            return
    temp_path = path.with_suffix(".json.tmp")
    temp_path.write_text(serialized, encoding="utf-8")
    temp_path.replace(path)


def main() -> int:
    args = parse_args()
    dataset_root = validate_dataset_target(args.root, args.resume)
    follower_port = find_port(FOLLOWER_USB_SERIAL)
    leader_port = find_port(LEADER_USB_SERIAL)
    if follower_port == leader_port:
        raise RuntimeError("Leader and follower resolved to the same serial port.")
    follower, leader = make_hardware(follower_port, leader_port)
    dashboard = RecordingWebDashboard(
        dataset_root=dataset_root,
        port=args.dashboard_port,
        open_browser=not args.no_browser,
    )
    info = existing_dataset_info(dataset_root)
    dashboard.set_setup_defaults(
        {
            "control_fps": info.get("fps", FPS) if info else FPS,
            "episodes": args.episodes,
            "episode_seconds": args.episode_seconds,
            "reset_seconds": args.reset_seconds,
            "preview_hz": DEFAULT_PREVIEW_HZ,
            "motion_triggered": DEFAULT_MOTION_TRIGGERED,
            "motion_body_threshold_deg_s": DEFAULT_MOTION_BODY_THRESHOLD_DEG_S,
            "motion_gripper_threshold_pct_s": DEFAULT_MOTION_GRIPPER_THRESHOLD_PCT_S,
            "motion_pre_roll_s": DEFAULT_MOTION_PRE_ROLL_S,
            "motion_stop_seconds": DEFAULT_MOTION_STOP_SECONDS,
        }
    )
    print(f"Follower : {follower_port} (USB serial {FOLLOWER_USB_SERIAL})")
    print(f"Leader   : {leader_port} (USB serial {LEADER_USB_SERIAL})")
    print(f"Dataset  : {dataset_root} ({'append' if args.resume else 'new'})")
    print(f"Dashboard: {dashboard.url} (localhost only)")
    print("Keep both motor power supplies and USB cables connected during shutdown.")
    leader_connected = False
    follower_connected = False
    connected_camera_names: list[str] = []
    listener = None
    dataset: LeRobotDataset | None = None
    primary_error: BaseException | None = None
    saved_in_this_run = 0
    try:
        dashboard.prepare()
        settings = wait_for_web_settings(
            dashboard,
            root=dataset_root,
            follower=follower,
            leader=leader,
        )
        calibration_targets = {
            "leader": [(leader, "Leader", leader_port)],
            "follower": [(follower, "Follower", follower_port)],
            "both": [(leader, "Leader", leader_port), (follower, "Follower", follower_port)],
        }.get(settings.calibration, [])
        for device, target, port in calibration_targets:
            run_web_calibration(device, target=target, port=port, dashboard=dashboard)
        validate_calibration_spans("Follower", follower.calibration)
        validate_calibration_spans("Leader", leader.calibration)
        dashboard.set_phase("CONNECTING", "Leader 포트와 calibration을 확인합니다.")
        leader.bus.connect()
        leader_connected = True
        leader.bus.disable_torque(num_retry=NUM_READ_RETRIES)
        apply_saved_calibration(leader, target="Leader", dashboard=dashboard)
        leader.configure()
        dashboard.set_phase("CONNECTING", "Follower 포트와 camera를 확인합니다.")
        follower.bus.connect()
        follower.bus.disable_torque(num_retry=NUM_READ_RETRIES)
        apply_saved_calibration(follower, target="Follower", dashboard=dashboard)
        connected_camera_names = connect_follower_cameras(follower)
        mean_brightness = check_camera_brightness(follower, args.allow_dark_camera)
        dashboard.log(f"Camera preflight: mean brightness {mean_brightness:.1f}/255")
        ranges = calibration_plot_ranges(follower)
        if not wait_for_safe_alignment_web(
            follower,
            leader,
            dashboard=dashboard,
            ranges=ranges,
        ):
            raise UserCancelled("Cancelled before follower motion was enabled.")
        safely_enable_follower(follower, follower_port)
        follower_connected = True
        features = make_dataset_features(follower)
        dataset = open_dataset(
            repo_id=args.repo_id,
            root=dataset_root,
            resume=args.resume,
            follower=follower,
            features=features,
            fps=settings.control_fps,
        )
        write_hardware_manifest(
            dataset_root,
            follower=follower,
            leader=leader,
            settings=settings,
        )
        with VideoEncodingManager(dataset):
            listener, events = init_keyboard_listener()
            saved_in_this_run = record_episodes_web(
                dataset=dataset,
                follower=follower,
                leader=leader,
                follower_port=follower_port,
                leader_port=leader_port,
                task=args.task,
                settings=settings,
                events=events,
                dashboard=dashboard,
                ranges=ranges,
            )
    except UserCancelled as exc:
        print(str(exc))
    except KeyboardInterrupt:
        print("Stopping recording; the unfinished episode will not be saved.")
    except BaseException as exc:
        primary_error = exc
        try:
            dashboard.set_phase("ERROR", str(exc))
            dashboard.log(str(exc), "error")
        except Exception:
            pass
    finally:
        disconnect_errors: list[str] = []
        if listener is not None:
            try:
                listener.stop()
            except Exception as exc:
                disconnect_errors.append(f"Keyboard listener stop failed: {exc}")
        if follower_connected:
            try:
                follower.disconnect()
            except Exception as exc:
                disconnect_errors.append(f"Follower disconnect failed on {follower_port}: {exc}")
        for name in reversed(connected_camera_names):
            camera = follower.cameras[name]
            if camera.is_connected:
                try:
                    camera.disconnect()
                except Exception as exc:
                    disconnect_errors.append(f"Camera {name!r} disconnect failed: {exc}")
        if follower.bus.is_connected:
            try:
                follower.bus.disconnect(disable_torque=True)
            except Exception as exc:
                disconnect_errors.append(f"Follower bus disconnect failed on {follower_port}: {exc}")
        if leader_connected:
            try:
                leader.disconnect()
            except Exception as exc:
                disconnect_errors.append(f"Leader disconnect failed on {leader_port}: {exc}")
        try:
            dashboard.close()
        except Exception as exc:
            disconnect_errors.append(f"Web dashboard close failed: {exc}")
        for error in disconnect_errors:
            print(error, file=sys.stderr)
        if disconnect_errors:
            print("Torque release was not confirmed. Turn off follower motor power.", file=sys.stderr)
    if primary_error is not None:
        raise primary_error
    if dataset is not None:
        print(
            f"Recording complete: {saved_in_this_run} new episode(s), "
            f"{dataset.num_episodes} total at {dataset_root}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
