"""Run diagnosed SO-101 teleoperation with stable USB port identification."""

from __future__ import annotations

import sys
import time

from lerobot.cameras import ColorMode, Cv2Backends
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
from serial.tools import list_ports


FOLLOWER_USB_SERIAL = "5B3D041258"
LEADER_USB_SERIAL = "5AAF218987"
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
CAMERA_FOURCC = "YUY2"
CAMERA_DARK_MEAN_THRESHOLD = 10.0
FPS = 20
NUM_READ_RETRIES = 5
MAX_START_ANGLE_DELTA_DEG = 20.0
MAX_START_GRIPPER_DELTA = 20.0
MIN_CALIBRATION_SPAN_TICKS = 100
MAX_RELATIVE_TARGET_DEG = {
    "shoulder_pan.pos": 2.0,
    "shoulder_lift.pos": 2.0,
    # The elbow carries more gravity load. A 2 degree position error can be too
    # small to start moving with the default P coefficient, so allow 5 degrees.
    "elbow_flex.pos": 5.0,
    "wrist_flex.pos": 2.0,
    "wrist_roll.pos": 2.0,
}
MAX_GRIPPER_RELATIVE_TARGET = 5.0
MOTOR_IDS = {
    "shoulder_pan.pos": 1,
    "shoulder_lift.pos": 2,
    "elbow_flex.pos": 3,
    "wrist_flex.pos": 4,
    "wrist_roll.pos": 5,
    "gripper.pos": 6,
}


def find_port(usb_serial: str) -> str:
    matches = [port for port in list_ports.comports() if port.serial_number == usb_serial]
    if len(matches) == 1:
        return matches[0].device

    detected = [
        f"{port.device}: serial={port.serial_number}, description={port.description}"
        for port in list_ports.comports()
    ]
    details = "\n".join(detected) if detected else "No serial ports detected."
    raise RuntimeError(
        f"Expected exactly one device with USB serial {usb_serial!r}, found {len(matches)}.\n"
        f"Detected ports:\n{details}"
    )


def validate_calibration_spans(role: str, calibration: dict) -> None:
    too_small = {
        motor: values.range_max - values.range_min
        for motor, values in calibration.items()
        if values.range_max - values.range_min < MIN_CALIBRATION_SPAN_TICKS
    }
    if too_small:
        details = ", ".join(f"{motor}={span} ticks" for motor, span in too_small.items())
        raise RuntimeError(
            f"{role} calibration range is too small: {details}. "
            "Recalibrate and move every prompted joint through its full mechanical range."
        )


def connect_follower_cameras(follower: SO101Follower) -> list[str]:
    """Connect and validate cameras before follower torque can be enabled."""
    connected: list[str] = []
    try:
        for name, camera in follower.cameras.items():
            camera.connect()
            connected.append(name)
            frame = camera.read_latest()
            mean_brightness = float(frame.mean())
            print(
                f"Camera {name!r}: connected, shape={frame.shape}, "
                f"mean_brightness={mean_brightness:.1f}/255"
            )
            if mean_brightness < CAMERA_DARK_MEAN_THRESHOLD:
                print(
                    f"WARNING: Camera {name!r} is almost black. "
                    "Remove the lens cover and check lighting/exposure before recording data."
                )
    except BaseException:
        for name in reversed(connected):
            camera = follower.cameras[name]
            if camera.is_connected:
                try:
                    camera.disconnect()
                except Exception:
                    pass
        raise
    return connected


def run_control_loop(
    follower: SO101Follower,
    leader: SO101Leader,
    follower_port: str,
    leader_port: str,
) -> None:
    frame_period = 1 / FPS
    frame_count = 0
    report_started = time.perf_counter()

    while True:
        frame_started = time.perf_counter()

        try:
            observation = follower.get_observation()
        except Exception as exc:
            raise ConnectionError(
                f"FOLLOWER READ failed on {follower_port} while reading Present_Position."
            ) from exc

        try:
            action = leader.get_action()
        except Exception as exc:
            raise ConnectionError(
                f"LEADER READ failed on {leader_port} while reading Present_Position."
            ) from exc

        try:
            safe_action = limit_action_step(action, observation)
            follower.send_action(safe_action)
        except Exception as exc:
            raise ConnectionError(
                f"FOLLOWER WRITE failed on {follower_port} while sending Goal_Position."
            ) from exc

        frame_count += 1
        now = time.perf_counter()
        if now - report_started >= 1.0:
            rate = frame_count / (now - report_started)
            print(f"\nTeleoperation running: {rate:.1f} Hz")
            print("ID  JOINT                 FOLLOWER     LEADER        CMD      ERROR")
            for key, target in action.items():
                current = observation[key]
                error = joint_position_delta(key, target, current)
                print(
                    f"{MOTOR_IDS[key]:>2}  {key.removesuffix('.pos'):<18} "
                    f"{current:>10.2f} {target:>10.2f} {safe_action[key]:>10.2f} {error:>10.2f}"
                )
            frame_count = 0
            report_started = now

        time.sleep(max(0.0, frame_period - (time.perf_counter() - frame_started)))


def shortest_angle_delta(target: float, current: float) -> float:
    return (target - current + 180.0) % 360.0 - 180.0


def joint_position_delta(key: str, target: float, current: float) -> float:
    """Return a linear joint delta, wrapping only the continuous wrist roll."""
    if key == "wrist_roll.pos":
        return shortest_angle_delta(target, current)
    return target - current


def limit_action_step(action: dict[str, float], observation: dict[str, float]) -> dict[str, float]:
    limited: dict[str, float] = {}
    for key, target in action.items():
        current = observation[key]
        if key == "gripper.pos":
            max_delta = MAX_GRIPPER_RELATIVE_TARGET
        else:
            max_delta = MAX_RELATIVE_TARGET_DEG[key]
        raw_delta = joint_position_delta(key, target, current)
        delta = max(-max_delta, min(max_delta, raw_delta))
        limited[key] = current + delta
    return limited


def safely_enable_follower(follower: SO101Follower, follower_port: str) -> None:
    """Set the current pose as the servo goal before follower torque is enabled."""
    if not follower.is_calibrated:
        raise RuntimeError(
            "Follower motor calibration does not match the saved calibration file. "
            "Run lerobot-calibrate for the follower before teleoperation."
        )

    try:
        current_raw = follower.bus.sync_read(
            "Present_Position", normalize=False, num_retry=NUM_READ_RETRIES
        )
        follower.bus.disable_torque(num_retry=NUM_READ_RETRIES)
        follower.bus.sync_write(
            "Goal_Position",
            current_raw,
            normalize=False,
            num_retry=NUM_READ_RETRIES,
        )
        follower.configure()
    except Exception as exc:
        raise ConnectionError(
            f"FOLLOWER SAFE ENABLE failed on {follower_port}; torque may not be in a known state."
        ) from exc


def wait_for_safe_alignment(follower: SO101Follower, leader: SO101Leader) -> bool:
    """Require the passive leader pose to match the follower before enabling motion."""
    while True:
        try:
            follower_pose = follower.get_observation()
        except Exception as exc:
            raise ConnectionError("FOLLOWER READ failed during startup alignment.") from exc
        try:
            leader_pose = leader.get_action()
        except Exception as exc:
            raise ConnectionError("LEADER READ failed during startup alignment.") from exc

        deltas: dict[str, float] = {}
        for key, target in leader_pose.items():
            current = follower_pose[key]
            deltas[key] = joint_position_delta(key, target, current)

        print("\nStartup pose difference (leader target - follower current):")
        for key, delta in deltas.items():
            unit = "%" if key == "gripper.pos" else "deg"
            print(f"  {key:<20} {delta:>8.2f} {unit}")

        angle_safe = all(
            abs(delta) <= MAX_START_ANGLE_DELTA_DEG
            for key, delta in deltas.items()
            if key != "gripper.pos"
        )
        gripper_safe = abs(deltas["gripper.pos"]) <= MAX_START_GRIPPER_DELTA
        if angle_safe and gripper_safe:
            answer = input("Alignment is safe. Press Enter to enable follower motion (q to cancel): ")
            return answer.strip().lower() != "q"

        print(
            "Follower motion remains disabled. Move the passive leader to match the follower pose "
            f"(limits: {MAX_START_ANGLE_DELTA_DEG:.0f} deg, {MAX_START_GRIPPER_DELTA:.0f}% gripper)."
        )
        answer = input("Press Enter to check alignment again (q to cancel): ")
        if answer.strip().lower() == "q":
            return False


def main() -> int:
    follower_port = find_port(FOLLOWER_USB_SERIAL)
    leader_port = find_port(LEADER_USB_SERIAL)
    if follower_port == leader_port:
        raise RuntimeError("Leader and follower resolved to the same serial port.")

    print(f"Follower: {follower_port} (USB serial {FOLLOWER_USB_SERIAL})")
    print(f"Leader  : {leader_port} (USB serial {LEADER_USB_SERIAL})")
    print(f"Control : {FPS} FPS, {NUM_READ_RETRIES} additional read retries")
    print("Limits  : elbow 5 deg, other joints 2 deg, gripper 5% from current position")
    print("Keep both motor power supplies and USB cables connected during shutdown.")
    answer = input("Clear the follower workspace, then press Enter to start (q to cancel): ")
    if answer.strip().lower() == "q":
        print("Cancelled.")
        return 0

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
    validate_calibration_spans("Follower", follower.calibration)
    validate_calibration_spans("Leader", leader.calibration)

    leader_connected = False
    follower_connected = False
    follower_bus_connected = False
    connected_camera_names: list[str] = []
    primary_error: BaseException | None = None

    try:
        try:
            leader.connect()
            leader_connected = True
        except Exception as exc:
            raise ConnectionError(f"LEADER CONNECT failed on {leader_port}.") from exc

        # Open the follower bus without configuring or enabling torque. This lets us
        # compare both poses before the first action can move the follower abruptly.
        try:
            follower.bus.connect()
            follower_bus_connected = True
        except Exception as exc:
            raise ConnectionError(f"FOLLOWER PREFLIGHT CONNECT failed on {follower_port}.") from exc

        try:
            connected_camera_names = connect_follower_cameras(follower)
        except Exception as exc:
            raise ConnectionError(
                f"CAMERA CONNECT failed for OpenCV index {CAMERA_INDEX}."
            ) from exc

        if not wait_for_safe_alignment(follower, leader):
            print("Cancelled before follower motion was enabled.")
            return 0

        safely_enable_follower(follower, follower_port)
        follower_connected = True
        follower_bus_connected = False

        print("Teleoperation started. Press Ctrl+C once to stop safely.")
        run_control_loop(follower, leader, follower_port, leader_port)
    except KeyboardInterrupt:
        print("Stopping teleoperation...")
    except BaseException as exc:
        primary_error = exc
    finally:
        disconnect_errors: list[str] = []
        if leader_connected:
            try:
                leader.disconnect()
            except Exception as exc:
                disconnect_errors.append(f"Leader disconnect failed on {leader_port}: {exc}")
        if follower_connected:
            try:
                follower.disconnect()
            except Exception as exc:
                disconnect_errors.append(f"Follower disconnect failed on {follower_port}: {exc}")
        else:
            for name in reversed(connected_camera_names):
                camera = follower.cameras[name]
                if camera.is_connected:
                    try:
                        camera.disconnect()
                    except Exception as exc:
                        disconnect_errors.append(f"Camera {name!r} disconnect failed: {exc}")
            if follower_bus_connected:
                try:
                    follower.bus.disconnect()
                except Exception as exc:
                    disconnect_errors.append(
                        f"Follower preflight disconnect failed on {follower_port}: {exc}"
                    )

        for error in disconnect_errors:
            print(error, file=sys.stderr)
        if disconnect_errors:
            print("Torque release was not confirmed. Turn off follower motor power.", file=sys.stderr)

    if primary_error is not None:
        raise primary_error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
