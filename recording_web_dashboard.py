"""Local-only web dashboard for SO-101 recording and trajectory replay."""

from __future__ import annotations

import json
import math
import queue
import re
import secrets
import threading
import time
import webbrowser
from collections.abc import Callable
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import cv2
import numpy as np


WEB_HOST = "127.0.0.1"
DEFAULT_WEB_PORT = 8765
CAMERA_PREVIEW_HZ = 10.0
MIN_SCENARIO_FPS = 5
MAX_SCENARIO_FPS = 20
MAX_LOG_LINES = 300
MAX_LIVE_TRACE_STEPS = 5000
SCENARIO_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


@dataclass(frozen=True)
class DashboardCommand:
    action: str
    scenario_id: str | None = None
    params: dict | None = None


class ScenarioLibrary:
    """Persist replay trajectories separately from the LeRobot episode dataset."""

    def __init__(self, dataset_root: Path) -> None:
        self.root = dataset_root / "scenarios"
        self._lock = threading.RLock()

    def _path(self, scenario_id: str) -> Path:
        if not SCENARIO_ID_PATTERN.fullmatch(scenario_id):
            raise ValueError(f"Invalid scenario id: {scenario_id!r}")
        return self.root / f"{scenario_id}.json"

    @staticmethod
    def _metadata(payload: dict) -> dict:
        capture_settings = payload.get("capture_settings")
        if isinstance(capture_settings, dict) and capture_settings.get("enabled") is True:
            capture_mode = "motion-triggered"
        elif isinstance(capture_settings, dict) and capture_settings.get("enabled") is False:
            capture_mode = "continuous"
        else:
            capture_mode = "legacy"
        return {
            "id": payload["id"],
            "episode_index": int(payload["episode_index"]),
            "task": str(payload["task"]),
            "fps": int(payload["fps"]),
            "steps": len(payload["states"]),
            "duration_s": len(payload["states"]) / int(payload["fps"]),
            "created_at": payload["created_at"],
            "capture_mode": capture_mode,
        }

    @staticmethod
    def _validate_payload(payload: object, *, expected_id: str | None = None) -> dict:
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise ValueError("Unsupported or invalid scenario schema.")

        scenario_id = payload.get("id")
        if not isinstance(scenario_id, str) or not SCENARIO_ID_PATTERN.fullmatch(scenario_id):
            raise ValueError("Scenario id is missing or invalid.")
        if expected_id is not None and scenario_id != expected_id:
            raise ValueError("Scenario id does not match its file name.")

        episode_index = payload.get("episode_index")
        fps = payload.get("fps")
        if isinstance(episode_index, bool) or not isinstance(episode_index, int) or episode_index < 0:
            raise ValueError("Scenario episode_index must be a non-negative integer.")
        if (
            isinstance(fps, bool)
            or not isinstance(fps, int)
            or not MIN_SCENARIO_FPS <= fps <= MAX_SCENARIO_FPS
        ):
            raise ValueError(
                f"Scenario fps must be an integer between {MIN_SCENARIO_FPS} and {MAX_SCENARIO_FPS}."
            )
        if not isinstance(payload.get("task"), str) or not payload["task"].strip():
            raise ValueError("Scenario task is missing.")
        if not isinstance(payload.get("created_at"), str) or not payload["created_at"]:
            raise ValueError("Scenario creation timestamp is missing.")

        capture_settings = payload.get("capture_settings")
        if capture_settings is not None:
            if not isinstance(capture_settings, dict) or not isinstance(
                capture_settings.get("enabled"), bool
            ):
                raise ValueError("Scenario capture settings are invalid.")
            for key in (
                "body_threshold_deg_s",
                "gripper_threshold_pct_s",
                "pre_roll_s",
                "stop_after_s",
            ):
                value = capture_settings.get(key)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or float(value) < 0
                ):
                    raise ValueError(f"Scenario capture setting {key!r} is invalid.")

        joint_names = payload.get("joint_names")
        if (
            not isinstance(joint_names, list)
            or not joint_names
            or any(not isinstance(name, str) or not name for name in joint_names)
            or len(set(joint_names)) != len(joint_names)
        ):
            raise ValueError("Scenario joint_names must be a non-empty unique string list.")

        states = payload.get("states")
        actions = payload.get("actions")
        leaders = payload.get("leaders")
        if not isinstance(states, list) or not isinstance(actions, list) or not states:
            raise ValueError("Scenario state/action data is missing.")
        if len(states) != len(actions):
            raise ValueError("Scenario state/action step counts do not match.")
        if leaders is not None and (
            not isinstance(leaders, list) or len(leaders) != len(states)
        ):
            raise ValueError("Scenario leader/state step counts do not match.")
        series = [("states", states), ("actions", actions)]
        if leaders is not None:
            series.append(("leaders", leaders))
        for series_name, rows in series:
            for row_index, row in enumerate(rows):
                if not isinstance(row, list) or len(row) != len(joint_names):
                    raise ValueError(
                        f"Scenario {series_name} row {row_index} does not match the joint count."
                    )
                for value in row:
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        raise ValueError(f"Scenario {series_name} contains a non-numeric value.")
                    if not math.isfinite(float(value)):
                        raise ValueError(f"Scenario {series_name} contains a non-finite value.")

        ranges = payload.get("ranges")
        if not isinstance(ranges, dict):
            raise ValueError("Scenario joint ranges are missing.")
        for joint_name in joint_names:
            joint_range = ranges.get(joint_name)
            if not isinstance(joint_range, dict):
                raise ValueError(f"Scenario range is missing for {joint_name}.")
            low, high = joint_range.get("min"), joint_range.get("max")
            if (
                isinstance(low, bool)
                or isinstance(high, bool)
                or not isinstance(low, (int, float))
                or not isinstance(high, (int, float))
                or not math.isfinite(float(low))
                or not math.isfinite(float(high))
                or float(low) >= float(high)
            ):
                raise ValueError(f"Scenario range is invalid for {joint_name}.")
        return payload

    def save(
        self,
        *,
        episode_index: int,
        task: str,
        fps: int,
        joint_names: list[str],
        ranges: dict[str, dict[str, float | str]],
        states: list[list[float]],
        actions: list[list[float]],
        leaders: list[list[float]] | None = None,
        capture_settings: dict | None = None,
    ) -> dict:
        scenario_id = f"episode-{episode_index:06d}"
        payload = {
            "schema_version": 1,
            "id": scenario_id,
            "episode_index": int(episode_index),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "task": task,
            "fps": int(fps),
            "joint_names": list(joint_names),
            "ranges": ranges,
            "states": [[float(value) for value in row] for row in states],
            "actions": [[float(value) for value in row] for row in actions],
        }
        if leaders is not None:
            payload["leaders"] = [[float(value) for value in row] for row in leaders]
        if capture_settings is not None:
            payload["capture_settings"] = dict(capture_settings)
        self._validate_payload(payload, expected_id=scenario_id)
        path = self._path(scenario_id)
        temp_path = path.with_suffix(".json.tmp")
        serialized = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        with self._lock:
            self.root.mkdir(parents=True, exist_ok=True)
            temp_path.write_text(serialized, encoding="utf-8")
            temp_path.replace(path)
        return self._metadata(payload)

    def get(self, scenario_id: str) -> dict:
        path = self._path(scenario_id)
        with self._lock:
            if not path.is_file():
                raise FileNotFoundError(f"Scenario not found: {scenario_id}")
            payload = json.loads(path.read_text(encoding="utf-8"))
        return self._validate_payload(payload, expected_id=scenario_id)

    def list(self) -> list[dict]:
        scenarios = []
        with self._lock:
            if not self.root.is_dir():
                return []
            paths = sorted(self.root.glob("*.json"))
            for path in paths:
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    scenarios.append(self._metadata(self._validate_payload(payload, expected_id=path.stem)))
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    continue
        return sorted(scenarios, key=lambda item: item["episode_index"], reverse=True)

    def delete(self, scenario_id: str) -> dict:
        path = self._path(scenario_id)
        with self._lock:
            payload = self.get(scenario_id)
            path.unlink()
        return self._metadata(payload)


class LatestFrameEncoder:
    """Encode only the newest RGB frame so a slow browser never backs up control."""

    def __init__(self, on_error: Callable[[str], None] | None = None) -> None:
        self._frames: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._jpeg = b""
        self._version = 0
        self._thread: threading.Thread | None = None
        self._closed = False
        self._on_error = on_error
        self._last_error_at = float("-inf")

    def start(self) -> None:
        if self._thread is not None:
            return
        thread = threading.Thread(target=self._run, name="dashboard-jpeg", daemon=True)
        try:
            thread.start()
        except Exception:
            self._thread = None
            raise
        self._thread = thread

    def submit(self, rgb_frame: np.ndarray) -> None:
        if self._closed:
            return
        frame = np.ascontiguousarray(np.asarray(rgb_frame).copy())
        try:
            self._frames.put_nowait(frame)
        except queue.Full:
            try:
                self._frames.get_nowait()
            except queue.Empty:
                pass
            try:
                self._frames.put_nowait(frame)
            except queue.Full:
                pass

    def _run(self) -> None:
        while True:
            frame = self._frames.get()
            if frame is None:
                return
            try:
                if frame.ndim != 3 or frame.shape[2] != 3 or not frame.size:
                    raise ValueError(f"expected a non-empty HxWx3 RGB frame, got {frame.shape}")
                if frame.dtype != np.uint8:
                    if not np.isfinite(frame).all():
                        raise ValueError("camera frame contains a non-finite value")
                    scale = 255.0 if float(np.max(frame)) <= 1.0 else 1.0
                    frame = np.clip(frame * scale, 0, 255).astype(np.uint8)
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 82])
                if not ok:
                    raise RuntimeError("OpenCV JPEG encoding returned failure")
                with self._lock:
                    self._jpeg = encoded.tobytes()
                    self._version += 1
            except Exception as exc:
                now = time.monotonic()
                if self._on_error is not None and now - self._last_error_at >= 5.0:
                    self._last_error_at = now
                    self._on_error(f"Camera preview encoding failed: {exc}")

    def snapshot(self) -> tuple[bytes, int]:
        with self._lock:
            return self._jpeg, self._version

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            while True:
                self._frames.get_nowait()
        except queue.Empty:
            pass
        try:
            self._frames.put_nowait(None)
        except queue.Full:
            pass
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)
            if thread.is_alive():
                if self._on_error is not None:
                    self._on_error("Camera preview worker did not stop within 2 seconds.")
            else:
                self._thread = None


class RecordingWebDashboard:
    """Thread-safe bridge between the browser and the single hardware owner loop."""

    COMMANDS = {
        "start_record",
        "finish_segment",
        "discard_segment",
        "save_episode",
        "discard_episode",
        "stop_collection",
        "run_scenario",
        "stop_scenario",
        "delete_scenario",
        "start_setup",
        "increase_episode_target",
        "calibration_continue",
        "calibration_finish",
        "confirm_alignment",
        "recheck_alignment",
        "reset_trace",
    }

    def __init__(
        self,
        *,
        dataset_root: Path,
        port: int = DEFAULT_WEB_PORT,
        open_browser: bool = True,
    ) -> None:
        self.host = WEB_HOST
        self.port = int(port)
        self.open_browser = open_browser
        self.camera_preview_hz = CAMERA_PREVIEW_HZ
        self.url = f"http://{self.host}:{self.port}/"
        self.token = secrets.token_urlsafe(32)
        self.library = ScenarioLibrary(dataset_root)
        self._commands: queue.Queue[DashboardCommand] = queue.Queue()
        self._lock = threading.RLock()
        self._logs: deque[dict] = deque(maxlen=MAX_LOG_LINES)
        self._log_seq = 0
        self._state = {
            "phase": "BOOT",
            "phase_detail": "Starting",
            "is_recording": False,
            "accepted_progress": "0/0",
            "saved_in_run": 0,
            "episode_target": 0,
            "remaining_s": 0.0,
            "control_hz": 0.0,
            "buffered_frames": 0,
            "motion_status": "OFF",
            "motion_body_speed_deg_s": 0.0,
            "motion_gripper_speed_pct_s": 0.0,
            "joint_rows": [],
            "ranges": {},
            "scenario_running": None,
            "playback_step": None,
            "playback_total_steps": 0,
            "scenarios": self.library.list(),
            "setup_defaults": {},
            "active_settings": {},
            "settings_locked": False,
            "calibration_target": None,
            "calibration_stage": None,
            "calibration_message": "",
            "alignment_safe": False,
        }
        self._trace = {
            "source": "live",
            "label": "Waiting for data",
            "joint_names": [],
            "ranges": {},
            "steps": deque(maxlen=MAX_LIVE_TRACE_STEPS),
            "states": deque(maxlen=MAX_LIVE_TRACE_STEPS),
            "actions": deque(maxlen=MAX_LIVE_TRACE_STEPS),
            "leaders": deque(maxlen=MAX_LIVE_TRACE_STEPS),
        }
        self._trace_step = 0
        self._trace_revision = 0
        self._frame_encoder = LatestFrameEncoder(lambda message: self.log(message, "error"))
        self._next_frame_submit = float("-inf")
        self._server: ThreadingHTTPServer | None = None
        self._server_thread: threading.Thread | None = None
        self._server_loop_started = False
        self._closing = False
        self._last_client_seen = 0.0
        self._html = ""
        self._joint_guide_image = b""

    def is_closing(self) -> bool:
        with self._lock:
            return self._closing

    def prepare(self) -> None:
        """Bind localhost and start workers before follower motion is enabled."""
        if self._server is not None:
            return
        asset = Path(__file__).with_name("recording_dashboard.html")
        self._html = asset.read_text(encoding="utf-8").replace("__DASHBOARD_TOKEN__", self.token)
        joint_guide_asset = (
            Path(__file__).resolve().parent
            / "docs"
            / "images"
            / "so101-follower-official.webp"
        )
        self._joint_guide_image = joint_guide_asset.read_bytes()
        dashboard = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "SO101Dashboard/1.0"

            def log_message(self, _format: str, *_args) -> None:
                return

            def _local_request(self) -> bool:
                host = self.headers.get("Host", "").split(":", 1)[0].strip("[]").lower()
                return host in {"127.0.0.1", "localhost", "::1"}

            def _origin_allowed(self) -> bool:
                origin = self.headers.get("Origin")
                if not origin:
                    return True
                parsed = urlparse(origin)
                return parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost", "::1"}

            def _json(self, payload: dict | list, status: int = HTTPStatus.OK) -> None:
                body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def _error(self, status: int, message: str) -> None:
                self._json({"error": message}, status)

            def _discard_small_request_body(self) -> None:
                """Avoid a Windows TCP reset when rejecting a small POST before parsing it."""
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    return
                if 0 < length <= 4096:
                    self.rfile.read(length)

            def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
                if not self._local_request():
                    self._error(HTTPStatus.FORBIDDEN, "Localhost requests only.")
                    return
                parsed_request = urlparse(self.path)
                path = parsed_request.path
                dashboard.touch_client()
                if path == "/":
                    body = dashboard._html.encode("utf-8")
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(body)
                elif path == "/api/state":
                    self._json(dashboard.state_snapshot())
                elif path == "/api/trace":
                    try:
                        query = parse_qs(parsed_request.query)
                        revision = int(query["revision"][0]) if "revision" in query else None
                        after_step = int(query["after"][0]) if "after" in query else None
                    except (KeyError, TypeError, ValueError):
                        self._error(HTTPStatus.BAD_REQUEST, "Invalid trace cursor.")
                        return
                    self._json(dashboard.trace_snapshot(revision=revision, after_step=after_step))
                elif path == "/camera.jpg":
                    jpeg, _version = dashboard._frame_encoder.snapshot()
                    if not jpeg:
                        self.send_response(HTTPStatus.NO_CONTENT)
                        self.end_headers()
                        return
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(jpeg)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(jpeg)
                elif path == "/assets/so101-follower-official.webp":
                    body = dashboard._joint_guide_image
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "image/webp")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", "public, max-age=3600")
                    self.send_header("X-Content-Type-Options", "nosniff")
                    self.end_headers()
                    self.wfile.write(body)
                elif path.startswith("/api/scenarios/"):
                    scenario_id = unquote(path.rsplit("/", 1)[-1])
                    try:
                        self._json(dashboard.library.get(scenario_id))
                    except (FileNotFoundError, ValueError) as exc:
                        self._error(HTTPStatus.NOT_FOUND, str(exc))
                else:
                    self._error(HTTPStatus.NOT_FOUND, "Not found.")

            def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
                if not self._local_request() or not self._origin_allowed():
                    self._discard_small_request_body()
                    self._error(HTTPStatus.FORBIDDEN, "Local dashboard origin required.")
                    return
                if not secrets.compare_digest(self.headers.get("X-Dashboard-Token", ""), dashboard.token):
                    self._discard_small_request_body()
                    self._error(HTTPStatus.FORBIDDEN, "Invalid dashboard token.")
                    return
                if dashboard.is_closing():
                    self._discard_small_request_body()
                    self._error(HTTPStatus.SERVICE_UNAVAILABLE, "Dashboard shutdown is in progress.")
                    return
                if urlparse(self.path).path != "/api/command":
                    self._discard_small_request_body()
                    self._error(HTTPStatus.NOT_FOUND, "Not found.")
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if length <= 0 or length > 4096:
                        raise ValueError("Invalid request size.")
                    payload = json.loads(self.rfile.read(length).decode("utf-8"))
                    action = str(payload["action"])
                    scenario_id = payload.get("scenario_id")
                    params = payload.get("params")
                    if action not in dashboard.COMMANDS:
                        raise ValueError(f"Unknown command: {action}")
                    if action in {"run_scenario", "delete_scenario"}:
                        if not isinstance(scenario_id, str) or not SCENARIO_ID_PATTERN.fullmatch(scenario_id):
                            raise ValueError("A valid scenario_id is required.")
                    else:
                        scenario_id = None
                    if action in {"start_setup", "increase_episode_target"}:
                        if not isinstance(params, dict):
                            raise ValueError(f"Parameters are required for {action}.")
                        if action == "increase_episode_target":
                            additional = params.get("additional_episodes")
                            if (
                                isinstance(additional, bool)
                                or not isinstance(additional, int)
                                or not 1 <= additional <= 1000
                            ):
                                raise ValueError("additional_episodes must be an integer from 1 to 1000.")
                    else:
                        params = None
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    self._error(HTTPStatus.BAD_REQUEST, str(exc))
                    return
                dashboard.touch_client()
                if action == "reset_trace":
                    dashboard.reset_trace()
                    dashboard.log("LIVE 관절 궤적을 초기화했습니다.")
                    self._json({"reset": True}, HTTPStatus.OK)
                    return
                dashboard._commands.put(
                    DashboardCommand(action=action, scenario_id=scenario_id, params=params)
                )
                self._json({"queued": True}, HTTPStatus.ACCEPTED)

        try:
            self._frame_encoder.start()
            self._server = ThreadingHTTPServer((self.host, self.port), Handler)
            self._server.daemon_threads = True
            self.port = int(self._server.server_address[1])
            self.url = f"http://{self.host}:{self.port}/"
            server = self._server
            self._server_thread = threading.Thread(
                target=lambda: server.serve_forever(poll_interval=0.05),
                name="dashboard-http",
                daemon=True,
            )
            self._server_thread.start()
            self._server_loop_started = True
        except Exception:
            if self._server is not None:
                self._server.server_close()
            self._server = None
            self._server_thread = None
            self._frame_encoder.close()
            raise
        self.log(f"Web dashboard ready: {self.url}")
        if self.open_browser:
            threading.Thread(target=lambda: webbrowser.open(self.url), name="dashboard-browser", daemon=True).start()

    def close(self) -> None:
        with self._lock:
            if self._closing:
                return
            self._closing = True
            server = self._server
            self._server = None
        if server is not None:
            if self._server_loop_started:
                server.shutdown()
            server.server_close()
        self._server_loop_started = False
        if self._server_thread is not None:
            self._server_thread.join(timeout=2.0)
        self._server_thread = None
        self._frame_encoder.close()

    def touch_client(self) -> None:
        with self._lock:
            self._last_client_seen = time.monotonic()

    def client_is_alive(self, timeout_s: float = 2.5) -> bool:
        with self._lock:
            return time.monotonic() - self._last_client_seen <= timeout_s

    def log(self, message: str, level: str = "info") -> None:
        timestamp = datetime.now().astimezone().strftime("%H:%M:%S")
        with self._lock:
            self._log_seq += 1
            self._logs.append(
                {"seq": self._log_seq, "time": timestamp, "level": level, "message": str(message)}
            )

    def set_phase(
        self,
        phase: str,
        detail: str,
        *,
        is_recording: bool = False,
        accepted_progress: str | None = None,
        scenario_running: str | None = None,
    ) -> None:
        with self._lock:
            self._state["phase"] = phase
            self._state["phase_detail"] = detail
            self._state["is_recording"] = is_recording
            self._state["scenario_running"] = scenario_running
            if phase != "PLAYBACK":
                self._state["playback_step"] = None
                self._state["playback_total_steps"] = 0
            if accepted_progress is not None:
                self._state["accepted_progress"] = accepted_progress

    def set_playback_progress(
        self,
        completed_steps: int | None,
        total_steps: int = 0,
    ) -> None:
        """Publish 1-based completed action count; zero means playback is ready to start."""
        if completed_steps is None:
            with self._lock:
                self._state["playback_step"] = None
                self._state["playback_total_steps"] = 0
            return
        if (
            isinstance(completed_steps, bool)
            or not isinstance(completed_steps, int)
            or isinstance(total_steps, bool)
            or not isinstance(total_steps, int)
            or total_steps <= 0
            or not 0 <= completed_steps <= total_steps
        ):
            raise ValueError("Playback progress must be an integer from zero through total_steps.")
        with self._lock:
            self._state["playback_step"] = completed_steps
            self._state["playback_total_steps"] = total_steps

    def finish_playback(self, detail: str) -> None:
        """Atomically clear transient playback state before the owner loop picks its next phase."""
        with self._lock:
            self._state["phase"] = "TRANSITION"
            self._state["phase_detail"] = str(detail)
            self._state["is_recording"] = False
            self._state["scenario_running"] = None
            self._state["playback_step"] = None
            self._state["playback_total_steps"] = 0
            self._state["remaining_s"] = 0.0
            self._state["control_hz"] = 0.0
            self._state["buffered_frames"] = 0

    def set_setup_defaults(self, defaults: dict) -> None:
        with self._lock:
            self._state["setup_defaults"] = dict(defaults)

    def lock_settings(self, settings: dict | None = None) -> None:
        with self._lock:
            active_settings = dict(settings or {})
            self._state["settings_locked"] = True
            self._state["active_settings"] = active_settings
            if "episodes" in active_settings:
                target = int(active_settings["episodes"])
                self._state["saved_in_run"] = 0
                self._state["episode_target"] = target
                self._state["accepted_progress"] = f"0/{target}"

    def collection_target(self) -> int:
        with self._lock:
            return int(self._state["episode_target"])

    def set_collection_progress(self, saved_in_run: int) -> None:
        saved = int(saved_in_run)
        if saved < 0:
            raise ValueError("Saved episode count must not be negative.")
        with self._lock:
            target = int(self._state["episode_target"])
            self._state["saved_in_run"] = saved
            self._state["accepted_progress"] = f"{saved}/{target}"

    def increase_episode_target(self, additional_episodes: int) -> int:
        if isinstance(additional_episodes, bool) or not isinstance(additional_episodes, int):
            raise ValueError("Additional episode count must be an integer.")
        if additional_episodes <= 0:
            raise ValueError("Additional episode count must be greater than zero.")
        with self._lock:
            current = int(self._state["episode_target"])
            target = current + additional_episodes
            if target > 1000:
                raise ValueError("The collection target cannot exceed 1000 episodes per run.")
            self._state["episode_target"] = target
            self._state["active_settings"]["episodes"] = target
            saved = int(self._state["saved_in_run"])
            self._state["accepted_progress"] = f"{saved}/{target}"
            return target

    def set_camera_preview_hz(self, hz: float) -> None:
        if not 1.0 <= float(hz) <= 30.0:
            raise ValueError("Camera preview rate must be between 1 and 30 Hz.")
        with self._lock:
            self.camera_preview_hz = float(hz)
            self._next_frame_submit = float("-inf")

    def set_calibration_stage(self, target: str | None, stage: str | None, message: str = "") -> None:
        with self._lock:
            self._state["calibration_target"] = target
            self._state["calibration_stage"] = stage
            self._state["calibration_message"] = message

    def set_alignment(self, safe: bool, message: str) -> None:
        with self._lock:
            self._state["alignment_safe"] = bool(safe)
            self._state["phase_detail"] = message

    def publish_joint_rows(self, joint_rows: list[dict]) -> None:
        with self._lock:
            self._state["joint_rows"] = joint_rows

    def set_motion_capture(
        self,
        status: str,
        *,
        body_speed_deg_s: float,
        gripper_speed_pct_s: float,
    ) -> None:
        with self._lock:
            self._state["motion_status"] = str(status)
            self._state["motion_body_speed_deg_s"] = round(float(body_speed_deg_s), 2)
            self._state["motion_gripper_speed_pct_s"] = round(float(gripper_speed_pct_s), 2)

    def refresh_scenarios(self) -> None:
        scenarios = self.library.list()
        with self._lock:
            self._state["scenarios"] = scenarios

    def begin_trace(
        self,
        *,
        label: str,
        joint_names: list[str],
        ranges: dict[str, dict[str, float | str]],
    ) -> None:
        with self._lock:
            self._trace = {
                "source": "live",
                "label": label,
                "joint_names": list(joint_names),
                "ranges": ranges,
                "steps": deque(maxlen=MAX_LIVE_TRACE_STEPS),
                "states": deque(maxlen=MAX_LIVE_TRACE_STEPS),
                "actions": deque(maxlen=MAX_LIVE_TRACE_STEPS),
                "leaders": deque(maxlen=MAX_LIVE_TRACE_STEPS),
            }
            self._trace_step = 0
            self._trace_revision += 1
            self._state["ranges"] = ranges

    def reset_trace(self) -> None:
        """Clear only the live display trace while preserving its schema and calibration ranges."""
        with self._lock:
            self._trace["steps"] = deque(maxlen=MAX_LIVE_TRACE_STEPS)
            self._trace["states"] = deque(maxlen=MAX_LIVE_TRACE_STEPS)
            self._trace["actions"] = deque(maxlen=MAX_LIVE_TRACE_STEPS)
            self._trace["leaders"] = deque(maxlen=MAX_LIVE_TRACE_STEPS)
            self._trace_step = 0
            self._trace_revision += 1

    def publish_sample(
        self,
        *,
        camera_rgb: np.ndarray,
        joint_rows: list[dict],
        state_values: list[float],
        action_values: list[float],
        leader_values: list[float] | None,
        remaining_s: float,
        control_hz: float,
        buffered_frames: int,
        append_trace: bool = True,
    ) -> None:
        now = time.monotonic()
        with self._lock:
            preview_hz = getattr(self, "camera_preview_hz", CAMERA_PREVIEW_HZ)
            period = 1.0 / preview_hz
            tolerance = min(0.005, period * 0.1)
            due = not math.isfinite(self._next_frame_submit) or now + tolerance >= self._next_frame_submit
            if due:
                if not math.isfinite(self._next_frame_submit):
                    self._next_frame_submit = now + period
                else:
                    skipped_periods = max(1, math.floor((now - self._next_frame_submit) / period) + 1)
                    self._next_frame_submit += skipped_periods * period
        if due:
            self._frame_encoder.submit(camera_rgb)
        with self._lock:
            self._state["remaining_s"] = round(float(remaining_s), 2)
            self._state["control_hz"] = round(float(control_hz), 2)
            self._state["buffered_frames"] = int(buffered_frames)
            self._state["joint_rows"] = joint_rows
            if append_trace:
                self._append_trace_sample_locked(state_values, action_values, leader_values)

    def _append_trace_sample_locked(
        self,
        state_values: list[float] | tuple[float, ...],
        action_values: list[float] | tuple[float, ...],
        leader_values: list[float] | tuple[float, ...] | None,
    ) -> None:
        self._trace["steps"].append(self._trace_step)
        self._trace["states"].append([float(value) for value in state_values])
        self._trace["actions"].append([float(value) for value in action_values])
        self._trace["leaders"].append(
            [float(value) for value in leader_values] if leader_values is not None else []
        )
        self._trace_step += 1

    def append_trace_samples(
        self,
        *,
        states: list[list[float]],
        actions: list[list[float]],
        leaders: list[list[float]],
    ) -> None:
        """Append a pre-roll batch without replaying stale camera or status samples."""
        if not (len(states) == len(actions) == len(leaders)):
            raise ValueError("Trace pre-roll state/action/leader lengths must match.")
        with self._lock:
            for state_values, action_values, leader_values in zip(states, actions, leaders, strict=True):
                self._append_trace_sample_locked(state_values, action_values, leader_values)

    def state_snapshot(self) -> dict:
        _jpeg, camera_version = self._frame_encoder.snapshot()
        with self._lock:
            return {
                **self._state,
                "logs": list(self._logs),
                "camera_version": camera_version,
                "dashboard_url": self.url,
            }

    def trace_snapshot(
        self,
        *,
        revision: int | None = None,
        after_step: int | None = None,
    ) -> dict:
        with self._lock:
            steps = list(self._trace["steps"])
            replace = revision != self._trace_revision or after_step is None
            if not replace and steps and after_step < steps[0] - 1:
                replace = True
            start_index = 0
            if not replace and after_step is not None:
                while start_index < len(steps) and steps[start_index] <= after_step:
                    start_index += 1
            return {
                "source": self._trace["source"],
                "label": self._trace["label"],
                "joint_names": list(self._trace["joint_names"]),
                "ranges": self._trace["ranges"],
                "revision": self._trace_revision,
                "replace": replace,
                "max_steps": MAX_LIVE_TRACE_STEPS,
                "steps": steps[start_index:],
                "states": list(self._trace["states"])[start_index:],
                "actions": list(self._trace["actions"])[start_index:],
                "leaders": list(self._trace["leaders"])[start_index:],
            }

    def drain_commands(self, max_items: int | None = None) -> list[DashboardCommand]:
        commands = []
        while max_items is None or len(commands) < max_items:
            try:
                commands.append(self._commands.get_nowait())
            except queue.Empty:
                break
        return commands
