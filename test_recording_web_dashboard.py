from __future__ import annotations

import json
import re
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

import numpy as np

from recording_web_dashboard import (
    DashboardCommand,
    LatestFrameEncoder,
    RecordingWebDashboard,
    ScenarioLibrary,
)
from run_record import (
    BufferedMotionSample,
    DEFAULT_EPISODES,
    DEFAULT_EPISODE_SECONDS,
    JOINT_NAMES,
    MotionCaptureSettings,
    MotionTriggeredCapture,
    RuntimeSettings,
    UserCancelled,
    calculate_max_sample_interval_s,
    calculate_sample_rate_hz,
    handle_episode_target_increase,
    measure_motion_speeds,
    parse_args,
    record_episodes_web,
    run_control_segment,
    run_scenario_playback,
    service_idle_mode,
    validate_runtime_settings,
    wait_for_web_settings,
)


def sample_ranges() -> dict[str, dict[str, float | str]]:
    return {
        key: {
            "min": 0.0 if key == "gripper.pos" else -180.0,
            "max": 100.0 if key == "gripper.pos" else 180.0,
            "unit": "%" if key == "gripper.pos" else "deg",
        }
        for key in JOINT_NAMES
    }


class ScenarioLibraryTest(unittest.TestCase):
    def test_save_reload_and_delete_replay_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "dataset"
            library = ScenarioLibrary(root)
            self.assertFalse(root.exists(), "listing scenarios must not pre-create a new dataset root")
            self.assertEqual(library.list(), [])
            states = [[float(index) for index in range(6)] for _ in range(3)]
            actions = [[float(index + 1) for index in range(6)] for _ in range(3)]
            leaders = [[float(index + 2) for index in range(6)] for _ in range(3)]
            metadata = library.save(
                episode_index=2,
                task="test task",
                fps=20,
                joint_names=list(JOINT_NAMES),
                ranges=sample_ranges(),
                states=states,
                actions=actions,
                leaders=leaders,
                capture_settings={
                    "enabled": True,
                    "body_threshold_deg_s": 4.0,
                    "gripper_threshold_pct_s": 5.0,
                    "pre_roll_s": 0.5,
                    "stop_after_s": 1.0,
                },
            )
            self.assertEqual(metadata["steps"], 3)
            self.assertEqual(metadata["capture_mode"], "motion-triggered")
            scenario = ScenarioLibrary(root).get("episode-000002")
            self.assertEqual(scenario["actions"], actions)
            self.assertEqual(scenario["leaders"], leaders)
            self.assertTrue(scenario["capture_settings"]["enabled"])
            self.assertEqual(len(library.list()), 1)
            removed = library.delete("episode-000002")
            self.assertEqual(removed["episode_index"], 2)
            self.assertEqual(library.list(), [])

    def test_legacy_scenario_without_leaders_remains_readable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "dataset"
            library = ScenarioLibrary(root)
            rows = [[float(index) for index in range(6)] for _ in range(2)]
            library.save(
                episode_index=0,
                task="legacy task",
                fps=20,
                joint_names=list(JOINT_NAMES),
                ranges=sample_ranges(),
                states=rows,
                actions=rows,
            )
            path = root / "scenarios" / "episode-000000.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload.pop("leaders", None)
            path.write_text(json.dumps(payload), encoding="utf-8")

            scenario = ScenarioLibrary(root).get("episode-000000")
            self.assertNotIn("leaders", scenario)
            self.assertEqual(scenario["actions"], rows)

    def test_scenario_rejects_leader_rows_that_do_not_match_the_timeline(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "dataset"
            library = ScenarioLibrary(root)
            rows = [[0.0] * 6 for _ in range(2)]
            with self.assertRaisesRegex(ValueError, "leader|step count"):
                library.save(
                    episode_index=0,
                    task="invalid leader timeline",
                    fps=20,
                    joint_names=list(JOINT_NAMES),
                    ranges=sample_ranges(),
                    states=rows,
                    actions=rows,
                    leaders=[[0.0] * 6],
                )

    def test_non_finite_scenario_is_rejected_before_writing_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "dataset"
            library = ScenarioLibrary(root)
            with self.assertRaisesRegex(ValueError, "non-finite"):
                library.save(
                    episode_index=0,
                    task="unsafe trajectory",
                    fps=20,
                    joint_names=list(JOINT_NAMES),
                    ranges=sample_ranges(),
                    states=[[0.0] * 6],
                    actions=[[float("nan"), 0.0, 0.0, 0.0, 0.0, 0.0]],
                )
            self.assertFalse((root / "scenarios" / "episode-000000.json").exists())


class DashboardHttpTest(unittest.TestCase):
    def test_dashboard_close_is_idempotent_and_marks_shutdown(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dashboard = RecordingWebDashboard(
                dataset_root=Path(temp_dir),
                port=0,
                open_browser=False,
            )
            dashboard.prepare()
            dashboard.close()
            self.assertTrue(dashboard.is_closing())
            dashboard.close()

    def test_excluded_compute_and_agent_commands_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dashboard = RecordingWebDashboard(
                dataset_root=Path(temp_dir),
                port=0,
                open_browser=False,
            )
            dashboard.prepare()
            try:
                for action in (
                    "start_training",
                    "stop_training",
                    "start_inference_test",
                    "stop_inference_test",
                    "start_policy_run",
                    "stop_policy_run",
                    "configure_agent",
                    "test_agent_decision",
                ):
                    with self.subTest(action=action):
                        request = urllib.request.Request(
                            dashboard.url + "api/command",
                            data=json.dumps({"action": action}).encode("utf-8"),
                            headers={
                                "Content-Type": "application/json",
                                "X-Dashboard-Token": dashboard.token,
                            },
                            method="POST",
                        )
                        with self.assertRaises(urllib.error.HTTPError) as context:
                            urllib.request.urlopen(request, timeout=2)
                        self.assertEqual(context.exception.code, 400)
                for endpoint in (
                    "api/training",
                    "api/inference-test",
                    "api/policy-run",
                    "api/agent",
                ):
                    with self.subTest(endpoint=endpoint):
                        with self.assertRaises(urllib.error.HTTPError) as context:
                            urllib.request.urlopen(dashboard.url + endpoint, timeout=2)
                        self.assertEqual(context.exception.code, 404)
                self.assertEqual(dashboard.drain_commands(), [])
            finally:
                dashboard.close()

    def test_local_http_state_trace_camera_and_token_command(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dashboard = RecordingWebDashboard(
                dataset_root=Path(temp_dir) / "dataset",
                port=0,
                open_browser=False,
            )
            dashboard.set_setup_defaults({"control_fps": 20})
            dashboard.prepare()
            try:
                dashboard.begin_trace(
                    label="synthetic",
                    joint_names=list(JOINT_NAMES),
                    ranges=sample_ranges(),
                )
                frame = np.full((480, 640, 3), 90, dtype=np.uint8)
                original = frame.copy()
                rows = [
                    {
                        "joint": key.removesuffix(".pos"),
                        "follower": 0.0,
                        "leader": 1.0,
                        "command": 0.5,
                        "error": 1.0,
                    }
                    for key in JOINT_NAMES
                ]
                dashboard.publish_sample(
                    camera_rgb=frame,
                    joint_rows=rows,
                    state_values=[0.0] * 6,
                    action_values=[0.5] * 6,
                    leader_values=[1.0] * 6,
                    remaining_s=1,
                    control_hz=20,
                    buffered_frames=1,
                )
                frame[:] = 0
                deadline = time.monotonic() + 2
                while dashboard.state_snapshot()["camera_version"] == 0 and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertTrue(np.all(original == 90))

                with urllib.request.urlopen(dashboard.url, timeout=2) as response:
                    html = response.read().decode("utf-8")
                self.assertIn("SO-101 Web Recorder", html)
                with urllib.request.urlopen(dashboard.url + "api/state", timeout=2) as response:
                    state = json.load(response)
                self.assertEqual(len(state["joint_rows"]), 6)
                with urllib.request.urlopen(dashboard.url + "api/trace", timeout=2) as response:
                    trace = json.load(response)
                self.assertEqual(trace["steps"], [0])
                self.assertEqual(trace["states"], [[0.0] * 6])
                self.assertEqual(trace["actions"], [[0.5] * 6])
                self.assertEqual(trace["leaders"], [[1.0] * 6])
                dashboard.publish_sample(
                    camera_rgb=original,
                    joint_rows=rows,
                    state_values=[1.0] * 6,
                    action_values=[1.5] * 6,
                    leader_values=[2.0] * 6,
                    remaining_s=0,
                    control_hz=20,
                    buffered_frames=2,
                )
                cursor_url = (
                    dashboard.url
                    + f"api/trace?revision={trace['revision']}&after={trace['steps'][-1]}"
                )
                with urllib.request.urlopen(cursor_url, timeout=2) as response:
                    delta = json.load(response)
                self.assertFalse(delta["replace"])
                self.assertEqual(delta["steps"], [1])
                self.assertEqual(delta["states"], [[1.0] * 6])
                self.assertEqual(delta["actions"], [[1.5] * 6])
                self.assertEqual(delta["leaders"], [[2.0] * 6])
                with urllib.request.urlopen(dashboard.url + "camera.jpg", timeout=2) as response:
                    self.assertGreater(len(response.read()), 100)

                body = json.dumps({"action": "start_record"}).encode("utf-8")
                bad_request = urllib.request.Request(
                    dashboard.url + "api/command",
                    data=body,
                    method="POST",
                    headers={"Content-Type": "application/json"},
                )
                with self.assertRaises(urllib.error.HTTPError) as context:
                    urllib.request.urlopen(bad_request, timeout=2)
                self.assertEqual(context.exception.code, 403)

                good_request = urllib.request.Request(
                    dashboard.url + "api/command",
                    data=body,
                    method="POST",
                    headers={
                        "Content-Type": "application/json",
                        "X-Dashboard-Token": dashboard.token,
                    },
                )
                with urllib.request.urlopen(good_request, timeout=2) as response:
                    self.assertEqual(response.status, 202)
                self.assertEqual(dashboard.drain_commands()[0].action, "start_record")

                target_body = json.dumps(
                    {
                        "action": "increase_episode_target",
                        "params": {"additional_episodes": 4},
                    }
                ).encode("utf-8")
                target_request = urllib.request.Request(
                    dashboard.url + "api/command",
                    data=target_body,
                    method="POST",
                    headers={
                        "Content-Type": "application/json",
                        "X-Dashboard-Token": dashboard.token,
                    },
                )
                with urllib.request.urlopen(target_request, timeout=2) as response:
                    self.assertEqual(response.status, 202)
                target_command = dashboard.drain_commands()[0]
                self.assertEqual(target_command.action, "increase_episode_target")
                self.assertEqual(target_command.params, {"additional_episodes": 4})

                invalid_target_request = urllib.request.Request(
                    dashboard.url + "api/command",
                    data=json.dumps(
                        {
                            "action": "increase_episode_target",
                            "params": {"additional_episodes": True},
                        }
                    ).encode("utf-8"),
                    method="POST",
                    headers={
                        "Content-Type": "application/json",
                        "X-Dashboard-Token": dashboard.token,
                    },
                )
                with self.assertRaises(urllib.error.HTTPError) as context:
                    urllib.request.urlopen(invalid_target_request, timeout=2)
                self.assertEqual(context.exception.code, 400)
            finally:
                dashboard.close()

    def test_official_so101_joint_guide_asset_is_served_locally(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dashboard = RecordingWebDashboard(
                dataset_root=Path(temp_dir) / "dataset",
                port=0,
                open_browser=False,
            )
            dashboard.prepare()
            try:
                asset_url = dashboard.url + "assets/so101-follower-official.webp"
                with urllib.request.urlopen(asset_url, timeout=2) as response:
                    body = response.read()
                    self.assertEqual(response.status, 200)
                    self.assertEqual(response.headers.get_content_type(), "image/webp")
                    self.assertEqual(response.headers["Cache-Control"], "public, max-age=3600")
                    self.assertEqual(response.headers["X-Content-Type-Options"], "nosniff")
                    self.assertEqual(response.headers["Content-Length"], str(len(body)))

                official_asset = (
                    Path(__file__).resolve().parent
                    / "docs"
                    / "images"
                    / "so101-follower-official.webp"
                )
                self.assertEqual(body, official_asset.read_bytes())
                self.assertGreater(len(body), 1_000)
                self.assertEqual(body[:4], b"RIFF")
                self.assertEqual(body[8:12], b"WEBP")

                with urllib.request.urlopen(dashboard.url, timeout=2) as response:
                    html = response.read().decode("utf-8")
                self.assertIn(
                    'src="/assets/so101-follower-official.webp"',
                    html,
                )
            finally:
                dashboard.close()

    def test_http_trace_reset_is_display_only_and_restarts_step_zero(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "dataset"
            dashboard = RecordingWebDashboard(
                dataset_root=root,
                port=0,
                open_browser=False,
            )
            dashboard.prepare()
            try:
                dashboard.begin_trace(
                    label="READY manual control",
                    joint_names=list(JOINT_NAMES),
                    ranges=sample_ranges(),
                )
                dashboard.append_trace_samples(
                    states=[[0.0] * 6, [1.0] * 6],
                    actions=[[0.5] * 6, [1.5] * 6],
                    leaders=[[1.0] * 6, [2.0] * 6],
                )
                scenario = dashboard.library.save(
                    episode_index=0,
                    task="reset must not delete saved data",
                    fps=20,
                    joint_names=list(JOINT_NAMES),
                    ranges=sample_ranges(),
                    states=[[0.0] * 6],
                    actions=[[0.5] * 6],
                    leaders=[[1.0] * 6],
                )
                scenario_path = root / "scenarios" / f"{scenario['id']}.json"
                scenario_before = scenario_path.read_bytes()
                previous = dashboard.trace_snapshot()

                request = urllib.request.Request(
                    dashboard.url + "api/command",
                    data=json.dumps({"action": "reset_trace"}).encode("utf-8"),
                    method="POST",
                    headers={
                        "Content-Type": "application/json",
                        "X-Dashboard-Token": dashboard.token,
                    },
                )
                with urllib.request.urlopen(request, timeout=2) as response:
                    self.assertEqual(response.status, 200)
                    self.assertTrue(json.load(response)["reset"])

                reset = dashboard.trace_snapshot(
                    revision=previous["revision"],
                    after_step=previous["steps"][-1],
                )
                self.assertTrue(reset["replace"])
                self.assertEqual(reset["revision"], previous["revision"] + 1)
                for field in ("steps", "states", "actions", "leaders"):
                    self.assertEqual(reset[field], [])
                self.assertEqual(dashboard.drain_commands(), [])
                self.assertEqual(scenario_path.read_bytes(), scenario_before)

                dashboard.append_trace_samples(
                    states=[[3.0] * 6],
                    actions=[[3.5] * 6],
                    leaders=[[4.0] * 6],
                )
                restarted = dashboard.trace_snapshot()
                self.assertEqual(restarted["steps"], [0])
                self.assertEqual(restarted["label"], "READY manual control")
            finally:
                dashboard.close()


class SettingsValidationTest(unittest.TestCase):
    def test_stop_before_connection_wins_over_queued_setup(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dashboard = RecordingWebDashboard(
                dataset_root=Path(temp_dir) / "dataset",
                port=0,
                open_browser=False,
            )
            dashboard._commands.put(DashboardCommand(action="start_setup", params={}))
            dashboard._commands.put(DashboardCommand(action="stop_collection"))

            with self.assertRaises(UserCancelled):
                wait_for_web_settings(
                    dashboard,
                    root=Path(temp_dir) / "dataset",
                    follower=object(),
                    leader=object(),
                )

            self.assertFalse(dashboard.state_snapshot()["settings_locked"])

    def test_resume_fps_and_recalibration_are_locked(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "meta").mkdir()
            (root / "meta" / "info.json").write_text(
                json.dumps({"fps": 20, "total_episodes": 2}),
                encoding="utf-8",
            )
            base = {
                "control_fps": 20,
                "episodes": 5,
                "episode_seconds": 30,
                "reset_seconds": 20,
                "preview_hz": 10,
                "motion_triggered": True,
                "motion_body_threshold_deg_s": 4,
                "motion_gripper_threshold_pct_s": 5,
                "motion_pre_roll_s": 0.5,
                "motion_stop_seconds": 1,
                "calibration": "none",
            }
            self.assertEqual(validate_runtime_settings(base, root).control_fps, 20)
            with self.assertRaisesRegex(ValueError, "fixed at 20 FPS"):
                validate_runtime_settings({**base, "control_fps": 10}, root)
            with self.assertRaisesRegex(ValueError, "coordinate system"):
                validate_runtime_settings({**base, "calibration": "both"}, root)
            with self.assertRaisesRegex(ValueError, "Invalid web settings"):
                validate_runtime_settings({**base, "episodes": True}, root)
            with self.assertRaisesRegex(ValueError, "Invalid web settings"):
                validate_runtime_settings({**base, "motion_triggered": "true"}, root)
            with self.assertRaisesRegex(ValueError, "Body motion threshold"):
                validate_runtime_settings({**base, "motion_body_threshold_deg_s": 0}, root)
            with self.assertRaisesRegex(ValueError, "Motion pre-roll"):
                validate_runtime_settings({**base, "motion_pre_roll_s": 1.1}, root)

    def test_episode_target_can_only_increase_and_progress_is_updated(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dashboard = RecordingWebDashboard(
                dataset_root=Path(temp_dir) / "dataset",
                port=0,
                open_browser=False,
            )
            dashboard.lock_settings({"episodes": 5, "preview_hz": 10})
            dashboard.set_collection_progress(5)
            handled = handle_episode_target_increase(
                DashboardCommand(
                    action="increase_episode_target",
                    params={"additional_episodes": 3},
                ),
                dashboard=dashboard,
            )
            self.assertTrue(handled)
            state = dashboard.state_snapshot()
            self.assertEqual(state["episode_target"], 8)
            self.assertEqual(state["saved_in_run"], 5)
            self.assertEqual(state["accepted_progress"], "5/8")
            self.assertEqual(state["active_settings"]["episodes"], 8)
            with self.assertRaisesRegex(ValueError, "cannot exceed 1000"):
                dashboard.increase_episode_target(1000)

    def test_completed_library_returns_to_collection_after_target_increase(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dashboard = RecordingWebDashboard(
                dataset_root=Path(temp_dir) / "dataset",
                port=0,
                open_browser=False,
            )
            dashboard.lock_settings({"episodes": 5})
            dashboard.set_collection_progress(5)
            dashboard._commands.put(
                DashboardCommand(
                    action="increase_episode_target",
                    params={"additional_episodes": 2},
                )
            )
            result = service_idle_mode(
                follower=object(),
                leader=object(),
                follower_port="FAKE_FOLLOWER",
                leader_port="FAKE_LEADER",
                dashboard=dashboard,
                ranges=sample_ranges(),
                events={
                    "exit_early": False,
                    "rerecord_episode": False,
                    "stop_recording": False,
                },
                fps=20,
                saved_in_run=5,
                allow_record=False,
            )
            self.assertEqual(result, "goal_changed")
            self.assertEqual(dashboard.collection_target(), 7)
            self.assertEqual(dashboard.state_snapshot()["accepted_progress"], "5/7")

    def test_record_start_resets_idle_trace_before_control_loop(self) -> None:
        class FakeDataset:
            num_episodes = 7

            def __init__(self) -> None:
                self.clear_calls = 0

            def clear_episode_buffer(self) -> None:
                self.clear_calls += 1

        with tempfile.TemporaryDirectory() as temp_dir:
            dashboard = RecordingWebDashboard(
                dataset_root=Path(temp_dir) / "dataset",
                port=0,
                open_browser=False,
            )
            dashboard.lock_settings({"episodes": 1})
            dashboard.begin_trace(
                label="READY manual control",
                joint_names=list(JOINT_NAMES),
                ranges=sample_ranges(),
            )
            dashboard.append_trace_samples(
                states=[[1.0] * 6],
                actions=[[1.0] * 6],
                leaders=[[1.0] * 6],
            )
            previous_revision = dashboard.trace_snapshot()["revision"]
            events = {
                "exit_early": False,
                "rerecord_episode": False,
                "stop_recording": False,
            }
            dataset = FakeDataset()
            captured: dict[str, object] = {}

            def inspect_record_start(**_kwargs) -> None:
                captured.update(dashboard.trace_snapshot())
                events["stop_recording"] = True

            settings = RuntimeSettings(
                control_fps=20,
                episodes=1,
                episode_seconds=30,
                reset_seconds=0,
                preview_hz=10,
                calibration="none",
                motion_capture=MotionCaptureSettings(True, 4.0, 5.0, 0.5, 1.0),
            )
            with (
                mock.patch("run_record.service_idle_mode", return_value="record"),
                mock.patch("run_record.run_control_segment", side_effect=inspect_record_start),
            ):
                saved = record_episodes_web(
                    dataset=dataset,
                    follower=object(),
                    leader=object(),
                    follower_port="FAKE_FOLLOWER",
                    leader_port="FAKE_LEADER",
                    task="trace reset test",
                    settings=settings,
                    events=events,
                    dashboard=dashboard,
                    ranges=sample_ranges(),
                )

            self.assertEqual(saved, 0)
            self.assertEqual(captured["label"], "RECORD episode 7")
            self.assertEqual(captured["revision"], previous_revision + 1)
            for field in ("steps", "states", "actions", "leaders"):
                self.assertEqual(captured[field], [])
            self.assertEqual(dataset.clear_calls, 1)


class MotionTriggeredCaptureTest(unittest.TestCase):
    @staticmethod
    def values(**updates: float) -> dict[str, float]:
        values = {key: 0.0 for key in JOINT_NAMES}
        values.update(updates)
        return values

    @staticmethod
    def sample(index: int, values: dict[str, float]) -> BufferedMotionSample:
        row = tuple(values[key] for key in JOINT_NAMES)
        return BufferedMotionSample(
            dataset_frame={"sample_index": index},
            state_values=row,
            action_values=row,
            leader_values=row,
            sample_time_s=index / 10,
        )

    def test_motion_speed_wraps_wrist_roll_and_separates_gripper_units(self) -> None:
        previous = self.values(**{"wrist_roll.pos": 179.0, "gripper.pos": 10.0})
        current = self.values(**{"wrist_roll.pos": -179.0, "gripper.pos": 11.0})
        body_speed, gripper_speed = measure_motion_speeds(previous, current, dt_s=0.05)
        self.assertAlmostEqual(body_speed, 40.0)
        self.assertAlmostEqual(gripper_speed, 20.0)

    def test_saved_rate_uses_only_captured_sample_times(self) -> None:
        self.assertAlmostEqual(calculate_sample_rate_hz([10.0, 10.05, 10.10]), 20.0)
        self.assertAlmostEqual(calculate_max_sample_interval_s([10.0, 10.05, 10.20]), 0.15)
        self.assertEqual(calculate_sample_rate_hz([]), 0.0)
        self.assertEqual(calculate_max_sample_interval_s([]), 0.0)

    def test_noise_and_one_tick_spike_do_not_trigger(self) -> None:
        settings = MotionCaptureSettings(True, 4.0, 5.0, 0.2, 0.2)
        gate = MotionTriggeredCapture(settings, fps=10)
        sequence = [0.0, 0.1, 1.0, 1.0, 1.1]
        updates = []
        for index, value in enumerate(sequence):
            values = self.values(**{"shoulder_pan.pos": value})
            updates.append(gate.update(values, self.sample(index, values)))
        self.assertFalse(gate.active)
        self.assertTrue(all(not update.samples for update in updates))

    def test_pre_roll_is_flushed_and_stationary_post_roll_auto_finishes(self) -> None:
        settings = MotionCaptureSettings(True, 4.0, 5.0, 0.2, 0.2)
        gate = MotionTriggeredCapture(settings, fps=10)
        sequence = [0.0, 0.1, 0.6, 1.1, 1.1, 1.1]
        saved_indices: list[int] = []
        updates = []
        for index, value in enumerate(sequence):
            values = self.values(**{"shoulder_pan.pos": value})
            update = gate.update(values, self.sample(index, values))
            updates.append(update)
            saved_indices.extend(int(sample.dataset_frame["sample_index"]) for sample in update.samples)
        self.assertTrue(updates[3].just_triggered)
        self.assertEqual(saved_indices, [0, 1, 2, 3, 4, 5])
        self.assertFalse(updates[4].should_finish)
        self.assertTrue(updates[5].should_finish)
        self.assertEqual(updates[5].status, "COMPLETE")

    def test_short_pause_is_preserved_and_stop_counter_resets_on_motion(self) -> None:
        settings = MotionCaptureSettings(True, 4.0, 5.0, 0.1, 0.3)
        gate = MotionTriggeredCapture(settings, fps=10)
        sequence = [0.0, 0.5, 1.0, 1.0, 1.0, 1.5, 1.5, 1.5, 1.5]
        updates = []
        for index, value in enumerate(sequence):
            values = self.values(**{"shoulder_pan.pos": value})
            updates.append(gate.update(values, self.sample(index, values)))
        self.assertFalse(any(update.should_finish for update in updates[:-1]))
        self.assertTrue(updates[-1].should_finish)
        self.assertEqual(
            sum(len(update.samples) for update in updates),
            len(sequence),
            "the short pause must remain inside the continuous saved window",
        )

    def test_large_leader_follower_error_prevents_false_auto_complete(self) -> None:
        settings = MotionCaptureSettings(True, 4.0, 5.0, 0.1, 0.2)
        gate = MotionTriggeredCapture(settings, fps=10)
        follower = self.values()
        leader_positions = [0.0, 1.0, 2.0, 10.0, 10.0, 10.0, 10.0]
        updates = []
        for index, value in enumerate(leader_positions):
            leader = self.values(**{"shoulder_pan.pos": value})
            updates.append(
                gate.update(
                    follower,
                    self.sample(index, follower),
                    leader_values=leader,
                    dt_s=0.1,
                )
            )
        self.assertTrue(gate.active)
        self.assertFalse(any(update.should_finish for update in updates))

    def test_gripper_motion_alone_can_trigger_capture(self) -> None:
        settings = MotionCaptureSettings(True, 4.0, 5.0, 0.1, 0.2)
        gate = MotionTriggeredCapture(settings, fps=10)
        updates = []
        for index, value in enumerate([0.0, 1.0, 2.0]):
            values = self.values(**{"gripper.pos": value})
            updates.append(gate.update(values, self.sample(index, values), dt_s=0.1))
        self.assertTrue(gate.active)
        self.assertTrue(updates[-1].just_triggered)

    def test_grasp_blocking_gripper_position_does_not_prevent_auto_complete(self) -> None:
        settings = MotionCaptureSettings(True, 4.0, 5.0, 0.1, 0.2)
        gate = MotionTriggeredCapture(settings, fps=10)
        follower_positions = [0.0, 1.0, 2.0, 2.0, 2.0]
        leader_positions = [0.0, 10.0, 20.0, 20.0, 20.0]
        updates = []
        for index, (follower_value, leader_value) in enumerate(
            zip(follower_positions, leader_positions, strict=True)
        ):
            follower = self.values(**{"gripper.pos": follower_value})
            leader = self.values(**{"gripper.pos": leader_value})
            updates.append(
                gate.update(
                    follower,
                    self.sample(index, follower),
                    leader_values=leader,
                    dt_s=0.1,
                )
            )
        self.assertTrue(updates[-1].should_finish)

    def test_control_loop_saves_only_one_continuous_motion_window(self) -> None:
        sequence = [0.0] * 10 + [0.4, 0.8, 0.8, 0.8]

        class FakeDataset:
            def __init__(self) -> None:
                self.features = {
                    "observation.state": {
                        "dtype": "float32",
                        "shape": (6,),
                        "names": list(JOINT_NAMES),
                    },
                    "observation.images.front": {
                        "dtype": "video",
                        "shape": (2, 2, 3),
                        "names": ["height", "width", "channels"],
                    },
                    "action": {
                        "dtype": "float32",
                        "shape": (6,),
                        "names": list(JOINT_NAMES),
                    },
                }
                self.frames: list[dict] = []

            def add_frame(self, frame: dict) -> None:
                self.frames.append(frame)

        class FakeFollower:
            def __init__(self) -> None:
                self.index = -1
                self.value = 0.0

            def get_observation(self) -> dict[str, object]:
                self.index += 1
                self.value = sequence[min(self.index, len(sequence) - 1)]
                observation: dict[str, object] = {
                    "front": np.full((2, 2, 3), self.index, dtype=np.uint8)
                }
                observation.update({key: 0.0 for key in JOINT_NAMES})
                observation["shoulder_pan.pos"] = self.value
                return observation

            def send_action(self, action: dict[str, float]) -> dict[str, float]:
                return dict(action)

        class FakeLeader:
            def __init__(self, follower: FakeFollower) -> None:
                self.follower = follower

            def get_action(self) -> dict[str, float]:
                action = {key: 0.0 for key in JOINT_NAMES}
                action["shoulder_pan.pos"] = self.follower.value
                return action

        with tempfile.TemporaryDirectory() as temp_dir:
            dashboard = RecordingWebDashboard(
                dataset_root=Path(temp_dir) / "dataset",
                port=0,
                open_browser=False,
            )
            dashboard.begin_trace(
                label="motion window",
                joint_names=list(JOINT_NAMES),
                ranges=sample_ranges(),
            )
            follower = FakeFollower()
            dataset = FakeDataset()
            with mock.patch("run_record.precise_sleep", return_value=None):
                result = run_control_segment(
                    follower=follower,
                    leader=FakeLeader(follower),
                    follower_port="FAKE_FOLLOWER",
                    leader_port="FAKE_LEADER",
                    duration_s=10.0,
                    fps=20,
                    events={
                        "exit_early": False,
                        "rerecord_episode": False,
                        "stop_recording": False,
                    },
                    label="MOTION",
                    dashboard=dashboard,
                    dataset=dataset,
                    task="motion test",
                    motion_capture=MotionCaptureSettings(True, 4.0, 5.0, 0.1, 0.1),
                )
            trace = dashboard.trace_snapshot()
            dashboard.close()
        self.assertEqual(result.control_frames, 14)
        self.assertEqual(result.saved_frames, 6)
        self.assertEqual(len(dataset.frames), 6)
        self.assertEqual(len(result.states), 6)
        self.assertEqual(int(dataset.frames[0]["observation.images.front"][0, 0, 0]), 8)
        self.assertEqual(trace["steps"], list(range(6)))
        self.assertEqual(trace["states"], [list(row) for row in result.states])

    def test_no_motion_keeps_dataset_and_trace_empty(self) -> None:
        events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}

        class FakeDataset:
            def __init__(self) -> None:
                self.features = {
                    "observation.state": {
                        "dtype": "float32",
                        "shape": (6,),
                        "names": list(JOINT_NAMES),
                    },
                    "observation.images.front": {
                        "dtype": "video",
                        "shape": (2, 2, 3),
                        "names": ["height", "width", "channels"],
                    },
                    "action": {
                        "dtype": "float32",
                        "shape": (6,),
                        "names": list(JOINT_NAMES),
                    },
                }
                self.frames: list[dict] = []

            def add_frame(self, frame: dict) -> None:
                self.frames.append(frame)

        class StationaryFollower:
            def __init__(self) -> None:
                self.frames = 0

            def get_observation(self) -> dict[str, object]:
                self.frames += 1
                if self.frames >= 8:
                    events["exit_early"] = True
                return {
                    "front": np.zeros((2, 2, 3), dtype=np.uint8),
                    **{key: 0.0 for key in JOINT_NAMES},
                }

            def send_action(self, action: dict[str, float]) -> dict[str, float]:
                return dict(action)

        class StationaryLeader:
            def get_action(self) -> dict[str, float]:
                return {key: 0.0 for key in JOINT_NAMES}

        with tempfile.TemporaryDirectory() as temp_dir:
            dashboard = RecordingWebDashboard(
                dataset_root=Path(temp_dir) / "dataset",
                port=0,
                open_browser=False,
            )
            dashboard.begin_trace(
                label="no motion",
                joint_names=list(JOINT_NAMES),
                ranges=sample_ranges(),
            )
            dataset = FakeDataset()
            with mock.patch("run_record.precise_sleep", return_value=None):
                result = run_control_segment(
                    follower=StationaryFollower(),
                    leader=StationaryLeader(),
                    follower_port="FAKE_FOLLOWER",
                    leader_port="FAKE_LEADER",
                    duration_s=10.0,
                    fps=20,
                    events=events,
                    label="NO MOTION",
                    dashboard=dashboard,
                    dataset=dataset,
                    task="no motion test",
                    motion_capture=MotionCaptureSettings(True, 4.0, 5.0, 0.1, 0.1),
                )
            trace = dashboard.trace_snapshot()
            state = dashboard.state_snapshot()
            dashboard.close()
        self.assertEqual(result.control_frames, 8)
        self.assertEqual(result.saved_frames, 0)
        self.assertEqual(dataset.frames, [])
        self.assertEqual(trace["steps"], [])
        self.assertEqual(state["motion_status"], "NO MOTION")


class ControlRateTest(unittest.TestCase):
    def test_bad_preview_frame_does_not_kill_encoder_worker(self) -> None:
        errors: list[str] = []
        encoder = LatestFrameEncoder(errors.append)
        encoder.start()
        try:
            encoder.submit(np.zeros((4, 4), dtype=np.uint8))
            error_deadline = time.monotonic() + 2
            while not errors and time.monotonic() < error_deadline:
                time.sleep(0.01)
            encoder.submit(np.full((8, 8, 3), 80, dtype=np.uint8))
            deadline = time.monotonic() + 2
            while encoder.snapshot()[1] < 1 and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual(encoder.snapshot()[1], 1)
            self.assertTrue(errors)
        finally:
            encoder.close()

    def test_non_finite_leader_action_is_rejected_before_motor_write(self) -> None:
        class FakeFollower:
            def __init__(self) -> None:
                self.sent: list[dict[str, float]] = []

            def get_observation(self) -> dict[str, object]:
                return {"front": np.zeros((2, 2, 3), dtype=np.uint8), **{key: 0.0 for key in JOINT_NAMES}}

            def send_action(self, action: dict[str, float]) -> dict[str, float]:
                self.sent.append(dict(action))
                return dict(action)

        class FakeLeader:
            def get_action(self) -> dict[str, float]:
                action = {key: 0.0 for key in JOINT_NAMES}
                action["shoulder_pan.pos"] = float("nan")
                return action

        with tempfile.TemporaryDirectory() as temp_dir:
            dashboard = RecordingWebDashboard(
                dataset_root=Path(temp_dir) / "dataset",
                port=0,
                open_browser=False,
            )
            follower = FakeFollower()
            with self.assertRaisesRegex(RuntimeError, "non-finite"):
                run_control_segment(
                    follower=follower,
                    leader=FakeLeader(),
                    follower_port="FAKE_FOLLOWER",
                    leader_port="FAKE_LEADER",
                    duration_s=0.1,
                    fps=20,
                    events={
                        "exit_early": False,
                        "rerecord_episode": False,
                        "stop_recording": False,
                    },
                    label="FINITE SAFETY",
                    dashboard=dashboard,
                )
            self.assertEqual(follower.sent, [])

    def test_preview_scheduler_does_not_alias_with_20hz_control(self) -> None:
        class CountingEncoder:
            def __init__(self) -> None:
                self.count = 0

            def submit(self, _frame: np.ndarray) -> None:
                self.count += 1

        with tempfile.TemporaryDirectory() as temp_dir:
            dashboard = RecordingWebDashboard(
                dataset_root=Path(temp_dir) / "dataset",
                port=0,
                open_browser=False,
            )
            encoder = CountingEncoder()
            dashboard._frame_encoder = encoder
            dashboard.set_camera_preview_hz(10)
            frame = np.zeros((2, 2, 3), dtype=np.uint8)
            rows = [{"joint": "x", "follower": 0, "leader": 0, "command": 0, "error": 0}]
            tick_times = [index / 20 for index in range(21)]
            with mock.patch("recording_web_dashboard.time.monotonic", side_effect=tick_times):
                for _ in tick_times:
                    dashboard.publish_sample(
                        camera_rgb=frame,
                        joint_rows=rows,
                        state_values=[],
                        action_values=[],
                        leader_values=None,
                        remaining_s=0,
                        control_hz=20,
                        buffered_frames=0,
                        append_trace=False,
                    )
            self.assertGreaterEqual(encoder.count, 10)
            self.assertLessEqual(encoder.count, 11)

    def test_20hz_control_survives_concurrent_http_and_jpeg_work(self) -> None:
        class FakeFollower:
            def __init__(self) -> None:
                self.step = 0
                self.image = np.full((480, 640, 3), 100, dtype=np.uint8)

            def get_observation(self) -> dict[str, object]:
                self.step += 1
                observation: dict[str, object] = {"front": self.image}
                observation.update({key: float(self.step % 20) for key in JOINT_NAMES})
                return observation

            def send_action(self, action: dict[str, float]) -> dict[str, float]:
                return dict(action)

        class FakeLeader:
            def get_action(self) -> dict[str, float]:
                return {key: 10.0 for key in JOINT_NAMES}

        with tempfile.TemporaryDirectory() as temp_dir:
            dashboard = RecordingWebDashboard(
                dataset_root=Path(temp_dir) / "dataset",
                port=0,
                open_browser=False,
            )
            dashboard.prepare()
            dashboard.begin_trace(
                label="20 Hz load",
                joint_names=list(JOINT_NAMES),
                ranges=sample_ranges(),
            )
            stop_poll = threading.Event()

            def poll() -> None:
                while not stop_poll.is_set():
                    for endpoint in ("api/state", "api/trace", "camera.jpg"):
                        try:
                            urllib.request.urlopen(dashboard.url + endpoint, timeout=1).read()
                        except urllib.error.HTTPError as exc:
                            if exc.code != 204:
                                raise
                    time.sleep(0.03)

            poller = threading.Thread(target=poll, daemon=True)
            poller.start()
            try:
                result = run_control_segment(
                    follower=FakeFollower(),
                    leader=FakeLeader(),
                    follower_port="FAKE_FOLLOWER",
                    leader_port="FAKE_LEADER",
                    duration_s=1.5,
                    fps=20,
                    events={
                        "exit_early": False,
                        "rerecord_episode": False,
                        "stop_recording": False,
                    },
                    label="SYNTHETIC",
                    dashboard=dashboard,
                )
                self.assertGreaterEqual(result.rate_hz, 18.0)
                self.assertGreaterEqual(result.control_frames, 27)
                messages = [entry["message"] for entry in dashboard.state_snapshot()["logs"]]
                self.assertFalse(
                    any(message.startswith("SYNTHETIC ") and " Hz | " in message for message in messages),
                    "periodic joint telemetry must not be copied into the event log",
                )
            finally:
                stop_poll.set()
                poller.join(timeout=2)
                dashboard.close()


class PlaybackSafetyTest(unittest.TestCase):
    class FakeFollower:
        def __init__(self, value: float = 0.0) -> None:
            self.value = value
            self.sent: list[dict[str, float]] = []
            self.image = np.full((480, 640, 3), 110, dtype=np.uint8)

        def get_observation(self) -> dict[str, object]:
            observation: dict[str, object] = {"front": self.image}
            observation.update({key: self.value for key in JOINT_NAMES})
            return observation

        def send_action(self, action: dict[str, float]) -> dict[str, float]:
            self.sent.append(dict(action))
            return dict(action)

    def make_scenario(self, dashboard: RecordingWebDashboard, steps: int = 100) -> None:
        dashboard.library.save(
            episode_index=0,
            task="playback test",
            fps=20,
            joint_names=list(JOINT_NAMES),
            ranges=sample_ranges(),
            states=[[0.0] * 6 for _ in range(steps)],
            actions=[[0.0] * 6 for _ in range(steps)],
        )

    def assert_playback_terminal_state(self, state: dict[str, object]) -> None:
        self.assertIsNone(state["scenario_running"])
        self.assertIsNone(state["playback_step"])
        self.assertEqual(state["playback_total_steps"], 0)
        self.assertEqual(state["remaining_s"], 0.0)
        self.assertEqual(state["control_hz"], 0.0)
        self.assertEqual(state["buffered_frames"], 0)

    def test_stop_is_processed_within_a_few_control_ticks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dashboard = RecordingWebDashboard(
                dataset_root=Path(temp_dir) / "dataset",
                port=0,
                open_browser=False,
            )
            dashboard.prepare()
            self.make_scenario(dashboard)
            dashboard.lock_settings({"episodes": 5})
            follower = self.FakeFollower()
            events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}
            dashboard.touch_client()
            def increase_then_stop() -> None:
                dashboard._commands.put(
                    DashboardCommand(
                        action="increase_episode_target",
                        params={"additional_episodes": 2},
                    )
                )
                dashboard._commands.put(DashboardCommand(action="stop_scenario"))

            timer = threading.Timer(0.12, increase_then_stop)
            timer.start()
            started = time.perf_counter()
            try:
                run_scenario_playback(
                    scenario_id="episode-000000",
                    follower=follower,
                    follower_port="FAKE",
                    dashboard=dashboard,
                    ranges=sample_ranges(),
                    events=events,
                )
                playback_elapsed = time.perf_counter() - started
            finally:
                timer.join(timeout=1)
                dashboard.close()
            self.assertLess(playback_elapsed, 0.5)
            self.assertGreaterEqual(len(follower.sent), 2)
            self.assertLess(len(follower.sent), 10)
            self.assertEqual(dashboard.collection_target(), 7)
            final_state = dashboard.state_snapshot()
            self.assert_playback_terminal_state(final_state)

    def test_first_pose_mismatch_refuses_playback(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dashboard = RecordingWebDashboard(
                dataset_root=Path(temp_dir) / "dataset",
                port=0,
                open_browser=False,
            )
            dashboard.prepare()
            self.make_scenario(dashboard, steps=3)
            follower = self.FakeFollower(value=90.0)
            dashboard.touch_client()
            try:
                run_scenario_playback(
                    scenario_id="episode-000000",
                    follower=follower,
                    follower_port="FAKE",
                    dashboard=dashboard,
                    ranges=sample_ranges(),
                    events={
                        "exit_early": False,
                        "rerecord_episode": False,
                        "stop_recording": False,
                    },
                )
            finally:
                dashboard.close()
            self.assertEqual(follower.sent, [])

    def test_playback_trace_keeps_follower_observation_and_leader_target_separate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dashboard = RecordingWebDashboard(
                dataset_root=Path(temp_dir) / "dataset",
                port=0,
                open_browser=False,
            )
            dashboard.prepare()
            target_rows = [[float(step)] * len(JOINT_NAMES) for step in range(3)]
            dashboard.library.save(
                episode_index=0,
                task="timeline playback test",
                fps=20,
                joint_names=list(JOINT_NAMES),
                ranges=sample_ranges(),
                states=[[0.0] * len(JOINT_NAMES) for _ in target_rows],
                actions=target_rows,
            )
            follower = self.FakeFollower(value=0.0)
            dashboard.touch_client()
            try:
                with mock.patch("run_record.precise_sleep", return_value=None):
                    run_scenario_playback(
                        scenario_id="episode-000000",
                        follower=follower,
                        follower_port="FAKE",
                        dashboard=dashboard,
                        ranges=sample_ranges(),
                        events={
                            "exit_early": False,
                            "rerecord_episode": False,
                            "stop_recording": False,
                        },
                    )
                trace = dashboard.trace_snapshot()
            finally:
                dashboard.close()

            self.assertEqual(trace["steps"], [0, 1, 2])
            self.assertEqual(
                trace["states"],
                [[0.0] * len(JOINT_NAMES) for _ in target_rows],
            )
            self.assertEqual(trace["leaders"], target_rows)
            self.assertEqual(trace["actions"], target_rows)

    def test_playback_progress_is_one_based_and_cleared_after_completion(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dashboard = RecordingWebDashboard(
                dataset_root=Path(temp_dir) / "dataset",
                port=0,
                open_browser=False,
            )
            dashboard.prepare()
            target_rows = [[float(step)] * len(JOINT_NAMES) for step in range(3)]
            dashboard.library.save(
                episode_index=0,
                task="playhead progress test",
                fps=20,
                joint_names=list(JOINT_NAMES),
                ranges=sample_ranges(),
                states=[[0.0] * len(JOINT_NAMES) for _ in target_rows],
                actions=target_rows,
            )

            class ProgressFollower(self.FakeFollower):
                def __init__(self) -> None:
                    super().__init__(value=0.0)
                    self.before_reads: list[tuple[str, int | None, int]] = []

                def get_observation(self) -> dict[str, object]:
                    state = dashboard.state_snapshot()
                    self.before_reads.append(
                        (
                            state["phase"],
                            state["playback_step"],
                            state["playback_total_steps"],
                        )
                    )
                    return super().get_observation()

            initial = dashboard.state_snapshot()
            self.assertIsNone(initial["playback_step"])
            self.assertEqual(initial["playback_total_steps"], 0)
            follower = ProgressFollower()
            after_transmissions: list[tuple[int | None, int]] = []
            dashboard.touch_client()
            try:
                def inspect_progress(_delay_s: float) -> None:
                    state = dashboard.state_snapshot()
                    after_transmissions.append(
                        (state["playback_step"], state["playback_total_steps"])
                    )

                with mock.patch("run_record.precise_sleep", side_effect=inspect_progress):
                    run_scenario_playback(
                        scenario_id="episode-000000",
                        follower=follower,
                        follower_port="FAKE",
                        dashboard=dashboard,
                        ranges=sample_ranges(),
                        events={
                            "exit_early": False,
                            "rerecord_episode": False,
                            "stop_recording": False,
                        },
                    )
                final_state = dashboard.state_snapshot()
            finally:
                dashboard.close()

            playback_reads = [
                (step, total)
                for phase, step, total in follower.before_reads
                if phase == "PLAYBACK"
            ]
            self.assertEqual(playback_reads[:3], [(0, 3), (1, 3), (2, 3)])
            self.assertEqual(after_transmissions, [(1, 3), (2, 3), (3, 3)])
            self.assert_playback_terminal_state(final_state)

    def test_playback_progress_is_cleared_when_motor_write_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dashboard = RecordingWebDashboard(
                dataset_root=Path(temp_dir) / "dataset",
                port=0,
                open_browser=False,
            )
            dashboard.prepare()
            self.make_scenario(dashboard, steps=3)

            class FailingFollower(self.FakeFollower):
                def __init__(self) -> None:
                    super().__init__(value=0.0)
                    self.write_count = 0

                def send_action(self, action: dict[str, float]) -> dict[str, float]:
                    self.write_count += 1
                    if self.write_count == 2:
                        raise RuntimeError("synthetic playback write failure")
                    return super().send_action(action)

            follower = FailingFollower()
            dashboard.touch_client()
            try:
                with mock.patch("run_record.precise_sleep", return_value=None):
                    with self.assertRaisesRegex(RuntimeError, "synthetic playback write failure"):
                        run_scenario_playback(
                            scenario_id="episode-000000",
                            follower=follower,
                            follower_port="FAKE",
                            dashboard=dashboard,
                            ranges=sample_ranges(),
                            events={
                                "exit_early": False,
                                "rerecord_episode": False,
                                "stop_recording": False,
                            },
                        )
                final_state = dashboard.state_snapshot()
            finally:
                dashboard.close()

            self.assert_playback_terminal_state(final_state)
            messages = [entry["message"] for entry in final_state["logs"]]
            self.assertTrue(
                any("오류로 중단" in message for message in messages),
                "a motor write exception must be reported as an interrupted playback",
            )
            self.assertFalse(
                any("재생 완료" in message for message in messages),
                "a motor write exception must never be logged as successful completion",
            )


class HtmlContractTest(unittest.TestCase):
    def test_public_scope_excludes_train_agent_inference_and_policy_run(self) -> None:
        root = Path(__file__).resolve().parent
        html = (root / "recording_dashboard.html").read_text(encoding="utf-8")
        dashboard_source = (root / "recording_web_dashboard.py").read_text(encoding="utf-8")
        recorder_source = (root / "run_record.py").read_text(encoding="utf-8")

        forbidden_surface = (
            'id="agentTab"',
            'id="trainTab"',
            'id="trainStart"',
            'id="testStart"',
            'id="policyRunStart"',
            '"start_training"',
            '"stop_training"',
            '"start_inference_test"',
            '"stop_inference_test"',
            '"start_policy_run"',
            '"stop_policy_run"',
            '"configure_agent"',
            '"test_agent_decision"',
            "/api/training",
            "/api/inference-test",
            "/api/policy-run",
            "/api/agent",
        )
        for forbidden in forbidden_surface:
            with self.subTest(forbidden=forbidden, source="html"):
                self.assertNotIn(forbidden, html)
            with self.subTest(forbidden=forbidden, source="dashboard"):
                self.assertNotIn(forbidden, dashboard_source)
            with self.subTest(forbidden=forbidden, source="recorder"):
                self.assertNotIn(forbidden, recorder_source)

        for forbidden_module in (
            "agent_workflow",
            "training_supervisor",
            "parallel_training_coordinator",
            "inference_test_supervisor",
            "policy_run_supervisor",
            "policy_inference_worker",
            "run_train",
            "train_xpu_backend",
            "run_inference_collect",
        ):
            with self.subTest(forbidden_module=forbidden_module):
                self.assertNotIn(forbidden_module, dashboard_source)
                self.assertNotIn(forbidden_module, recorder_source)
                self.assertFalse((root / f"{forbidden_module}.py").exists())

    def test_compact_dashboard_and_live_trace_reset_contract(self) -> None:
        html = Path("recording_dashboard.html").read_text(encoding="utf-8")
        compact_html = "".join(html.split())

        self.assertEqual(html.count('id="resetTrace"'), 1)
        self.assertIn('class="joint-table"', html)
        self.assertIn("const TIMELINE_HEIGHT = 185", html)
        self.assertIn("--camera-panel-height:clamp(220px,min(42vh,70vw),560px)", compact_html)
        self.assertIn("height:var(--camera-panel-height)", compact_html)
        self.assertIn(".joint-table{align-self:start;", compact_html)
        self.assertNotIn("height:260px", compact_html)
        self.assertIn("#leaderChart,#followerChart{display:block;height:185px;}", compact_html)
        reset_start = compact_html.index("$('resetTrace').onclick=")
        reset_end = compact_html.index("$('startRecord').onclick=", reset_start)
        reset_handler = compact_html[reset_start:reset_end]
        self.assertIn("command('reset_trace')", reset_handler)
        self.assertIn("activateLiveView({clear:true", reset_handler)
        self.assertNotIn("delete_scenario", reset_handler)
        start_end = compact_html.index("$('finishSegment').onclick=", reset_end)
        start_handler = compact_html[reset_end:start_end]
        self.assertIn("command('start_record')", start_handler)
        self.assertIn("activateLiveView({clear:true", start_handler)

    def test_scenario_and_event_log_are_single_accessible_right_sidebar_tabs(self) -> None:
        html = Path("recording_dashboard.html").read_text(encoding="utf-8")
        compact_html = "".join(html.split())

        left_start = html.index('<section class="left">')
        aside_start = html.index("<aside", left_start)
        aside_end = html.index("</aside>", aside_start) + len("</aside>")
        left_html = html[left_start:aside_start]
        aside_html = html[aside_start:aside_end]

        self.assertNotIn('id="log"', left_html, "the old full-width log card must be removed")
        for element_id in (
            "scenarioTab",
            "logTab",
            "scenarioPanel",
            "logPanel",
            "scenarioList",
            "log",
            "logUnread",
        ):
            self.assertEqual(
                html.count(f'id="{element_id}"'),
                1,
                f"sidebar element {element_id!r} must exist exactly once",
            )
            self.assertIn(f'id="{element_id}"', aside_html)

        self.assertEqual(html.count('class="side-tab"'), 2)
        self.assertEqual(html.count('class="side-tab-panel"'), 2)
        self.assertEqual(html.count('role="tablist"'), 1)
        self.assertIn(
            'id="scenarioTab" class="side-tab" role="tab" '
            'aria-controls="scenarioPanel" aria-selected="true" tabindex="0"',
            html,
        )
        self.assertIn(
            'id="logTab" class="side-tab" role="tab" '
            'aria-controls="logPanel" aria-selected="false" tabindex="-1"',
            html,
        )
        self.assertIn(
            'id="scenarioPanel" class="side-tab-panel" role="tabpanel" '
            'aria-labelledby="scenarioTab"',
            html,
        )
        self.assertIn(
            'id="logPanel" class="side-tab-panel" role="tabpanel" '
            'aria-labelledby="logTab" hidden',
            html,
        )
        self.assertIn(".side-tab-panel[hidden]{display:none;}", compact_html)
        self.assertIn(".log-unread[hidden]{display:none;}", compact_html)

        toggle_start = compact_html.index("functionactivateSideTab(")
        toggle_end = compact_html.index("asyncfunctioncommand(", toggle_start)
        toggle_handler = compact_html[toggle_start:toggle_end]
        for contract in (
            "$('scenarioTab')",
            "$('scenarioPanel')",
            "$('logTab')",
            "$('logPanel')",
            "entry.button.setAttribute('aria-selected',String(selected))",
            "entry.button.tabIndex=selected?0:-1",
            "entry.panel.hidden=!selected",
            "if(name==='log')",
            "unreadLogCount=0",
            "requestAnimationFrame(",
            "log.scrollTop=log.scrollHeight",
        ):
            self.assertIn(contract, toggle_handler)

        self.assertIn(
            "$('scenarioTab').onclick=()=>activateSideTab('scenario')",
            compact_html,
        )
        self.assertIn(
            "$('logTab').onclick=()=>activateSideTab('log')",
            compact_html,
        )
        self.assertIn("constsideTabOrder=['scenario','log']", compact_html)
        self.assertIn(".addEventListener('keydown',event=>", compact_html)
        for key_name in ("ArrowLeft", "ArrowRight", "Home", "End"):
            self.assertIn(f"event.key==='{key_name}'", compact_html)
        self.assertIn("activateSideTab(sideTabOrder[nextIndex],{focus:true})", compact_html)

        update_start = compact_html.index("functionupdateState(s){")
        update_end = compact_html.index("functionupdateModal(s){", update_start)
        update_handler = compact_html[update_start:update_end]
        for contract in (
            "s.logs.at(-1)?.seq||0",
            "constlog=$('log')",
            "s.logs.filter(entry=>Number(entry.seq)>lastLogSeq).length",
            "s.logs.map(",
            "x.level.toUpperCase()",
            "x.message",
            "log.textContent=",
            "log.scrollTop=log.scrollHeight",
            "if(activeSideTab!=='log')unreadLogCount+=newLogCount",
            "lastLogSeq=newestLogSeq",
            "updateLogUnread()",
        ):
            self.assertIn(contract, update_handler)

        render_start = compact_html.index("functionrenderScenarios(items){")
        render_end = compact_html.index("asyncfunctionselectScenario(", render_start)
        render_handler = compact_html[render_start:render_end]
        for contract in (
            "scenarioRenderSignature",
            "document.activeElement?.dataset?.scenarioId",
            "document.createElement('button')",
            "button.type='button'",
            "button.dataset.scenarioId=item.id",
            "button.setAttribute('aria-pressed'",
            "if(signature===scenarioRenderSignature)return",
            "focusTarget.focus({preventScroll:true})",
        ):
            self.assertIn(contract, render_handler)
        self.assertNotIn("document.createElement('div');div.className='scenario'", render_handler)

    def test_so101_joint_guide_has_six_canonical_markers_labels_and_colors(self) -> None:
        html = Path("recording_dashboard.html").read_text(encoding="utf-8")
        compact_html = "".join(html.split())
        expected_joints = (
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        )
        expected_labels = (
            "베이스 회전",
            "어깨 들기",
            "팔꿈치 굽힘",
            "손목 굽힘",
            "손목 회전",
            "그리퍼 열기/닫기",
        )

        self.assertEqual(html.count('id="jointGuide"'), 1)
        self.assertEqual(html.count('id="jointGuideTitle"'), 1)
        guide_id = html.index('id="jointGuide"')
        guide_start = html.rfind("<figure", 0, guide_id)
        guide_end = html.index("</figure>", guide_id) + len("</figure>")
        guide = html[guide_start:guide_end]
        self.assertIn('aria-labelledby="jointGuideTitle"', guide)
        self.assertIn(
            'class="joint-guide-image" src="/assets/so101-follower-official.webp"',
            guide,
        )
        self.assertIn('alt="실제 SO-101 Follower 암의 측면 모습"', guide)
        self.assertIn('href="https://huggingface.co/docs/lerobot/so101"', guide)
        self.assertIn('target="_blank" rel="noopener noreferrer"', guide)

        marker_tags = re.findall(
            r'<span class="joint-guide-marker"[^>]*>\s*[1-6]\s*</span>',
            guide,
        )
        item_tags = re.findall(
            r'<li class="joint-guide-item"[^>]*>.*?</li>',
            guide,
            flags=re.DOTALL,
        )
        self.assertEqual(len(marker_tags), 6)
        self.assertEqual(len(item_tags), 6)
        self.assertEqual(
            [re.search(r'data-joint="([^"]+)"', tag).group(1) for tag in marker_tags],
            list(expected_joints),
        )
        self.assertEqual(
            [re.search(r'data-joint="([^"]+)"', tag).group(1) for tag in item_tags],
            list(expected_joints),
        )
        self.assertTrue(all('aria-hidden="true"' in tag for tag in marker_tags))
        self.assertEqual(
            [re.search(r'>\s*([1-6])\s*</span>$', tag).group(1) for tag in marker_tags],
            ["1", "2", "3", "4", "5", "6"],
        )
        self.assertEqual(
            re.findall(r'<span class="joint-guide-number">\s*([1-6])\s*</span>', guide),
            ["1", "2", "3", "4", "5", "6"],
        )
        self.assertEqual(
            re.findall(r'<code class="joint-guide-key">\s*([^<]+?)\s*</code>', guide),
            list(expected_joints),
        )
        self.assertEqual(
            re.findall(r'<span class="joint-guide-label">\s*([^<]+?)\s*</span>', guide),
            list(expected_labels),
        )

        palette = (
            "constJOINT_COLORS=Object.freeze("
            "['#36d7e8','#ffad42','#c792ea','#5bd88f','#ff6b8a','#ffd166']);"
        )
        joint_keys = (
            "constJOINT_KEYS=Object.freeze("
            "['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper']);"
        )
        self.assertEqual(compact_html.count(palette), 1)
        self.assertEqual(compact_html.count(joint_keys), 1)
        color_start = compact_html.index("functionjointColor(")
        color_end = compact_html.index("functionupdateLogUnread(", color_start)
        color_contract = compact_html[color_start:color_end]
        for contract in (
            "replace(/\\.pos$/,'')",
            "JOINT_KEYS.indexOf(key)",
            "JOINT_COLORS[(canonical>=0?canonical:index)%JOINT_COLORS.length]",
            "querySelectorAll('.joint-guide-marker,.joint-guide-item')",
            "style.setProperty('--joint-color',jointColor(element.dataset.joint,index))",
        ):
            self.assertIn(contract, color_contract)
        self.assertIn("syncJointGuideColors();", compact_html)

    def test_joint_guide_and_sidebar_reflow_without_changing_camera_or_timeline(self) -> None:
        html = Path("recording_dashboard.html").read_text(encoding="utf-8")
        compact_html = "".join(html.split())

        self.assertEqual(
            compact_html.count(
                "--camera-panel-height:clamp(220px,min(42vh,70vw),560px)"
            ),
            1,
        )
        self.assertIn(".camera-wrap{display:grid;", compact_html)
        self.assertIn("height:var(--camera-panel-height)", compact_html)
        self.assertIn("#camera{width:100%;height:100%;", compact_html)
        self.assertIn(
            ".trajectory-layout{display:grid;grid-template-columns:clamp(230px,20vw,290px)minmax(0,1fr);min-width:0;}",
            compact_html,
        )

        tablet_start = compact_html.index("@media(max-width:1050px){")
        phone_start = compact_html.index("@media(max-width:760px){", tablet_start)
        tablet_css = compact_html[tablet_start:phone_start]
        phone_css = compact_html[phone_start:compact_html.index("</style>", phone_start)]
        for contract in (
            "main{grid-template-columns:1fr}",
            ".camera-wrap{grid-template-columns:1fr;height:auto;min-height:0}",
            "#camera{height:var(--camera-panel-height)}",
            "aside{position:static;order:-1}",
            ".side-tab-panel:not([hidden]){height:min(320px,42vh);min-height:220px}",
        ):
            self.assertIn(contract, tablet_css)
        for contract in (
            ".trajectory-layout{grid-template-columns:1fr}",
            ".joint-guide{border-right:0;border-bottom:1pxsolidvar(--line)}",
            ".joint-guide-visual{max-width:420px;margin:0auto}",
            ".joint-guide-list{grid-template-columns:repeat(2,minmax(0,1fr))}",
        ):
            self.assertIn(contract, phone_css)

        for element_id in (
            "chartScroll",
            "chartTrack",
            "leaderChart",
            "followerChart",
            "leaderLegend",
            "followerLegend",
        ):
            self.assertEqual(html.count(f'id="{element_id}"'), 1)
        for function_name in (
            "applyLiveTrace",
            "scheduleGraphDraw",
            "renderJointLegends",
            "drawPlayhead",
            "drawTimeline",
            "ensurePlayheadVisible",
            "drawGraph",
            "activateLiveView",
        ):
            self.assertEqual(
                compact_html.count(f"function{function_name}("),
                1,
                f"{function_name} must survive the layout-only refactor",
            )
        self.assertIn("window.addEventListener('resize',drawGraph)", compact_html)

    def test_playback_playhead_is_shared_red_progress_and_auto_scrolled(self) -> None:
        html = Path("recording_dashboard.html").read_text(encoding="utf-8")

        self.assertEqual(html.count('id="playbackProgress"'), 1)
        self.assertIn('id="playbackProgress" class="playback-progress" hidden', html)
        for contract_name in (
            "PLAYHEAD_COLOR",
            "getActivePlayhead",
            "drawPlayhead",
            "ensurePlayheadVisible",
            "playback_step",
            "playback_total_steps",
        ):
            self.assertIn(contract_name, html)

        color_lines = [line.lower() for line in html.splitlines() if "playhead_color" in line.lower()]
        self.assertTrue(
            any("#f" in line or "red" in line or "rgb(2" in line for line in color_lines),
            "the shared playhead color must be visibly red",
        )
        self.assertGreaterEqual(
            html.count("drawPlayhead"),
            2,
            "both canvases must be rendered through the shared playhead drawing path",
        )
        self.assertGreaterEqual(
            html.count("ensurePlayheadVisible"),
            2,
            "playhead visibility logic must be defined and invoked during playback updates",
        )
        compact_html = "".join(html.split())
        self.assertTrue(
            any(
                expression in compact_html
                for expression in ("step-1", "current-1", "rawStep-1", "playbackStep-1")
            ),
            "the 1-based backend step must map to a 0-based x index",
        )
        self.assertIn("$('chartScroll')", html)
        self.assertIn("playheadIndex", html)

    def test_playhead_follow_can_be_disabled_by_manual_navigation_and_zoom_keeps_source_step(self) -> None:
        html = Path("recording_dashboard.html").read_text(encoding="utf-8")
        compact_html = "".join(html.split())

        self.assertEqual(html.count('id="followPlayhead"'), 1)
        self.assertIn(">진행선따라가기</button>", compact_html)
        for contract_name in (
            "playheadFollow",
            "activePlaybackRun",
            "programmaticTimelineScroll",
            "setTimelineScrollLeft",
            "disablePlayheadFollow",
            "updatePlayheadFollowButton",
        ):
            self.assertIn(contract_name, html)

        self.assertIn(
            "if(!programmaticTimelineScroll)disablePlayheadFollow()",
            compact_html,
            "a genuine scrollbar move must stop automatic playhead following",
        )
        for event_name in ("wheel", "pointerdown", "touchstart"):
            self.assertIn(
                f"$('chartScroll').addEventListener('{event_name}',disablePlayheadFollow",
                compact_html,
                f"manual {event_name} navigation must stop automatic playhead following",
            )
        self.assertIn(
            "if(!skipTimelineScroll&&playheadFollow)ensurePlayheadVisible",
            compact_html,
            "drawing must move the shared scrollbar only while follow mode is enabled",
        )
        self.assertIn(
            "$('followPlayhead').onclick=()=>{if(!getActivePlayhead())return;playheadFollow=true;",
            compact_html,
            "the follow button must explicitly re-enable automatic playhead following",
        )

        zoom_start = compact_html.index("$('timelineZoom').addEventListener('change',()=>{")
        zoom_end = compact_html.index("$('followPlayhead').onclick", zoom_start)
        zoom_handler = compact_html[zoom_start:zoom_end]
        ordered_markers = (
            "constsourceStep=",
            "drawGraph({skipTimelineScroll:true})",
            "setTimelineScrollLeft(",
            "if(playheadFollow)ensurePlayheadVisible(",
        )
        marker_positions = [zoom_handler.index(marker) for marker in ordered_markers]
        self.assertEqual(
            marker_positions,
            sorted(marker_positions),
            "zoom must calculate the source-step anchor, redraw, restore that anchor, then follow the playhead",
        )

    def test_leader_and_follower_timelines_share_joint_palette_and_legends(self) -> None:
        html = Path("recording_dashboard.html").read_text(encoding="utf-8")

        for element_id in (
            "leaderChart",
            "followerChart",
            "chartScroll",
            "chartTrack",
            "leaderLegend",
            "followerLegend",
        ):
            self.assertEqual(
                html.count(f'id="{element_id}"'),
                1,
                f"timeline element {element_id!r} must exist exactly once",
            )

        self.assertNotIn('id="leaderChartScroll"', html)
        self.assertNotIn('id="followerChartScroll"', html)
        self.assertEqual(html.count('class="joint-legend"'), 2)
        self.assertIn("JOINT_COLORS", html)
        self.assertGreaterEqual(
            html.count("JOINT_COLORS"),
            2,
            "both timelines must obtain joint colors from one shared palette",
        )
        compact_html = "".join(html.split())
        self.assertIn("drawTimeline($('leaderChart'),leaderRows,options)", compact_html)
        self.assertIn("drawTimeline($('followerChart'),graphData.states||[],options)", compact_html)
        self.assertIn(
            "leaders:data.leaders||data.actions",
            compact_html,
            "saved scenarios must use recorded leader rows and fall back to actions for legacy files",
        )

    def test_timeline_zoom_is_display_only_and_does_not_change_robot_playback_speed(self) -> None:
        html = Path("recording_dashboard.html").read_text(encoding="utf-8")

        self.assertEqual(html.count('id="timelineZoom"'), 1)
        self.assertTrue(
            '<select id="timelineZoom"' in html or '<input id="timelineZoom"' in html,
            "timelineZoom must be a local display control",
        )
        zoom_lines = [line for line in html.splitlines() if "timelineZoom" in line]
        self.assertTrue(
            any(
                "addEventListener" in line and ("input" in line or "change" in line)
                for line in zoom_lines
            ),
            "timeline zoom must redraw in response to display-slider input",
        )
        self.assertTrue(
            all("command(" not in line for line in zoom_lines),
            "timeline zoom must not send a robot command",
        )
        self.assertNotIn("speed_multiplier", html)
        self.assertNotIn("playback_speed", html)
        self.assertNotIn("playbackSpeed", html)
        self.assertIn("command('run_scenario',selectedScenario)", html)

    def test_graph_scroll_menu_and_calibration_controls_exist(self) -> None:
        html = Path("recording_dashboard.html").read_text(encoding="utf-8")
        compact_html = "".join(html.split())
        self.assertEqual(DEFAULT_EPISODES, 200)
        self.assertEqual(DEFAULT_EPISODE_SECONDS, 60.0)
        with mock.patch("sys.argv", ["run_record.py", "--task", "default check"]):
            args = parse_args()
        self.assertEqual(args.episodes, 200)
        self.assertEqual(args.episode_seconds, 60.0)
        self.assertIn("overflow-x:auto", html)
        self.assertIn('id="settingsMenu"', html)
        self.assertIn('id="calibrationMenu"', html)
        self.assertIn('id="runScenario"', html)
        self.assertIn('id="stopScenario"', html)
        self.assertIn('id="deleteScenario"', html)
        self.assertIn('id="calibrationRows"', html)
        self.assertIn('id="runtimeSettingsModal"', html)
        self.assertIn('id="additionalEpisodes"', html)
        self.assertIn('id="extendTarget"', html)
        self.assertIn('id="stopCollection" class="danger">프로그램 종료</button>', html)
        self.assertIn("increase_episode_target", html)
        self.assertIn('#alignmentModal th:last-child, #alignmentModal td:last-child', html)
        self.assertIn('id="controlFps" type="number" min="5" max="20" step="1" value="20"', html)
        self.assertIn('id="targetEpisodes" type="number" min="1" max="1000" step="1" value="200"', html)
        self.assertIn('id="episodeSeconds" type="number" min="1" max="300" step="1" value="60"', html)
        self.assertIn("$('targetEpisodes').value=s.setup_defaults.episodes??200", compact_html)
        self.assertIn("$('episodeSeconds').value=s.setup_defaults.episode_seconds??60", compact_html)
        self.assertIn('id="resetSeconds" type="number" min="0" max="600" step="1" value="20"', html)
        self.assertIn('id="previewHz" type="number" min="1" max="10" step="1" value="10"', html)
        self.assertIn('id="motionTriggered"', html)
        self.assertIn('id="motionBodyThreshold" type="number" min="0.1" max="180" step="0.1" value="4"', html)
        self.assertIn('id="motionGripperThreshold" type="number" min="0.1" max="200" step="0.1" value="5"', html)
        self.assertIn('id="motionPreRoll" type="number" min="0" max="1" step="0.1" value="0.5"', html)
        self.assertIn('id="motionStopSeconds" type="number" min="0.1" max="10" step="0.1" value="1"', html)
        self.assertIn('id="motionStatus"', html)
        self.assertIn('<option value="none" selected>기존 calibration 사용</option>', html)


if __name__ == "__main__":
    unittest.main()
