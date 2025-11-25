"""
Comprehensive tests for MONITORING-001: Enhanced Monitoring Integration

Tests structured JSON logging, periodic health checks, task completion rate metrics,
gate failure tracking and alerting, and integration hooks.

Coverage targets:
- EventType enum: All event types exist
- MonitoringEvent: JSON serialization
- MetricsSnapshot: to_dict() method
- MonitoringMetrics: All metric tracking and calculations
- StructuredLogger: Event logging and rotation
- MonitoringHook: Hook dispatch and failure isolation
- Health check integration: Interval-based triggering
"""

import json
import pytest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open

# Import from orchestrator_loop
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from autonomous.orchestrator_loop import (
    EventType,
    MonitoringEvent,
    MetricsSnapshot,
    MonitoringMetrics,
    StructuredLogger,
    MonitoringHook,
    LoggingHook,
    MonitoringHookDispatcher
)


# ============================================================================
# EventType Enum Tests
# ============================================================================


class TestEventType:
    """Tests for EventType enum."""

    def test_all_event_types_exist(self):
        """Test that all expected event types are defined."""
        expected_types = [
            "CYCLE_START", "CYCLE_END", "HEALTH_CHECK", "TASK_STARTED",
            "TASK_COMPLETED", "TASK_FAILED", "GATE_PASSED", "GATE_FAILED",
            "INSTANCE_SPAWNED", "INSTANCE_STUCK", "INSTANCE_CRASHED",
            "INTERVENTION_SENT", "ORCHESTRATOR_STARTED", "ORCHESTRATOR_SHUTDOWN"
        ]
        for event_type in expected_types:
            assert hasattr(EventType, event_type), f"Missing event type: {event_type}"

    def test_event_type_values(self):
        """Test that event type values are correctly formatted (snake_case)."""
        assert EventType.CYCLE_START.value == "cycle_start"
        assert EventType.CYCLE_END.value == "cycle_end"
        assert EventType.HEALTH_CHECK.value == "health_check"
        assert EventType.TASK_COMPLETED.value == "task_completed"
        assert EventType.GATE_FAILED.value == "gate_failed"


# ============================================================================
# MonitoringEvent Tests
# ============================================================================


class TestMonitoringEvent:
    """Tests for MonitoringEvent dataclass."""

    def test_to_json_valid_format(self):
        """Test that to_json() produces valid JSON."""
        event = MonitoringEvent(
            event_type=EventType.CYCLE_START,
            timestamp="2025-11-25T20:00:00Z",
            cycle=1,
            payload={"uptime": "1m 0s"}
        )
        json_str = event.to_json()

        # Should be valid JSON
        data = json.loads(json_str)
        assert data["event"] == "cycle_start"
        assert data["cycle"] == 1
        assert data["uptime"] == "1m 0s"
        assert data["timestamp"] == "2025-11-25T20:00:00Z"

    def test_to_json_with_optional_fields(self):
        """Test JSON serialization with optional context fields."""
        event = MonitoringEvent(
            event_type=EventType.TASK_COMPLETED,
            timestamp="2025-11-25T20:00:00Z",
            cycle=5,
            payload={"duration_seconds": 120.5, "success": True},
            task_id="TASK-001",
            instance_id="instance-001",
            gate_name=None
        )
        json_str = event.to_json()
        data = json.loads(json_str)

        assert data["event"] == "task_completed"
        assert data["task_id"] == "TASK-001"
        assert data["instance_id"] == "instance-001"
        assert "gate_name" not in data  # None values excluded
        assert data["duration_seconds"] == 120.5
        assert data["success"] is True

    def test_to_json_gate_failure_event(self):
        """Test JSON format for gate failure event."""
        event = MonitoringEvent(
            event_type=EventType.GATE_FAILED,
            timestamp="2025-11-25T20:00:00Z",
            cycle=10,
            payload={"issues": ["Missing tests", "Low coverage"]},
            gate_name="design_review"
        )
        json_str = event.to_json()
        data = json.loads(json_str)

        assert data["event"] == "gate_failed"
        assert data["gate_name"] == "design_review"
        assert data["issues"] == ["Missing tests", "Low coverage"]

    def test_to_json_empty_payload(self):
        """Test JSON serialization with empty payload."""
        event = MonitoringEvent(
            event_type=EventType.ORCHESTRATOR_SHUTDOWN,
            timestamp="2025-11-25T20:00:00Z",
            cycle=100,
            payload={}
        )
        json_str = event.to_json()
        data = json.loads(json_str)

        assert data["event"] == "orchestrator_shutdown"
        assert data["cycle"] == 100
        # Empty payload should not add extra keys
        assert len(data) == 3  # event, timestamp, cycle only


# ============================================================================
# MetricsSnapshot Tests
# ============================================================================


class TestMetricsSnapshot:
    """Tests for MetricsSnapshot dataclass."""

    def test_to_dict_all_fields(self):
        """Test that to_dict() includes all fields."""
        snapshot = MetricsSnapshot(
            timestamp="2025-11-25T20:00:00Z",
            uptime_seconds=3600,
            cycle_count=60,
            tasks_completed=10,
            tasks_failed=2,
            tasks_in_progress=1,
            completion_rate=83.33,
            avg_task_duration_seconds=120.5,
            gates_passed=15,
            gates_failed=3,
            gate_failure_rate=16.67,
            instances_spawned=5,
            instances_crashed=1,
            interventions_sent=2,
            health_status="HEALTHY",
            active_instances=2,
            available_tasks=5
        )

        data = snapshot.to_dict()

        assert data["timestamp"] == "2025-11-25T20:00:00Z"
        assert data["uptime_seconds"] == 3600
        assert data["cycle_count"] == 60
        assert data["tasks_completed"] == 10
        assert data["tasks_failed"] == 2
        assert data["completion_rate"] == 83.33
        assert data["gates_failed"] == 3
        assert data["health_status"] == "HEALTHY"

    def test_to_dict_default_values(self):
        """Test to_dict() with default field values."""
        snapshot = MetricsSnapshot(
            timestamp="2025-11-25T20:00:00Z",
            uptime_seconds=0,
            cycle_count=0
        )

        data = snapshot.to_dict()

        assert data["tasks_completed"] == 0
        assert data["tasks_failed"] == 0
        assert data["completion_rate"] == 0.0
        assert data["health_status"] == "HEALTHY"


# ============================================================================
# MonitoringMetrics Tests
# ============================================================================


class TestMonitoringMetrics:
    """Tests for MonitoringMetrics class."""

    @pytest.fixture
    def metrics(self, tmp_path):
        """Create metrics instance with temporary storage."""
        with patch.object(MonitoringMetrics, 'METRICS_FILE', str(tmp_path / 'metrics.json')):
            return MonitoringMetrics()

    def test_record_task_completed_success(self, metrics):
        """Test recording successful task completion."""
        metrics.record_task_completed("TASK-001", 120.5, success=True)

        assert metrics.tasks_completed == 1
        assert metrics.tasks_failed == 0
        assert len(metrics.task_durations) == 1
        assert metrics.task_durations[0] == 120.5

    def test_record_task_completed_failure(self, metrics):
        """Test recording failed task completion."""
        metrics.record_task_completed("TASK-002", 60.0, success=False)

        assert metrics.tasks_completed == 0
        assert metrics.tasks_failed == 1
        assert len(metrics.task_durations) == 1
        assert metrics.task_durations[0] == 60.0

    def test_record_multiple_tasks(self, metrics):
        """Test recording multiple task completions."""
        metrics.record_task_completed("TASK-001", 100.0, success=True)
        metrics.record_task_completed("TASK-002", 200.0, success=True)
        metrics.record_task_completed("TASK-003", 50.0, success=False)

        assert metrics.tasks_completed == 2
        assert metrics.tasks_failed == 1
        assert len(metrics.task_durations) == 3

    def test_completion_rate_all_success(self, metrics):
        """Test completion rate with 100% success."""
        metrics.record_task_completed("TASK-001", 100.0, success=True)
        metrics.record_task_completed("TASK-002", 100.0, success=True)

        assert metrics.get_completion_rate() == 100.0

    def test_completion_rate_half_failed(self, metrics):
        """Test completion rate with 50% failure."""
        metrics.record_task_completed("TASK-001", 100.0, success=True)
        metrics.record_task_completed("TASK-002", 100.0, success=False)

        assert metrics.get_completion_rate() == 50.0

    def test_completion_rate_all_failed(self, metrics):
        """Test completion rate with 100% failure."""
        metrics.record_task_completed("TASK-001", 100.0, success=False)
        metrics.record_task_completed("TASK-002", 100.0, success=False)

        assert metrics.get_completion_rate() == 0.0

    def test_completion_rate_no_tasks(self, metrics):
        """Test completion rate with zero tasks (edge case)."""
        # No tasks recorded - should return 100% (no failures)
        assert metrics.get_completion_rate() == 100.0

    def test_record_gate_result_passed(self, metrics):
        """Test recording gate pass."""
        metrics.record_gate_result("design_review", passed=True)

        assert metrics.gates_passed == 1
        assert metrics.gates_failed == 0
        assert len(metrics.recent_gate_failures) == 0

    def test_record_gate_result_failed(self, metrics):
        """Test recording gate failure."""
        metrics.record_gate_result("design_review", passed=False, issues=["Missing tests"])

        assert metrics.gates_passed == 0
        assert metrics.gates_failed == 1
        assert len(metrics.recent_gate_failures) == 1
        assert metrics.recent_gate_failures[0]["gate_name"] == "design_review"
        assert metrics.recent_gate_failures[0]["issues"] == ["Missing tests"]

    def test_gate_failure_rate_calculation(self, metrics):
        """Test gate failure rate calculation."""
        metrics.record_gate_result("gate1", passed=True)
        metrics.record_gate_result("gate2", passed=True)
        metrics.record_gate_result("gate3", passed=False)
        metrics.record_gate_result("gate4", passed=False)

        # 2 failed out of 4 total = 50%
        assert metrics.get_gate_failure_rate() == 50.0

    def test_gate_failure_rate_no_gates(self, metrics):
        """Test gate failure rate with no gates (edge case)."""
        assert metrics.get_gate_failure_rate() == 0.0

    def test_recent_gate_failure_tracking(self, metrics):
        """Test that gate failures outside 60-minute window are cleaned up."""
        # Use real datetime for this test - easier than mocking
        base_time = datetime.utcnow()

        # Record failure at time 0
        metrics.record_gate_result("gate1", passed=False, issues=["Issue 1"])

        # Manually add old failure (outside 60-minute window)
        old_failure = {
            "gate_name": "gate_old",
            "timestamp": (base_time - timedelta(minutes=65)).isoformat(),
            "issues": ["Old issue"]
        }
        metrics.recent_gate_failures.insert(0, old_failure)

        # Should have 2 failures
        assert len(metrics.recent_gate_failures) == 2

        # Clean old failures
        metrics._clean_old_gate_failures()

        # Old failure should be removed, only recent one remains
        assert len(metrics.recent_gate_failures) == 1
        assert metrics.recent_gate_failures[0]["gate_name"] == "gate1"

    def test_avg_task_duration_calculation(self, metrics):
        """Test average task duration calculation."""
        metrics.record_task_completed("TASK-001", 100.0, success=True)
        metrics.record_task_completed("TASK-002", 200.0, success=True)
        metrics.record_task_completed("TASK-003", 300.0, success=True)

        assert metrics.get_avg_task_duration() == 200.0

    def test_avg_task_duration_no_tasks(self, metrics):
        """Test average duration with no tasks (edge case)."""
        assert metrics.get_avg_task_duration() == 0.0

    def test_task_duration_history_limit(self, metrics):
        """Test that task duration history is limited to max_duration_history."""
        # Record 150 tasks (exceeds 100 limit)
        for i in range(150):
            metrics.record_task_completed(f"TASK-{i}", float(i), success=True)

        # Should only keep last 100
        assert len(metrics.task_durations) == 100
        assert metrics.task_durations[0] == 50.0  # First kept duration
        assert metrics.task_durations[-1] == 149.0  # Last duration

    def test_record_instance_spawned(self, metrics):
        """Test recording instance spawn."""
        metrics.record_instance_spawned()
        metrics.record_instance_spawned()

        assert metrics.instances_spawned == 2

    def test_record_instance_crashed(self, metrics):
        """Test recording instance crash."""
        metrics.record_instance_crashed()
        metrics.record_instance_crashed()

        assert metrics.instances_crashed == 2

    def test_record_intervention(self, metrics):
        """Test recording intervention."""
        metrics.record_intervention()
        metrics.record_intervention()
        metrics.record_intervention()

        assert metrics.interventions_sent == 3

    def test_get_snapshot_all_fields_populated(self, metrics):
        """Test that get_snapshot() populates all fields correctly."""
        base_time = datetime.utcnow()

        # Record some data
        metrics.start_time = base_time - timedelta(hours=1)
        metrics.record_task_completed("TASK-001", 120.0, success=True)
        metrics.record_task_completed("TASK-002", 60.0, success=False)
        metrics.record_gate_result("gate1", passed=True)
        # Don't test gate failure here to avoid the datetime mocking issue
        metrics.gates_failed = 1  # Set directly
        metrics.record_instance_spawned()
        metrics.record_instance_crashed()
        metrics.record_intervention()

        snapshot = metrics.get_snapshot(
            cycle_count=10,
            health_status="HEALTHY",
            active_instances=2,
            available_tasks=5
        )

        # Uptime should be approximately 1 hour (3600 seconds)
        assert 3595 <= snapshot.uptime_seconds <= 3605  # Allow 5 second tolerance
        assert snapshot.cycle_count == 10
        assert snapshot.tasks_completed == 1
        assert snapshot.tasks_failed == 1
        assert snapshot.tasks_in_progress == 2  # active_instances
        assert snapshot.completion_rate == 50.0
        assert snapshot.avg_task_duration_seconds == 90.0  # (120 + 60) / 2
        assert snapshot.gates_passed == 1
        assert snapshot.gates_failed == 1
        assert snapshot.gate_failure_rate == 50.0
        assert snapshot.instances_spawned == 1
        assert snapshot.instances_crashed == 1
        assert snapshot.interventions_sent == 1
        assert snapshot.health_status == "HEALTHY"
        assert snapshot.active_instances == 2
        assert snapshot.available_tasks == 5

    def test_persistence_round_trip(self, tmp_path):
        """Test saving and loading metrics (persistence round-trip)."""
        metrics_file = tmp_path / "metrics.json"

        with patch.object(MonitoringMetrics, 'METRICS_FILE', str(metrics_file)):
            # Create metrics and record data
            metrics1 = MonitoringMetrics()
            metrics1.record_task_completed("TASK-001", 100.0, success=True)
            metrics1.record_task_completed("TASK-002", 200.0, success=False)
            metrics1.record_gate_result("gate1", passed=False, issues=["Issue 1"])
            metrics1.record_instance_spawned()
            metrics1.record_intervention()

            # File should exist
            assert metrics_file.exists()

            # Load into new instance
            metrics2 = MonitoringMetrics()

            # Verify data loaded correctly
            assert metrics2.tasks_completed == 1
            assert metrics2.tasks_failed == 1
            assert metrics2.gates_failed == 1
            assert metrics2.instances_spawned == 1
            assert metrics2.interventions_sent == 1
            assert len(metrics2.task_durations) == 2
            assert len(metrics2.recent_gate_failures) == 1

    def test_persistence_old_data_ignored(self, tmp_path):
        """Test that old persisted data (>1 hour) is ignored."""
        metrics_file = tmp_path / "metrics.json"

        # Create old persisted data (2 hours old)
        old_time = datetime.utcnow() - timedelta(hours=2)
        old_data = {
            "start_time": old_time.isoformat(),
            "tasks_completed": 100,
            "tasks_failed": 50,
            "persisted_at": old_time.isoformat()
        }

        with open(metrics_file, 'w') as f:
            json.dump(old_data, f)

        with patch.object(MonitoringMetrics, 'METRICS_FILE', str(metrics_file)):
            # Load metrics - should start fresh
            metrics = MonitoringMetrics()

            # Should have default values, not loaded values
            assert metrics.tasks_completed == 0
            assert metrics.tasks_failed == 0


# ============================================================================
# StructuredLogger Tests
# ============================================================================


class TestStructuredLogger:
    """Tests for StructuredLogger class."""

    @pytest.fixture
    def logger(self, tmp_path):
        """Create logger with temporary directory."""
        return StructuredLogger(tmp_path)

    def test_log_event_creates_jsonl_file(self, logger, tmp_path):
        """Test that logging event creates JSONL file."""
        event = MonitoringEvent(
            event_type=EventType.CYCLE_START,
            timestamp="2025-11-25T20:00:00Z",
            cycle=1,
            payload={"uptime": "1m 0s"}
        )

        logger.log_event(event)

        events_file = tmp_path / "orchestrator_events.jsonl"
        assert events_file.exists()

    def test_log_event_valid_json_format(self, logger, tmp_path):
        """Test that logged events are valid JSON lines."""
        event1 = MonitoringEvent(
            event_type=EventType.CYCLE_START,
            timestamp="2025-11-25T20:00:00Z",
            cycle=1,
            payload={"uptime": "1m 0s"}
        )
        event2 = MonitoringEvent(
            event_type=EventType.TASK_COMPLETED,
            timestamp="2025-11-25T20:01:00Z",
            cycle=1,
            payload={"duration": 60.0},
            task_id="TASK-001"
        )

        logger.log_event(event1)
        logger.log_event(event2)

        events_file = tmp_path / "orchestrator_events.jsonl"
        with open(events_file, 'r') as f:
            lines = f.readlines()

        assert len(lines) == 2

        # Each line should be valid JSON
        data1 = json.loads(lines[0])
        data2 = json.loads(lines[1])

        assert data1["event"] == "cycle_start"
        assert data2["event"] == "task_completed"
        assert data2["task_id"] == "TASK-001"

    def test_log_rotation_triggers_at_size_limit(self, logger, tmp_path):
        """Test that log rotation triggers when file exceeds size limit."""
        events_file = tmp_path / "orchestrator_events.jsonl"

        # Create a large file (over 100MB threshold)
        with open(events_file, 'w') as f:
            # Write 110MB of data
            f.write("x" * (110 * 1024 * 1024))

        # Trigger rotation
        logger.rotate_logs(max_size_mb=100, max_files=10)

        # Original file should be rotated to .1
        rotated_file = tmp_path / "orchestrator_events.jsonl.1"
        assert rotated_file.exists()

        # New file should not exist yet (only created on next log)
        assert not events_file.exists()

    def test_log_rotation_preserves_multiple_generations(self, tmp_path):
        """Test that log rotation maintains multiple generations."""
        logger = StructuredLogger(tmp_path)
        events_file = tmp_path / "orchestrator_events.jsonl"

        # Create rotated files
        for i in range(1, 4):
            rotated = tmp_path / f"orchestrator_events.jsonl.{i}"
            with open(rotated, 'w') as f:
                f.write(f"generation {i}")

        # Create current file over size limit
        with open(events_file, 'w') as f:
            f.write("x" * (110 * 1024 * 1024))

        # Rotate
        logger.rotate_logs(max_size_mb=100, max_files=10)

        # Files should shift: .1 -> .2, .2 -> .3, .3 -> .4, current -> .1
        assert (tmp_path / "orchestrator_events.jsonl.1").exists()
        assert (tmp_path / "orchestrator_events.jsonl.2").exists()
        assert (tmp_path / "orchestrator_events.jsonl.3").exists()
        assert (tmp_path / "orchestrator_events.jsonl.4").exists()

    def test_log_rotation_deletes_oldest(self, tmp_path):
        """Test that oldest log is deleted when max_files limit is reached."""
        logger = StructuredLogger(tmp_path)
        events_file = tmp_path / "orchestrator_events.jsonl"

        # Test the actual rotation logic:
        # With max_files=10, rotation keeps files .1 through .9, deletes .10
        # Let's create files .1 through .9
        for i in range(1, 10):
            rotated = tmp_path / f"orchestrator_events.jsonl.{i}"
            with open(rotated, 'w') as f:
                f.write(f"generation {i}")

        # Create current file over size limit
        with open(events_file, 'w') as f:
            f.write("x" * (110 * 1024 * 1024))

        # Rotate with max_files=10
        # Expected: .1->.2, .2->.3, ..., .9->.10, current->.1
        # Then delete .10 (oldest)
        logger.rotate_logs(max_size_mb=100, max_files=10)

        # File .10 was created (.9 moved to it) then deleted
        assert not (tmp_path / "orchestrator_events.jsonl.10").exists()

        # File .1 should exist (current was moved there)
        assert (tmp_path / "orchestrator_events.jsonl.1").exists()

        # File .9 should exist (.8 moved there)
        assert (tmp_path / "orchestrator_events.jsonl.9").exists()

        # Verify the delete logic works correctly
        # Create another current file and rotate - .10 should be deleted again
        with open(events_file, 'w') as f:
            f.write("x" * (110 * 1024 * 1024))

        logger.rotate_logs(max_size_mb=100, max_files=10)

        # .10 should still not exist (gets deleted each rotation when at max)
        assert not (tmp_path / "orchestrator_events.jsonl.10").exists()
        assert not (tmp_path / "orchestrator_events.jsonl.11").exists()


# ============================================================================
# MonitoringHook Tests
# ============================================================================


class TestLoggingHook:
    """Tests for LoggingHook (default hook implementation)."""

    @pytest.fixture
    def hook(self):
        """Create logging hook instance."""
        return LoggingHook()

    def test_logging_hook_on_task_complete_success(self, hook, caplog):
        """Test logging hook for successful task completion."""
        import logging
        caplog.set_level(logging.INFO)

        hook.on_task_complete("TASK-001", 120.5, success=True, details={})

        assert "Task TASK-001 SUCCESS" in caplog.text
        assert "120.5s" in caplog.text

    def test_logging_hook_on_task_complete_failure(self, hook, caplog):
        """Test logging hook for failed task completion."""
        import logging
        caplog.set_level(logging.INFO)

        hook.on_task_complete("TASK-002", 60.0, success=False, details={})

        assert "Task TASK-002 FAILED" in caplog.text
        assert "60.0s" in caplog.text

    def test_logging_hook_on_gate_failure(self, hook, caplog):
        """Test logging hook for gate failure."""
        import logging
        caplog.set_level(logging.WARNING)

        hook.on_gate_failure(
            "design_review",
            issues=["Missing tests", "Low coverage"],
            failure_count=3,
            threshold=5
        )

        assert "Gate design_review failed" in caplog.text
        assert "3/5" in caplog.text
        assert "Missing tests" in caplog.text

    def test_logging_hook_on_gate_failure_threshold_exceeded(self, hook, caplog):
        """Test logging hook when gate failure threshold exceeded."""
        import logging
        caplog.set_level(logging.WARNING)

        hook.on_gate_failure(
            "design_review",
            issues=["Critical issue"],
            failure_count=6,
            threshold=5
        )

        assert "ALERT" in caplog.text
        assert "threshold exceeded" in caplog.text
        assert "6 >= 5" in caplog.text

    def test_logging_hook_on_health_check(self, hook, caplog):
        """Test logging hook for health check."""
        import logging
        caplog.set_level(logging.DEBUG)

        metrics = MetricsSnapshot(
            timestamp="2025-11-25T20:00:00Z",
            uptime_seconds=3600,
            cycle_count=60,
            completion_rate=85.5,
            health_status="HEALTHY",
            active_instances=2,
            available_tasks=5
        )

        hook.on_health_check(metrics)

        assert "Health check" in caplog.text
        assert "HEALTHY" in caplog.text
        assert "85.5%" in caplog.text
        assert "active=2" in caplog.text


# ============================================================================
# MonitoringHookDispatcher Tests
# ============================================================================


class TestMonitoringHookDispatcher:
    """Tests for MonitoringHookDispatcher."""

    @pytest.fixture
    def dispatcher(self):
        """Create dispatcher instance."""
        return MonitoringHookDispatcher()

    def test_dispatch_to_all_registered_hooks(self, dispatcher):
        """Test that events dispatch to all registered hooks."""
        # Create mock hooks
        hook1 = Mock(spec=MonitoringHook)
        hook2 = Mock(spec=MonitoringHook)

        dispatcher.register_hook(hook1)
        dispatcher.register_hook(hook2)

        # Dispatch task completion
        dispatcher.dispatch_task_complete("TASK-001", 120.0, success=True, details={})

        # Both hooks should be called
        hook1.on_task_complete.assert_called_once_with("TASK-001", 120.0, True, {})
        hook2.on_task_complete.assert_called_once_with("TASK-001", 120.0, True, {})

    def test_dispatch_gate_failure_to_all_hooks(self, dispatcher):
        """Test gate failure dispatch to all hooks."""
        hook1 = Mock(spec=MonitoringHook)
        hook2 = Mock(spec=MonitoringHook)

        dispatcher.register_hook(hook1)
        dispatcher.register_hook(hook2)

        dispatcher.dispatch_gate_failure("gate1", ["Issue 1"], 3, 5)

        hook1.on_gate_failure.assert_called_once_with("gate1", ["Issue 1"], 3, 5)
        hook2.on_gate_failure.assert_called_once_with("gate1", ["Issue 1"], 3, 5)

    def test_dispatch_health_check_to_all_hooks(self, dispatcher):
        """Test health check dispatch to all hooks."""
        hook1 = Mock(spec=MonitoringHook)
        hook2 = Mock(spec=MonitoringHook)

        dispatcher.register_hook(hook1)
        dispatcher.register_hook(hook2)

        metrics = MetricsSnapshot(
            timestamp="2025-11-25T20:00:00Z",
            uptime_seconds=3600,
            cycle_count=60
        )

        dispatcher.dispatch_health_check(metrics)

        hook1.on_health_check.assert_called_once_with(metrics)
        hook2.on_health_check.assert_called_once_with(metrics)

    def test_hook_failure_isolation(self, dispatcher, caplog):
        """Test that failing hook doesn't crash dispatcher or affect other hooks."""
        # Create hooks - first one fails
        hook1 = Mock(spec=MonitoringHook)
        hook1.on_task_complete.side_effect = Exception("Hook 1 failed!")

        hook2 = Mock(spec=MonitoringHook)

        dispatcher.register_hook(hook1)
        dispatcher.register_hook(hook2)

        # Dispatch - should not raise exception
        dispatcher.dispatch_task_complete("TASK-001", 120.0, success=True, details={})

        # First hook should have been called (and failed)
        hook1.on_task_complete.assert_called_once()

        # Second hook should still be called despite first one failing
        hook2.on_task_complete.assert_called_once_with("TASK-001", 120.0, True, {})

        # Error should be logged
        assert "Hook" in caplog.text
        assert "failed" in caplog.text

    def test_dispatcher_includes_default_logging_hook(self, dispatcher):
        """Test that dispatcher includes default logging hook."""
        # New dispatcher should have 1 hook (default LoggingHook)
        assert len(dispatcher.hooks) == 1
        assert isinstance(dispatcher.hooks[0], LoggingHook)

    def test_register_hook_adds_to_list(self, dispatcher):
        """Test that register_hook adds hook to list."""
        initial_count = len(dispatcher.hooks)

        new_hook = Mock(spec=MonitoringHook)
        dispatcher.register_hook(new_hook)

        assert len(dispatcher.hooks) == initial_count + 1
        assert new_hook in dispatcher.hooks


# ============================================================================
# Health Check Integration Tests
# ============================================================================


class TestHealthCheckIntegration:
    """Tests for health check interval-based triggering."""

    def test_health_check_interval_elapsed(self, tmp_path):
        """Test that health check triggers when interval has elapsed."""
        from autonomous.orchestrator_loop import OrchestratorLoop

        # Create orchestrator with 5-second health check interval
        config_file = tmp_path / "config.json"
        with open(config_file, 'w') as f:
            json.dump({"health_check_interval": 5}, f)

        with patch('autonomous.orchestrator_loop.DistributedLockManager'), \
             patch('autonomous.orchestrator_loop.InstanceRegistry'), \
             patch('autonomous.orchestrator_loop.TaskCoordinator'), \
             patch('autonomous.orchestrator_loop.MessageQueue'), \
             patch('autonomous.orchestrator_loop.OrchestratorMonitor'):

            loop = OrchestratorLoop(config_path=config_file)

            # Mock time passage
            base_time = datetime(2025, 11, 25, 20, 0, 0)

            with patch('autonomous.orchestrator_loop.datetime') as mock_dt:
                mock_dt.utcnow.return_value = base_time

                # First call - should trigger (no last_health_check)
                dashboard = {
                    "active_instances": [],
                    "available_tasks": 0
                }
                system_health = {"health_status": "HEALTHY"}

                loop.last_health_check = None

                # Mock hook dispatcher
                loop.hook_dispatcher.dispatch_health_check = Mock()

                loop._maybe_health_check(dashboard, system_health)

                # Should have triggered
                assert loop.last_health_check is not None
                loop.hook_dispatcher.dispatch_health_check.assert_called_once()

    def test_health_check_not_triggered_before_interval(self, tmp_path):
        """Test that health check does NOT trigger before interval elapses."""
        from autonomous.orchestrator_loop import OrchestratorLoop

        config_file = tmp_path / "config.json"
        with open(config_file, 'w') as f:
            json.dump({"health_check_interval": 300}, f)  # 5 minutes

        with patch('autonomous.orchestrator_loop.DistributedLockManager'), \
             patch('autonomous.orchestrator_loop.InstanceRegistry'), \
             patch('autonomous.orchestrator_loop.TaskCoordinator'), \
             patch('autonomous.orchestrator_loop.MessageQueue'), \
             patch('autonomous.orchestrator_loop.OrchestratorMonitor'):

            loop = OrchestratorLoop(config_path=config_file)

            base_time = datetime(2025, 11, 25, 20, 0, 0)

            # Set last health check 1 minute ago
            loop.last_health_check = base_time - timedelta(minutes=1)

            with patch('autonomous.orchestrator_loop.datetime') as mock_dt:
                mock_dt.utcnow.return_value = base_time

                dashboard = {
                    "active_instances": [],
                    "available_tasks": 0
                }
                system_health = {"health_status": "HEALTHY"}

                # Mock hook dispatcher
                loop.hook_dispatcher.dispatch_health_check = Mock()

                loop._maybe_health_check(dashboard, system_health)

                # Should NOT have triggered (only 1 minute elapsed, need 5)
                loop.hook_dispatcher.dispatch_health_check.assert_not_called()

    def test_health_check_dispatches_to_hooks(self, tmp_path):
        """Test that health check dispatches metrics to hooks."""
        from autonomous.orchestrator_loop import OrchestratorLoop

        config_file = tmp_path / "config.json"
        with open(config_file, 'w') as f:
            json.dump({"health_check_interval": 5}, f)

        with patch('autonomous.orchestrator_loop.DistributedLockManager'), \
             patch('autonomous.orchestrator_loop.InstanceRegistry'), \
             patch('autonomous.orchestrator_loop.TaskCoordinator'), \
             patch('autonomous.orchestrator_loop.MessageQueue'), \
             patch('autonomous.orchestrator_loop.OrchestratorMonitor'):

            loop = OrchestratorLoop(config_path=config_file)

            # Add mock hook
            mock_hook = Mock(spec=MonitoringHook)
            loop.hook_dispatcher.register_hook(mock_hook)

            base_time = datetime(2025, 11, 25, 20, 0, 0)
            loop.last_health_check = None

            with patch('autonomous.orchestrator_loop.datetime') as mock_dt:
                mock_dt.utcnow.return_value = base_time

                dashboard = {
                    "active_instances": [{"id": "inst-1"}],
                    "available_tasks": 5
                }
                system_health = {"health_status": "HEALTHY"}

                loop._maybe_health_check(dashboard, system_health)

                # Mock hook should have been called
                mock_hook.on_health_check.assert_called_once()

                # Check that metrics snapshot was passed
                call_args = mock_hook.on_health_check.call_args[0]
                metrics = call_args[0]

                assert isinstance(metrics, MetricsSnapshot)
                assert metrics.health_status == "HEALTHY"
                assert metrics.active_instances == 1
                assert metrics.available_tasks == 5


# ============================================================================
# Integration Tests
# ============================================================================


class TestMonitoringIntegration:
    """End-to-end integration tests for monitoring system."""

    def test_complete_monitoring_flow(self, tmp_path):
        """Test complete monitoring flow: metrics -> events -> hooks."""
        # Setup - use temporary directory for metrics
        with patch.object(MonitoringMetrics, 'METRICS_FILE', str(tmp_path / 'metrics.json')):
            metrics = MonitoringMetrics()
            logger = StructuredLogger(tmp_path)
            dispatcher = MonitoringHookDispatcher()

            mock_hook = Mock(spec=MonitoringHook)
            dispatcher.register_hook(mock_hook)

            # Simulate task completion
            metrics.record_task_completed("TASK-001", 120.5, success=True)

            event = MonitoringEvent(
                event_type=EventType.TASK_COMPLETED,
                timestamp=datetime.utcnow().isoformat(),
                cycle=1,
                payload={"duration_seconds": 120.5, "success": True},
                task_id="TASK-001"
            )

            logger.log_event(event)
            dispatcher.dispatch_task_complete("TASK-001", 120.5, success=True, details={})

            # Verify
            assert metrics.tasks_completed == 1
            assert (tmp_path / "orchestrator_events.jsonl").exists()
            mock_hook.on_task_complete.assert_called_once()

    def test_gate_failure_alerting_flow(self, tmp_path):
        """Test gate failure tracking and alerting flow."""
        with patch.object(MonitoringMetrics, 'METRICS_FILE', str(tmp_path / 'metrics.json')):
            metrics = MonitoringMetrics()
            dispatcher = MonitoringHookDispatcher()

            mock_hook = Mock(spec=MonitoringHook)
            dispatcher.register_hook(mock_hook)

            # Record multiple gate failures
            for i in range(6):
                metrics.record_gate_result(f"gate{i}", passed=False, issues=[f"Issue {i}"])

            # Check failure count and dispatch alert
            failure_count = metrics.get_recent_gate_failure_count()
            threshold = 5

            if failure_count >= threshold:
                dispatcher.dispatch_gate_failure(
                    "gate5",
                    ["Issue 5"],
                    failure_count,
                    threshold
                )

            # Verify
            assert failure_count == 6
            assert failure_count >= threshold
            mock_hook.on_gate_failure.assert_called_once_with(
                "gate5",
                ["Issue 5"],
                6,
                5
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
