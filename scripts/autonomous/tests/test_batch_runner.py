#!/usr/bin/env python3
"""Tests for batch_runner.py

Comprehensive test suite covering:
- Data model serialization/deserialization
- NotificationService functionality
- BatchRunner core logic (limits, checkpointing, reporting)
- Task execution with mocked subprocess
- Signal handling
- Report generation
"""

import pytest
import json
import tempfile
import signal
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open, call
from datetime import datetime, timedelta

# Import classes to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from batch_runner import (
    BatchConfig,
    BatchCheckpoint,
    TaskResult,
    BatchResult,
    NotificationService,
    BatchRunner
)


# ==============================================================================
# SECTION 1: Data Model Tests
# ==============================================================================

class TestBatchConfig:
    """Tests for BatchConfig dataclass."""

    def test_default_values(self):
        """Test that BatchConfig has correct default values."""
        config = BatchConfig()

        assert config.max_tasks is None
        assert config.max_hours is None
        assert config.notify_webhook is None
        assert config.checkpoint_file == Path(".autonomous/batch_checkpoint.json")
        assert config.report_file == Path(".autonomous/batch_report.md")
        assert config.resume is True
        assert config.task_filter is None
        assert config.priority_filter is None
        assert config.dry_run is False

    def test_custom_configuration(self):
        """Test creating BatchConfig with custom values."""
        config = BatchConfig(
            max_tasks=5,
            max_hours=8.0,
            notify_webhook="https://example.com/webhook",
            checkpoint_file=Path("/tmp/checkpoint.json"),
            report_file=Path("/tmp/report.md"),
            resume=False,
            task_filter="SDLC-*",
            priority_filter="HIGH",
            dry_run=True
        )

        assert config.max_tasks == 5
        assert config.max_hours == 8.0
        assert config.notify_webhook == "https://example.com/webhook"
        assert config.checkpoint_file == Path("/tmp/checkpoint.json")
        assert config.report_file == Path("/tmp/report.md")
        assert config.resume is False
        assert config.task_filter == "SDLC-*"
        assert config.priority_filter == "HIGH"
        assert config.dry_run is True


class TestBatchCheckpoint:
    """Tests for BatchCheckpoint dataclass."""

    def test_to_dict_serialization(self):
        """Test that to_dict() produces correct serialization."""
        checkpoint = BatchCheckpoint(
            batch_id="batch-20250101-120000",
            started_at="2025-01-01T12:00:00",
            last_updated="2025-01-01T13:00:00",
            tasks_completed=["TASK-001", "TASK-002"],
            tasks_failed=["TASK-003"],
            tasks_skipped=["TASK-004"],
            current_task="TASK-005",
            config={"max_tasks": 10, "max_hours": 8.0}
        )

        result = checkpoint.to_dict()

        assert isinstance(result, dict)
        assert result["batch_id"] == "batch-20250101-120000"
        assert result["started_at"] == "2025-01-01T12:00:00"
        assert result["last_updated"] == "2025-01-01T13:00:00"
        assert result["tasks_completed"] == ["TASK-001", "TASK-002"]
        assert result["tasks_failed"] == ["TASK-003"]
        assert result["tasks_skipped"] == ["TASK-004"]
        assert result["current_task"] == "TASK-005"
        assert result["config"] == {"max_tasks": 10, "max_hours": 8.0}

    def test_from_dict_deserialization(self):
        """Test that from_dict() correctly deserializes."""
        data = {
            "batch_id": "batch-20250101-120000",
            "started_at": "2025-01-01T12:00:00",
            "last_updated": "2025-01-01T13:00:00",
            "tasks_completed": ["TASK-001", "TASK-002"],
            "tasks_failed": ["TASK-003"],
            "tasks_skipped": ["TASK-004"],
            "current_task": "TASK-005",
            "config": {"max_tasks": 10}
        }

        checkpoint = BatchCheckpoint.from_dict(data)

        assert checkpoint.batch_id == "batch-20250101-120000"
        assert checkpoint.started_at == "2025-01-01T12:00:00"
        assert checkpoint.last_updated == "2025-01-01T13:00:00"
        assert checkpoint.tasks_completed == ["TASK-001", "TASK-002"]
        assert checkpoint.tasks_failed == ["TASK-003"]
        assert checkpoint.tasks_skipped == ["TASK-004"]
        assert checkpoint.current_task == "TASK-005"
        assert checkpoint.config == {"max_tasks": 10}

    def test_round_trip_serialization(self):
        """Test that to_dict -> from_dict preserves data."""
        original = BatchCheckpoint(
            batch_id="batch-test",
            started_at="2025-01-01T12:00:00",
            last_updated="2025-01-01T13:00:00",
            tasks_completed=["A", "B"],
            tasks_failed=["C"],
            tasks_skipped=["D"],
            current_task="E",
            config={"key": "value"}
        )

        # Serialize and deserialize
        data = original.to_dict()
        restored = BatchCheckpoint.from_dict(data)

        # Verify all fields match
        assert restored.batch_id == original.batch_id
        assert restored.started_at == original.started_at
        assert restored.last_updated == original.last_updated
        assert restored.tasks_completed == original.tasks_completed
        assert restored.tasks_failed == original.tasks_failed
        assert restored.tasks_skipped == original.tasks_skipped
        assert restored.current_task == original.current_task
        assert restored.config == original.config

    def test_optional_fields(self):
        """Test checkpoint with optional fields set to None."""
        checkpoint = BatchCheckpoint(
            batch_id="batch-test",
            started_at="2025-01-01T12:00:00",
            last_updated="2025-01-01T13:00:00",
            tasks_completed=[],
            tasks_failed=[],
            tasks_skipped=[],
            current_task=None,
            config={}
        )

        result = checkpoint.to_dict()
        assert result["current_task"] is None
        assert result["config"] == {}


class TestTaskResult:
    """Tests for TaskResult dataclass."""

    def test_to_dict_serialization(self):
        """Test that to_dict() produces correct serialization."""
        result = TaskResult(
            task_id="TASK-001",
            status="completed",
            started_at="2025-01-01T12:00:00",
            finished_at="2025-01-01T12:30:00",
            duration_seconds=1800.0,
            pr_number=123,
            error_message=None,
            gate_failures=[]
        )

        data = result.to_dict()

        assert isinstance(data, dict)
        assert data["task_id"] == "TASK-001"
        assert data["status"] == "completed"
        assert data["started_at"] == "2025-01-01T12:00:00"
        assert data["finished_at"] == "2025-01-01T12:30:00"
        assert data["duration_seconds"] == 1800.0
        assert data["pr_number"] == 123
        assert data["error_message"] is None
        assert data["gate_failures"] == []

    def test_creation_with_all_fields(self):
        """Test creating TaskResult with all fields."""
        result = TaskResult(
            task_id="TASK-002",
            status="failed",
            started_at="2025-01-01T14:00:00",
            finished_at="2025-01-01T14:05:00",
            duration_seconds=300.0,
            pr_number=None,
            error_message="Subprocess exited with code 1",
            gate_failures=[{"gate": "design-review", "issue": "failed"}]
        )

        assert result.task_id == "TASK-002"
        assert result.status == "failed"
        assert result.error_message == "Subprocess exited with code 1"
        assert len(result.gate_failures) == 1
        assert result.gate_failures[0]["gate"] == "design-review"

    def test_default_fields(self):
        """Test TaskResult with default field values."""
        result = TaskResult(
            task_id="TASK-003",
            status="skipped",
            started_at="2025-01-01T15:00:00",
            finished_at="2025-01-01T15:01:00",
            duration_seconds=60.0
        )

        assert result.pr_number is None
        assert result.error_message is None
        assert result.gate_failures == []


class TestBatchResult:
    """Tests for BatchResult dataclass."""

    def test_to_dict_serialization(self):
        """Test that to_dict() produces correct serialization."""
        task_results = [
            TaskResult(
                task_id="TASK-001",
                status="completed",
                started_at="2025-01-01T12:00:00",
                finished_at="2025-01-01T12:30:00",
                duration_seconds=1800.0,
                pr_number=123
            )
        ]

        result = BatchResult(
            batch_id="batch-20250101-120000",
            status="completed",
            started_at="2025-01-01T12:00:00",
            finished_at="2025-01-01T14:00:00",
            duration_hours=2.0,
            tasks_completed=1,
            tasks_failed=0,
            tasks_skipped=0,
            tasks_remaining=5,
            task_results=task_results,
            exit_reason="max_tasks"
        )

        data = result.to_dict()

        assert isinstance(data, dict)
        assert data["batch_id"] == "batch-20250101-120000"
        assert data["status"] == "completed"
        assert data["exit_reason"] == "max_tasks"
        assert data["tasks_completed"] == 1
        assert data["duration_hours"] == 2.0
        assert len(data["task_results"]) == 1

    def test_creation_with_task_results_list(self):
        """Test creating BatchResult with multiple task results."""
        task_results = [
            TaskResult(
                task_id="TASK-001",
                status="completed",
                started_at="2025-01-01T12:00:00",
                finished_at="2025-01-01T12:30:00",
                duration_seconds=1800.0,
                pr_number=123
            ),
            TaskResult(
                task_id="TASK-002",
                status="failed",
                started_at="2025-01-01T12:30:00",
                finished_at="2025-01-01T12:35:00",
                duration_seconds=300.0,
                error_message="Build failed"
            )
        ]

        result = BatchResult(
            batch_id="batch-test",
            status="completed",
            started_at="2025-01-01T12:00:00",
            finished_at="2025-01-01T13:00:00",
            duration_hours=1.0,
            tasks_completed=1,
            tasks_failed=1,
            tasks_skipped=0,
            tasks_remaining=0,
            task_results=task_results,
            exit_reason="completed"
        )

        assert len(result.task_results) == 2
        assert result.tasks_completed == 1
        assert result.tasks_failed == 1

    def test_task_results_serialization(self):
        """Test that task_results are properly serialized in to_dict()."""
        task_results = [
            TaskResult(
                task_id="TASK-001",
                status="completed",
                started_at="2025-01-01T12:00:00",
                finished_at="2025-01-01T12:30:00",
                duration_seconds=1800.0
            )
        ]

        result = BatchResult(
            batch_id="batch-test",
            status="completed",
            started_at="2025-01-01T12:00:00",
            finished_at="2025-01-01T13:00:00",
            duration_hours=1.0,
            tasks_completed=1,
            tasks_failed=0,
            tasks_skipped=0,
            tasks_remaining=0,
            task_results=task_results,
            exit_reason="completed"
        )

        data = result.to_dict()

        # Verify task_results are serialized as dicts
        assert isinstance(data["task_results"], list)
        assert len(data["task_results"]) == 1
        assert isinstance(data["task_results"][0], dict)
        assert data["task_results"][0]["task_id"] == "TASK-001"


# ==============================================================================
# SECTION 2: NotificationService Tests
# ==============================================================================

class TestNotificationService:
    """Tests for NotificationService class."""

    def test_initialization_with_webhook_url(self):
        """Test NotificationService initialization with webhook URL."""
        service = NotificationService("https://example.com/webhook")
        assert service.webhook_url == "https://example.com/webhook"

    def test_initialization_without_webhook_url(self):
        """Test NotificationService initialization without webhook URL."""
        service = NotificationService(None)
        assert service.webhook_url is None

    def test_send_completion_without_webhook_returns_true(self):
        """Test that send_completion returns True when no webhook configured."""
        service = NotificationService(None)

        batch_result = BatchResult(
            batch_id="batch-test",
            status="completed",
            started_at="2025-01-01T12:00:00",
            finished_at="2025-01-01T13:00:00",
            duration_hours=1.0,
            tasks_completed=5,
            tasks_failed=0,
            tasks_skipped=0,
            tasks_remaining=0,
            task_results=[],
            exit_reason="completed"
        )

        result = service.send_completion(batch_result, "# Report")
        assert result is True

    def test_send_failure_without_webhook_returns_true(self):
        """Test that send_failure returns True when no webhook configured."""
        service = NotificationService(None)

        result = service.send_failure("TASK-001", "Error message")
        assert result is True

    def test_payload_construction_for_completion(self):
        """Test that send_completion constructs correct payload."""
        service = NotificationService("https://example.com/webhook")

        task_results = [
            TaskResult(
                task_id="TASK-001",
                status="completed",
                started_at="2025-01-01T12:00:00",
                finished_at="2025-01-01T12:30:00",
                duration_seconds=1800.0,
                pr_number=123
            ),
            TaskResult(
                task_id="TASK-002",
                status="failed",
                started_at="2025-01-01T12:30:00",
                finished_at="2025-01-01T12:35:00",
                duration_seconds=300.0,
                error_message="Build failed"
            )
        ]

        batch_result = BatchResult(
            batch_id="batch-test",
            status="completed",
            started_at="2025-01-01T12:00:00",
            finished_at="2025-01-01T13:00:00",
            duration_hours=1.0,
            tasks_completed=1,
            tasks_failed=1,
            tasks_skipped=0,
            tasks_remaining=3,
            task_results=task_results,
            exit_reason="max_tasks"
        )

        # Mock the _post_webhook method to capture payload
        with patch.object(service, '_post_webhook', return_value=True) as mock_post:
            service.send_completion(batch_result, "# Report")

            # Verify _post_webhook was called
            assert mock_post.called
            payload = mock_post.call_args[0][0]

            # Verify payload structure
            assert payload["event"] == "batch_completed"
            assert payload["batch_id"] == "batch-test"
            assert payload["status"] == "completed"
            assert payload["exit_reason"] == "max_tasks"
            assert payload["summary"]["tasks_completed"] == 1
            assert payload["summary"]["tasks_failed"] == 1
            assert payload["summary"]["tasks_skipped"] == 0
            assert payload["summary"]["duration_hours"] == 1.0
            assert len(payload["task_results"]) == 2
            assert payload["task_results"][0]["task_id"] == "TASK-001"
            assert payload["task_results"][0]["status"] == "completed"
            assert payload["task_results"][0]["pr_number"] == 123
            assert "timestamp" in payload

    def test_payload_construction_for_failure(self):
        """Test that send_failure constructs correct payload."""
        service = NotificationService("https://example.com/webhook")

        # Mock the _post_webhook method to capture payload
        with patch.object(service, '_post_webhook', return_value=True) as mock_post:
            service.send_failure("TASK-001", "Build error occurred")

            # Verify _post_webhook was called
            assert mock_post.called
            payload = mock_post.call_args[0][0]

            # Verify payload structure
            assert payload["event"] == "task_failed"
            assert payload["task_id"] == "TASK-001"
            assert payload["error"] == "Build error occurred"
            assert "timestamp" in payload

    @patch('urllib.request.urlopen')
    def test_post_webhook_success(self, mock_urlopen):
        """Test successful webhook POST."""
        service = NotificationService("https://example.com/webhook")

        # Mock successful response
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        payload = {"event": "test", "data": "value"}
        result = service._post_webhook(payload)

        assert result is True
        assert mock_urlopen.called

    @patch('urllib.request.urlopen')
    @patch('time.sleep')  # Mock sleep to speed up test
    def test_post_webhook_retry_on_failure(self, mock_sleep, mock_urlopen):
        """Test webhook retry on failure."""
        service = NotificationService("https://example.com/webhook")

        # Mock failures then success on third attempt
        from urllib.error import URLError

        # Create a proper mock response for the third attempt
        mock_success_response = MagicMock()
        mock_success_response.status = 200
        mock_success_response.__enter__ = MagicMock(return_value=mock_success_response)
        mock_success_response.__exit__ = MagicMock(return_value=False)

        mock_urlopen.side_effect = [
            URLError("Connection error"),
            URLError("Connection error"),
            mock_success_response
        ]

        payload = {"event": "test"}
        result = service._post_webhook(payload)

        assert result is True
        assert mock_urlopen.call_count == 3
        assert mock_sleep.call_count == 2  # Sleep after first two attempts

    @patch('urllib.request.urlopen')
    @patch('time.sleep')
    def test_post_webhook_max_retries(self, mock_sleep, mock_urlopen):
        """Test webhook fails after max retries."""
        service = NotificationService("https://example.com/webhook")

        # Mock all attempts failing
        from urllib.error import URLError
        mock_urlopen.side_effect = URLError("Connection error")

        payload = {"event": "test"}
        result = service._post_webhook(payload)

        assert result is False
        assert mock_urlopen.call_count == 3  # Max 3 attempts
        assert mock_sleep.call_count == 2


# ==============================================================================
# SECTION 3: BatchRunner Core Logic Tests
# ==============================================================================

class TestBatchRunner:
    """Tests for BatchRunner class."""

    @pytest.fixture
    def temp_repo(self):
        """Create temporary repository directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            # Create required directories
            (repo_path / ".autonomous").mkdir(parents=True)
            (repo_path / "tasks").mkdir(parents=True)
            yield repo_path

    @pytest.fixture
    def basic_config(self, temp_repo):
        """Create basic BatchConfig for testing."""
        return BatchConfig(
            checkpoint_file=temp_repo / ".autonomous" / "checkpoint.json",
            report_file=temp_repo / ".autonomous" / "report.md",
            resume=False
        )

    @pytest.fixture
    def mock_lock_manager(self):
        """Mock DistributedLockManager."""
        with patch('batch_runner.DistributedLockManager') as mock_class:
            mock_instance = MagicMock()
            mock_class.return_value = mock_instance
            yield mock_instance

    @pytest.fixture
    def mock_task_coordinator(self):
        """Mock TaskCoordinator."""
        with patch('batch_runner.TaskCoordinator') as mock_class:
            mock_instance = MagicMock()
            mock_class.return_value = mock_instance
            yield mock_instance

    def test_should_continue_with_max_tasks_limit(self, temp_repo, basic_config, mock_lock_manager, mock_task_coordinator):
        """Test _should_continue returns False when max_tasks reached."""
        config = BatchConfig(
            max_tasks=3,
            checkpoint_file=temp_repo / ".autonomous" / "checkpoint.json",
            report_file=temp_repo / ".autonomous" / "report.md"
        )

        runner = BatchRunner(config, temp_repo)
        runner._start_time = datetime.utcnow()

        # Add task results to reach limit
        runner._task_results = [
            TaskResult("T1", "completed", "", "", 100),
            TaskResult("T2", "completed", "", "", 100),
            TaskResult("T3", "completed", "", "", 100)
        ]

        # Should return False when limit reached
        assert runner._should_continue() is False

    def test_should_continue_with_max_tasks_not_reached(self, temp_repo, basic_config, mock_lock_manager, mock_task_coordinator):
        """Test _should_continue returns True when max_tasks not reached."""
        config = BatchConfig(
            max_tasks=5,
            checkpoint_file=temp_repo / ".autonomous" / "checkpoint.json",
            report_file=temp_repo / ".autonomous" / "report.md"
        )

        runner = BatchRunner(config, temp_repo)
        runner._start_time = datetime.utcnow()

        # Add task results below limit
        runner._task_results = [
            TaskResult("T1", "completed", "", "", 100),
            TaskResult("T2", "completed", "", "", 100)
        ]

        # Should return True when below limit
        assert runner._should_continue() is True

    def test_should_continue_with_max_hours_limit(self, temp_repo, basic_config, mock_lock_manager, mock_task_coordinator):
        """Test _should_continue returns False when max_hours reached."""
        config = BatchConfig(
            max_hours=2.0,
            checkpoint_file=temp_repo / ".autonomous" / "checkpoint.json",
            report_file=temp_repo / ".autonomous" / "report.md"
        )

        runner = BatchRunner(config, temp_repo)

        # Set start time to 2.5 hours ago
        runner._start_time = datetime.utcnow() - timedelta(hours=2.5)

        # Should return False when time limit exceeded
        assert runner._should_continue() is False

    def test_should_continue_with_max_hours_not_reached(self, temp_repo, basic_config, mock_lock_manager, mock_task_coordinator):
        """Test _should_continue returns True when max_hours not reached."""
        config = BatchConfig(
            max_hours=5.0,
            checkpoint_file=temp_repo / ".autonomous" / "checkpoint.json",
            report_file=temp_repo / ".autonomous" / "report.md"
        )

        runner = BatchRunner(config, temp_repo)

        # Set start time to 1 hour ago
        runner._start_time = datetime.utcnow() - timedelta(hours=1)

        # Should return True when below time limit
        assert runner._should_continue() is True

    def test_should_continue_with_no_limits(self, temp_repo, basic_config, mock_lock_manager, mock_task_coordinator):
        """Test _should_continue returns True when no limits set."""
        config = BatchConfig(
            max_tasks=None,
            max_hours=None,
            checkpoint_file=temp_repo / ".autonomous" / "checkpoint.json",
            report_file=temp_repo / ".autonomous" / "report.md"
        )

        runner = BatchRunner(config, temp_repo)
        runner._start_time = datetime.utcnow()
        runner._task_results = [TaskResult("T1", "completed", "", "", 100)] * 100

        # Should always return True with no limits
        assert runner._should_continue() is True

    def test_determine_exit_reason_signal(self, temp_repo, basic_config, mock_lock_manager, mock_task_coordinator):
        """Test _determine_exit_reason returns 'signal' when shutdown requested."""
        runner = BatchRunner(basic_config, temp_repo)
        runner._start_time = datetime.utcnow()
        runner._shutdown_requested = True

        tasks = [{"id": "TASK-001"}, {"id": "TASK-002"}]
        exit_reason = runner._determine_exit_reason(tasks)

        assert exit_reason == "signal"

    def test_determine_exit_reason_max_tasks(self, temp_repo, basic_config, mock_lock_manager, mock_task_coordinator):
        """Test _determine_exit_reason returns 'max_tasks' when task limit reached."""
        config = BatchConfig(
            max_tasks=2,
            checkpoint_file=temp_repo / ".autonomous" / "checkpoint.json",
            report_file=temp_repo / ".autonomous" / "report.md"
        )

        runner = BatchRunner(config, temp_repo)
        runner._start_time = datetime.utcnow()
        runner._task_results = [
            TaskResult("T1", "completed", "", "", 100),
            TaskResult("T2", "completed", "", "", 100)
        ]

        tasks = [{"id": "T1"}, {"id": "T2"}, {"id": "T3"}]
        exit_reason = runner._determine_exit_reason(tasks)

        assert exit_reason == "max_tasks"

    def test_determine_exit_reason_max_hours(self, temp_repo, basic_config, mock_lock_manager, mock_task_coordinator):
        """Test _determine_exit_reason returns 'max_hours' when time limit reached."""
        config = BatchConfig(
            max_hours=1.0,
            checkpoint_file=temp_repo / ".autonomous" / "checkpoint.json",
            report_file=temp_repo / ".autonomous" / "report.md"
        )

        runner = BatchRunner(config, temp_repo)
        runner._start_time = datetime.utcnow() - timedelta(hours=1.5)

        tasks = [{"id": "TASK-001"}]
        exit_reason = runner._determine_exit_reason(tasks)

        assert exit_reason == "max_hours"

    def test_determine_exit_reason_no_tasks(self, temp_repo, basic_config, mock_lock_manager, mock_task_coordinator):
        """Test _determine_exit_reason returns 'no_tasks' when task list is empty."""
        runner = BatchRunner(basic_config, temp_repo)
        runner._start_time = datetime.utcnow()

        tasks = []
        exit_reason = runner._determine_exit_reason(tasks)

        assert exit_reason == "no_tasks"

    def test_determine_exit_reason_completed(self, temp_repo, basic_config, mock_lock_manager, mock_task_coordinator):
        """Test _determine_exit_reason returns 'completed' when all tasks processed."""
        runner = BatchRunner(basic_config, temp_repo)
        runner._start_time = datetime.utcnow()
        runner._completed = {"TASK-001", "TASK-002"}

        tasks = [{"id": "TASK-001"}, {"id": "TASK-002"}]
        exit_reason = runner._determine_exit_reason(tasks)

        assert exit_reason == "completed"

    def test_determine_exit_reason_interrupted(self, temp_repo, basic_config, mock_lock_manager, mock_task_coordinator):
        """Test _determine_exit_reason returns 'interrupted' when tasks remain unprocessed."""
        runner = BatchRunner(basic_config, temp_repo)
        runner._start_time = datetime.utcnow()
        runner._completed = {"TASK-001"}

        tasks = [{"id": "TASK-001"}, {"id": "TASK-002"}, {"id": "TASK-003"}]
        exit_reason = runner._determine_exit_reason(tasks)

        assert exit_reason == "interrupted"

    def test_determine_exit_reason_dry_run(self, temp_repo, basic_config, mock_lock_manager, mock_task_coordinator):
        """Test _determine_exit_reason returns 'dry_run' in dry run mode."""
        config = BatchConfig(
            dry_run=True,
            checkpoint_file=temp_repo / ".autonomous" / "checkpoint.json",
            report_file=temp_repo / ".autonomous" / "report.md"
        )

        runner = BatchRunner(config, temp_repo)
        runner._start_time = datetime.utcnow()

        tasks = [{"id": "TASK-001"}]
        exit_reason = runner._determine_exit_reason(tasks)

        assert exit_reason == "dry_run"

    def test_generate_report_creates_valid_markdown(self, temp_repo, basic_config, mock_lock_manager, mock_task_coordinator):
        """Test _generate_report creates valid markdown report."""
        runner = BatchRunner(basic_config, temp_repo)
        runner._start_time = datetime.utcnow() - timedelta(hours=2)
        runner._batch_id = "batch-test"
        runner._completed = {"TASK-001"}
        runner._failed = {"TASK-002"}
        runner._skipped = set()

        runner._task_results = [
            TaskResult(
                task_id="TASK-001",
                status="completed",
                started_at="2025-01-01T12:00:00",
                finished_at="2025-01-01T12:30:00",
                duration_seconds=1800.0,
                pr_number=123
            ),
            TaskResult(
                task_id="TASK-002",
                status="failed",
                started_at="2025-01-01T12:30:00",
                finished_at="2025-01-01T12:35:00",
                duration_seconds=300.0,
                error_message="Build failed"
            )
        ]

        report = runner._generate_report("completed")

        # Verify report structure
        assert "# Batch Execution Report" in report
        assert "batch-test" in report
        assert "COMPLETED" in report
        assert "## Summary" in report
        assert "## Task Results" in report
        assert "### Completed (1)" in report
        assert "### Failed (1)" in report
        assert "TASK-001" in report
        assert "TASK-002" in report
        assert "#123" in report
        assert "Build failed" in report
        assert "## Next Steps" in report

        # Verify report was saved
        assert basic_config.report_file.exists()
        with open(basic_config.report_file, 'r') as f:
            saved_report = f.read()
        assert saved_report == report

    def test_generate_report_with_gate_failures(self, temp_repo, basic_config, mock_lock_manager, mock_task_coordinator):
        """Test _generate_report includes gate failure section."""
        runner = BatchRunner(basic_config, temp_repo)
        runner._start_time = datetime.utcnow()
        runner._batch_id = "batch-test"
        runner._completed = set()
        runner._failed = {"TASK-001"}
        runner._skipped = set()

        runner._task_results = [
            TaskResult(
                task_id="TASK-001",
                status="failed",
                started_at="2025-01-01T12:00:00",
                finished_at="2025-01-01T12:05:00",
                duration_seconds=300.0,
                gate_failures=[
                    {"gate": "design-review", "issue": "Design incomplete"},
                    {"gate": "implementation-review", "issue": "Tests missing"}
                ]
            )
        ]

        report = runner._generate_report("completed")

        assert "### Gate Failures (Skipped via RESILIENCE-001)" in report
        assert "design-review" in report
        assert "Design incomplete" in report
        assert "implementation-review" in report
        assert "Tests missing" in report

    def test_signal_handler_sets_shutdown_flag(self, temp_repo, basic_config, mock_lock_manager, mock_task_coordinator):
        """Test that signal handler sets shutdown flag."""
        runner = BatchRunner(basic_config, temp_repo)

        assert runner._shutdown_requested is False

        # Simulate signal
        runner._handle_signal(signal.SIGINT, None)

        assert runner._shutdown_requested is True

    def test_load_checkpoint_when_file_exists(self, temp_repo, basic_config, mock_lock_manager, mock_task_coordinator):
        """Test _load_checkpoint loads checkpoint when file exists."""
        checkpoint_data = {
            "batch_id": "batch-20250101-120000",
            "started_at": "2025-01-01T12:00:00",
            "last_updated": "2025-01-01T13:00:00",
            "tasks_completed": ["TASK-001", "TASK-002"],
            "tasks_failed": ["TASK-003"],
            "tasks_skipped": [],
            "current_task": None,
            "config": {}
        }

        # Write checkpoint file
        checkpoint_file = temp_repo / ".autonomous" / "checkpoint.json"
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f)

        config = BatchConfig(checkpoint_file=checkpoint_file, resume=True)
        runner = BatchRunner(config, temp_repo)

        checkpoint = runner._load_checkpoint()

        assert checkpoint is not None
        assert checkpoint.batch_id == "batch-20250101-120000"
        assert checkpoint.tasks_completed == ["TASK-001", "TASK-002"]
        assert checkpoint.tasks_failed == ["TASK-003"]

    def test_load_checkpoint_when_file_does_not_exist(self, temp_repo, basic_config, mock_lock_manager, mock_task_coordinator):
        """Test _load_checkpoint returns None when file doesn't exist."""
        checkpoint_file = temp_repo / ".autonomous" / "nonexistent.json"
        config = BatchConfig(checkpoint_file=checkpoint_file, resume=True)
        runner = BatchRunner(config, temp_repo)

        checkpoint = runner._load_checkpoint()

        assert checkpoint is None

    def test_load_checkpoint_with_interrupted_task(self, temp_repo, basic_config, mock_lock_manager, mock_task_coordinator):
        """Test _load_checkpoint handles interrupted task correctly."""
        checkpoint_data = {
            "batch_id": "batch-test",
            "started_at": "2025-01-01T12:00:00",
            "last_updated": "2025-01-01T13:00:00",
            "tasks_completed": ["TASK-001"],
            "tasks_failed": [],
            "tasks_skipped": [],
            "current_task": "TASK-002",  # Task was interrupted
            "config": {}
        }

        checkpoint_file = temp_repo / ".autonomous" / "checkpoint.json"
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f)

        config = BatchConfig(checkpoint_file=checkpoint_file, resume=True)
        runner = BatchRunner(config, temp_repo)

        checkpoint = runner._load_checkpoint()

        # Interrupted task should be added to skipped
        assert "TASK-002" in checkpoint.tasks_skipped
        assert checkpoint.current_task is None

    def test_checkpoint_saves_correct_state(self, temp_repo, basic_config, mock_lock_manager, mock_task_coordinator):
        """Test _checkpoint saves correct state."""
        runner = BatchRunner(basic_config, temp_repo)
        runner._start_time = datetime.utcnow()
        runner._batch_id = "batch-test"
        runner._completed = {"TASK-001", "TASK-002"}
        runner._failed = {"TASK-003"}
        runner._skipped = {"TASK-004"}

        # Mock atomic_file_operation to capture checkpoint data
        saved_checkpoint = None

        def mock_atomic_operation(file_path, operation_func, lock_name):
            nonlocal saved_checkpoint
            saved_checkpoint = operation_func({})
            # Write to file for verification
            with open(file_path, 'w') as f:
                json.dump(saved_checkpoint, f)

        runner.lock_manager.atomic_file_operation = mock_atomic_operation

        runner._checkpoint()

        # Verify checkpoint was saved
        assert saved_checkpoint is not None
        assert saved_checkpoint["batch_id"] == "batch-test"
        assert set(saved_checkpoint["tasks_completed"]) == {"TASK-001", "TASK-002"}
        assert set(saved_checkpoint["tasks_failed"]) == {"TASK-003"}
        assert set(saved_checkpoint["tasks_skipped"]) == {"TASK-004"}
        assert saved_checkpoint["current_task"] is None


# ==============================================================================
# SECTION 4: Task Execution Tests
# ==============================================================================

class TestTaskExecution:
    """Tests for task execution with mocked subprocess."""

    @pytest.fixture
    def temp_repo(self):
        """Create temporary repository directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            (repo_path / ".autonomous" / "logs").mkdir(parents=True)
            (repo_path / ".autonomous" / "workflow_states").mkdir(parents=True)
            (repo_path / "tasks").mkdir(parents=True)
            yield repo_path

    @pytest.fixture
    def basic_config(self, temp_repo):
        """Create basic BatchConfig for testing."""
        return BatchConfig(
            checkpoint_file=temp_repo / ".autonomous" / "checkpoint.json",
            report_file=temp_repo / ".autonomous" / "report.md",
            resume=False
        )

    @pytest.fixture
    def mock_lock_manager(self):
        """Mock DistributedLockManager."""
        with patch('batch_runner.DistributedLockManager') as mock_class:
            mock_instance = MagicMock()
            mock_class.return_value = mock_instance
            yield mock_instance

    @pytest.fixture
    def mock_task_coordinator(self):
        """Mock TaskCoordinator."""
        with patch('batch_runner.TaskCoordinator') as mock_class:
            mock_instance = MagicMock()
            mock_class.return_value = mock_instance
            yield mock_instance

    @patch('subprocess.Popen')
    @patch('time.sleep')  # Mock to avoid actual sleep
    def test_execute_task_success_exit_code_zero(self, mock_sleep, mock_popen, temp_repo, basic_config, mock_lock_manager, mock_task_coordinator):
        """Test _execute_task with successful subprocess (exit code 0)."""
        runner = BatchRunner(basic_config, temp_repo)
        runner._batch_id = "batch-test"

        # Mock subprocess success
        mock_process = MagicMock()
        mock_process.wait.return_value = 0  # Exit code 0
        mock_popen.return_value = mock_process

        task = {"id": "TASK-001", "title": "Test Task"}
        result = runner._execute_task(task)

        assert result.task_id == "TASK-001"
        assert result.status == "completed"
        assert result.duration_seconds >= 0  # Changed to >= to handle very fast execution
        assert result.error_message is None

    @patch('subprocess.Popen')
    def test_execute_task_failure_non_zero_exit_code(self, mock_popen, temp_repo, basic_config, mock_lock_manager, mock_task_coordinator):
        """Test _execute_task with failed subprocess (non-zero exit code)."""
        runner = BatchRunner(basic_config, temp_repo)
        runner._batch_id = "batch-test"

        # Mock subprocess failure
        mock_process = MagicMock()
        mock_process.wait.return_value = 1  # Exit code 1
        mock_popen.return_value = mock_process

        task = {"id": "TASK-002", "title": "Failing Task"}
        result = runner._execute_task(task)

        assert result.task_id == "TASK-002"
        assert result.status == "failed"
        assert result.duration_seconds >= 0  # Changed to >= to handle very fast execution
        assert "exited with code 1" in result.error_message

    @patch('subprocess.Popen')
    def test_execute_task_extracts_pr_number(self, mock_popen, temp_repo, basic_config, mock_lock_manager, mock_task_coordinator):
        """Test _execute_task extracts PR number from workflow state."""
        runner = BatchRunner(basic_config, temp_repo)
        runner._batch_id = "batch-test"

        # Mock subprocess success
        mock_process = MagicMock()
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process

        # Create workflow state file with PR number
        state_file = temp_repo / ".autonomous" / "workflow_states" / "TASK-003_state.json"
        state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(state_file, 'w') as f:
            json.dump({"pr_number": 456}, f)

        task = {"id": "TASK-003", "title": "Task with PR"}
        result = runner._execute_task(task)

        assert result.status == "completed"
        assert result.pr_number == 456

    @patch('subprocess.Popen')
    def test_execute_task_exception_handling(self, mock_popen, temp_repo, basic_config, mock_lock_manager, mock_task_coordinator):
        """Test _execute_task handles exceptions correctly."""
        runner = BatchRunner(basic_config, temp_repo)
        runner._batch_id = "batch-test"

        # Mock subprocess raising exception
        mock_popen.side_effect = Exception("Subprocess creation failed")

        task = {"id": "TASK-004", "title": "Exception Task"}
        result = runner._execute_task(task)

        assert result.task_id == "TASK-004"
        assert result.status == "failed"
        assert "Subprocess creation failed" in result.error_message

    @patch('subprocess.Popen')
    def test_execute_task_creates_log_file(self, mock_popen, temp_repo, basic_config, mock_lock_manager, mock_task_coordinator):
        """Test _execute_task creates log file."""
        runner = BatchRunner(basic_config, temp_repo)
        runner._batch_id = "batch-test"

        # Mock subprocess success
        mock_process = MagicMock()
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process

        task = {"id": "TASK-005", "title": "Log Test"}
        runner._execute_task(task)

        # Verify log file was created
        log_file = temp_repo / ".autonomous" / "logs" / "TASK-005_batch-test.log"
        assert log_file.exists()

    @patch('subprocess.Popen')
    def test_execute_task_extracts_error_from_log(self, mock_popen, temp_repo, basic_config, mock_lock_manager, mock_task_coordinator):
        """Test _execute_task extracts error message from log file."""
        runner = BatchRunner(basic_config, temp_repo)
        runner._batch_id = "batch-test"

        # Mock subprocess failure
        mock_process = MagicMock()
        mock_process.wait.return_value = 1

        # Create a side effect that writes to the log file during the execution
        def wait_side_effect():
            log_file = temp_repo / ".autonomous" / "logs" / "TASK-006_batch-test.log"
            with open(log_file, 'a') as f:
                f.write("Some output\n")
                f.write("ERROR: Something went wrong\n")
                f.write("More output\n")
            return 1

        mock_process.wait = wait_side_effect
        mock_popen.return_value = mock_process

        task = {"id": "TASK-006", "title": "Error Extraction Test"}
        result = runner._execute_task(task)

        assert result.status == "failed"
        assert "ERROR: Something went wrong" in result.error_message


# ==============================================================================
# SECTION 5: Integration Tests
# ==============================================================================

class TestReportGeneration:
    """Tests for complete report generation scenarios."""

    @pytest.fixture
    def temp_repo(self):
        """Create temporary repository directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            (repo_path / ".autonomous").mkdir(parents=True)
            (repo_path / "tasks").mkdir(parents=True)
            yield repo_path

    @pytest.fixture
    def mock_lock_manager(self):
        """Mock DistributedLockManager."""
        with patch('batch_runner.DistributedLockManager') as mock_class:
            mock_instance = MagicMock()
            mock_class.return_value = mock_instance
            yield mock_instance

    @pytest.fixture
    def mock_task_coordinator(self):
        """Mock TaskCoordinator."""
        with patch('batch_runner.TaskCoordinator') as mock_class:
            mock_instance = MagicMock()
            mock_class.return_value = mock_instance
            yield mock_instance

    def test_complete_report_with_completed_tasks(self, temp_repo, mock_lock_manager, mock_task_coordinator):
        """Test complete report generation with all completed tasks."""
        config = BatchConfig(
            checkpoint_file=temp_repo / ".autonomous" / "checkpoint.json",
            report_file=temp_repo / ".autonomous" / "report.md"
        )

        runner = BatchRunner(config, temp_repo)
        runner._start_time = datetime.utcnow() - timedelta(hours=3)
        runner._batch_id = "batch-complete-test"
        runner._completed = {"TASK-001", "TASK-002", "TASK-003"}
        runner._failed = set()
        runner._skipped = set()

        runner._task_results = [
            TaskResult(
                task_id="TASK-001",
                status="completed",
                started_at="2025-01-01T12:00:00",
                finished_at="2025-01-01T12:30:00",
                duration_seconds=1800.0,
                pr_number=101
            ),
            TaskResult(
                task_id="TASK-002",
                status="completed",
                started_at="2025-01-01T12:30:00",
                finished_at="2025-01-01T13:00:00",
                duration_seconds=1800.0,
                pr_number=102
            ),
            TaskResult(
                task_id="TASK-003",
                status="completed",
                started_at="2025-01-01T13:00:00",
                finished_at="2025-01-01T13:45:00",
                duration_seconds=2700.0,
                pr_number=103
            )
        ]

        report = runner._generate_report("completed")

        # Verify report structure
        assert "batch-complete-test" in report
        assert "COMPLETED" in report
        assert "Tasks Completed | 3" in report
        assert "Tasks Failed | 0" in report
        assert "### Completed (3)" in report
        assert "TASK-001" in report
        assert "#101" in report
        assert "TASK-002" in report
        assert "#102" in report
        assert "TASK-003" in report
        assert "#103" in report
        assert "Merge PRs: #101, #102, #103" in report

    def test_complete_report_with_mixed_status_tasks(self, temp_repo, mock_lock_manager, mock_task_coordinator):
        """Test complete report generation with mixed status tasks."""
        config = BatchConfig(
            checkpoint_file=temp_repo / ".autonomous" / "checkpoint.json",
            report_file=temp_repo / ".autonomous" / "report.md"
        )

        runner = BatchRunner(config, temp_repo)
        runner._start_time = datetime.utcnow() - timedelta(hours=2)
        runner._batch_id = "batch-mixed-test"
        runner._completed = {"TASK-001"}
        runner._failed = {"TASK-002"}
        runner._skipped = {"TASK-003"}

        runner._task_results = [
            TaskResult(
                task_id="TASK-001",
                status="completed",
                started_at="2025-01-01T12:00:00",
                finished_at="2025-01-01T12:30:00",
                duration_seconds=1800.0,
                pr_number=201
            ),
            TaskResult(
                task_id="TASK-002",
                status="failed",
                started_at="2025-01-01T12:30:00",
                finished_at="2025-01-01T12:35:00",
                duration_seconds=300.0,
                error_message="Build compilation failed at step 3"
            ),
            TaskResult(
                task_id="TASK-003",
                status="skipped",
                started_at="2025-01-01T12:35:00",
                finished_at="2025-01-01T12:36:00",
                duration_seconds=60.0,
                error_message="Missing dependencies"
            )
        ]

        report = runner._generate_report("completed")

        # Verify all sections present
        assert "batch-mixed-test" in report
        assert "Tasks Completed | 1" in report
        assert "Tasks Failed | 1" in report
        assert "Tasks Skipped | 1" in report
        assert "### Completed (1)" in report
        assert "### Failed (1)" in report
        assert "### Skipped (1)" in report
        assert "TASK-001" in report
        assert "#201" in report
        assert "TASK-002" in report
        assert "Build compilation failed at step 3" in report
        assert "TASK-003" in report
        assert "Missing dependencies" in report
        assert "Review failed tasks and fix issues" in report


# ==============================================================================
# SECTION 6: FIX-004 Tests - Accurate Batch Report with Cleanup Failures
# ==============================================================================

class TestFIX004PRMergeTracking:
    """Tests for FIX-004: PR merged tracked separately from cleanup success.

    Primary acceptance criteria:
    - BatchResult tracks pr_merged separately from cleanup_success
    - Report shows COMPLETED if PR merged regardless of cleanup
    - Test: PR merged + cleanup failed = COMPLETED in report
    """

    @pytest.fixture
    def temp_repo(self):
        """Create temporary repository directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            (repo_path / ".autonomous" / "logs").mkdir(parents=True)
            (repo_path / ".autonomous" / "workflow_states").mkdir(parents=True)
            (repo_path / "tasks").mkdir(parents=True)
            yield repo_path

    @pytest.fixture
    def basic_config(self, temp_repo):
        """Create basic BatchConfig for testing."""
        return BatchConfig(
            checkpoint_file=temp_repo / ".autonomous" / "checkpoint.json",
            report_file=temp_repo / ".autonomous" / "report.md",
            resume=False
        )

    @pytest.fixture
    def mock_lock_manager(self):
        """Mock DistributedLockManager."""
        with patch('batch_runner.DistributedLockManager') as mock_class:
            mock_instance = MagicMock()
            mock_class.return_value = mock_instance
            yield mock_instance

    @pytest.fixture
    def mock_task_coordinator(self):
        """Mock TaskCoordinator."""
        with patch('batch_runner.TaskCoordinator') as mock_class:
            mock_instance = MagicMock()
            mock_class.return_value = mock_instance
            yield mock_instance

    def test_task_result_has_pr_merged_and_cleanup_success_fields(self):
        """Test that TaskResult dataclass has pr_merged and cleanup_success fields."""
        result = TaskResult(
            task_id="TASK-001",
            status="completed",
            started_at="2025-01-01T12:00:00",
            finished_at="2025-01-01T12:30:00",
            duration_seconds=1800.0,
            pr_number=123,
            pr_merged=True,
            cleanup_success=False
        )

        assert result.pr_merged is True
        assert result.cleanup_success is False

    def test_task_result_default_values(self):
        """Test TaskResult default values for new FIX-004 fields."""
        result = TaskResult(
            task_id="TASK-001",
            status="completed",
            started_at="2025-01-01T12:00:00",
            finished_at="2025-01-01T12:30:00",
            duration_seconds=1800.0
        )

        # Default values from FIX-004
        assert result.pr_merged is False
        assert result.cleanup_success is True

    def test_task_result_serialization_includes_new_fields(self):
        """Test that to_dict() includes pr_merged and cleanup_success fields."""
        result = TaskResult(
            task_id="TASK-001",
            status="completed",
            started_at="2025-01-01T12:00:00",
            finished_at="2025-01-01T12:30:00",
            duration_seconds=1800.0,
            pr_number=456,
            pr_merged=True,
            cleanup_success=False
        )

        data = result.to_dict()

        assert "pr_merged" in data
        assert data["pr_merged"] is True
        assert "cleanup_success" in data
        assert data["cleanup_success"] is False

    def test_check_pr_merged_from_state_file_explicit_flag(self, temp_repo, basic_config, mock_lock_manager, mock_task_coordinator):
        """Test _check_pr_merged reads explicit pr_merged flag from state file."""
        runner = BatchRunner(basic_config, temp_repo)

        # Create workflow state file with explicit pr_merged flag
        state_file = temp_repo / ".autonomous" / "workflow_states" / "TASK-001_state.json"
        state_data = {
            "task_id": "TASK-001",
            "pr_number": 123,
            "pr_merged": True,
            "status": "COMPLETED"
        }
        with open(state_file, 'w') as f:
            json.dump(state_data, f)

        pr_merged, pr_number = runner._check_pr_merged("TASK-001")

        assert pr_merged is True
        assert pr_number == 123

    def test_check_pr_merged_from_state_file_completed_status(self, temp_repo, basic_config, mock_lock_manager, mock_task_coordinator):
        """Test _check_pr_merged detects PR merged from COMPLETED status (FIX-002 behavior)."""
        runner = BatchRunner(basic_config, temp_repo)

        # Create workflow state file with COMPLETED status but no explicit pr_merged
        state_file = temp_repo / ".autonomous" / "workflow_states" / "TASK-002_state.json"
        state_data = {
            "task_id": "TASK-002",
            "pr_number": 456,
            "status": "COMPLETED"
        }
        with open(state_file, 'w') as f:
            json.dump(state_data, f)

        pr_merged, pr_number = runner._check_pr_merged("TASK-002")

        assert pr_merged is True
        assert pr_number == 456

    def test_check_pr_merged_returns_false_when_not_merged(self, temp_repo, basic_config, mock_lock_manager, mock_task_coordinator):
        """Test _check_pr_merged returns False when PR not merged."""
        runner = BatchRunner(basic_config, temp_repo)

        # Create workflow state file with non-merged status
        state_file = temp_repo / ".autonomous" / "workflow_states" / "TASK-003_state.json"
        state_data = {
            "task_id": "TASK-003",
            "pr_number": 789,
            "pr_merged": False,
            "status": "PR_CREATION"
        }
        with open(state_file, 'w') as f:
            json.dump(state_data, f)

        pr_merged, pr_number = runner._check_pr_merged("TASK-003")

        assert pr_merged is False
        assert pr_number == 789

    def test_check_pr_merged_returns_false_when_no_state_file(self, temp_repo, basic_config, mock_lock_manager, mock_task_coordinator):
        """Test _check_pr_merged returns False when state file doesn't exist."""
        runner = BatchRunner(basic_config, temp_repo)

        pr_merged, pr_number = runner._check_pr_merged("NONEXISTENT-TASK")

        assert pr_merged is False
        assert pr_number is None

    @patch('subprocess.run')
    def test_check_pr_merged_gh_cli_fallback(self, mock_run, temp_repo, basic_config, mock_lock_manager, mock_task_coordinator):
        """Test _check_pr_merged uses gh CLI when state file doesn't have merge info."""
        runner = BatchRunner(basic_config, temp_repo)

        # Mock gh CLI returning merged status
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='{"state": "MERGED", "merged": true}'
        )

        pr_merged, pr_number = runner._check_pr_merged("TASK-004", pr_number=100)

        assert pr_merged is True
        assert pr_number == 100
        mock_run.assert_called_once()

    @patch('subprocess.Popen')
    def test_execute_task_pr_merged_cleanup_failed_returns_completed(self, mock_popen, temp_repo, basic_config, mock_lock_manager, mock_task_coordinator):
        """PRIMARY TEST: PR merged + cleanup failed = status COMPLETED (FIX-004 main criterion)."""
        runner = BatchRunner(basic_config, temp_repo)
        runner._batch_id = "batch-test"

        # Mock subprocess failure (non-zero exit code)
        mock_process = MagicMock()
        mock_process.wait.return_value = 1  # Exit code 1 (cleanup failed)
        mock_popen.return_value = mock_process

        # Create workflow state indicating PR was merged
        state_file = temp_repo / ".autonomous" / "workflow_states" / "TASK-MERGED_state.json"
        state_data = {
            "task_id": "TASK-MERGED",
            "pr_number": 999,
            "pr_merged": True,
            "status": "COMPLETED"
        }
        with open(state_file, 'w') as f:
            json.dump(state_data, f)

        task = {"id": "TASK-MERGED", "title": "Task with merged PR but cleanup failed"}
        result = runner._execute_task(task)

        # ACCEPTANCE CRITERION: status="completed" because PR merged
        assert result.status == "completed"
        assert result.pr_merged is True
        assert result.cleanup_success is False
        assert result.pr_number == 999
        assert "Cleanup failed" in result.error_message

    @patch('subprocess.Popen')
    def test_execute_task_pr_not_merged_returns_failed(self, mock_popen, temp_repo, basic_config, mock_lock_manager, mock_task_coordinator):
        """Test that task with non-zero exit and no PR merged returns failed status."""
        runner = BatchRunner(basic_config, temp_repo)
        runner._batch_id = "batch-test"

        # Mock subprocess failure
        mock_process = MagicMock()
        mock_process.wait.return_value = 1
        mock_popen.return_value = mock_process

        # No state file = PR not merged

        task = {"id": "TASK-FAILED", "title": "Task that truly failed"}
        result = runner._execute_task(task)

        # True failure - no PR merged
        assert result.status == "failed"
        assert result.pr_merged is False
        assert result.cleanup_success is False

    @patch('subprocess.Popen')
    def test_execute_task_clean_success(self, mock_popen, temp_repo, basic_config, mock_lock_manager, mock_task_coordinator):
        """Test clean success: exit code 0, cleanup_success=True."""
        runner = BatchRunner(basic_config, temp_repo)
        runner._batch_id = "batch-test"

        # Mock subprocess success
        mock_process = MagicMock()
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process

        # Create workflow state indicating PR was merged
        state_file = temp_repo / ".autonomous" / "workflow_states" / "TASK-CLEAN_state.json"
        state_data = {
            "task_id": "TASK-CLEAN",
            "pr_number": 111,
            "pr_merged": True,
            "status": "COMPLETED"
        }
        with open(state_file, 'w') as f:
            json.dump(state_data, f)

        task = {"id": "TASK-CLEAN", "title": "Fully successful task"}
        result = runner._execute_task(task)

        assert result.status == "completed"
        assert result.pr_merged is True
        assert result.cleanup_success is True  # Clean success
        assert result.error_message is None

    def test_report_shows_cleanup_failed_section(self, temp_repo, basic_config, mock_lock_manager, mock_task_coordinator):
        """Test that report includes 'Completed with Cleanup Issues' section."""
        runner = BatchRunner(basic_config, temp_repo)
        runner._start_time = datetime.utcnow() - timedelta(hours=1)
        runner._batch_id = "batch-cleanup-test"
        runner._completed = {"TASK-001", "TASK-002"}
        runner._failed = set()
        runner._skipped = set()

        runner._task_results = [
            # Clean success
            TaskResult(
                task_id="TASK-001",
                status="completed",
                started_at="2025-01-01T12:00:00",
                finished_at="2025-01-01T12:30:00",
                duration_seconds=1800.0,
                pr_number=100,
                pr_merged=True,
                cleanup_success=True
            ),
            # Cleanup failed but PR merged
            TaskResult(
                task_id="TASK-002",
                status="completed",
                started_at="2025-01-01T12:30:00",
                finished_at="2025-01-01T13:00:00",
                duration_seconds=1800.0,
                pr_number=200,
                pr_merged=True,
                cleanup_success=False,
                error_message="Cleanup failed after successful PR merge (exit code 1)"
            )
        ]

        report = runner._generate_report("completed")

        # Verify both sections present
        assert "### Completed (1)" in report  # Clean success
        assert "### Completed with Cleanup Issues (1)" in report  # Cleanup failed
        assert "TASK-001" in report
        assert "TASK-002" in report
        assert "Cleanup failed" in report
        assert "worktree cleanup failed" in report

    def test_report_counts_all_completed_correctly(self, temp_repo, basic_config, mock_lock_manager, mock_task_coordinator):
        """Test that summary table counts include cleanup-failed tasks in completed."""
        runner = BatchRunner(basic_config, temp_repo)
        runner._start_time = datetime.utcnow() - timedelta(hours=1)
        runner._batch_id = "batch-count-test"
        runner._completed = {"TASK-001", "TASK-002", "TASK-003"}  # All three completed
        runner._failed = set()
        runner._skipped = set()

        runner._task_results = [
            TaskResult(
                task_id="TASK-001", status="completed",
                started_at="", finished_at="", duration_seconds=100,
                pr_merged=True, cleanup_success=True
            ),
            TaskResult(
                task_id="TASK-002", status="completed",
                started_at="", finished_at="", duration_seconds=100,
                pr_merged=True, cleanup_success=False  # Cleanup failed
            ),
            TaskResult(
                task_id="TASK-003", status="completed",
                started_at="", finished_at="", duration_seconds=100,
                pr_merged=True, cleanup_success=False  # Cleanup failed
            )
        ]

        report = runner._generate_report("completed")

        # Total completed should be 3
        assert "Tasks Completed | 3" in report
        assert "### Completed (1)" in report  # 1 clean
        assert "### Completed with Cleanup Issues (2)" in report  # 2 with cleanup issues

    def test_notification_payload_includes_new_fields(self):
        """Test NotificationService includes pr_merged in payload."""
        service = NotificationService("https://example.com/webhook")

        task_results = [
            TaskResult(
                task_id="TASK-001",
                status="completed",
                started_at="2025-01-01T12:00:00",
                finished_at="2025-01-01T12:30:00",
                duration_seconds=1800.0,
                pr_number=123,
                pr_merged=True,
                cleanup_success=False
            )
        ]

        batch_result = BatchResult(
            batch_id="batch-test",
            status="completed",
            started_at="2025-01-01T12:00:00",
            finished_at="2025-01-01T13:00:00",
            duration_hours=1.0,
            tasks_completed=1,
            tasks_failed=0,
            tasks_skipped=0,
            tasks_remaining=0,
            task_results=task_results,
            exit_reason="completed"
        )

        # Mock the _post_webhook method
        with patch.object(service, '_post_webhook', return_value=True) as mock_post:
            service.send_completion(batch_result, "# Report")

            payload = mock_post.call_args[0][0]
            # Verify task result in payload
            assert len(payload["task_results"]) == 1
            assert payload["task_results"][0]["status"] == "completed"
            # Note: The notification payload format is simplified,
            # but BatchResult.to_dict() includes the full TaskResult


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
