#!/usr/bin/env python3
"""
Unit tests for orchestrator monitoring interface.

Tests the OrchestratorMonitor class and its query methods for
instance/task status and system health monitoring.
"""

import json
import pytest
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from autonomous.monitor import OrchestratorMonitor, format_dashboard
from autonomous.coordination import (
    DistributedLockManager,
    InstanceRegistry,
    InstanceStatus,
    TaskCoordinator,
    MessageQueue,
    MessageType
)


@pytest.fixture
def temp_repo():
    """Create temporary repository directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir)

        # Create required directories
        (repo_path / ".autonomous").mkdir(parents=True)
        (repo_path / "tasks").mkdir(parents=True)

        yield repo_path


@pytest.fixture
def monitor(temp_repo):
    """Create monitor instance with temporary repository."""
    return OrchestratorMonitor(main_repo=temp_repo)


@pytest.fixture
def populated_monitor(temp_repo):
    """Create monitor with populated test data."""
    lock_manager = DistributedLockManager(temp_repo)
    registry = InstanceRegistry(lock_manager)
    coordinator = TaskCoordinator(lock_manager)
    message_queue = MessageQueue(lock_manager)

    # Register test instance
    instance_id = registry.register_instance(
        "20251112-120000",
        "TASK-058",
        temp_repo / "worktree"
    )

    # Create test task queue
    task_queue_data = {
        "backlog": [
            {"id": "TASK-001", "status": "READY", "priority": "HIGH"},
            {"id": "TASK-002", "status": "READY", "priority": "MEDIUM"}
        ],
        "in_progress": [
            {
                "id": "TASK-058",
                "status": "IN_PROGRESS",
                "assigned_to": instance_id,
                "assigned_at": datetime.utcnow().isoformat(),
                "priority": "HIGH"
            }
        ],
        "completed": []
    }

    task_queue_path = temp_repo / "tasks" / "task_queue.json"
    with open(task_queue_path, 'w') as f:
        json.dump(task_queue_data, f)

    # Send test message
    message_queue.broadcast(
        instance_id,
        MessageType.INSTANCE_STARTED,
        {"task_id": "TASK-058"}
    )

    monitor = OrchestratorMonitor(main_repo=temp_repo)
    monitor._test_instance_id = instance_id  # Save for cleanup
    monitor._test_registry = registry  # Save for cleanup

    yield monitor

    # Cleanup
    registry.shutdown()


class TestOrchestratorMonitor:
    """Test OrchestratorMonitor class."""

    def test_initialization(self, temp_repo):
        """Test monitor initialization."""
        monitor = OrchestratorMonitor(main_repo=temp_repo)

        assert monitor.main_repo == temp_repo
        assert isinstance(monitor.lock_manager, DistributedLockManager)
        assert isinstance(monitor.registry, InstanceRegistry)
        assert isinstance(monitor.coordinator, TaskCoordinator)
        assert isinstance(monitor.queue, MessageQueue)

    def test_get_instance_dashboard_empty(self, monitor):
        """Test dashboard with no instances."""
        dashboard = monitor.get_instance_dashboard()

        assert isinstance(dashboard, dict)
        assert dashboard["active_instances"] == []
        assert dashboard["claimed_tasks"] == {}
        assert dashboard["recent_messages"] == []
        assert dashboard["stale_instances"] == []
        assert dashboard["available_tasks"] == 0
        assert dashboard["resource_usage"]["max_instances"] == 3
        assert dashboard["resource_usage"]["current_instances"] == 0
        assert dashboard["resource_usage"]["available_slots"] == 3
        assert "timestamp" in dashboard

    def test_get_instance_dashboard_with_data(self, populated_monitor):
        """Test dashboard with populated data."""
        dashboard = populated_monitor.get_instance_dashboard()

        assert len(dashboard["active_instances"]) == 1
        assert dashboard["active_instances"][0]["task"] == "TASK-058"
        assert dashboard["active_instances"][0]["status"] == InstanceStatus.STARTING.value

        assert "TASK-058" in dashboard["claimed_tasks"]
        assert dashboard["claimed_tasks"]["TASK-058"]["status"] == "IN_PROGRESS"

        assert dashboard["available_tasks"] == 2  # TASK-001, TASK-002

        assert len(dashboard["recent_messages"]) == 1
        assert dashboard["recent_messages"][0]["type"] == MessageType.INSTANCE_STARTED.value

        assert dashboard["resource_usage"]["current_instances"] == 1
        assert dashboard["resource_usage"]["available_slots"] == 2

    def test_get_instance_status_found(self, populated_monitor):
        """Test getting status of existing instance."""
        instance_id = populated_monitor._test_instance_id

        status = populated_monitor.get_instance_status(instance_id)

        assert status is not None
        assert status["instance_id"] == instance_id
        assert status["task_id"] == "TASK-058"
        assert status["status"] == InstanceStatus.STARTING.value
        assert "heartbeat_seconds_ago" in status
        assert status["is_stale"] is False
        assert "pid" in status
        assert "hostname" in status

    def test_get_instance_status_not_found(self, monitor):
        """Test getting status of non-existent instance."""
        status = monitor.get_instance_status("instance-nonexistent")

        assert status is None

    def test_get_task_status_found(self, populated_monitor):
        """Test getting status of existing task."""
        status = populated_monitor.get_task_status("TASK-058")

        assert status is not None
        assert status["id"] == "TASK-058"
        assert status["status"] == "IN_PROGRESS"
        assert status["assigned_to"] == populated_monitor._test_instance_id

    def test_get_task_status_not_found(self, monitor):
        """Test getting status of non-existent task."""
        status = monitor.get_task_status("TASK-NONEXISTENT")

        assert status is None

    def test_get_system_health_healthy(self, populated_monitor):
        """Test system health check with healthy system."""
        health = populated_monitor.get_system_health()

        assert isinstance(health, dict)
        assert health["total_instances"] == 1
        assert health["active_instances"] == 1
        assert health["stale_instances"] == 0
        assert health["tasks_in_progress"] == 1
        assert health["tasks_available"] == 2
        assert 0 <= health["resource_utilization"] <= 100
        assert health["health_status"] in ["HEALTHY", "DEGRADED", "UNHEALTHY"]
        assert isinstance(health["issues"], list)
        assert "timestamp" in health

    def test_get_system_health_empty(self, monitor):
        """Test system health check with no activity."""
        health = monitor.get_system_health()

        assert health["total_instances"] == 0
        assert health["active_instances"] == 0
        assert health["stale_instances"] == 0
        assert health["tasks_in_progress"] == 0
        assert health["health_status"] == "HEALTHY"

    def test_format_dashboard(self, populated_monitor):
        """Test dashboard formatting."""
        dashboard = populated_monitor.get_instance_dashboard()
        formatted = format_dashboard(dashboard)

        assert isinstance(formatted, str)
        assert "ORCHESTRATOR DASHBOARD" in formatted
        assert "Resource Usage:" in formatted
        assert "Active Instances:" in formatted
        assert "TASK-058" in formatted
        assert "Claimed Tasks:" in formatted

    def test_stale_instance_detection(self, temp_repo):
        """Test detection of stale instances."""
        lock_manager = DistributedLockManager(temp_repo)
        registry = InstanceRegistry(lock_manager)

        # Create instance with old heartbeat
        instance_id = f"instance-20251112-000000"
        instances_data = {
            "instances": {
                instance_id: {
                    "instance_id": instance_id,
                    "task_id": "TASK-001",
                    "status": InstanceStatus.EXECUTING.value,
                    "last_heartbeat": (datetime.utcnow() - timedelta(seconds=120)).isoformat(),
                    "start_time": datetime.utcnow().isoformat(),
                    "worktree_path": str(temp_repo / "worktree"),
                    "current_phase": "EXECUTION",
                    "gates_passed": [],
                    "pid": 12345,
                    "hostname": "test-host"
                }
            },
            "resource_limits": {
                "max_instances": 3,
                "current_count": 1,
                "queue": []
            }
        }

        registry_path = temp_repo / ".autonomous" / "instances.json"
        with open(registry_path, 'w') as f:
            json.dump(instances_data, f)

        monitor = OrchestratorMonitor(main_repo=temp_repo)
        dashboard = monitor.get_instance_dashboard()

        # Should be detected as stale (>60s since last heartbeat)
        assert len(dashboard["stale_instances"]) == 1
        assert dashboard["stale_instances"][0]["id"] == instance_id


def test_cli_help():
    """Test CLI help output."""
    import subprocess

    result = subprocess.run(
        ["python", "-m", "scripts.autonomous.monitor", "--help"],
        capture_output=True,
        text=True
    )

    assert result.returncode == 0
    assert "--dashboard" in result.stdout
    assert "--instance" in result.stdout
    assert "--task" in result.stdout
    assert "--health" in result.stdout


def test_cli_health_check():
    """Test CLI health check command."""
    import subprocess

    result = subprocess.run(
        ["python", "-m", "scripts.autonomous.monitor", "--health"],
        capture_output=True,
        text=True,
        cwd="C:/Users/tomas/GitHub/mmm-agents"
    )

    assert result.returncode == 0
    assert "System Health:" in result.stdout


def test_cli_dashboard():
    """Test CLI dashboard command."""
    import subprocess

    result = subprocess.run(
        ["python", "-m", "scripts.autonomous.monitor", "--dashboard"],
        capture_output=True,
        text=True,
        cwd="C:/Users/tomas/GitHub/mmm-agents"
    )

    assert result.returncode == 0
    assert "ORCHESTRATOR DASHBOARD" in result.stdout


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
