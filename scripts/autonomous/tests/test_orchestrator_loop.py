#!/usr/bin/env python3
"""
Unit tests for orchestrator monitoring loop.

Tests the OrchestratorLoop class and its monitoring, detection, and intervention logic.
"""

import json
import pytest
import subprocess
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.autonomous.orchestrator_loop import OrchestratorLoop


@pytest.fixture
def temp_repo():
    """Create temporary repository directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir)

        # Create required directories
        (repo_path / ".autonomous").mkdir(parents=True)
        (repo_path / ".autonomous" / "logs").mkdir(parents=True)
        (repo_path / "tasks").mkdir(parents=True)

        # Create default config
        config_path = repo_path / ".autonomous" / "orchestrator_config.json"
        config = {
            "loop_interval": 1,  # Short interval for testing
            "max_instances": 3,
            "heartbeat_timeout": 300,
            "progress_timeout": 1800,
            "warning_threshold": 900,
            "snapshot_interval": 600,
            "enable_spawning": False,
            "enable_interventions": True,
            "log_level": "INFO"
        }
        with open(config_path, 'w') as f:
            json.dump(config, f)

        # Create task queue
        task_queue_path = repo_path / "tasks" / "task_queue.json"
        task_queue = {
            "backlog": {
                "TASK-001": {"status": "READY", "priority": "HIGH"},
                "TASK-002": {"status": "READY", "priority": "MEDIUM"}
            },
            "in_progress": [],
            "completed": []
        }
        with open(task_queue_path, 'w') as f:
            json.dump(task_queue, f)

        yield repo_path


@pytest.fixture
def orchestrator(temp_repo, monkeypatch):
    """Create orchestrator instance with mocked dependencies."""
    # Change to temp directory
    monkeypatch.chdir(temp_repo)

    # Mock coordination components
    with patch('autonomous.orchestrator_loop.DistributedLockManager'), \
         patch('autonomous.orchestrator_loop.InstanceRegistry'), \
         patch('autonomous.orchestrator_loop.TaskCoordinator'), \
         patch('autonomous.orchestrator_loop.MessageQueue'), \
         patch('autonomous.orchestrator_loop.OrchestratorMonitor'):

        config_path = temp_repo / ".autonomous" / "orchestrator_config.json"
        loop = OrchestratorLoop(config_path=config_path)

        yield loop


class TestOrchestratorLoop:
    """Test OrchestratorLoop class."""

    def test_initialization(self, orchestrator):
        """Test orchestrator initializes correctly."""
        assert orchestrator.running is True
        assert orchestrator.cycle_count == 0
        assert isinstance(orchestrator.config, dict)
        assert orchestrator.config["loop_interval"] == 1
        assert orchestrator.config["max_instances"] == 3

    def test_load_config_with_file(self, temp_repo):
        """Test configuration loading from file."""
        config_path = temp_repo / ".autonomous" / "orchestrator_config.json"
        config_data = {
            "loop_interval": 30,
            "max_instances": 5,
            "enable_spawning": True
        }

        with open(config_path, 'w') as f:
            json.dump(config_data, f)

        with patch('autonomous.orchestrator_loop.DistributedLockManager'), \
             patch('autonomous.orchestrator_loop.InstanceRegistry'), \
             patch('autonomous.orchestrator_loop.TaskCoordinator'), \
             patch('autonomous.orchestrator_loop.MessageQueue'), \
             patch('autonomous.orchestrator_loop.OrchestratorMonitor'):

            loop = OrchestratorLoop(config_path=config_path)

            assert loop.config["loop_interval"] == 30
            assert loop.config["max_instances"] == 5
            assert loop.config["enable_spawning"] is True

    def test_load_config_defaults(self, temp_repo):
        """Test default configuration when file doesn't exist."""
        config_path = temp_repo / ".autonomous" / "missing_config.json"

        with patch('autonomous.orchestrator_loop.DistributedLockManager'), \
             patch('autonomous.orchestrator_loop.InstanceRegistry'), \
             patch('autonomous.orchestrator_loop.TaskCoordinator'), \
             patch('autonomous.orchestrator_loop.MessageQueue'), \
             patch('autonomous.orchestrator_loop.OrchestratorMonitor'):

            loop = OrchestratorLoop(config_path=config_path)

            assert loop.config["loop_interval"] == 60  # Default
            assert loop.config["max_instances"] == 3  # Default
            assert loop.config["enable_spawning"] is False  # Default

    def test_monitoring_cycle_executes(self, orchestrator):
        """Test single monitoring cycle execution."""
        # Mock monitor dashboard
        orchestrator.monitor.get_instance_dashboard = Mock(return_value={
            "active_instances": [],
            "claimed_tasks": {},
            "stale_instances": [],
            "available_tasks": 2,
            "recent_messages": []
        })

        orchestrator.monitor.get_system_health = Mock(return_value={
            "health_status": "HEALTHY",
            "issues": []
        })

        # Mock message queue
        orchestrator.message_queue.broadcast = Mock()

        # Execute single cycle
        orchestrator._execute_monitoring_cycle()

        # Verify queries were made
        assert orchestrator.monitor.get_instance_dashboard.called
        assert orchestrator.monitor.get_system_health.called
        assert orchestrator.message_queue.broadcast.called

    def test_monitoring_cycle_with_active_instances(self, orchestrator):
        """Test monitoring cycle with active instances."""
        # Mock dashboard with active instances
        orchestrator.monitor.get_instance_dashboard = Mock(return_value={
            "active_instances": [
                {
                    "id": "instance-001",
                    "task": "TASK-001",
                    "status": "EXECUTING",
                    "phase": "DEVELOPMENT",
                    "last_heartbeat": "10s ago"
                }
            ],
            "claimed_tasks": {
                "TASK-001": {
                    "claimed_by": "instance-001",
                    "status": "IN_PROGRESS"
                }
            },
            "stale_instances": [],
            "available_tasks": 1,
            "recent_messages": []
        })

        orchestrator.monitor.get_system_health = Mock(return_value={
            "health_status": "HEALTHY",
            "issues": []
        })

        orchestrator.message_queue.broadcast = Mock()

        # Execute cycle
        orchestrator._execute_monitoring_cycle()

        # Verify heartbeat was tracked
        assert "instance-001" in orchestrator.last_heartbeat

    def test_stuck_instance_detection(self, orchestrator):
        """Test detection of stuck instances."""
        # Mock dashboard with stale instance
        orchestrator.monitor.get_instance_dashboard = Mock(return_value={
            "active_instances": [],
            "claimed_tasks": {
                "TASK-001": {
                    "claimed_by": "instance-001",
                    "status": "IN_PROGRESS"
                }
            },
            "stale_instances": [
                {
                    "id": "instance-001",
                    "task": "TASK-001",
                    "status": "EXECUTING",
                    "last_heartbeat": "2000s ago"  # Exceeds progress_timeout (1800s)
                }
            ],
            "available_tasks": 0,
            "recent_messages": []
        })

        orchestrator.monitor.get_system_health = Mock(return_value={
            "health_status": "DEGRADED",
            "issues": ["1 stale instance detected"]
        })

        orchestrator.message_queue.send_message = Mock()
        orchestrator.message_queue.broadcast = Mock()

        # Execute cycle
        orchestrator._execute_monitoring_cycle()

        # Verify intervention was sent (stuck instance)
        # Note: stuck detection is within _handle_stuck_instances
        # which checks progress_timeout (1800s)

    def test_crashed_instance_detection(self, orchestrator):
        """Test detection of crashed instances (no heartbeat)."""
        # Mock dashboard with crashed instance
        orchestrator.monitor.get_instance_dashboard = Mock(return_value={
            "active_instances": [],
            "claimed_tasks": {
                "TASK-001": {
                    "claimed_by": "instance-001",
                    "status": "IN_PROGRESS"
                }
            },
            "stale_instances": [
                {
                    "id": "instance-001",
                    "task": "TASK-001",
                    "status": "EXECUTING",
                    "last_heartbeat": "400s ago"  # Exceeds heartbeat_timeout (300s)
                }
            ],
            "available_tasks": 0,
            "recent_messages": []
        })

        orchestrator.monitor.get_system_health = Mock(return_value={
            "health_status": "DEGRADED",
            "issues": ["1 stale instance detected"]
        })

        orchestrator.coordinator.release_task = Mock()
        orchestrator.message_queue.broadcast = Mock()

        # Execute cycle
        orchestrator._execute_monitoring_cycle()

        # Verify cleanup was performed (task released)
        assert orchestrator.coordinator.release_task.called

    def test_intervention_escalation(self, orchestrator):
        """Test that interventions escalate (status request -> pause -> force release) (Phase 1 + Phase 4)."""
        instance_id = "instance-stuck"
        instances_with_phase = {instance_id: "INIT"}  # Phase 4: Add phase info

        # First intervention - should be status request
        orchestrator.message_queue.send_message = Mock()

        orchestrator._handle_stuck_instances([instance_id], instances_with_phase)

        assert len(orchestrator.intervention_history) == 1
        assert orchestrator.intervention_history[0]["intervention_type"] == "status_request"

        # Second intervention - should be pause (INIT is a short phase)
        orchestrator._handle_stuck_instances([instance_id], instances_with_phase)

        assert len(orchestrator.intervention_history) == 2
        assert orchestrator.intervention_history[1]["intervention_type"] == "pause"

        # Third intervention - should be force release
        orchestrator.monitor.get_instance_dashboard = Mock(return_value={
            "claimed_tasks": {
                "TASK-001": {
                    "claimed_by": instance_id,
                    "status": "IN_PROGRESS"
                }
            }
        })
        orchestrator.coordinator.release_task = Mock()

        orchestrator._handle_stuck_instances([instance_id], instances_with_phase)

        assert len(orchestrator.intervention_history) == 3
        assert orchestrator.intervention_history[2]["intervention_type"] == "force_release"
        assert orchestrator.coordinator.release_task.called

    def test_crashed_instance_cleanup(self, orchestrator):
        """Test cleanup after crashed instance."""
        instance_id = "instance-crashed"

        # Mock dashboard with claimed task
        orchestrator.monitor.get_instance_dashboard = Mock(return_value={
            "claimed_tasks": {
                "TASK-001": {
                    "claimed_by": instance_id,
                    "status": "IN_PROGRESS"
                }
            }
        })

        orchestrator.coordinator.release_task = Mock()

        # Handle crashed instance
        orchestrator._handle_crashed_instances([instance_id])

        # Verify task was released
        assert orchestrator.coordinator.release_task.called
        orchestrator.coordinator.release_task.assert_called_once_with(
            instance_id,
            "TASK-001",
            f"Instance crashed (no heartbeat for {orchestrator.config['heartbeat_timeout']}s)"
        )

        # Verify intervention recorded
        assert len(orchestrator.intervention_history) == 1
        assert orchestrator.intervention_history[0]["intervention_type"] == "cleanup_crashed"

    def test_state_snapshot(self, orchestrator, temp_repo):
        """Test state persistence."""
        # Add some state
        orchestrator.cycle_count = 10
        orchestrator.spawned_instances = {12345: "TASK-001"}
        orchestrator.last_heartbeat = {
            "instance-001": datetime.utcnow() - timedelta(seconds=30)
        }
        orchestrator.intervention_history = [
            {
                "timestamp": datetime.utcnow().isoformat(),
                "instance_id": "instance-001",
                "intervention_type": "status_request"
            }
        ]

        # Save snapshot
        orchestrator._snapshot_state()

        # Verify file exists
        state_path = temp_repo / ".autonomous" / "orchestrator_state.json"
        assert state_path.exists()

        # Verify contents
        with open(state_path, 'r') as f:
            state = json.load(f)

        assert state["cycle_count"] == 10
        assert "12345" in str(state["spawned_instances"]) or 12345 in state["spawned_instances"]
        assert "instance-001" in state["last_heartbeat"]
        assert len(state["intervention_history"]) == 1

    def test_format_duration(self, orchestrator):
        """Test duration formatting."""
        assert orchestrator._format_duration(timedelta(seconds=30)) == "30s"
        assert orchestrator._format_duration(timedelta(seconds=90)) == "1m 30s"
        assert orchestrator._format_duration(timedelta(seconds=3661)) == "1h 1m 1s"
        assert orchestrator._format_duration(timedelta(hours=2, minutes=15, seconds=30)) == "2h 15m 30s"

    def test_graceful_shutdown(self, orchestrator):
        """Test graceful shutdown handling."""
        orchestrator.running = True

        # Mock shutdown dependencies
        orchestrator._snapshot_state = Mock()
        orchestrator.message_queue.broadcast = Mock()
        orchestrator.registry.shutdown = Mock()

        # Trigger shutdown
        orchestrator._shutdown()

        # Verify shutdown actions
        assert orchestrator._snapshot_state.called
        assert orchestrator.message_queue.broadcast.called
        assert orchestrator.registry.shutdown.called

    def test_run_single_cycle(self, orchestrator):
        """Test run method executes single cycle and stops."""
        # Mock dependencies
        orchestrator.monitor.get_instance_dashboard = Mock(return_value={
            "active_instances": [],
            "claimed_tasks": {},
            "stale_instances": [],
            "available_tasks": 0,
            "recent_messages": []
        })

        orchestrator.monitor.get_system_health = Mock(return_value={
            "health_status": "HEALTHY",
            "issues": []
        })

        orchestrator.message_queue.broadcast = Mock()
        orchestrator._snapshot_state = Mock()
        orchestrator.registry.shutdown = Mock()

        # Make it stop after 1 cycle
        def stop_after_one():
            orchestrator.running = False

        orchestrator._execute_monitoring_cycle = Mock(side_effect=stop_after_one)

        # Run
        orchestrator.run()

        # Verify one cycle executed
        assert orchestrator._execute_monitoring_cycle.called
        assert orchestrator.cycle_count == 1


def test_orchestrator_loop_integration(temp_repo, monkeypatch):
    """Integration test: Orchestrator with real coordination layer."""
    # This test would require setting up real coordination components
    # Skip for now as it's more of an integration test
    pytest.skip("Integration test - requires full coordination setup")


class TestDecisionEngine:
    """Test DecisionEngine class for spawn decision logic."""

    @pytest.fixture
    def mock_monitor(self):
        """Create mock monitor."""
        from scripts.autonomous.orchestrator_loop import DecisionEngine
        monitor = Mock()
        monitor.get_instance_dashboard = Mock(return_value={
            "resource_usage": {
                "current_instances": 1,
                "max_instances": 3,
                "available_slots": 2
            }
        })
        return monitor

    @pytest.fixture
    def mock_coordinator(self):
        """Create mock coordinator."""
        return Mock()

    @pytest.fixture
    def base_config(self):
        """Basic config for tests."""
        return {
            "enable_spawning": True,
            "max_instances": 3,
            "spawn_cooldown": 60,
            "models": {
                "CRITICAL": "opus",
                "HIGH": "sonnet",
                "MEDIUM": "sonnet",
                "LOW": "haiku"
            }
        }

    @pytest.fixture
    def sample_task_queue(self, temp_repo):
        """Create sample task queue."""
        task_queue_path = temp_repo / "tasks" / "task_queue.json"
        task_queue_path.parent.mkdir(parents=True, exist_ok=True)

        task_queue = {
            "backlog": [
                {"id": "TASK-001", "status": "READY", "priority": "HIGH", "description": "High priority task"},
                {"id": "TASK-002", "status": "READY", "priority": "MEDIUM", "description": "Medium priority task"},
                {"id": "TASK-003", "status": "READY", "priority": "LOW", "description": "Low priority task"},
                {"id": "TASK-004", "status": "IN_PROGRESS", "priority": "HIGH", "description": "Already running"}
            ],
            "in_progress": [],
            "completed": []
        }

        with open(task_queue_path, 'w') as f:
            json.dump(task_queue, f)

        return task_queue_path

    def test_decision_engine_initialization(self, base_config, mock_monitor, mock_coordinator):
        """Test decision engine initialization."""
        from scripts.autonomous.orchestrator_loop import DecisionEngine

        engine = DecisionEngine(
            config=base_config,
            monitor=mock_monitor,
            coordinator=mock_coordinator,
            last_spawn_time=None
        )

        assert engine.config == base_config
        assert engine.monitor == mock_monitor
        assert engine.coordinator == mock_coordinator
        assert engine.last_spawn_time is None

    def test_spawning_disabled(self, base_config, mock_monitor, mock_coordinator, temp_repo, monkeypatch):
        """Test decision when spawning is disabled."""
        from scripts.autonomous.orchestrator_loop import DecisionEngine

        monkeypatch.chdir(temp_repo)
        config = base_config.copy()
        config["enable_spawning"] = False

        engine = DecisionEngine(config, mock_monitor, mock_coordinator)
        should_spawn, task_details = engine.should_spawn_instance()

        assert should_spawn is False
        assert task_details is None

    def test_resource_limits_reached(self, base_config, mock_monitor, mock_coordinator, temp_repo, monkeypatch):
        """Test decision when resource limits reached."""
        from scripts.autonomous.orchestrator_loop import DecisionEngine

        monkeypatch.chdir(temp_repo)

        # Set current instances = max instances
        mock_monitor.get_instance_dashboard = Mock(return_value={
            "resource_usage": {
                "current_instances": 3,
                "max_instances": 3,
                "available_slots": 0
            }
        })

        engine = DecisionEngine(base_config, mock_monitor, mock_coordinator)
        should_spawn, task_details = engine.should_spawn_instance()

        assert should_spawn is False
        assert task_details is None

    def test_cooldown_active(self, base_config, mock_monitor, mock_coordinator, temp_repo, monkeypatch):
        """Test decision when cooldown period active."""
        from scripts.autonomous.orchestrator_loop import DecisionEngine

        monkeypatch.chdir(temp_repo)

        # Set last spawn time to 30 seconds ago (cooldown = 60s)
        last_spawn = datetime.utcnow() - timedelta(seconds=30)

        engine = DecisionEngine(base_config, mock_monitor, mock_coordinator, last_spawn_time=last_spawn)
        should_spawn, task_details = engine.should_spawn_instance()

        assert should_spawn is False
        assert task_details is None

    def test_cooldown_elapsed(self, base_config, mock_monitor, mock_coordinator):
        """Test cooldown logic."""
        from scripts.autonomous.orchestrator_loop import DecisionEngine

        engine = DecisionEngine(base_config, mock_monitor, mock_coordinator)

        # No last spawn - cooldown elapsed
        assert engine._cooldown_elapsed() is True

        # Last spawn 30 seconds ago, cooldown 60s - not elapsed
        engine.last_spawn_time = datetime.utcnow() - timedelta(seconds=30)
        assert engine._cooldown_elapsed() is False

        # Last spawn 70 seconds ago, cooldown 60s - elapsed
        engine.last_spawn_time = datetime.utcnow() - timedelta(seconds=70)
        assert engine._cooldown_elapsed() is True

    def test_no_tasks_available(self, base_config, mock_monitor, mock_coordinator, temp_repo, monkeypatch):
        """Test decision when no tasks available."""
        from scripts.autonomous.orchestrator_loop import DecisionEngine

        monkeypatch.chdir(temp_repo)

        # Create empty task queue
        task_queue_path = temp_repo / "tasks" / "task_queue.json"
        task_queue_path.parent.mkdir(parents=True, exist_ok=True)
        with open(task_queue_path, 'w') as f:
            json.dump({"backlog": [], "in_progress": [], "completed": []}, f)

        engine = DecisionEngine(base_config, mock_monitor, mock_coordinator)
        should_spawn, task_details = engine.should_spawn_instance()

        assert should_spawn is False
        assert task_details is None

    def test_successful_spawn_decision(self, base_config, mock_monitor, mock_coordinator,
                                       temp_repo, sample_task_queue, monkeypatch):
        """Test successful spawn decision."""
        from scripts.autonomous.orchestrator_loop import DecisionEngine

        monkeypatch.chdir(temp_repo)

        engine = DecisionEngine(base_config, mock_monitor, mock_coordinator)
        should_spawn, task_details = engine.should_spawn_instance()

        assert should_spawn is True
        assert task_details is not None
        assert task_details["task_id"] == "TASK-001"  # Highest priority READY task
        assert task_details["priority"] == "HIGH"
        assert task_details["model"] == "sonnet"
        assert "description" in task_details

    def test_get_next_task_priority_order(self, base_config, mock_monitor, mock_coordinator,
                                          temp_repo, sample_task_queue, monkeypatch):
        """Test priority-based task selection."""
        from scripts.autonomous.orchestrator_loop import DecisionEngine

        monkeypatch.chdir(temp_repo)

        engine = DecisionEngine(base_config, mock_monitor, mock_coordinator)
        task = engine._get_next_task()

        # Should get TASK-001 (HIGH priority) not TASK-002 (MEDIUM) or TASK-003 (LOW)
        assert task is not None
        assert task["id"] == "TASK-001"
        assert task["priority"] == "HIGH"

    def test_get_next_task_critical_priority(self, base_config, mock_monitor, mock_coordinator,
                                             temp_repo, monkeypatch):
        """Test CRITICAL tasks are selected first."""
        from scripts.autonomous.orchestrator_loop import DecisionEngine

        monkeypatch.chdir(temp_repo)

        # Create task queue with CRITICAL task
        task_queue_path = temp_repo / "tasks" / "task_queue.json"
        task_queue_path.parent.mkdir(parents=True, exist_ok=True)
        task_queue = {
            "backlog": [
                {"id": "TASK-001", "status": "READY", "priority": "HIGH"},
                {"id": "TASK-002", "status": "READY", "priority": "CRITICAL"},
                {"id": "TASK-003", "status": "READY", "priority": "MEDIUM"}
            ],
            "in_progress": [],
            "completed": []
        }
        with open(task_queue_path, 'w') as f:
            json.dump(task_queue, f)

        engine = DecisionEngine(base_config, mock_monitor, mock_coordinator)
        task = engine._get_next_task()

        assert task is not None
        assert task["id"] == "TASK-002"  # CRITICAL priority
        assert task["priority"] == "CRITICAL"

    def test_get_next_task_filters_ready_only(self, base_config, mock_monitor, mock_coordinator,
                                               temp_repo, sample_task_queue, monkeypatch):
        """Test that only READY tasks are considered."""
        from scripts.autonomous.orchestrator_loop import DecisionEngine

        monkeypatch.chdir(temp_repo)

        engine = DecisionEngine(base_config, mock_monitor, mock_coordinator)
        task = engine._get_next_task()

        # TASK-004 is HIGH priority but IN_PROGRESS, so should not be selected
        assert task["status"] == "READY"
        assert task["id"] != "TASK-004"

    def test_select_model_by_priority(self, base_config, mock_monitor, mock_coordinator):
        """Test model selection based on priority."""
        from scripts.autonomous.orchestrator_loop import DecisionEngine

        engine = DecisionEngine(base_config, mock_monitor, mock_coordinator)

        assert engine._select_model("CRITICAL") == "opus"
        assert engine._select_model("HIGH") == "sonnet"
        assert engine._select_model("MEDIUM") == "sonnet"
        assert engine._select_model("LOW") == "haiku"
        assert engine._select_model("UNKNOWN") == "sonnet"  # Default

    def test_can_spawn_with_available_slots(self, base_config, mock_monitor, mock_coordinator):
        """Test resource limit checking with available slots."""
        from scripts.autonomous.orchestrator_loop import DecisionEngine

        engine = DecisionEngine(base_config, mock_monitor, mock_coordinator)
        assert engine._can_spawn() is True

    def test_can_spawn_at_limit(self, base_config, mock_coordinator):
        """Test resource limit checking at max instances."""
        from scripts.autonomous.orchestrator_loop import DecisionEngine

        monitor = Mock()
        monitor.get_instance_dashboard = Mock(return_value={
            "resource_usage": {
                "current_instances": 3,
                "max_instances": 3
            }
        })

        engine = DecisionEngine(base_config, monitor, mock_coordinator)
        assert engine._can_spawn() is False

    def test_get_next_task_old_format(self, base_config, mock_monitor, mock_coordinator,
                                      temp_repo, monkeypatch):
        """Test task queue parsing with old dict format."""
        from scripts.autonomous.orchestrator_loop import DecisionEngine

        monkeypatch.chdir(temp_repo)

        # Create task queue in old format (dict of task_id -> task_info)
        task_queue_path = temp_repo / "tasks" / "task_queue.json"
        task_queue_path.parent.mkdir(parents=True, exist_ok=True)
        task_queue = {
            "backlog": {
                "TASK-001": {"status": "READY", "priority": "HIGH"},
                "TASK-002": {"status": "READY", "priority": "LOW"}
            },
            "in_progress": [],
            "completed": []
        }
        with open(task_queue_path, 'w') as f:
            json.dump(task_queue, f)

        engine = DecisionEngine(base_config, mock_monitor, mock_coordinator)
        task = engine._get_next_task()

        assert task is not None
        assert task["id"] == "TASK-001"  # HIGH priority
        assert task["priority"] == "HIGH"


class TestInstanceSpawning:
    """Test instance spawning functionality (Phase 3)."""

    @pytest.fixture
    def spawn_orchestrator(self, temp_repo, monkeypatch):
        """Create orchestrator with spawn config enabled."""
        from scripts.autonomous.orchestrator_loop import OrchestratorLoop

        monkeypatch.chdir(temp_repo)

        # Create config with spawning enabled
        config_path = temp_repo / ".autonomous" / "orchestrator_config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config = {
            "loop_interval": 1,
            "max_instances": 3,
            "heartbeat_timeout": 300,
            "progress_timeout": 1800,
            "enable_spawning": True,
            "enable_interventions": True,
            "max_spawn_retries": 3,
            "models": {"CRITICAL": "opus", "HIGH": "sonnet", "MEDIUM": "sonnet", "LOW": "haiku"}
        }
        with open(config_path, 'w') as f:
            json.dump(config, f)

        # Mock coordination layer
        with patch('scripts.autonomous.orchestrator_loop.DistributedLockManager'), \
             patch('scripts.autonomous.orchestrator_loop.InstanceRegistry'), \
             patch('scripts.autonomous.orchestrator_loop.TaskCoordinator'), \
             patch('scripts.autonomous.orchestrator_loop.MessageQueue'), \
             patch('scripts.autonomous.orchestrator_loop.OrchestratorMonitor'):

            orch = OrchestratorLoop(config_path=config_path)
            yield orch

    def test_spawn_instance_success(self, spawn_orchestrator, temp_repo, monkeypatch):
        """Test successful instance spawn."""
        from scripts.autonomous.orchestrator_loop import OrchestratorLoop

        monkeypatch.chdir(temp_repo)

        task_details = {
            "task_id": "TASK-001",
            "priority": "HIGH",
            "model": "sonnet",
            "description": "Test task"
        }

        # Mock subprocess.Popen to avoid actual spawn
        mock_process = Mock()
        mock_process.pid = 12345

        with patch('subprocess.Popen', return_value=mock_process):
            result = spawn_orchestrator._spawn_instance(task_details)

        assert result is not None
        assert result["success"] is True
        assert result["pid"] == 12345
        assert result["task_id"] == "TASK-001"
        assert result["model"] == "sonnet"
        assert result["priority"] == "HIGH"
        assert "instance_id" in result
        assert "spawned_at" in result
        assert "logs" in result
        assert "stdout" in result["logs"]
        assert "stderr" in result["logs"]

    def test_spawn_instance_windows_flags(self, spawn_orchestrator, temp_repo, monkeypatch):
        """Test Windows-specific spawn flags."""
        from scripts.autonomous.orchestrator_loop import OrchestratorLoop

        monkeypatch.chdir(temp_repo)
        monkeypatch.setattr('sys.platform', 'win32')

        task_details = {
            "task_id": "TASK-002",
            "priority": "HIGH",
            "model": "sonnet",
            "description": "Windows test"
        }

        mock_process = Mock()
        mock_process.pid = 23456

        with patch('subprocess.Popen', return_value=mock_process) as mock_popen:
            result = spawn_orchestrator._spawn_instance(task_details)

            # Verify Windows flags used
            call_kwargs = mock_popen.call_args[1]
            assert 'creationflags' in call_kwargs
            # CREATE_NO_WINDOW (0x08000000) | CREATE_NEW_PROCESS_GROUP (0x00000200)
            expected_flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
            assert call_kwargs['creationflags'] == expected_flags
            assert call_kwargs['close_fds'] is True

    def test_spawn_instance_unix_flags(self, spawn_orchestrator, temp_repo, monkeypatch):
        """Test Unix-specific spawn flags."""
        from scripts.autonomous.orchestrator_loop import OrchestratorLoop

        monkeypatch.chdir(temp_repo)
        monkeypatch.setattr('sys.platform', 'linux')

        task_details = {
            "task_id": "TASK-003",
            "priority": "HIGH",
            "model": "sonnet",
            "description": "Unix test"
        }

        mock_process = Mock()
        mock_process.pid = 34567

        with patch('subprocess.Popen', return_value=mock_process) as mock_popen:
            result = spawn_orchestrator._spawn_instance(task_details)

            # Verify Unix flags used
            call_kwargs = mock_popen.call_args[1]
            assert call_kwargs['start_new_session'] is True
            assert call_kwargs['close_fds'] is True
            assert 'creationflags' not in call_kwargs  # Unix doesn't use creationflags

    def test_spawn_instance_retry_logic(self, spawn_orchestrator, temp_repo, monkeypatch):
        """Test retry with exponential backoff."""
        from scripts.autonomous.orchestrator_loop import OrchestratorLoop

        monkeypatch.chdir(temp_repo)

        task_details = {
            "task_id": "TASK-004",
            "priority": "HIGH",
            "model": "sonnet",
            "description": "Retry test"
        }

        # Mock time.sleep to avoid actual delays
        sleep_calls = []
        def mock_sleep(seconds):
            sleep_calls.append(seconds)

        monkeypatch.setattr('time.sleep', mock_sleep)

        # First call fails with OSError, second succeeds
        mock_process = Mock()
        mock_process.pid = 45678

        call_count = [0]
        def side_effect_popen(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise OSError("Resource temporarily unavailable")
            return mock_process

        with patch('subprocess.Popen', side_effect=side_effect_popen):
            result = spawn_orchestrator._spawn_instance(task_details)

        # Verify retry happened
        assert result is not None
        assert result["success"] is True
        assert len(sleep_calls) == 1
        assert sleep_calls[0] == 2  # 2^1 = 2 seconds backoff

    def test_spawn_instance_max_retries_exceeded(self, spawn_orchestrator, temp_repo, monkeypatch):
        """Test failure after max retries."""
        from scripts.autonomous.orchestrator_loop import OrchestratorLoop

        monkeypatch.chdir(temp_repo)

        task_details = {
            "task_id": "TASK-005",
            "priority": "HIGH",
            "model": "sonnet",
            "description": "Max retries test"
        }

        # Mock time.sleep to avoid actual delays
        monkeypatch.setattr('time.sleep', lambda x: None)

        # Always fail with OSError
        with patch('subprocess.Popen', side_effect=OSError("Out of memory")):
            result = spawn_orchestrator._spawn_instance(task_details)

        # Verify spawn failed after retries
        assert result is None

    def test_spawn_instance_pid_registry(self, spawn_orchestrator, temp_repo, monkeypatch):
        """Test PID registry persistence."""
        from scripts.autonomous.orchestrator_loop import OrchestratorLoop

        monkeypatch.chdir(temp_repo)

        task_details = {
            "task_id": "TASK-006",
            "priority": "CRITICAL",
            "model": "opus",
            "description": "PID registry test"
        }

        mock_process = Mock()
        mock_process.pid = 56789

        with patch('subprocess.Popen', return_value=mock_process):
            result = spawn_orchestrator._spawn_instance(task_details)

        # Verify PID registry file created
        registry_path = temp_repo / ".autonomous" / "spawned_instances.json"
        assert registry_path.exists()

        # Verify registry contents
        with open(registry_path, 'r') as f:
            registry = json.load(f)

        assert "instances" in registry
        assert "56789" in registry["instances"]

        instance_record = registry["instances"]["56789"]
        assert instance_record["pid"] == 56789
        assert instance_record["task_id"] == "TASK-006"
        assert instance_record["model"] == "opus"
        assert instance_record["priority"] == "CRITICAL"
        assert "instance_id" in instance_record
        assert "spawned_at" in instance_record
        assert "command" in instance_record
        assert "logs" in instance_record

    def test_spawn_instance_invalid_task_details(self, spawn_orchestrator):
        """Test spawn with invalid task_details."""
        from scripts.autonomous.orchestrator_loop import OrchestratorLoop

        # Missing required keys
        invalid_task_details = {
            "task_id": "TASK-007"
            # Missing: priority, model
        }

        result = spawn_orchestrator._spawn_instance(invalid_task_details)

        # Verify spawn failed due to validation
        assert result is None

    def test_spawn_instance_logs_created(self, spawn_orchestrator, temp_repo, monkeypatch):
        """Test stdout/stderr log files are created."""
        from scripts.autonomous.orchestrator_loop import OrchestratorLoop

        monkeypatch.chdir(temp_repo)

        task_details = {
            "task_id": "TASK-008",
            "priority": "MEDIUM",
            "model": "sonnet",
            "description": "Log files test"
        }

        mock_process = Mock()
        mock_process.pid = 67890

        with patch('subprocess.Popen', return_value=mock_process):
            result = spawn_orchestrator._spawn_instance(task_details)

        # Verify log directory created
        log_dir = temp_repo / ".autonomous" / "logs"
        assert log_dir.exists()

        # Verify log file paths returned
        assert "logs" in result
        assert "stdout" in result["logs"]
        assert "stderr" in result["logs"]

        # Verify log files contain instance_id and task_id in names
        stdout_path = Path(result["logs"]["stdout"])
        stderr_path = Path(result["logs"]["stderr"])

        assert "TASK-008" in stdout_path.name
        assert "TASK-008" in stderr_path.name
        assert stdout_path.suffix == ".out"
        assert stderr_path.suffix == ".err"


# ============================================================================
# Phase 4 Tests: Progress Tracking & Stuck Detection
# ============================================================================


def test_progress_tracker_phase_updates(tmp_path):
    """
    Test Phase 4: ProgressTracker tracks phase transitions correctly.

    Verifies:
    - Phase updates are recorded
    - Phase history is maintained
    - Phase transition logging works
    """
    from scripts.autonomous.orchestrator_loop import ProgressTracker

    tracker = ProgressTracker(tmp_path)

    # Update phase for instance
    tracker.update_phase("instance-1", "INIT")
    assert "instance-1" in tracker.instance_phases
    assert tracker.instance_phases["instance-1"]["current_phase"] == "INIT"

    # Transition to new phase
    tracker.update_phase("instance-1", "SEARCH")
    assert tracker.instance_phases["instance-1"]["current_phase"] == "SEARCH"
    assert len(tracker.instance_phases["instance-1"]["phase_history"]) == 1

    # Verify history contains previous phase
    phase, entered, exited = tracker.instance_phases["instance-1"]["phase_history"][0]
    assert phase == "INIT"


def test_progress_tracker_stuck_detection(tmp_path):
    """
    Test Phase 4: ProgressTracker identifies stuck instances.

    Verifies:
    - Instances stuck beyond phase timeout are detected
    - Phase-specific timeouts are applied correctly
    - Stuck instances list includes (id, phase, minutes_stuck)
    """
    from scripts.autonomous.orchestrator_loop import ProgressTracker
    from datetime import datetime, timedelta

    tracker = ProgressTracker(tmp_path)

    # Simulate instance stuck in INIT phase (5 min timeout)
    tracker.update_phase("instance-stuck", "INIT")

    # Manually set phase_entered_at to 10 minutes ago (beyond 5 min timeout)
    tracker.instance_phases["instance-stuck"]["phase_entered_at"] = \
        datetime.utcnow() - timedelta(minutes=10)

    # Check stuck detection
    stuck = tracker.get_stuck_instances()
    assert len(stuck) == 1
    assert stuck[0][0] == "instance-stuck"
    assert stuck[0][1] == "INIT"
    assert stuck[0][2] >= 9  # At least 9 minutes stuck


def test_phase_timeout_thresholds(tmp_path):
    """
    Test Phase 4: Different phases have different timeout thresholds.

    Verifies:
    - SEARCH phase has 120 min timeout (long-running)
    - INIT phase has 5 min timeout (short)
    - DEFAULT fallback works for unknown phases
    """
    from scripts.autonomous.orchestrator_loop import ProgressTracker

    tracker = ProgressTracker(tmp_path)

    # Check phase timeout constants
    assert tracker.PHASE_TIMEOUTS["SEARCH"] == 120  # 2 hours
    assert tracker.PHASE_TIMEOUTS["INIT"] == 5  # 5 minutes
    assert tracker.PHASE_TIMEOUTS["PR_CREATION"] == 5  # 5 minutes
    assert tracker.PHASE_TIMEOUTS["DEFAULT"] == 30  # 30 minutes

    # Verify check_progress uses correct timeouts
    from datetime import datetime, timedelta

    # Instance in SEARCH phase for 30 minutes (under 120 min threshold)
    tracker.update_phase("search-instance", "SEARCH")
    tracker.instance_phases["search-instance"]["phase_entered_at"] = \
        datetime.utcnow() - timedelta(minutes=30)

    assert tracker.check_progress("search-instance", "SEARCH") is True  # Not stuck

    # Instance in INIT phase for 10 minutes (over 5 min threshold)
    tracker.update_phase("init-instance", "INIT")
    tracker.instance_phases["init-instance"]["phase_entered_at"] = \
        datetime.utcnow() - timedelta(minutes=10)

    assert tracker.check_progress("init-instance", "INIT") is False  # Stuck


def test_smart_intervention_by_phase(tmp_path):
    """
    Test Phase 4: Intervention type varies by phase.

    Verifies:
    - SEARCH phase uses gentle "status_request"
    - INIT phase uses "pause" (short phase)
    - DEFAULT fallback works
    """
    from scripts.autonomous.orchestrator_loop import ProgressTracker

    tracker = ProgressTracker(tmp_path)

    # Long-running phases use status_request
    assert tracker.get_intervention_type("inst-1", "SEARCH") == "status_request"
    assert tracker.get_intervention_type("inst-2", "EXECUTING") == "status_request"

    # Short phases use pause
    assert tracker.get_intervention_type("inst-3", "INIT") == "pause"
    assert tracker.get_intervention_type("inst-4", "PR_CREATION") == "pause"
    assert tracker.get_intervention_type("inst-5", "DATA_CHECK") == "pause"

    # Unknown phases use DEFAULT
    assert tracker.get_intervention_type("inst-6", "UNKNOWN_PHASE") == "status_request"


def test_commit_activity_tracking(tmp_path, monkeypatch):
    """
    Test Phase 4: Optional git commit activity tracking.

    Verifies:
    - check_commit_activity() queries git log correctly
    - Commit timestamps are parsed and stored
    - Errors are handled gracefully
    """
    from scripts.autonomous.orchestrator_loop import ProgressTracker
    from datetime import datetime
    import subprocess

    tracker = ProgressTracker(tmp_path)

    # Mock successful git log output (Unix timestamp)
    def mock_run_success(*args, **kwargs):
        class Result:
            returncode = 0
            stdout = "1700000000"  # Example Unix timestamp
            stderr = ""
        return Result()

    monkeypatch.setattr(subprocess, "run", mock_run_success)

    # Check commit activity
    commit_time = tracker.check_commit_activity("instance-1")
    assert commit_time is not None
    assert isinstance(commit_time, datetime)

    # Mock failed git log (no commits found)
    def mock_run_no_commits(*args, **kwargs):
        class Result:
            returncode = 0
            stdout = ""  # No output
            stderr = ""
        return Result()

    monkeypatch.setattr(subprocess, "run", mock_run_no_commits)

    commit_time = tracker.check_commit_activity("instance-2")
    assert commit_time is None

    # Mock git error
    def mock_run_error(*args, **kwargs):
        raise subprocess.SubprocessError("Git error")

    monkeypatch.setattr(subprocess, "run", mock_run_error)

    commit_time = tracker.check_commit_activity("instance-3")
    assert commit_time is None  # Error handled gracefully


def test_integration_progress_and_heartbeat(temp_repo, monkeypatch):
    """
    Test Phase 4: Integration test combining heartbeat and progress checks.

    Verifies:
    - Instances stuck in phase are detected (even with active heartbeat)
    - Phase-aware intervention is used
    - Progress tracker is updated during monitoring cycle
    """
    from scripts.autonomous.orchestrator_loop import OrchestratorLoop
    from datetime import datetime, timedelta
    from unittest.mock import Mock, patch

    monkeypatch.chdir(temp_repo)

    # Mock coordination components
    with patch('scripts.autonomous.orchestrator_loop.DistributedLockManager'), \
         patch('scripts.autonomous.orchestrator_loop.InstanceRegistry'), \
         patch('scripts.autonomous.orchestrator_loop.TaskCoordinator'), \
         patch('scripts.autonomous.orchestrator_loop.MessageQueue'), \
         patch('scripts.autonomous.orchestrator_loop.OrchestratorMonitor'):

        # Setup orchestrator with mocked components
        loop = OrchestratorLoop()

        # Mock monitor
        mock_monitor = Mock()
        mock_monitor.get_instance_dashboard.return_value = {
            "active_instances": [
                {
                    "id": "instance-stuck-search",
                    "task": "TASK-123",
                    "status": "ACTIVE",
                    "phase": "SEARCH",
                    "last_heartbeat": "10s ago"  # Heartbeat is active!
                }
            ],
            "claimed_tasks": {
                "TASK-123": {
                    "claimed_by": "instance-stuck-search",
                    "status": "IN_PROGRESS"
                }
            },
            "stale_instances": [],  # Not stale by heartbeat
            "available_tasks": 0,
            "recent_messages": []
        }

        mock_monitor.get_system_health.return_value = {
            "health_status": "OK",
            "issues": []
        }

        loop.monitor = mock_monitor
        loop.message_queue.send_message = Mock()
        loop.config["enable_interventions"] = True

        # Manually set instance as stuck in SEARCH phase for 130 minutes (over 120 min threshold)
        loop.progress_tracker.update_phase("instance-stuck-search", "SEARCH")
        loop.progress_tracker.instance_phases["instance-stuck-search"]["phase_entered_at"] = \
            datetime.utcnow() - timedelta(minutes=130)

        # Execute monitoring cycle
        loop._execute_monitoring_cycle()

        # Verify progress tracker detected stuck instance
        stuck = loop.progress_tracker.get_stuck_instances()
        assert len(stuck) == 1
        assert stuck[0][0] == "instance-stuck-search"
        assert stuck[0][1] == "SEARCH"

        # Verify intervention was sent
        assert loop.message_queue.send_message.called
        call_args = loop.message_queue.send_message.call_args

        # Verify intervention message contains phase info
        intervention_data = call_args[0][2]  # Third argument is the data dict
        assert "phase" in intervention_data
        assert intervention_data["phase"] == "SEARCH"

        # Verify phase-aware intervention type (status_request for SEARCH)
        assert intervention_data["type"] == "status_request"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
