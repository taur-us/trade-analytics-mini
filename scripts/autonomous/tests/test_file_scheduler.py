#!/usr/bin/env python3
"""
Tests for Intelligent Task Scheduling with File Affinity (SCHEDULER-001).

Comprehensive test coverage for:
- FileAffinityAnalyzer: Glob expansion, file overlap, conflict detection
- FileLockRegistry: Lock acquisition, release, cleanup
- FileAwareScheduler: Task selection with conflict filtering
- FileAwareDecisionEngine: Full decision engine integration

Test scenarios include:
- Two tasks with overlapping files queued sequentially
- Two tasks with no overlap run in parallel
- Tasks without affects_files backwards compatibility
- Lock cleanup on instance crash
- Glob pattern expansion
"""

import json
import pytest
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set
from unittest.mock import MagicMock, patch

# Import test targets
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from autonomous.file_scheduler import (
    FileAffinityMode,
    FileLock,
    FileConflict,
    SchedulingDecision,
    FileAffinityAnalyzer,
    FileLockRegistry,
    FileAwareScheduler,
    FileAwareDecisionEngine,
    create_file_aware_decision_engine,
    FileSchedulingError,
    FileLockConflictError,
    FileLockTimeoutError,
)


# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture
def temp_repo():
    """Create a temporary repository with test file structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir)

        # Create standard directory structure
        (repo_path / "scripts" / "autonomous").mkdir(parents=True)
        (repo_path / "scripts" / "module_a").mkdir(parents=True)
        (repo_path / "scripts" / "module_b").mkdir(parents=True)
        (repo_path / "docs").mkdir(parents=True)
        (repo_path / "tests").mkdir(parents=True)
        (repo_path / ".autonomous").mkdir(parents=True)
        (repo_path / "tasks").mkdir(parents=True)

        # Create test files
        (repo_path / "scripts" / "autonomous" / "file1.py").touch()
        (repo_path / "scripts" / "autonomous" / "file2.py").touch()
        (repo_path / "scripts" / "autonomous" / "shared.py").touch()
        (repo_path / "scripts" / "module_a" / "main.py").touch()
        (repo_path / "scripts" / "module_a" / "utils.py").touch()
        (repo_path / "scripts" / "module_b" / "main.py").touch()
        (repo_path / "scripts" / "module_b" / "utils.py").touch()
        (repo_path / "docs" / "README.md").touch()
        (repo_path / "docs" / "SCHEDULER-001.md").touch()
        (repo_path / "tests" / "test_main.py").touch()

        yield repo_path


@pytest.fixture
def mock_lock_manager(temp_repo):
    """Create a mock DistributedLockManager for testing."""
    class MockLockManager:
        def __init__(self, main_repo: Path):
            self.main_repo = main_repo
            self._locks = {}

        def acquire_lock(self, lock_name: str, timeout: Optional[float] = None):
            """Mock lock acquisition."""
            class MockLock:
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    pass
            return MockLock()

        def atomic_file_operation(self, file_path: Path, operation, lock_name: str = None,
                                  max_retries: int = 3, create_if_missing: bool = True):
            """Execute operation atomically on JSON file."""
            file_path = Path(file_path)

            # Load or create data
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            elif create_if_missing:
                data = {}
            else:
                raise FileNotFoundError(f"File {file_path} does not exist")

            # Apply operation
            result = operation(data)

            # Handle None return (in-place modification)
            if result is None:
                result = data

            # Save result
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2)

            return result

    return MockLockManager(temp_repo)


@pytest.fixture
def task_queue(temp_repo):
    """Create a test task queue with various tasks."""
    queue = {
        "backlog": [
            {
                "id": "TASK-001",
                "title": "Task with file affinity",
                "priority": "HIGH",
                "status": "READY",
                "affects_files": ["scripts/autonomous/*.py"],
                "file_affinity_mode": "exclusive"
            },
            {
                "id": "TASK-002",
                "title": "Task with overlapping files",
                "priority": "MEDIUM",
                "status": "READY",
                "affects_files": ["scripts/autonomous/shared.py", "scripts/autonomous/other.py"],
                "file_affinity_mode": "exclusive"
            },
            {
                "id": "TASK-003",
                "title": "Task with no overlap (module_a)",
                "priority": "HIGH",
                "status": "READY",
                "affects_files": ["scripts/module_a/*.py"],
                "file_affinity_mode": "exclusive"
            },
            {
                "id": "TASK-004",
                "title": "Task with no overlap (module_b)",
                "priority": "MEDIUM",
                "status": "READY",
                "affects_files": ["scripts/module_b/*.py"],
                "file_affinity_mode": "exclusive"
            },
            {
                "id": "TASK-005",
                "title": "Task without affects_files (legacy)",
                "priority": "LOW",
                "status": "READY"
            },
            {
                "id": "TASK-006",
                "title": "Not ready task",
                "priority": "HIGH",
                "status": "IN_PROGRESS",
                "affects_files": ["docs/*.md"]
            }
        ],
        "in_progress": [],
        "completed": []
    }

    queue_path = temp_repo / "tasks" / "task_queue.json"
    with open(queue_path, 'w', encoding='utf-8') as f:
        json.dump(queue, f, indent=2)

    return queue_path


# ==============================================================================
# DATA MODEL TESTS
# ==============================================================================

class TestDataModels:
    """Tests for data model serialization and deserialization."""

    def test_file_lock_serialization(self):
        """Test FileLock to_dict and from_dict."""
        lock = FileLock(
            task_id="TASK-001",
            instance_id="instance-123",
            files={"file1.py", "file2.py"},
            mode=FileAffinityMode.EXCLUSIVE,
            acquired_at="2025-11-25T12:00:00",
            expires_at="2025-11-26T12:00:00"
        )

        # Serialize
        lock_dict = lock.to_dict()
        assert lock_dict["task_id"] == "TASK-001"
        assert lock_dict["instance_id"] == "instance-123"
        assert set(lock_dict["files"]) == {"file1.py", "file2.py"}
        assert lock_dict["mode"] == "exclusive"
        assert lock_dict["acquired_at"] == "2025-11-25T12:00:00"
        assert lock_dict["expires_at"] == "2025-11-26T12:00:00"

        # Deserialize
        restored = FileLock.from_dict(lock_dict)
        assert restored.task_id == lock.task_id
        assert restored.instance_id == lock.instance_id
        assert restored.files == lock.files
        assert restored.mode == lock.mode
        assert restored.acquired_at == lock.acquired_at
        assert restored.expires_at == lock.expires_at

    def test_file_conflict_serialization(self):
        """Test FileConflict to_dict."""
        conflict = FileConflict(
            candidate_task_id="TASK-002",
            blocking_task_id="TASK-001",
            conflicting_files={"shared.py"},
            blocking_instance_id="instance-123"
        )

        conflict_dict = conflict.to_dict()
        assert conflict_dict["candidate_task_id"] == "TASK-002"
        assert conflict_dict["blocking_task_id"] == "TASK-001"
        assert conflict_dict["conflicting_files"] == ["shared.py"]
        assert conflict_dict["blocking_instance_id"] == "instance-123"

    def test_scheduling_decision_serialization(self):
        """Test SchedulingDecision to_dict."""
        decision = SchedulingDecision(
            task_id="TASK-001",
            can_schedule=True,
            reason="Selected TASK-001 (priority: HIGH)",
            conflicts=[],
            safe_alternatives=["TASK-003", "TASK-004"]
        )

        decision_dict = decision.to_dict()
        assert decision_dict["task_id"] == "TASK-001"
        assert decision_dict["can_schedule"] is True
        assert "TASK-001" in decision_dict["reason"]
        assert decision_dict["conflicts"] == []
        assert decision_dict["safe_alternatives"] == ["TASK-003", "TASK-004"]


# ==============================================================================
# FILE AFFINITY ANALYZER TESTS
# ==============================================================================

class TestFileAffinityAnalyzer:
    """Tests for file affinity analysis functionality."""

    def test_expand_glob_patterns_single_pattern(self, temp_repo):
        """Test glob expansion with a single pattern."""
        analyzer = FileAffinityAnalyzer(temp_repo)

        files = analyzer.expand_glob_patterns(["scripts/autonomous/*.py"])

        assert "scripts/autonomous/file1.py" in files
        assert "scripts/autonomous/file2.py" in files
        assert "scripts/autonomous/shared.py" in files
        assert len(files) == 3

    def test_expand_glob_patterns_multiple_patterns(self, temp_repo):
        """Test glob expansion with multiple patterns."""
        analyzer = FileAffinityAnalyzer(temp_repo)

        files = analyzer.expand_glob_patterns([
            "scripts/autonomous/*.py",
            "docs/*.md"
        ])

        # Should include all autonomous files and docs
        assert "scripts/autonomous/file1.py" in files
        assert "docs/README.md" in files
        assert "docs/SCHEDULER-001.md" in files

    def test_expand_glob_patterns_recursive(self, temp_repo):
        """Test recursive glob pattern expansion."""
        analyzer = FileAffinityAnalyzer(temp_repo)

        files = analyzer.expand_glob_patterns(["scripts/**/*.py"])

        # Should find files in all subdirectories
        assert "scripts/autonomous/file1.py" in files
        assert "scripts/module_a/main.py" in files
        assert "scripts/module_b/main.py" in files

    def test_expand_glob_patterns_no_match(self, temp_repo):
        """Test glob expansion with no matches."""
        analyzer = FileAffinityAnalyzer(temp_repo)

        files = analyzer.expand_glob_patterns(["nonexistent/*.xyz"])

        assert files == set()

    def test_expand_glob_patterns_empty_list(self, temp_repo):
        """Test glob expansion with empty pattern list."""
        analyzer = FileAffinityAnalyzer(temp_repo)

        files = analyzer.expand_glob_patterns([])

        assert files == set()

    def test_compute_file_overlap_with_intersection(self, temp_repo):
        """Test file overlap computation with overlapping sets."""
        analyzer = FileAffinityAnalyzer(temp_repo)

        files_a = {"file1.py", "file2.py", "shared.py"}
        files_b = {"file3.py", "shared.py", "file4.py"}

        overlap = analyzer.compute_file_overlap(files_a, files_b)

        assert overlap == {"shared.py"}

    def test_compute_file_overlap_no_intersection(self, temp_repo):
        """Test file overlap computation with disjoint sets."""
        analyzer = FileAffinityAnalyzer(temp_repo)

        files_a = {"file1.py", "file2.py"}
        files_b = {"file3.py", "file4.py"}

        overlap = analyzer.compute_file_overlap(files_a, files_b)

        assert overlap == set()

    def test_detect_conflicts_with_overlap(self, temp_repo):
        """Test conflict detection with overlapping files."""
        analyzer = FileAffinityAnalyzer(temp_repo)

        candidate_task = {
            "id": "TASK-B",
            "affects_files": ["scripts/autonomous/shared.py"]
        }

        active_locks = [
            FileLock(
                task_id="TASK-A",
                instance_id="instance-123",
                files={"scripts/autonomous/shared.py", "scripts/autonomous/other.py"},
                mode=FileAffinityMode.EXCLUSIVE,
                acquired_at="2025-11-25T12:00:00"
            )
        ]

        conflicts = analyzer.detect_conflicts(candidate_task, active_locks)

        assert len(conflicts) == 1
        assert conflicts[0].candidate_task_id == "TASK-B"
        assert conflicts[0].blocking_task_id == "TASK-A"
        assert "scripts/autonomous/shared.py" in conflicts[0].conflicting_files

    def test_detect_conflicts_no_overlap(self, temp_repo):
        """Test conflict detection with no overlapping files."""
        analyzer = FileAffinityAnalyzer(temp_repo)

        candidate_task = {
            "id": "TASK-B",
            "affects_files": ["scripts/module_b/*.py"]
        }

        active_locks = [
            FileLock(
                task_id="TASK-A",
                instance_id="instance-123",
                files={"scripts/module_a/main.py", "scripts/module_a/utils.py"},
                mode=FileAffinityMode.EXCLUSIVE,
                acquired_at="2025-11-25T12:00:00"
            )
        ]

        conflicts = analyzer.detect_conflicts(candidate_task, active_locks)

        assert len(conflicts) == 0

    def test_detect_conflicts_no_affects_files(self, temp_repo):
        """Test conflict detection for task without affects_files."""
        analyzer = FileAffinityAnalyzer(temp_repo)

        candidate_task = {
            "id": "TASK-B",
            # No affects_files field
        }

        active_locks = [
            FileLock(
                task_id="TASK-A",
                instance_id="instance-123",
                files={"scripts/autonomous/shared.py"},
                mode=FileAffinityMode.EXCLUSIVE,
                acquired_at="2025-11-25T12:00:00"
            )
        ]

        conflicts = analyzer.detect_conflicts(candidate_task, active_locks)

        # Should return no conflicts (backwards compatible)
        assert len(conflicts) == 0


# ==============================================================================
# FILE LOCK REGISTRY TESTS
# ==============================================================================

class TestFileLockRegistry:
    """Tests for file lock registry functionality."""

    def test_acquire_and_release_locks(self, temp_repo, mock_lock_manager):
        """Test successful lock acquisition and release."""
        registry = FileLockRegistry(mock_lock_manager, temp_repo)

        # Acquire lock
        success, lock, conflicts = registry.acquire_file_locks(
            task_id="TASK-001",
            instance_id="instance-123",
            affects_files=["scripts/autonomous/*.py"],
        )

        assert success is True
        assert lock is not None
        assert "scripts/autonomous/file1.py" in lock.files
        assert "scripts/autonomous/file2.py" in lock.files
        assert "scripts/autonomous/shared.py" in lock.files
        assert lock.task_id == "TASK-001"
        assert lock.instance_id == "instance-123"
        assert lock.mode == FileAffinityMode.EXCLUSIVE
        assert len(conflicts) == 0

        # Release lock
        released = registry.release_file_locks("TASK-001", "instance-123")
        assert released is True

        # Verify locks are cleared
        active_locks = registry.get_active_locks()
        assert len(active_locks) == 0

    def test_conflict_detection_on_acquire(self, temp_repo, mock_lock_manager):
        """Test that conflicts prevent lock acquisition."""
        registry = FileLockRegistry(mock_lock_manager, temp_repo)

        # First lock succeeds
        success1, lock1, conflicts1 = registry.acquire_file_locks(
            task_id="TASK-A",
            instance_id="instance-123",
            affects_files=["scripts/autonomous/shared.py"],
        )
        assert success1 is True
        assert len(conflicts1) == 0

        # Second lock fails with conflict
        success2, lock2, conflicts2 = registry.acquire_file_locks(
            task_id="TASK-B",
            instance_id="instance-456",
            affects_files=["scripts/autonomous/shared.py"],
        )
        assert success2 is False
        assert lock2 is None
        assert len(conflicts2) == 1
        assert conflicts2[0].blocking_task_id == "TASK-A"

    def test_get_active_locks(self, temp_repo, mock_lock_manager):
        """Test retrieving active locks."""
        registry = FileLockRegistry(mock_lock_manager, temp_repo)

        # Initially no locks
        assert len(registry.get_active_locks()) == 0

        # Acquire some locks
        registry.acquire_file_locks(
            task_id="TASK-A",
            instance_id="instance-123",
            affects_files=["scripts/module_a/*.py"],
        )
        registry.acquire_file_locks(
            task_id="TASK-B",
            instance_id="instance-456",
            affects_files=["scripts/module_b/*.py"],
        )

        locks = registry.get_active_locks()
        assert len(locks) == 2
        task_ids = {l.task_id for l in locks}
        assert task_ids == {"TASK-A", "TASK-B"}

    def test_get_locked_files(self, temp_repo, mock_lock_manager):
        """Test getting set of all locked files."""
        registry = FileLockRegistry(mock_lock_manager, temp_repo)

        # Acquire locks on different modules
        registry.acquire_file_locks(
            task_id="TASK-A",
            instance_id="instance-123",
            affects_files=["scripts/module_a/*.py"],
        )

        locked_files = registry.get_locked_files()

        assert "scripts/module_a/main.py" in locked_files
        assert "scripts/module_a/utils.py" in locked_files

    def test_cleanup_expired_locks(self, temp_repo, mock_lock_manager):
        """Test automatic cleanup of expired locks."""
        registry = FileLockRegistry(mock_lock_manager, temp_repo)

        # Create lock with very short expiration (already expired)
        registry.acquire_file_locks(
            task_id="TASK-A",
            instance_id="instance-123",
            affects_files=["scripts/autonomous/*.py"],
            timeout_hours=0.0001  # ~0.36 seconds - will expire immediately
        )

        # Manually expire the lock by modifying the file
        locks_file = temp_repo / ".autonomous" / "file_locks.json"
        with open(locks_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Set expiration to the past
        past_time = (datetime.utcnow() - timedelta(hours=1)).isoformat()
        data['locks'][0]['expires_at'] = past_time

        with open(locks_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

        # Run cleanup
        removed = registry.cleanup_expired_locks()

        assert removed == 1
        assert len(registry.get_active_locks()) == 0

    def test_cleanup_instance_locks(self, temp_repo, mock_lock_manager):
        """Test cleanup of locks for crashed instance."""
        registry = FileLockRegistry(mock_lock_manager, temp_repo)

        # Create locks for different instances
        registry.acquire_file_locks(
            task_id="TASK-A",
            instance_id="instance-crashed",
            affects_files=["scripts/module_a/*.py"],
        )
        registry.acquire_file_locks(
            task_id="TASK-B",
            instance_id="instance-active",
            affects_files=["scripts/module_b/*.py"],
        )

        # Cleanup crashed instance
        removed = registry.cleanup_instance_locks("instance-crashed")

        assert removed == 1

        # Verify only active instance lock remains
        locks = registry.get_active_locks()
        assert len(locks) == 1
        assert locks[0].instance_id == "instance-active"

    def test_acquire_empty_pattern(self, temp_repo, mock_lock_manager):
        """Test acquiring lock with pattern that matches no files."""
        registry = FileLockRegistry(mock_lock_manager, temp_repo)

        success, lock, conflicts = registry.acquire_file_locks(
            task_id="TASK-A",
            instance_id="instance-123",
            affects_files=["nonexistent/*.xyz"],
        )

        # Should succeed with empty file set
        assert success is True
        assert lock is not None
        assert lock.files == set()


# ==============================================================================
# FILE-AWARE SCHEDULER TESTS
# ==============================================================================

class TestFileAwareScheduler:
    """Tests for file-aware scheduler functionality."""

    def test_get_safe_tasks_no_locks(self, temp_repo, mock_lock_manager, task_queue):
        """Test getting safe tasks when no locks held."""
        scheduler = FileAwareScheduler(mock_lock_manager, temp_repo, task_queue)

        safe_tasks = scheduler.get_safe_tasks()

        # All READY tasks should be safe
        safe_task_ids = {t["id"] for t in safe_tasks}
        assert "TASK-001" in safe_task_ids
        assert "TASK-002" in safe_task_ids
        assert "TASK-003" in safe_task_ids
        assert "TASK-004" in safe_task_ids
        assert "TASK-005" in safe_task_ids  # Legacy task without affects_files
        assert "TASK-006" not in safe_task_ids  # IN_PROGRESS, not READY

    def test_get_safe_tasks_with_locks(self, temp_repo, mock_lock_manager, task_queue):
        """Test getting safe tasks with active locks."""
        # Create locks first
        lock_registry = FileLockRegistry(mock_lock_manager, temp_repo)
        lock_registry.acquire_file_locks(
            task_id="TASK-001",
            instance_id="instance-123",
            affects_files=["scripts/autonomous/*.py"],
        )

        scheduler = FileAwareScheduler(mock_lock_manager, temp_repo, task_queue)
        safe_tasks = scheduler.get_safe_tasks()

        safe_task_ids = {t["id"] for t in safe_tasks}

        # TASK-002 conflicts with TASK-001 (overlapping files)
        assert "TASK-001" not in safe_task_ids  # Already has lock (would conflict with self)
        assert "TASK-002" not in safe_task_ids  # Overlaps with TASK-001
        assert "TASK-003" in safe_task_ids  # No overlap (module_a)
        assert "TASK-004" in safe_task_ids  # No overlap (module_b)
        assert "TASK-005" in safe_task_ids  # No affects_files, always safe

    def test_select_next_task_priority_ordering(self, temp_repo, mock_lock_manager, task_queue):
        """Test that select_next_task respects priority ordering."""
        scheduler = FileAwareScheduler(mock_lock_manager, temp_repo, task_queue)

        decision = scheduler.select_next_task()

        assert decision.can_schedule is True
        # TASK-001 and TASK-003 are both HIGH priority, but TASK-001 comes first in queue
        assert decision.task_id == "TASK-001"
        assert "priority: HIGH" in decision.reason

    def test_select_next_task_with_conflicts(self, temp_repo, mock_lock_manager, task_queue):
        """Test select_next_task with some tasks conflicting."""
        # Lock module_a files
        lock_registry = FileLockRegistry(mock_lock_manager, temp_repo)
        lock_registry.acquire_file_locks(
            task_id="ACTIVE-TASK",
            instance_id="instance-active",
            affects_files=["scripts/autonomous/*.py"],
        )

        scheduler = FileAwareScheduler(mock_lock_manager, temp_repo, task_queue)
        decision = scheduler.select_next_task()

        # Should select TASK-003 (HIGH priority, module_a, no conflict)
        assert decision.can_schedule is True
        assert decision.task_id == "TASK-003"

    def test_select_next_task_all_conflict(self, temp_repo, mock_lock_manager, task_queue):
        """Test select_next_task when all tasks conflict."""
        lock_registry = FileLockRegistry(mock_lock_manager, temp_repo)

        # Lock all possible files
        lock_registry.acquire_file_locks(
            task_id="BLOCKER-1",
            instance_id="instance-1",
            affects_files=["scripts/**/*.py"],  # All Python files
        )
        lock_registry.acquire_file_locks(
            task_id="BLOCKER-2",
            instance_id="instance-2",
            affects_files=["docs/*.md"],
        )

        scheduler = FileAwareScheduler(mock_lock_manager, temp_repo, task_queue)
        decision = scheduler.select_next_task()

        # TASK-005 has no affects_files, so it should still be selectable
        assert decision.can_schedule is True
        assert decision.task_id == "TASK-005"

    def test_select_next_task_no_ready_tasks(self, temp_repo, mock_lock_manager):
        """Test select_next_task when no READY tasks exist."""
        # Create empty task queue
        queue_path = temp_repo / "tasks" / "task_queue.json"
        with open(queue_path, 'w', encoding='utf-8') as f:
            json.dump({"backlog": [], "in_progress": [], "completed": []}, f)

        scheduler = FileAwareScheduler(mock_lock_manager, temp_repo, queue_path)
        decision = scheduler.select_next_task()

        assert decision.can_schedule is False
        assert decision.task_id is None
        assert "No READY tasks" in decision.reason


# ==============================================================================
# INTEGRATION TESTS
# ==============================================================================

class TestIntegration:
    """Integration tests for complete file-aware scheduling scenarios."""

    def test_overlapping_tasks_queued_sequentially(self, temp_repo, mock_lock_manager, task_queue):
        """
        Test: Two tasks with overlapping files are queued sequentially.

        Scenario:
        - TASK-A affects scripts/autonomous/*.py (starts first, holds lock)
        - TASK-B affects scripts/autonomous/shared.py (conflicts, must wait)

        Expected: TASK-A starts, TASK-B waits, TASK-B starts after TASK-A completes
        """
        lock_registry = FileLockRegistry(mock_lock_manager, temp_repo)
        scheduler = FileAwareScheduler(mock_lock_manager, temp_repo, task_queue)

        # Step 1: First task acquires lock
        success_a, lock_a, _ = lock_registry.acquire_file_locks(
            task_id="TASK-001",
            instance_id="instance-A",
            affects_files=["scripts/autonomous/*.py"],
        )
        assert success_a is True

        # Step 2: TASK-002 tries to acquire - should fail (conflicts)
        success_b, lock_b, conflicts_b = lock_registry.acquire_file_locks(
            task_id="TASK-002",
            instance_id="instance-B",
            affects_files=["scripts/autonomous/shared.py"],
        )
        assert success_b is False
        assert len(conflicts_b) == 1

        # Step 3: Scheduler should not select TASK-002
        decision = scheduler.select_next_task()
        assert decision.task_id != "TASK-002"  # TASK-002 blocked

        # Step 4: TASK-001 completes and releases lock
        lock_registry.release_file_locks("TASK-001", "instance-A")

        # Step 5: Now TASK-002 can acquire lock
        success_b2, lock_b2, conflicts_b2 = lock_registry.acquire_file_locks(
            task_id="TASK-002",
            instance_id="instance-B",
            affects_files=["scripts/autonomous/shared.py"],
        )
        assert success_b2 is True
        assert len(conflicts_b2) == 0

    def test_non_overlapping_tasks_run_parallel(self, temp_repo, mock_lock_manager, task_queue):
        """
        Test: Two tasks with no overlap run in parallel.

        Scenario:
        - TASK-A affects scripts/module_a/*.py
        - TASK-B affects scripts/module_b/*.py

        Expected: Both tasks can start simultaneously
        """
        lock_registry = FileLockRegistry(mock_lock_manager, temp_repo)

        # Both tasks acquire locks simultaneously - should succeed
        success_a, lock_a, conflicts_a = lock_registry.acquire_file_locks(
            task_id="TASK-003",
            instance_id="instance-A",
            affects_files=["scripts/module_a/*.py"],
        )
        success_b, lock_b, conflicts_b = lock_registry.acquire_file_locks(
            task_id="TASK-004",
            instance_id="instance-B",
            affects_files=["scripts/module_b/*.py"],
        )

        # Both should succeed - no overlap
        assert success_a is True
        assert success_b is True
        assert len(conflicts_a) == 0
        assert len(conflicts_b) == 0

        # Both locks should be active
        active_locks = lock_registry.get_active_locks()
        assert len(active_locks) == 2

    def test_task_without_affects_files_runs_alongside(self, temp_repo, mock_lock_manager, task_queue):
        """
        Test: Task without affects_files can run alongside any task.

        Scenario:
        - TASK-A affects scripts/*.py (has lock)
        - TASK-B has no affects_files (legacy task)

        Expected: Both can run in parallel
        """
        lock_registry = FileLockRegistry(mock_lock_manager, temp_repo)

        # Lock all scripts
        lock_registry.acquire_file_locks(
            task_id="TASK-001",
            instance_id="instance-A",
            affects_files=["scripts/**/*.py"],
        )

        # Legacy task can still acquire "lock" (empty file set)
        success, lock, conflicts = lock_registry.acquire_file_locks(
            task_id="TASK-005",
            instance_id="instance-B",
            affects_files=[],  # No files
        )

        assert success is True
        assert len(conflicts) == 0

    def test_lock_cleanup_on_crash(self, temp_repo, mock_lock_manager, task_queue):
        """
        Test: Locks are released when instance crashes.

        Scenario:
        - Instance-A acquires lock for TASK-A
        - Instance-A crashes (detected by heartbeat timeout)
        - Orchestrator cleans up locks
        - TASK-B can now acquire the same files
        """
        lock_registry = FileLockRegistry(mock_lock_manager, temp_repo)

        # Instance-A acquires lock
        lock_registry.acquire_file_locks(
            task_id="TASK-001",
            instance_id="instance-crashed",
            affects_files=["scripts/autonomous/*.py"],
        )

        # Verify lock is held
        assert len(lock_registry.get_active_locks()) == 1

        # Simulate crash cleanup
        removed = lock_registry.cleanup_instance_locks("instance-crashed")
        assert removed == 1

        # New task can now acquire
        success, lock, conflicts = lock_registry.acquire_file_locks(
            task_id="TASK-002",
            instance_id="instance-new",
            affects_files=["scripts/autonomous/shared.py"],
        )
        assert success is True


# ==============================================================================
# ERROR HANDLING TESTS
# ==============================================================================

class TestErrorHandling:
    """Tests for error handling scenarios."""

    def test_file_lock_conflict_error(self):
        """Test FileLockConflictError exception."""
        conflicts = [
            FileConflict(
                candidate_task_id="TASK-B",
                blocking_task_id="TASK-A",
                conflicting_files={"shared.py"},
                blocking_instance_id="instance-123"
            )
        ]

        error = FileLockConflictError("TASK-B", conflicts)

        assert error.task_id == "TASK-B"
        assert len(error.conflicts) == 1
        assert "Cannot acquire locks" in str(error)
        assert "1 conflicts" in str(error)

    def test_file_lock_timeout_error(self):
        """Test FileLockTimeoutError exception."""
        error = FileLockTimeoutError("TASK-A", 30.0)

        assert error.task_id == "TASK-A"
        assert error.timeout == 30.0
        assert "timed out" in str(error)

    def test_missing_task_queue(self, temp_repo, mock_lock_manager):
        """Test scheduler behavior with missing task queue."""
        # Path to non-existent file
        missing_path = temp_repo / "tasks" / "nonexistent.json"

        scheduler = FileAwareScheduler(mock_lock_manager, temp_repo, missing_path)
        decision = scheduler.select_next_task()

        assert decision.can_schedule is False
        assert "No READY tasks" in decision.reason


# ==============================================================================
# FILE-AWARE DECISION ENGINE TESTS
# ==============================================================================

class TestFileAwareDecisionEngine:
    """Tests for FileAwareDecisionEngine integration."""

    def test_should_spawn_instance_basic(self, temp_repo, mock_lock_manager, task_queue):
        """Test basic spawn decision with file affinity."""
        # Create mocks for monitor and coordinator
        mock_monitor = MagicMock()
        mock_monitor.get_instance_dashboard.return_value = {
            "resource_usage": {"current_instances": 0}
        }
        mock_coordinator = MagicMock()

        config = {
            "enable_spawning": True,
            "max_instances": 3,
            "spawn_cooldown": 0,  # No cooldown
            "models": {"HIGH": "sonnet", "MEDIUM": "sonnet", "LOW": "haiku"}
        }

        engine = FileAwareDecisionEngine(
            config=config,
            monitor=mock_monitor,
            coordinator=mock_coordinator,
            lock_manager=mock_lock_manager,
            main_repo=temp_repo,
            task_queue_path=task_queue,
            last_spawn_time=None
        )

        should_spawn, task_details = engine.should_spawn_instance()

        assert should_spawn is True
        assert task_details is not None
        assert task_details["task_id"] == "TASK-001"  # Highest priority READY task
        assert task_details["priority"] == "HIGH"
        assert task_details["model"] == "sonnet"
        assert "affects_files" in task_details

    def test_should_spawn_instance_spawning_disabled(self, temp_repo, mock_lock_manager, task_queue):
        """Test spawn decision when spawning is disabled."""
        mock_monitor = MagicMock()
        mock_coordinator = MagicMock()

        config = {"enable_spawning": False}

        engine = FileAwareDecisionEngine(
            config=config,
            monitor=mock_monitor,
            coordinator=mock_coordinator,
            lock_manager=mock_lock_manager,
            main_repo=temp_repo,
            task_queue_path=task_queue,
        )

        should_spawn, task_details = engine.should_spawn_instance()

        assert should_spawn is False
        assert task_details is None

    def test_should_spawn_instance_resource_limit(self, temp_repo, mock_lock_manager, task_queue):
        """Test spawn decision when at resource limit."""
        mock_monitor = MagicMock()
        mock_monitor.get_instance_dashboard.return_value = {
            "resource_usage": {"current_instances": 3}  # At limit
        }
        mock_coordinator = MagicMock()

        config = {
            "enable_spawning": True,
            "max_instances": 3,
        }

        engine = FileAwareDecisionEngine(
            config=config,
            monitor=mock_monitor,
            coordinator=mock_coordinator,
            lock_manager=mock_lock_manager,
            main_repo=temp_repo,
            task_queue_path=task_queue,
        )

        should_spawn, task_details = engine.should_spawn_instance()

        assert should_spawn is False
        assert task_details is None

    def test_should_spawn_instance_with_conflicts(self, temp_repo, mock_lock_manager, task_queue):
        """Test spawn decision when top priority tasks conflict."""
        # Create locks that block top tasks
        lock_registry = FileLockRegistry(mock_lock_manager, temp_repo)
        lock_registry.acquire_file_locks(
            task_id="BLOCKER",
            instance_id="instance-blocker",
            affects_files=["scripts/autonomous/*.py"],
        )

        mock_monitor = MagicMock()
        mock_monitor.get_instance_dashboard.return_value = {
            "resource_usage": {"current_instances": 1}
        }
        mock_coordinator = MagicMock()

        config = {
            "enable_spawning": True,
            "max_instances": 3,
            "spawn_cooldown": 0,
            "models": {"HIGH": "sonnet"}
        }

        engine = FileAwareDecisionEngine(
            config=config,
            monitor=mock_monitor,
            coordinator=mock_coordinator,
            lock_manager=mock_lock_manager,
            main_repo=temp_repo,
            task_queue_path=task_queue,
        )

        should_spawn, task_details = engine.should_spawn_instance()

        # Should select TASK-003 (module_a, no conflict) instead of TASK-001/TASK-002
        assert should_spawn is True
        assert task_details["task_id"] == "TASK-003"


# ==============================================================================
# FACTORY FUNCTION TEST
# ==============================================================================

class TestFactoryFunction:
    """Test for factory function."""

    def test_create_file_aware_decision_engine(self, temp_repo, mock_lock_manager, task_queue):
        """Test factory function creates proper engine."""
        mock_monitor = MagicMock()
        mock_coordinator = MagicMock()
        config = {"enable_spawning": True, "max_instances": 3}

        engine = create_file_aware_decision_engine(
            config=config,
            monitor=mock_monitor,
            coordinator=mock_coordinator,
            lock_manager=mock_lock_manager,
            main_repo=temp_repo,
            task_queue_path=task_queue,
        )

        assert isinstance(engine, FileAwareDecisionEngine)
        assert engine.config == config
        assert engine.monitor == mock_monitor


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
