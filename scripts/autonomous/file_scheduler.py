#!/usr/bin/env python3
"""
Intelligent Task Scheduling with File Affinity.

Implements file-based conflict detection, lock management,
and scheduling decisions for safe parallel task execution.

SCHEDULER-001: Intelligent Task Scheduling with File Affinity
"""

import fnmatch
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ==============================================================================
# SECTION 1: Data Models
# ==============================================================================

class FileAffinityMode(Enum):
    """File access modes for tasks."""
    EXCLUSIVE = "exclusive"  # Task has exclusive access to files
    SHARED = "shared"        # Multiple tasks can read (future enhancement)
    NONE = "none"            # No file affinity tracking


@dataclass
class FileLock:
    """Represents a lock on a set of files."""
    task_id: str
    instance_id: str
    files: Set[str]
    mode: FileAffinityMode
    acquired_at: str
    expires_at: Optional[str] = None  # For timeout-based cleanup

    def to_dict(self) -> Dict:
        return {
            'task_id': self.task_id,
            'instance_id': self.instance_id,
            'files': list(self.files),
            'mode': self.mode.value,
            'acquired_at': self.acquired_at,
            'expires_at': self.expires_at,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'FileLock':
        return cls(
            task_id=data['task_id'],
            instance_id=data['instance_id'],
            files=set(data['files']),
            mode=FileAffinityMode(data.get('mode', 'exclusive')),
            acquired_at=data['acquired_at'],
            expires_at=data.get('expires_at'),
        )


@dataclass
class FileConflict:
    """Represents a conflict between tasks over files."""
    candidate_task_id: str
    blocking_task_id: str
    conflicting_files: Set[str]
    blocking_instance_id: str

    def to_dict(self) -> Dict:
        return {
            'candidate_task_id': self.candidate_task_id,
            'blocking_task_id': self.blocking_task_id,
            'conflicting_files': list(self.conflicting_files),
            'blocking_instance_id': self.blocking_instance_id,
        }


@dataclass
class SchedulingDecision:
    """Result of a scheduling decision."""
    task_id: Optional[str]
    can_schedule: bool
    reason: str
    conflicts: List[FileConflict] = field(default_factory=list)
    safe_alternatives: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            'task_id': self.task_id,
            'can_schedule': self.can_schedule,
            'reason': self.reason,
            'conflicts': [c.to_dict() for c in self.conflicts],
            'safe_alternatives': self.safe_alternatives,
        }


# ==============================================================================
# SECTION 2: File Affinity Analyzer
# ==============================================================================

class FileAffinityAnalyzer:
    """Analyzes file dependencies and detects conflicts."""

    def __init__(self, repo_path: Path):
        """Initialize analyzer with repository path.

        Args:
            repo_path: Path to repository root for glob expansion
        """
        self.repo_path = Path(repo_path).resolve()

    def expand_glob_patterns(self, patterns: List[str]) -> Set[str]:
        """Expand glob patterns to actual file paths.

        Args:
            patterns: List of glob patterns (e.g., ["scripts/*.py", "docs/*.md"])

        Returns:
            Set of relative file paths matching the patterns
        """
        expanded = set()

        for pattern in patterns:
            # Handle absolute patterns
            if pattern.startswith('/'):
                pattern = pattern[1:]

            # Use pathlib glob for expansion
            matches = list(self.repo_path.glob(pattern))

            for match in matches:
                if match.is_file():
                    # Store as relative path with forward slashes
                    rel_path = match.relative_to(self.repo_path)
                    expanded.add(str(rel_path).replace('\\', '/'))

        logger.debug(f"Expanded {len(patterns)} patterns to {len(expanded)} files")
        return expanded

    def compute_file_overlap(self, files_a: Set[str], files_b: Set[str]) -> Set[str]:
        """Compute intersection of two file sets.

        Args:
            files_a: First set of file paths
            files_b: Second set of file paths

        Returns:
            Set of files that appear in both sets
        """
        return files_a & files_b

    def detect_conflicts(self,
                        candidate_task: Dict,
                        active_locks: List[FileLock]) -> List[FileConflict]:
        """Detect file conflicts between candidate task and active locks.

        Args:
            candidate_task: Task dict with 'affects_files' field
            active_locks: List of currently held file locks

        Returns:
            List of FileConflict objects for any conflicts found
        """
        conflicts = []

        # Get candidate's file set
        candidate_patterns = candidate_task.get('affects_files', [])
        if not candidate_patterns:
            return []  # No file affinity, no conflicts

        candidate_files = self.expand_glob_patterns(candidate_patterns)

        # Check against each active lock
        for lock in active_locks:
            overlap = self.compute_file_overlap(candidate_files, lock.files)

            if overlap:
                conflict = FileConflict(
                    candidate_task_id=candidate_task['id'],
                    blocking_task_id=lock.task_id,
                    conflicting_files=overlap,
                    blocking_instance_id=lock.instance_id,
                )
                conflicts.append(conflict)
                logger.info(f"Conflict detected: {candidate_task['id']} conflicts with "
                           f"{lock.task_id} on {len(overlap)} files")

        return conflicts


# ==============================================================================
# SECTION 3: File Lock Registry
# ==============================================================================

class FileLockRegistry:
    """Manages file locks for task execution."""

    def __init__(self, lock_manager, main_repo: Path):
        """Initialize registry with lock manager.

        Args:
            lock_manager: DistributedLockManager instance
            main_repo: Path to main repository
        """
        self.lock_manager = lock_manager
        self.main_repo = Path(main_repo).resolve()
        self.locks_file = self.main_repo / ".autonomous" / "file_locks.json"
        self.analyzer = FileAffinityAnalyzer(self.main_repo)

    def acquire_file_locks(self,
                          task_id: str,
                          instance_id: str,
                          affects_files: List[str],
                          mode: FileAffinityMode = FileAffinityMode.EXCLUSIVE,
                          timeout_hours: float = 24.0) -> Tuple[bool, Optional[FileLock], List[FileConflict]]:
        """Atomically acquire locks on a set of files.

        Args:
            task_id: Task requesting the locks
            instance_id: Instance that will hold the locks
            affects_files: List of file patterns to lock
            mode: Lock mode (exclusive/shared)
            timeout_hours: Lock expiration time in hours

        Returns:
            Tuple of (success, FileLock if success else None, conflicts if failed)
        """
        acquired_lock = None
        conflicts_found = []

        def _acquire(data: Dict) -> Dict:
            nonlocal acquired_lock, conflicts_found

            locks = data.setdefault('locks', [])
            lock_objects = [FileLock.from_dict(l) for l in locks]

            # Expand patterns to files
            files_to_lock = self.analyzer.expand_glob_patterns(affects_files)

            if not files_to_lock:
                # No files to lock (patterns matched nothing)
                acquired_lock = FileLock(
                    task_id=task_id,
                    instance_id=instance_id,
                    files=set(),
                    mode=mode,
                    acquired_at=datetime.utcnow().isoformat(),
                    expires_at=None,
                )
                return data

            # Check for conflicts
            for lock in lock_objects:
                overlap = files_to_lock & lock.files
                if overlap and lock.mode == FileAffinityMode.EXCLUSIVE:
                    conflicts_found.append(FileConflict(
                        candidate_task_id=task_id,
                        blocking_task_id=lock.task_id,
                        conflicting_files=overlap,
                        blocking_instance_id=lock.instance_id,
                    ))

            if conflicts_found:
                return data  # Don't modify, return conflicts

            # Calculate expiration
            expires_at = (datetime.utcnow() + timedelta(hours=timeout_hours)).isoformat()

            # Create new lock
            new_lock = FileLock(
                task_id=task_id,
                instance_id=instance_id,
                files=files_to_lock,
                mode=mode,
                acquired_at=datetime.utcnow().isoformat(),
                expires_at=expires_at,
            )

            # Add to registry
            locks.append(new_lock.to_dict())
            data['locks'] = locks
            acquired_lock = new_lock

            logger.info(f"Acquired locks for {task_id}: {len(files_to_lock)} files")
            return data

        self.lock_manager.atomic_file_operation(
            self.locks_file,
            _acquire,
            lock_name="file_locks",
            create_if_missing=True
        )

        if acquired_lock:
            return True, acquired_lock, []
        else:
            return False, None, conflicts_found

    def release_file_locks(self, task_id: str, instance_id: str) -> bool:
        """Release all file locks held by a task.

        Args:
            task_id: Task releasing the locks
            instance_id: Instance that held the locks

        Returns:
            True if locks were released
        """
        released = False

        def _release(data: Dict) -> Dict:
            nonlocal released
            locks = data.get('locks', [])

            # Filter out locks for this task/instance
            remaining = [
                l for l in locks
                if not (l['task_id'] == task_id and l['instance_id'] == instance_id)
            ]

            if len(remaining) < len(locks):
                released = True
                logger.info(f"Released file locks for {task_id}")

            data['locks'] = remaining
            return data

        self.lock_manager.atomic_file_operation(
            self.locks_file,
            _release,
            lock_name="file_locks",
            create_if_missing=True
        )

        return released

    def get_active_locks(self) -> List[FileLock]:
        """Get all currently active file locks.

        Returns:
            List of FileLock objects
        """
        def _get_locks(data: Dict) -> Dict:
            return data

        data = self.lock_manager.atomic_file_operation(
            self.locks_file,
            _get_locks,
            lock_name="file_locks",
            create_if_missing=True
        )

        locks = data.get('locks', [])
        return [FileLock.from_dict(l) for l in locks]

    def get_locked_files(self) -> Set[str]:
        """Get set of all currently locked files.

        Returns:
            Set of file paths that are currently locked
        """
        locks = self.get_active_locks()
        all_files = set()
        for lock in locks:
            all_files.update(lock.files)
        return all_files

    def cleanup_expired_locks(self) -> int:
        """Remove locks that have expired.

        Returns:
            Number of locks removed
        """
        removed_count = 0

        def _cleanup(data: Dict) -> Dict:
            nonlocal removed_count
            locks = data.get('locks', [])
            now = datetime.utcnow()

            remaining = []
            for lock_dict in locks:
                expires_at = lock_dict.get('expires_at')
                if expires_at:
                    expiry = datetime.fromisoformat(expires_at)
                    if expiry < now:
                        removed_count += 1
                        logger.warning(f"Cleaned up expired lock for {lock_dict['task_id']}")
                        continue
                remaining.append(lock_dict)

            data['locks'] = remaining
            return data

        self.lock_manager.atomic_file_operation(
            self.locks_file,
            _cleanup,
            lock_name="file_locks",
            create_if_missing=True
        )

        return removed_count

    def cleanup_instance_locks(self, instance_id: str) -> int:
        """Remove all locks held by a specific instance (e.g., after crash).

        Args:
            instance_id: Instance whose locks should be released

        Returns:
            Number of locks removed
        """
        removed_count = 0

        def _cleanup(data: Dict) -> Dict:
            nonlocal removed_count
            locks = data.get('locks', [])

            remaining = []
            for lock_dict in locks:
                if lock_dict['instance_id'] == instance_id:
                    removed_count += 1
                    logger.warning(f"Cleaned up lock for crashed instance {instance_id}: {lock_dict['task_id']}")
                    continue
                remaining.append(lock_dict)

            data['locks'] = remaining
            return data

        self.lock_manager.atomic_file_operation(
            self.locks_file,
            _cleanup,
            lock_name="file_locks",
            create_if_missing=True
        )

        return removed_count


# ==============================================================================
# SECTION 4: File-Aware Scheduler
# ==============================================================================

class FileAwareScheduler:
    """Scheduler that considers file affinity for task selection."""

    # Priority ordering (higher = more important)
    PRIORITY_ORDER = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]

    def __init__(self, lock_manager, main_repo: Path, task_queue_path: Path):
        """Initialize scheduler.

        Args:
            lock_manager: DistributedLockManager instance
            main_repo: Path to main repository
            task_queue_path: Path to task_queue.json
        """
        self.lock_manager = lock_manager
        self.main_repo = Path(main_repo).resolve()
        self.task_queue_path = task_queue_path
        self.lock_registry = FileLockRegistry(lock_manager, main_repo)
        self.analyzer = FileAffinityAnalyzer(main_repo)

    def get_safe_tasks(self) -> List[Dict]:
        """Get all READY tasks that don't conflict with active locks.

        Returns:
            List of task dicts that are safe to start (no file conflicts)
        """
        # Get READY tasks from backlog
        ready_tasks = self._get_ready_tasks()

        if not ready_tasks:
            return []

        # Get active locks
        active_locks = self.lock_registry.get_active_locks()

        if not active_locks:
            # No locks held, all tasks are safe
            return ready_tasks

        # Filter to non-conflicting tasks
        safe_tasks = []
        for task in ready_tasks:
            conflicts = self.analyzer.detect_conflicts(task, active_locks)
            if not conflicts:
                safe_tasks.append(task)
            else:
                logger.debug(f"Task {task['id']} excluded due to file conflicts")

        logger.info(f"Found {len(safe_tasks)}/{len(ready_tasks)} safe tasks (no file conflicts)")
        return safe_tasks

    def select_next_task(self) -> SchedulingDecision:
        """Select the next task to execute, considering file affinity.

        Returns:
            SchedulingDecision with task selection result
        """
        # Get safe tasks
        safe_tasks = self.get_safe_tasks()

        if not safe_tasks:
            # Check if there are any READY tasks at all
            ready_tasks = self._get_ready_tasks()

            if not ready_tasks:
                return SchedulingDecision(
                    task_id=None,
                    can_schedule=False,
                    reason="No READY tasks in backlog",
                )
            else:
                # Tasks exist but all conflict
                active_locks = self.lock_registry.get_active_locks()
                conflicts = []
                for task in ready_tasks:
                    conflicts.extend(self.analyzer.detect_conflicts(task, active_locks))

                return SchedulingDecision(
                    task_id=None,
                    can_schedule=False,
                    reason=f"All {len(ready_tasks)} READY tasks conflict with active locks",
                    conflicts=conflicts,
                )

        # Sort by priority
        def priority_rank(task):
            priority = task.get("priority", "LOW")
            try:
                return self.PRIORITY_ORDER.index(priority)
            except ValueError:
                return len(self.PRIORITY_ORDER)

        safe_tasks.sort(key=priority_rank)

        # Select highest priority safe task
        selected = safe_tasks[0]

        return SchedulingDecision(
            task_id=selected['id'],
            can_schedule=True,
            reason=f"Selected {selected['id']} (priority: {selected.get('priority', 'LOW')})",
            safe_alternatives=[t['id'] for t in safe_tasks[1:5]],  # Top 5 alternatives
        )

    def _get_ready_tasks(self) -> List[Dict]:
        """Get all READY tasks from backlog.

        Returns:
            List of task dicts with status == "READY"
        """
        try:
            if not self.task_queue_path.exists():
                return []

            with open(self.task_queue_path, 'r', encoding='utf-8') as f:
                task_queue = json.load(f)

            backlog = task_queue.get("backlog", [])

            # Handle both list and dict formats
            if isinstance(backlog, dict):
                backlog = [{"id": k, **v} for k, v in backlog.items()]

            # Filter to READY tasks
            return [t for t in backlog if t.get("status") == "READY"]

        except (IOError, json.JSONDecodeError) as e:
            logger.error(f"Failed to load task queue: {e}")
            return []


# ==============================================================================
# SECTION 5: Integration with Orchestrator
# ==============================================================================

def create_file_aware_decision_engine(config: dict,
                                       monitor,
                                       coordinator,
                                       lock_manager,
                                       main_repo: Path,
                                       task_queue_path: Path,
                                       last_spawn_time=None):
    """Factory function to create a file-aware decision engine.

    This wraps the existing DecisionEngine with file affinity capabilities.

    Args:
        config: Orchestrator configuration
        monitor: OrchestratorMonitor instance
        coordinator: TaskCoordinator instance
        lock_manager: DistributedLockManager instance
        main_repo: Path to main repository
        task_queue_path: Path to task_queue.json
        last_spawn_time: Last spawn timestamp

    Returns:
        FileAwareDecisionEngine instance
    """
    return FileAwareDecisionEngine(
        config=config,
        monitor=monitor,
        coordinator=coordinator,
        lock_manager=lock_manager,
        main_repo=main_repo,
        task_queue_path=task_queue_path,
        last_spawn_time=last_spawn_time,
    )


class FileAwareDecisionEngine:
    """Enhanced decision engine with file affinity support."""

    PRIORITY_ORDER = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]

    def __init__(self, config: dict, monitor, coordinator, lock_manager,
                 main_repo: Path, task_queue_path: Path, last_spawn_time=None):
        self.config = config
        self.monitor = monitor
        self.coordinator = coordinator
        self.lock_manager = lock_manager
        self.main_repo = main_repo
        self.task_queue_path = task_queue_path
        self.last_spawn_time = last_spawn_time

        # Initialize file affinity components
        self.scheduler = FileAwareScheduler(lock_manager, main_repo, task_queue_path)
        self.lock_registry = FileLockRegistry(lock_manager, main_repo)

    def should_spawn_instance(self) -> tuple[bool, Optional[dict]]:
        """Decide whether to spawn a new instance (file-affinity aware).

        Returns:
            Tuple of (should_spawn: bool, task_details: Optional[dict])
        """
        # 1. Check if spawning is enabled
        if not self.config.get("enable_spawning", False):
            return False, None

        # 2. Check resource limits
        if not self._can_spawn():
            logger.debug("Cannot spawn: resource limits reached")
            return False, None

        # 3. Check spawn cooldown
        if not self._cooldown_elapsed():
            logger.debug("Cannot spawn: cooldown period active")
            return False, None

        # 4. Cleanup expired locks before selecting
        expired = self.lock_registry.cleanup_expired_locks()
        if expired > 0:
            logger.info(f"Cleaned up {expired} expired file locks")

        # 5. Select next task using file-aware scheduler
        decision = self.scheduler.select_next_task()

        if not decision.can_schedule:
            logger.debug(f"Cannot spawn: {decision.reason}")
            return False, None

        # 6. Get full task details
        task = self._get_task_by_id(decision.task_id)
        if not task:
            logger.error(f"Task {decision.task_id} not found")
            return False, None

        # 7. Select model based on priority
        model = self._select_model(task.get("priority", "LOW"))

        # 8. Build task details
        task_details = {
            "task_id": task["id"],
            "priority": task.get("priority", "LOW"),
            "model": model,
            "description": task.get("description", f"Execute {task['id']}"),
            "affects_files": task.get("affects_files", []),
        }

        logger.info(f"Decision: SPAWN instance for {task_details['task_id']} " +
                   f"(priority={task_details['priority']}, model={task_details['model']}, " +
                   f"affects_files={len(task_details['affects_files'])} patterns)")

        return True, task_details

    def _can_spawn(self) -> bool:
        """Check if we can spawn another instance."""
        try:
            dashboard = self.monitor.get_instance_dashboard()
            current_instances = dashboard["resource_usage"]["current_instances"]
            max_instances = self.config.get("max_instances", 3)
            return current_instances < max_instances
        except Exception as e:
            logger.error(f"Failed to check resource limits: {e}")
            return False

    def _cooldown_elapsed(self) -> bool:
        """Check if spawn cooldown period has elapsed."""
        if self.last_spawn_time is None:
            return True
        cooldown = self.config.get("spawn_cooldown", 60)
        elapsed = (datetime.utcnow() - self.last_spawn_time).total_seconds()
        return elapsed >= cooldown

    def _get_task_by_id(self, task_id: str) -> Optional[Dict]:
        """Get task details by ID."""
        try:
            with open(self.task_queue_path, 'r', encoding='utf-8') as f:
                task_queue = json.load(f)

            backlog = task_queue.get("backlog", [])
            if isinstance(backlog, dict):
                backlog = [{"id": k, **v} for k, v in backlog.items()]

            for task in backlog:
                if task.get("id") == task_id:
                    return task

            return None
        except Exception as e:
            logger.error(f"Failed to get task {task_id}: {e}")
            return None

    def _select_model(self, priority: str) -> str:
        """Select Claude model based on task priority."""
        models = self.config.get("models", {
            "CRITICAL": "opus",
            "HIGH": "sonnet",
            "MEDIUM": "sonnet",
            "LOW": "haiku"
        })
        return models.get(priority, "sonnet")


# ==============================================================================
# SECTION 6: Error Classes
# ==============================================================================

class FileSchedulingError(Exception):
    """Base exception for file scheduling errors."""
    pass


class FileLockConflictError(FileSchedulingError):
    """Raised when file lock cannot be acquired due to conflicts."""
    def __init__(self, task_id: str, conflicts: List[FileConflict]):
        self.task_id = task_id
        self.conflicts = conflicts
        super().__init__(f"Cannot acquire locks for {task_id}: {len(conflicts)} conflicts")


class FileLockTimeoutError(FileSchedulingError):
    """Raised when lock acquisition times out."""
    def __init__(self, task_id: str, timeout: float):
        self.task_id = task_id
        self.timeout = timeout
        super().__init__(f"Lock acquisition timed out for {task_id} after {timeout}s")


# ==============================================================================
# EXPORTS
# ==============================================================================

__all__ = [
    "FileAffinityMode",
    "FileLock",
    "FileConflict",
    "SchedulingDecision",
    "FileAffinityAnalyzer",
    "FileLockRegistry",
    "FileAwareScheduler",
    "FileAwareDecisionEngine",
    "create_file_aware_decision_engine",
    "FileSchedulingError",
    "FileLockConflictError",
    "FileLockTimeoutError",
]
