#!/usr/bin/env python3
"""
Windows-compatible worktree cleanup with retry and deferred queue.

FIX-003: Handles file lock issues on Windows by implementing:
1. Retry with exponential backoff (3 attempts, 2s/4s/8s)
2. Deferred cleanup queue for failed removals
3. Background cleanup process for queue processing
"""

import gc
import json
import logging
import os
import platform
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


@dataclass
class DeferredCleanup:
    """Record of a failed worktree cleanup for later retry."""
    worktree_path: str
    instance_dir: str
    task_id: str
    session_id: str
    branch_name: str
    first_failure: str
    last_attempt: str
    attempt_count: int
    failure_reason: str
    priority: str = "normal"

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'worktree_path': self.worktree_path,
            'instance_dir': self.instance_dir,
            'task_id': self.task_id,
            'session_id': self.session_id,
            'branch_name': self.branch_name,
            'first_failure': self.first_failure,
            'last_attempt': self.last_attempt,
            'attempt_count': self.attempt_count,
            'failure_reason': self.failure_reason,
            'priority': self.priority,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'DeferredCleanup':
        """Create from dictionary."""
        return cls(**data)


class WorktreeCleanupManager:
    """Manages worktree cleanup with retry and deferred queue."""

    DEFERRED_FILE = ".autonomous/deferred_cleanup.json"
    MAX_ATTEMPTS = 3
    BACKOFF_BASE = 2  # seconds

    def __init__(self, main_repo: Path):
        """Initialize cleanup manager.

        Args:
            main_repo: Path to the main repository
        """
        self.main_repo = Path(main_repo).resolve()
        self.deferred_path = self.main_repo / self.DEFERRED_FILE
        self._ensure_deferred_file()

    def _ensure_deferred_file(self) -> None:
        """Create deferred cleanup file if it doesn't exist."""
        self.deferred_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.deferred_path.exists():
            self._save_deferred_queue([])

    def cleanup_worktree(
        self,
        worktree_path: Path,
        instance_dir: Path,
        task_id: str,
        session_id: str,
        branch_name: str,
    ) -> bool:
        """Attempt worktree cleanup with retry logic.

        Args:
            worktree_path: Path to the worktree to remove
            instance_dir: Path to the instance directory
            task_id: Task identifier
            session_id: Session identifier
            branch_name: Git branch name

        Returns:
            True if cleanup succeeded, False if deferred
        """
        worktree_path = Path(worktree_path).resolve()
        instance_dir = Path(instance_dir).resolve()

        logger.info(f"Starting worktree cleanup: {worktree_path}")

        for attempt in range(1, self.MAX_ATTEMPTS + 1):
            success, error = self._attempt_cleanup(worktree_path)

            if success:
                logger.info(f"✓ Worktree removed on attempt {attempt}: {worktree_path}")
                # Also clean up instance directory if empty
                self._cleanup_instance_dir(instance_dir)
                return True

            # Failed - check if we should retry
            if attempt < self.MAX_ATTEMPTS:
                wait_time = self.BACKOFF_BASE ** attempt  # 2s, 4s, 8s
                logger.warning(
                    f"Worktree cleanup failed (attempt {attempt}/{self.MAX_ATTEMPTS}): {error}"
                )
                logger.warning(f"Retrying in {wait_time}s...")

                # Try to release file handles before retry
                self._release_handles()
                time.sleep(wait_time)
            else:
                # Final failure - add to deferred queue
                logger.warning(
                    f"Worktree cleanup failed after {self.MAX_ATTEMPTS} attempts: {error}"
                )
                logger.warning(f"Adding to deferred cleanup queue: {worktree_path}")

                self._add_to_deferred_queue(
                    worktree_path=str(worktree_path),
                    instance_dir=str(instance_dir),
                    task_id=task_id,
                    session_id=session_id,
                    branch_name=branch_name,
                    failure_reason=error,
                )

                logger.warning(f"Manual cleanup may be needed: {worktree_path}")
                return False

        return False

    def _attempt_cleanup(self, worktree_path: Path) -> Tuple[bool, str]:
        """Attempt a single worktree cleanup.

        Args:
            worktree_path: Path to the worktree

        Returns:
            Tuple of (success: bool, error_message: str)
        """
        if not worktree_path.exists():
            return True, ""

        try:
            # Step 1: Try git worktree remove
            cmd = f'git worktree remove "{worktree_path}" --force'
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                cwd=str(self.main_repo),
                timeout=60,
            )

            if result.returncode == 0:
                return True, ""

            error = result.stderr.strip() or result.stdout.strip()

            # Step 2: If git worktree failed, try manual removal
            if worktree_path.exists():
                logger.debug(f"Git worktree remove failed, trying manual removal: {error}")
                return self._manual_remove(worktree_path)

            return True, ""  # Already removed

        except subprocess.TimeoutExpired:
            return False, "Cleanup command timed out after 60s"
        except Exception as e:
            return False, str(e)

    def _manual_remove(self, worktree_path: Path) -> Tuple[bool, str]:
        """Manually remove worktree directory.

        Args:
            worktree_path: Path to the worktree

        Returns:
            Tuple of (success: bool, error_message: str)
        """
        try:
            # On Windows, try to make files writable first
            if platform.system() == 'Windows':
                self._make_writable(worktree_path)

            shutil.rmtree(worktree_path, onerror=self._remove_readonly)

            # Also prune git worktree references
            subprocess.run(
                "git worktree prune",
                shell=True,
                capture_output=True,
                cwd=str(self.main_repo),
                timeout=30,
            )

            return True, ""

        except Exception as e:
            return False, str(e)

    def _make_writable(self, path: Path) -> None:
        """Make all files in path writable (Windows fix).

        Args:
            path: Directory path
        """
        import stat
        for root, dirs, files in os.walk(path):
            for d in dirs:
                try:
                    os.chmod(os.path.join(root, d), stat.S_IRWXU)
                except Exception:
                    pass
            for f in files:
                try:
                    os.chmod(os.path.join(root, f), stat.S_IRWXU)
                except Exception:
                    pass

    def _remove_readonly(self, func, path, excinfo) -> None:
        """Error handler for shutil.rmtree to handle read-only files.

        Args:
            func: Function that raised the error
            path: Path that caused the error
            excinfo: Exception info
        """
        import stat
        try:
            os.chmod(path, stat.S_IRWXU)
            func(path)
        except Exception:
            pass

    def _release_handles(self) -> None:
        """Attempt to release file handles before retry."""
        # Force garbage collection to release any Python file handles
        gc.collect()

        # On Windows, give the system a moment to release handles
        if platform.system() == 'Windows':
            time.sleep(0.5)

    def _cleanup_instance_dir(self, instance_dir: Path) -> None:
        """Clean up instance directory if empty.

        Args:
            instance_dir: Path to instance directory
        """
        try:
            if instance_dir.exists():
                # Check if directory is empty (except for .autonomous)
                contents = list(instance_dir.iterdir())
                if not contents or (len(contents) == 1 and contents[0].name == '.autonomous'):
                    shutil.rmtree(instance_dir, onerror=self._remove_readonly)
                    logger.info(f"✓ Instance directory removed: {instance_dir}")
        except Exception as e:
            logger.debug(f"Could not remove instance directory: {e}")

    def _add_to_deferred_queue(
        self,
        worktree_path: str,
        instance_dir: str,
        task_id: str,
        session_id: str,
        branch_name: str,
        failure_reason: str,
    ) -> None:
        """Add failed cleanup to deferred queue.

        Args:
            worktree_path: Path to the worktree
            instance_dir: Path to the instance directory
            task_id: Task identifier
            session_id: Session identifier
            branch_name: Git branch name
            failure_reason: Reason for failure
        """
        queue = self._load_deferred_queue()

        # Check if already in queue (update if so)
        existing_idx = None
        for idx, item in enumerate(queue):
            if item.get('worktree_path') == worktree_path:
                existing_idx = idx
                break

        now = datetime.utcnow().isoformat()

        if existing_idx is not None:
            # Update existing entry
            queue[existing_idx]['last_attempt'] = now
            queue[existing_idx]['attempt_count'] += 1
            queue[existing_idx]['failure_reason'] = failure_reason
        else:
            # Add new entry
            deferred = DeferredCleanup(
                worktree_path=worktree_path,
                instance_dir=instance_dir,
                task_id=task_id,
                session_id=session_id,
                branch_name=branch_name,
                first_failure=now,
                last_attempt=now,
                attempt_count=1,
                failure_reason=failure_reason,
                priority="normal",
            )
            queue.append(deferred.to_dict())

        self._save_deferred_queue(queue)

    def _load_deferred_queue(self) -> List[Dict]:
        """Load deferred cleanup queue from file.

        Returns:
            List of deferred cleanup records
        """
        try:
            if self.deferred_path.exists():
                with open(self.deferred_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get('queue', [])
            return []
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load deferred queue: {e}")
            return []

    def _save_deferred_queue(self, queue: List[Dict]) -> None:
        """Save deferred cleanup queue to file.

        Args:
            queue: List of deferred cleanup records
        """
        try:
            data = {
                'version': '1.0',
                'last_updated': datetime.utcnow().isoformat(),
                'queue': queue,
            }
            with open(self.deferred_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except IOError as e:
            logger.error(f"Failed to save deferred queue: {e}")

    def process_deferred_queue(self) -> Tuple[int, int]:
        """Process deferred cleanup queue.

        Called by orchestrator loop or maintenance process.

        Returns:
            Tuple of (succeeded_count, failed_count)
        """
        queue = self._load_deferred_queue()
        if not queue:
            return 0, 0

        logger.info(f"Processing {len(queue)} deferred cleanups...")

        succeeded = 0
        failed = 0
        remaining = []

        for item in queue:
            worktree_path = Path(item['worktree_path'])

            # Skip if already removed
            if not worktree_path.exists():
                logger.info(f"✓ Deferred cleanup already removed: {worktree_path}")
                succeeded += 1
                continue

            # Attempt cleanup
            success, error = self._attempt_cleanup(worktree_path)

            if success:
                logger.info(f"✓ Deferred cleanup succeeded: {worktree_path}")
                # Clean up instance dir too
                self._cleanup_instance_dir(Path(item['instance_dir']))
                succeeded += 1
            else:
                # Update and keep in queue
                item['last_attempt'] = datetime.utcnow().isoformat()
                item['attempt_count'] += 1
                item['failure_reason'] = error
                remaining.append(item)
                failed += 1
                logger.warning(f"Deferred cleanup still failing: {worktree_path}")

        self._save_deferred_queue(remaining)

        logger.info(f"Deferred cleanup complete: {succeeded} succeeded, {failed} still pending")
        return succeeded, failed

    def get_queue_status(self) -> Dict:
        """Get status of deferred cleanup queue.

        Returns:
            Dictionary with queue status information
        """
        queue = self._load_deferred_queue()
        return {
            'count': len(queue),
            'items': [
                {
                    'worktree_path': item['worktree_path'],
                    'task_id': item['task_id'],
                    'attempt_count': item['attempt_count'],
                    'first_failure': item['first_failure'],
                }
                for item in queue
            ],
        }
