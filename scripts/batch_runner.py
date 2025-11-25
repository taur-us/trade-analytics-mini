#!/usr/bin/env python3
"""
Batch Runner for Autonomous Task Execution

Runs multiple tasks sequentially with:
- Task count limit (--max-tasks)
- Time limit (--max-hours)
- Progress checkpointing
- Graceful shutdown on SIGTERM/SIGINT
- Summary report generation
- Optional webhook notification

Usage:
    python -m scripts.batch_runner --max-tasks=5 --max-hours=8 --notify-webhook=URL
"""

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

from autonomous.coordination import (
    DistributedLockManager,
    TaskCoordinator
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==============================================================================
# SECTION 1: Data Models
# ==============================================================================

@dataclass
class BatchConfig:
    """Configuration for batch execution."""
    max_tasks: Optional[int] = None          # Max tasks to run (None = unlimited)
    max_hours: Optional[float] = None        # Max hours to run (None = unlimited)
    notify_webhook: Optional[str] = None     # Webhook URL for notifications
    checkpoint_file: Path = Path(".autonomous/batch_checkpoint.json")
    report_file: Path = Path(".autonomous/batch_report.md")
    resume: bool = True                      # Resume from checkpoint if exists
    task_filter: Optional[str] = None        # Filter tasks by ID pattern (e.g., "SDLC-*")
    priority_filter: Optional[str] = None    # Filter by priority (CRITICAL, HIGH, etc.)
    dry_run: bool = False                    # Preview tasks without executing


@dataclass
class BatchCheckpoint:
    """Checkpoint state for resumable batch execution."""
    batch_id: str                            # Unique batch identifier
    started_at: str                          # ISO timestamp
    last_updated: str                        # ISO timestamp
    tasks_completed: List[str]               # Task IDs completed
    tasks_failed: List[str]                  # Task IDs failed
    tasks_skipped: List[str]                 # Task IDs skipped
    current_task: Optional[str] = None       # Task currently executing (if interrupted)
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Serialize for JSON storage."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'BatchCheckpoint':
        """Deserialize from JSON."""
        return cls(**data)


@dataclass
class TaskResult:
    """Result of single task execution.

    FIX-004: Added pr_merged and cleanup_success fields to track PR merge status
    independently from cleanup success. This allows accurate batch reporting when
    PR was merged but cleanup failed (e.g., Windows file locks).
    """
    task_id: str
    status: str                              # "completed", "failed", "skipped"
    started_at: str
    finished_at: str
    duration_seconds: float
    pr_number: Optional[int] = None
    pr_merged: bool = False                  # FIX-004: Whether PR was merged
    cleanup_success: bool = True             # FIX-004: Whether cleanup succeeded
    error_message: Optional[str] = None
    gate_failures: List[Dict] = field(default_factory=list)  # From RESILIENCE-001

    def to_dict(self) -> Dict:
        """Serialize to dict."""
        return asdict(self)


@dataclass
class BatchResult:
    """Final batch execution result."""
    batch_id: str
    status: str                              # "completed", "interrupted", "failed"
    started_at: str
    finished_at: str
    duration_hours: float
    tasks_completed: int
    tasks_failed: int
    tasks_skipped: int
    tasks_remaining: int
    task_results: List[TaskResult]
    exit_reason: str                         # "max_tasks", "max_hours", "signal", "error", "no_tasks"

    def to_dict(self) -> Dict:
        """Serialize to dict."""
        result = asdict(self)
        result['task_results'] = [r.to_dict() if hasattr(r, 'to_dict') else r for r in self.task_results]
        return result


# ==============================================================================
# SECTION 2: Notification Service
# ==============================================================================

class NotificationService:
    """Handles notifications via webhook."""

    def __init__(self, webhook_url: Optional[str] = None):
        """Initialize notification service.

        Args:
            webhook_url: Optional webhook URL for notifications
        """
        self.webhook_url = webhook_url

    def send_completion(self, result: BatchResult, report: str) -> bool:
        """Send completion notification.

        Args:
            result: BatchResult with execution summary
            report: Markdown summary report

        Returns:
            True if notification sent successfully, False otherwise
        """
        if not self.webhook_url:
            return True

        payload = {
            "event": "batch_completed",
            "batch_id": result.batch_id,
            "status": result.status,
            "exit_reason": result.exit_reason,
            "summary": {
                "tasks_completed": result.tasks_completed,
                "tasks_failed": result.tasks_failed,
                "tasks_skipped": result.tasks_skipped,
                "duration_hours": result.duration_hours
            },
            "task_results": [
                {
                    "task_id": r.task_id,
                    "status": r.status,
                    "pr_number": r.pr_number,
                    "duration_seconds": r.duration_seconds
                }
                for r in result.task_results
            ],
            "timestamp": datetime.utcnow().isoformat()
        }

        return self._post_webhook(payload)

    def send_failure(self, task_id: str, error: str) -> bool:
        """Send failure notification.

        Args:
            task_id: Task ID that failed
            error: Error message

        Returns:
            True if notification sent successfully, False otherwise
        """
        if not self.webhook_url:
            return True

        payload = {
            "event": "task_failed",
            "task_id": task_id,
            "error": error,
            "timestamp": datetime.utcnow().isoformat()
        }

        return self._post_webhook(payload)

    def _post_webhook(self, payload: Dict) -> bool:
        """POST JSON payload to webhook URL with retry.

        Args:
            payload: JSON payload to send

        Returns:
            True if sent successfully, False otherwise
        """
        for attempt in range(3):
            try:
                data = json.dumps(payload).encode('utf-8')
                req = urllib.request.Request(
                    self.webhook_url,
                    data=data,
                    headers={'Content-Type': 'application/json'}
                )

                with urllib.request.urlopen(req, timeout=10) as response:
                    if response.status == 200:
                        logger.info(f"Webhook notification sent successfully (attempt {attempt + 1})")
                        return True
                    else:
                        logger.warning(f"Webhook returned status {response.status}")

            except urllib.error.URLError as e:
                logger.warning(f"Webhook attempt {attempt + 1} failed: {e}")
                if attempt < 2:  # Don't sleep after last attempt
                    time.sleep(2 ** attempt)

        logger.error("Failed to send webhook notification after 3 attempts")
        return False


# ==============================================================================
# SECTION 3: Batch Runner
# ==============================================================================

class BatchRunner:
    """Orchestrates batch execution of autonomous tasks."""

    def __init__(self, config: BatchConfig, main_repo: Path):
        """Initialize batch runner with configuration.

        Args:
            config: BatchConfig with execution settings
            main_repo: Path to main repository root
        """
        self.config = config
        self.main_repo = Path(main_repo).resolve()

        # Initialize coordination components
        self.lock_manager = DistributedLockManager(self.main_repo)
        self.task_coordinator = TaskCoordinator(self.lock_manager)
        self.notifier = NotificationService(config.notify_webhook)

        # Runtime state
        self._start_time: Optional[datetime] = None
        self._batch_id: Optional[str] = None
        self._shutdown_requested = False
        self._current_process: Optional[subprocess.Popen] = None

        # Task tracking
        self._completed: set = set()
        self._failed: set = set()
        self._skipped: set = set()
        self._task_results: List[TaskResult] = []

        # Register signal handlers
        signal.signal(signal.SIGINT, self._handle_signal)
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, self._handle_signal)

    def run(self) -> BatchResult:
        """Execute batch and return results.

        Main execution loop:
        1. Load checkpoint (if resume=True and exists)
        2. Get available tasks from task queue
        3. For each task (until limit reached or no tasks):
           a. Check time limit
           b. Check signal flag (graceful shutdown)
           c. Execute task via SpawnOrchestrator
           d. Record result
           e. Save checkpoint
        4. Generate summary report
        5. Send notification (if configured)
        6. Return BatchResult

        Returns:
            BatchResult with execution summary
        """
        self._start_time = datetime.utcnow()
        self._batch_id = f"batch-{self._start_time.strftime('%Y%m%d-%H%M%S')}"

        logger.info(f"Starting batch execution: {self._batch_id}")
        logger.info(f"Config: max_tasks={self.config.max_tasks}, max_hours={self.config.max_hours}")

        # Load checkpoint if resume enabled
        checkpoint = None
        if self.config.resume:
            checkpoint = self._load_checkpoint()
            if checkpoint:
                logger.info(f"Resuming batch {checkpoint.batch_id}")
                self._batch_id = checkpoint.batch_id
                self._completed = set(checkpoint.tasks_completed)
                self._failed = set(checkpoint.tasks_failed)
                self._skipped = set(checkpoint.tasks_skipped)
                logger.info(f"Progress: {len(self._completed)} completed, {len(self._failed)} failed, {len(self._skipped)} skipped")

        # Get available tasks
        tasks = self._get_available_tasks()
        if not tasks:
            logger.info("No tasks available in backlog")
            return self._create_result("no_tasks")

        logger.info(f"Found {len(tasks)} tasks in backlog")

        # Dry run mode - just preview
        if self.config.dry_run:
            logger.info("DRY RUN MODE - Preview only")
            self._preview_tasks(tasks)
            return self._create_result("dry_run")

        # Main execution loop
        for task in tasks:
            # Check limits
            if not self._should_continue():
                logger.info("Stopping: limits reached")
                break

            # Check shutdown flag
            if self._shutdown_requested:
                logger.info("Shutdown requested - stopping after current task")
                break

            # Skip already processed
            if task["id"] in self._completed | self._failed | self._skipped:
                logger.debug(f"Skipping already processed task: {task['id']}")
                continue

            # Execute task
            logger.info(f"=" * 80)
            logger.info(f"Starting task {len(self._task_results) + 1}/{len(tasks)}: {task['id']}")
            logger.info(f"=" * 80)

            result = self._execute_task(task)
            self._task_results.append(result)

            # Update tracking
            if result.status == "completed":
                self._completed.add(task["id"])
                logger.info(f"Task {task['id']} COMPLETED (PR #{result.pr_number})" if result.pr_number else f"Task {task['id']} COMPLETED")
            elif result.status == "failed":
                self._failed.add(task["id"])
                logger.error(f"Task {task['id']} FAILED: {result.error_message}")
            else:
                self._skipped.add(task["id"])
                logger.warning(f"Task {task['id']} SKIPPED: {result.error_message}")

            # Checkpoint after each task
            self._checkpoint()

        # Determine exit reason
        exit_reason = self._determine_exit_reason(tasks)
        logger.info(f"Batch execution complete. Exit reason: {exit_reason}")

        # Generate report
        report = self._generate_report(exit_reason)
        logger.info(f"Summary report generated: {self.config.report_file}")

        # Send notification
        batch_result = self._create_result(exit_reason)
        if self.config.notify_webhook:
            success = self.notifier.send_completion(batch_result, report)
            logger.info(f"Webhook notification: {'sent' if success else 'failed'}")

        return batch_result

    def _get_available_tasks(self) -> List[Dict]:
        """Get available tasks from task queue.

        Returns:
            List of task dicts from backlog
        """
        # Read task queue directly (read-only operation)
        task_queue_path = self.main_repo / "tasks" / "task_queue.json"

        if not task_queue_path.exists():
            logger.warning(f"Task queue file not found: {task_queue_path}")
            return []

        with open(task_queue_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        backlog = data.get("backlog", [])

        # Apply filters
        filtered = []
        for task in backlog:
            # Skip completed tasks (already in completed status)
            if task.get("status") == "COMPLETED":
                continue

            # Apply task ID filter
            if self.config.task_filter:
                import fnmatch
                if not fnmatch.fnmatch(task["id"], self.config.task_filter):
                    continue

            # Apply priority filter
            if self.config.priority_filter:
                if task.get("priority") != self.config.priority_filter:
                    continue

            filtered.append(task)

        return filtered

    def _execute_task(self, task: Dict) -> TaskResult:
        """Execute single task via SpawnOrchestrator subprocess.

        Args:
            task: Task dict from task_queue.json

        Returns:
            TaskResult with execution outcome
        """
        task_id = task["id"]
        start_time = datetime.utcnow()

        # Build command
        cmd = [
            sys.executable,
            "-m",
            "scripts.spawn_orchestrator",
            task_id
        ]

        # Create logs directory
        logs_dir = self.main_repo / ".autonomous" / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        log_file = logs_dir / f"{task_id}_{self._batch_id}.log"

        try:
            # Execute subprocess
            logger.info(f"Executing: {' '.join(cmd)}")
            logger.info(f"Log file: {log_file}")

            with open(log_file, 'w', encoding='utf-8') as log:
                # Run subprocess with output to log file
                self._current_process = subprocess.Popen(
                    cmd,
                    cwd=str(self.main_repo),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True
                )

                # Wait for completion
                exit_code = self._current_process.wait()
                self._current_process = None

            # Calculate duration
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()

            # FIX-004: Check PR merge status to determine success
            pr_number = self._extract_pr_number(task_id)
            pr_merged, pr_number = self._check_pr_merged(task_id, pr_number)

            if exit_code == 0:
                # Clean success - PR merged and cleanup succeeded
                return TaskResult(
                    task_id=task_id,
                    status="completed",
                    started_at=start_time.isoformat(),
                    finished_at=end_time.isoformat(),
                    duration_seconds=duration,
                    pr_number=pr_number,
                    pr_merged=pr_merged,
                    cleanup_success=True
                )
            else:
                # Non-zero exit code - check if PR was merged despite failure
                if pr_merged:
                    # FIX-004: PR merged successfully, only cleanup failed
                    logger.info(f"Task {task_id}: PR #{pr_number} merged, cleanup may have failed")
                    return TaskResult(
                        task_id=task_id,
                        status="completed",  # COMPLETED because PR merged
                        started_at=start_time.isoformat(),
                        finished_at=end_time.isoformat(),
                        duration_seconds=duration,
                        pr_number=pr_number,
                        pr_merged=True,
                        cleanup_success=False,  # Cleanup failed
                        error_message=f"Cleanup failed after successful PR merge (exit code {exit_code})"
                    )
                else:
                    # True failure - no PR merged
                    error_msg = f"Subprocess exited with code {exit_code}"

                    # Try to extract error from log
                    try:
                        with open(log_file, 'r', encoding='utf-8') as f:
                            log_content = f.read()
                            # Look for error patterns
                            if "ERROR" in log_content:
                                lines = log_content.split('\n')
                                error_lines = [l for l in lines if "ERROR" in l]
                                if error_lines:
                                    error_msg = error_lines[-1]  # Last error
                    except Exception:
                        pass

                    return TaskResult(
                        task_id=task_id,
                        status="failed",
                        started_at=start_time.isoformat(),
                        finished_at=end_time.isoformat(),
                        duration_seconds=duration,
                        pr_number=pr_number,
                        pr_merged=False,
                        cleanup_success=False,
                        error_message=error_msg
                    )

        except Exception as e:
            # Unexpected error
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()

            logger.error(f"Task execution error: {e}", exc_info=True)

            return TaskResult(
                task_id=task_id,
                status="failed",
                started_at=start_time.isoformat(),
                finished_at=end_time.isoformat(),
                duration_seconds=duration,
                error_message=str(e)
            )

    def _extract_pr_number(self, task_id: str) -> Optional[int]:
        """Extract PR number from workflow state if available.

        Args:
            task_id: Task ID

        Returns:
            PR number or None
        """
        try:
            # Look for workflow state file
            state_file = self.main_repo / ".autonomous" / "workflow_states" / f"{task_id}_state.json"
            if state_file.exists():
                with open(state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                    return state.get("pr_number")
        except Exception:
            pass

        return None

    def _check_pr_merged(self, task_id: str, pr_number: Optional[int] = None) -> tuple:
        """Check if PR for task was merged (FIX-004).

        Uses multiple strategies to determine PR merge status:
        1. Read from workflow state file (fastest, most reliable)
        2. Query GitHub CLI for PR status (fallback)

        Args:
            task_id: Task ID to check
            pr_number: Optional PR number (if known)

        Returns:
            Tuple of (pr_merged: bool, pr_number: Optional[int])
        """
        # Strategy 1: Check workflow state file
        state_file = self.main_repo / ".autonomous" / "workflow_states" / f"{task_id}_state.json"
        if state_file.exists():
            try:
                with open(state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)

                # Check for explicit merge status (FIX-004)
                if state.get("pr_merged"):
                    return True, state.get("pr_number")

                # Check if task marked completed (FIX-002 behavior)
                if state.get("status") == "COMPLETED":
                    return True, state.get("pr_number")

                # Get PR number if not provided
                if pr_number is None:
                    pr_number = state.get("pr_number")
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to read workflow state: {e}")

        # Strategy 2: Query GitHub CLI
        if pr_number:
            try:
                result = subprocess.run(
                    ["gh", "pr", "view", str(pr_number), "--json", "state,merged"],
                    cwd=str(self.main_repo),
                    capture_output=True,
                    text=True,
                    timeout=30
                )

                if result.returncode == 0 and result.stdout.strip():
                    pr_data = json.loads(result.stdout.strip())
                    merged = pr_data.get("merged", False)
                    state = pr_data.get("state", "")

                    if merged or state == "MERGED":
                        return True, pr_number
            except subprocess.TimeoutExpired:
                logger.warning("Timeout querying PR status via gh CLI")
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse PR status JSON: {e}")
            except Exception as e:
                logger.warning(f"Failed to query PR status via gh CLI: {e}")

        return False, pr_number

    def _should_continue(self) -> bool:
        """Check if we should process more tasks.

        Returns:
            True if should continue, False if limits reached
        """
        # Check task count limit
        if self.config.max_tasks is not None:
            if len(self._task_results) >= self.config.max_tasks:
                logger.info(f"Task limit reached: {len(self._task_results)}/{self.config.max_tasks}")
                return False

        # Check time limit
        if self.config.max_hours is not None and self._start_time:
            elapsed = datetime.utcnow() - self._start_time
            elapsed_hours = elapsed.total_seconds() / 3600

            if elapsed_hours >= self.config.max_hours:
                logger.info(f"Time limit reached: {elapsed_hours:.2f}/{self.config.max_hours} hours")
                return False

        return True

    def _handle_signal(self, signum, frame):
        """Handle shutdown signal gracefully.

        Args:
            signum: Signal number
            frame: Stack frame
        """
        signal_name = signal.Signals(signum).name if hasattr(signal, 'Signals') else str(signum)
        logger.warning(f"Received {signal_name} - completing current task and exiting")
        self._shutdown_requested = True

        # Do NOT kill current subprocess - let it complete
        # The main loop will check _shutdown_requested before starting next task

    def _checkpoint(self) -> None:
        """Save progress checkpoint after each task."""
        checkpoint = BatchCheckpoint(
            batch_id=self._batch_id,
            started_at=self._start_time.isoformat(),
            last_updated=datetime.utcnow().isoformat(),
            tasks_completed=list(self._completed),
            tasks_failed=list(self._failed),
            tasks_skipped=list(self._skipped),
            current_task=None,
            config={
                "max_tasks": self.config.max_tasks,
                "max_hours": self.config.max_hours,
                "notify_webhook": self.config.notify_webhook is not None
            }
        )

        # Ensure directory exists
        self.config.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)

        # Atomic write using coordination layer
        def _write_checkpoint(data: Dict) -> Dict:
            return checkpoint.to_dict()

        self.lock_manager.atomic_file_operation(
            self.config.checkpoint_file,
            _write_checkpoint,
            lock_name="batch_checkpoint"
        )

        logger.debug(f"Checkpoint saved: {len(self._completed)} completed, {len(self._failed)} failed, {len(self._skipped)} skipped")

    def _load_checkpoint(self) -> Optional[BatchCheckpoint]:
        """Load checkpoint for resume capability.

        Returns:
            BatchCheckpoint or None if not found
        """
        if not self.config.checkpoint_file.exists():
            return None

        try:
            with open(self.config.checkpoint_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            checkpoint = BatchCheckpoint.from_dict(data)
            logger.info(f"Loaded checkpoint: {checkpoint.batch_id}")

            # Check for interrupted task
            if checkpoint.current_task:
                logger.warning(f"Task {checkpoint.current_task} was interrupted - will be skipped")
                checkpoint.tasks_skipped.append(checkpoint.current_task)
                checkpoint.current_task = None

            return checkpoint

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}", exc_info=True)
            return None

    def _determine_exit_reason(self, tasks: List[Dict]) -> str:
        """Determine why batch execution ended.

        Args:
            tasks: List of available tasks

        Returns:
            Exit reason string
        """
        if self._shutdown_requested:
            return "signal"

        if self.config.dry_run:
            return "dry_run"

        if self.config.max_tasks and len(self._task_results) >= self.config.max_tasks:
            return "max_tasks"

        if self.config.max_hours:
            elapsed = datetime.utcnow() - self._start_time
            elapsed_hours = elapsed.total_seconds() / 3600
            if elapsed_hours >= self.config.max_hours:
                return "max_hours"

        # Check if there are unprocessed tasks remaining
        processed = self._completed | self._failed | self._skipped
        remaining = [t for t in tasks if t["id"] not in processed]

        if remaining:
            return "interrupted"

        if not tasks:
            return "no_tasks"

        return "completed"

    def _generate_report(self, exit_reason: str) -> str:
        """Generate markdown summary report.

        Args:
            exit_reason: Why batch execution ended

        Returns:
            Markdown report content
        """
        end_time = datetime.utcnow()
        duration = end_time - self._start_time
        duration_hours = duration.total_seconds() / 3600

        # Determine status
        if exit_reason == "signal":
            status = "INTERRUPTED"
        elif exit_reason in ["completed", "max_tasks", "max_hours"]:
            status = "COMPLETED"
        elif exit_reason == "no_tasks":
            status = "NO_TASKS"
        elif exit_reason == "dry_run":
            status = "DRY_RUN"
        else:
            status = "FAILED"

        # Build report
        report = f"""# Batch Execution Report

**Batch ID**: {self._batch_id}
**Status**: {status}
**Exit Reason**: {exit_reason}

## Summary

| Metric | Value |
|--------|-------|
| Started | {self._start_time.strftime('%Y-%m-%d %H:%M:%S')} |
| Finished | {end_time.strftime('%Y-%m-%d %H:%M:%S')} |
| Duration | {int(duration_hours)}h {int((duration_hours % 1) * 60)}m |
| Tasks Completed | {len(self._completed)} |
| Tasks Failed | {len(self._failed)} |
| Tasks Skipped | {len(self._skipped)} |

## Task Results

"""

        # Completed tasks (clean success)
        clean_completed = [r for r in self._task_results if r.status == "completed" and r.cleanup_success]
        if clean_completed:
            report += f"### Completed ({len(clean_completed)})\n\n"
            report += "| Task | Duration | PR |\n"
            report += "|------|----------|-----|\n"
            for r in clean_completed:
                duration_min = int(r.duration_seconds / 60)
                pr = f"#{r.pr_number}" if r.pr_number else "N/A"
                report += f"| {r.task_id} | {duration_min}m | {pr} |\n"
            report += "\n"

        # FIX-004: Tasks with cleanup failures (PR merged but cleanup failed)
        cleanup_failed = [r for r in self._task_results if r.status == "completed" and not r.cleanup_success]
        if cleanup_failed:
            report += f"### Completed with Cleanup Issues ({len(cleanup_failed)})\n\n"
            report += "| Task | Duration | PR | Issue |\n"
            report += "|------|----------|-----|-------|\n"
            for r in cleanup_failed:
                duration_min = int(r.duration_seconds / 60)
                pr = f"#{r.pr_number}" if r.pr_number else "N/A"
                issue = "Cleanup failed" if r.error_message else "Unknown"
                report += f"| {r.task_id} | {duration_min}m | {pr} | {issue} |\n"
            report += "\n"
            report += "> **Note**: These tasks completed successfully (PR merged) but worktree cleanup failed.\n"
            report += "> Manual cleanup may be required.\n\n"

        # Failed tasks
        failed_results = [r for r in self._task_results if r.status == "failed"]
        if failed_results:
            report += f"### Failed ({len(failed_results)})\n\n"
            report += "| Task | Duration | Error |\n"
            report += "|------|----------|-------|\n"
            for r in failed_results:
                duration_min = int(r.duration_seconds / 60)
                error = r.error_message[:50] if r.error_message else "Unknown"
                report += f"| {r.task_id} | {duration_min}m | {error} |\n"
            report += "\n"

        # Skipped tasks
        skipped_results = [r for r in self._task_results if r.status == "skipped"]
        if skipped_results:
            report += f"### Skipped ({len(skipped_results)})\n\n"
            report += "| Task | Reason |\n"
            report += "|------|--------|\n"
            for r in skipped_results:
                reason = r.error_message[:50] if r.error_message else "Unknown"
                report += f"| {r.task_id} | {reason} |\n"
            report += "\n"

        # Gate failures (if any)
        gate_failures = []
        for r in self._task_results:
            if r.gate_failures:
                gate_failures.extend([(r.task_id, gf) for gf in r.gate_failures])

        if gate_failures:
            report += f"### Gate Failures (Skipped via RESILIENCE-001)\n\n"
            report += "| Task | Gate | Issue |\n"
            report += "|------|------|-------|\n"
            for task_id, gf in gate_failures:
                gate = gf.get("gate", "Unknown")
                issue = gf.get("issue", "Unknown")
                report += f"| {task_id} | {gate} | {issue} |\n"
            report += "\n"

        # Next steps
        report += "## Next Steps\n\n"
        if failed_results:
            report += "- Review failed tasks and fix issues\n"
        # FIX-004: Include all completed tasks (both clean and cleanup-failed) for PR merge list
        all_completed = [r for r in self._task_results if r.status == "completed"]
        if all_completed:
            # Only show PRs that weren't merged (for auto_merge_disabled cases)
            unmerged_prs = [f"#{r.pr_number}" for r in all_completed if r.pr_number and not r.pr_merged]
            if unmerged_prs:
                report += f"- Merge PRs: {', '.join(unmerged_prs)}\n"
        if cleanup_failed:
            report += "- Clean up orphaned worktrees for tasks with cleanup failures\n"
        if exit_reason == "signal":
            report += "- Resume batch execution to complete remaining tasks\n"

        # Save report
        self.config.report_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config.report_file, 'w', encoding='utf-8') as f:
            f.write(report)

        return report

    def _create_result(self, exit_reason: str) -> BatchResult:
        """Create BatchResult from current state.

        Args:
            exit_reason: Why batch execution ended

        Returns:
            BatchResult with execution summary
        """
        end_time = datetime.utcnow()
        duration = end_time - self._start_time if self._start_time else timedelta(0)
        duration_hours = duration.total_seconds() / 3600

        # Determine status
        if exit_reason == "signal":
            status = "interrupted"
        elif exit_reason in ["completed", "max_tasks", "max_hours", "no_tasks"]:
            status = "completed"
        else:
            status = "failed"

        # Count remaining tasks (tasks in backlog not yet processed)
        all_tasks = self._get_available_tasks()
        processed = self._completed | self._failed | self._skipped
        remaining = len([t for t in all_tasks if t["id"] not in processed])

        return BatchResult(
            batch_id=self._batch_id,
            status=status,
            started_at=self._start_time.isoformat() if self._start_time else "",
            finished_at=end_time.isoformat(),
            duration_hours=duration_hours,
            tasks_completed=len(self._completed),
            tasks_failed=len(self._failed),
            tasks_skipped=len(self._skipped),
            tasks_remaining=remaining,
            task_results=self._task_results,
            exit_reason=exit_reason
        )

    def _preview_tasks(self, tasks: List[Dict]) -> None:
        """Preview tasks in dry run mode.

        Args:
            tasks: List of tasks to preview
        """
        logger.info(f"\n{'=' * 80}")
        logger.info(f"BATCH PREVIEW - {len(tasks)} tasks")
        logger.info(f"{'=' * 80}\n")

        for i, task in enumerate(tasks, 1):
            # Skip already processed
            if task["id"] in self._completed | self._failed | self._skipped:
                logger.info(f"{i}. {task['id']}: {task.get('title', 'No title')} [ALREADY PROCESSED]")
                continue

            logger.info(f"{i}. {task['id']}: {task.get('title', 'No title')}")
            logger.info(f"   Priority: {task.get('priority', 'N/A')}")
            logger.info(f"   Estimated: {task.get('estimated_hours', 'N/A')}h")

            if self.config.max_tasks and i >= self.config.max_tasks:
                logger.info(f"\n   ... (max_tasks={self.config.max_tasks} limit)")
                break

        logger.info(f"\n{'=' * 80}\n")


# ==============================================================================
# SECTION 4: CLI Interface
# ==============================================================================

def main():
    """Main entry point for batch runner."""
    parser = argparse.ArgumentParser(
        description="Batch Runner for Autonomous Task Execution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run 3 tasks max
  python -m scripts.batch_runner --max-tasks=3

  # Run for 8 hours max
  python -m scripts.batch_runner --max-hours=8

  # Long-running batch with notifications
  python -m scripts.batch_runner --max-hours=10 --notify-webhook=https://hooks.slack.com/xxx

  # Dry run to preview
  python -m scripts.batch_runner --max-tasks=5 --dry-run
        """
    )

    parser.add_argument(
        '--max-tasks',
        type=int,
        default=None,
        help='Maximum number of tasks to execute (default: unlimited)'
    )

    parser.add_argument(
        '--max-hours',
        type=float,
        default=None,
        help='Maximum hours to run (default: unlimited)'
    )

    parser.add_argument(
        '--notify-webhook',
        type=str,
        default=None,
        help='Webhook URL for completion/failure notifications'
    )

    parser.add_argument(
        '--resume',
        action='store_true',
        default=True,
        help='Resume from checkpoint (default: True)'
    )

    parser.add_argument(
        '--no-resume',
        action='store_false',
        dest='resume',
        help='Do not resume from checkpoint'
    )

    parser.add_argument(
        '--filter',
        type=str,
        default=None,
        dest='task_filter',
        help='Filter tasks by ID pattern (e.g., "SDLC-*")'
    )

    parser.add_argument(
        '--priority',
        type=str,
        default=None,
        dest='priority_filter',
        help='Filter by priority (CRITICAL, HIGH, MEDIUM, LOW)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        default=False,
        help='Preview tasks without executing'
    )

    parser.add_argument(
        '--checkpoint',
        type=Path,
        default=Path(".autonomous/batch_checkpoint.json"),
        dest='checkpoint_file',
        help='Custom checkpoint file path'
    )

    parser.add_argument(
        '--report',
        type=Path,
        default=Path(".autonomous/batch_report.md"),
        dest='report_file',
        help='Custom report file path'
    )

    args = parser.parse_args()

    # Create config
    config = BatchConfig(
        max_tasks=args.max_tasks,
        max_hours=args.max_hours,
        notify_webhook=args.notify_webhook,
        checkpoint_file=args.checkpoint_file,
        report_file=args.report_file,
        resume=args.resume,
        task_filter=args.task_filter,
        priority_filter=args.priority_filter,
        dry_run=args.dry_run
    )

    # Determine main repo (current directory or parent of scripts/)
    main_repo = Path.cwd()
    if main_repo.name == "scripts":
        main_repo = main_repo.parent

    # Create and run batch runner
    runner = BatchRunner(config, main_repo)

    try:
        result = runner.run()

        # Print summary
        logger.info(f"\n{'=' * 80}")
        logger.info(f"BATCH EXECUTION COMPLETE")
        logger.info(f"{'=' * 80}")
        logger.info(f"Status: {result.status}")
        logger.info(f"Exit Reason: {result.exit_reason}")
        logger.info(f"Tasks Completed: {result.tasks_completed}")
        logger.info(f"Tasks Failed: {result.tasks_failed}")
        logger.info(f"Tasks Skipped: {result.tasks_skipped}")
        logger.info(f"Duration: {result.duration_hours:.2f} hours")
        logger.info(f"Report: {config.report_file}")
        logger.info(f"{'=' * 80}\n")

        # Exit code: 0 if all tasks completed, 1 if any failed
        sys.exit(0 if result.tasks_failed == 0 else 1)

    except Exception as e:
        logger.error(f"Batch execution error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
