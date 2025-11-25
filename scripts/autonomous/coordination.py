#!/usr/bin/env python3
"""
Instance Coordination Components for Autonomous Workflow

Provides distributed coordination primitives for managing concurrent Claude Code instances:
- DistributedLockManager: Cross-platform file locking
- InstanceRegistry: Instance lifecycle and heartbeat monitoring
- TaskCoordinator: Atomic task queue operations
- MessageQueue: Inter-instance communication

All components use filelock for cross-platform file locking with automatic retry
and exponential backoff for robustness.
"""

import json
import os
import socket
import sys
import threading
import time
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Third-party imports
try:
    from filelock import FileLock, Timeout
except ImportError:
    print("Error: filelock library required. Install with: pip install filelock")
    sys.exit(1)


# ==============================================================================
# SECTION 1: Distributed Lock Manager
# ==============================================================================

class DistributedLockManager:
    """
    Thread-safe, process-safe distributed locking using filelock.

    This class provides atomic operations on shared files to prevent
    race conditions when multiple instances access the same resources.

    Features:
    - Cross-platform file locking (Windows, Linux, macOS)
    - Automatic timeout detection and retry with exponential backoff
    - Network drive detection for adjusted timeouts
    - Atomic read-modify-write operations on JSON files
    """

    def __init__(self, main_repo: Path):
        """
        Initialize lock manager with main repository path.

        Args:
            main_repo: Path to main repository root
        """
        self.main_repo = Path(main_repo)
        self.locks_dir = self.main_repo / ".autonomous" / "locks"
        self.locks_dir.mkdir(parents=True, exist_ok=True)

        # Detect if we're on a network drive for timeout adjustment
        self.is_network_drive = self._detect_network_drive()
        self.default_timeout = 30 if self.is_network_drive else 5

    def _detect_network_drive(self) -> bool:
        """
        Detect if repository is on a network drive.

        Returns:
            True if on network drive, False otherwise
        """
        try:
            if sys.platform == "win32":
                # Windows: Check drive type
                import string
                drive = str(self.main_repo.drive)
                if drive and drive[0] in string.ascii_letters:
                    # Simple heuristic: network drives often start with high letters
                    # More sophisticated: use win32api if available
                    return drive[0].upper() > 'D'
            else:
                # Unix: Check mount point
                import subprocess
                result = subprocess.run(
                    ["df", "-T", str(self.main_repo)],
                    capture_output=True, text=True, timeout=5
                )
                # Check for network filesystems
                return any(fs in result.stdout.lower()
                          for fs in ["nfs", "cifs", "smb", "sshfs"])
        except Exception:
            # If detection fails, assume local drive
            return False

        return False

    def acquire_lock(self, lock_name: str, timeout: Optional[float] = None) -> FileLock:
        """
        Acquire a named lock with timeout.

        Args:
            lock_name: Name of the lock file (e.g., "task_queue", "instances")
            timeout: Lock acquisition timeout in seconds (defaults to 5s local, 30s network)

        Returns:
            FileLock object (use as context manager)

        Raises:
            RuntimeError: If lock cannot be acquired within timeout

        Example:
            with lock_manager.acquire_lock("task_queue"):
                # Critical section - task_queue.json is locked
                pass
        """
        if timeout is None:
            timeout = self.default_timeout

        lock_path = self.locks_dir / f"{lock_name}.lock"
        lock = FileLock(str(lock_path), timeout=timeout)

        try:
            lock.acquire()
            return lock
        except Timeout:
            raise RuntimeError(
                f"Could not acquire {lock_name} lock within {timeout}s. "
                f"Another instance may be holding it."
            )

    def atomic_file_operation(self,
                              file_path: Path,
                              operation: Callable[[Dict], Dict],
                              lock_name: Optional[str] = None,
                              max_retries: int = 3,
                              create_if_missing: bool = True) -> Any:
        """
        Perform atomic read-modify-write operation on JSON file.

        This method ensures atomicity by:
        1. Acquiring distributed lock
        2. Reading current state
        3. Applying operation function
        4. Writing result atomically (temp file + rename)
        5. Releasing lock

        Args:
            file_path: Path to JSON file
            operation: Function that takes current data and returns modified data
            lock_name: Name for the lock (defaults to filename)
            max_retries: Number of retry attempts on lock timeout
            create_if_missing: Create file with empty dict if it doesn't exist

        Returns:
            Result of the operation function

        Raises:
            RuntimeError: If lock cannot be acquired after max_retries
            FileNotFoundError: If file doesn't exist and create_if_missing=False

        Example:
            def increment_counter(data):
                data['counter'] = data.get('counter', 0) + 1
                return data

            lock_manager.atomic_file_operation(
                Path("counter.json"),
                increment_counter
            )
        """
        if lock_name is None:
            lock_name = file_path.stem

        for attempt in range(max_retries):
            try:
                with self.acquire_lock(lock_name, timeout=self.default_timeout):
                    # Read current state
                    if file_path.exists():
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                    elif create_if_missing:
                        data = {}
                    else:
                        raise FileNotFoundError(f"File {file_path} does not exist")

                    # Apply operation (modifies data, may return modified copy)
                    result = operation(data)

                    # CRITICAL FIX: Determine what to save
                    # If operation returns None, it modified data in place, save data
                    # Otherwise, save the returned result
                    data_to_save = data if result is None else result
                    if result is None:
                        result = data  # Return the modified data

                    # Write atomically using temp file + rename
                    temp_path = file_path.with_suffix('.tmp')
                    with open(temp_path, 'w', encoding='utf-8') as f:
                        json.dump(data_to_save, f, indent=2, default=str)

                    # Atomic rename (works on Windows and Unix)
                    temp_path.replace(file_path)

                    return result

            except RuntimeError as e:
                if attempt < max_retries - 1:
                    # Exponential backoff: 2^attempt seconds
                    wait_time = 2 ** attempt
                    print(f"Lock timeout, retrying in {wait_time}s... ({attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    raise


# ==============================================================================
# SECTION 2: Instance Registry with Heartbeat
# ==============================================================================

class InstanceStatus(Enum):
    """Instance lifecycle states."""
    STARTING = "STARTING"
    EXECUTING = "EXECUTING"
    VALIDATING = "VALIDATING"
    COMPLETING = "COMPLETING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    STALE = "STALE"


class InstanceRegistry:
    """
    Manages instance registration, heartbeats, and lifecycle.

    Tracks all active instances, detects stale instances, and
    manages resource limits (max concurrent instances).

    Features:
    - Automatic heartbeat thread (10s interval)
    - Stale instance detection (60s threshold)
    - Resource limits with priority queueing
    - Automatic cleanup of crashed instances
    """

    HEARTBEAT_INTERVAL = 10  # seconds
    STALE_THRESHOLD = 60     # seconds - mark as stale if no heartbeat
    REMOVAL_THRESHOLD = 300  # seconds - remove stale instances after 5 minutes

    def __init__(self, lock_manager: DistributedLockManager):
        """
        Initialize registry with lock manager.

        Args:
            lock_manager: DistributedLockManager instance
        """
        self.lock_manager = lock_manager
        self.registry_path = lock_manager.main_repo / ".autonomous" / "instances.json"
        self.instance_id: Optional[str] = None
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._stop_heartbeat = threading.Event()

    def register_instance(self,
                          session_id: str,
                          task_id: str,
                          worktree_path: Path) -> str:
        """
        Register new instance and start heartbeat.

        Args:
            session_id: Unique session identifier (YYYYMMDD-HHMMSS)
            task_id: Task being executed (e.g., "TASK-058")
            worktree_path: Path to instance worktree

        Returns:
            Instance ID (format: "instance-{session_id}")

        Raises:
            RuntimeError: If registration fails or resource limit reached
        """
        instance_id = f"instance-{session_id}"
        limit_reached = False

        def _register(data: Dict) -> None:  # Return None to save in-place modifications
            nonlocal limit_reached

            instances = data.setdefault("instances", {})
            limits = data.setdefault("resource_limits", {
                "max_instances": 3,
                "current_count": 0,
                "queue": []
            })

            # Count active instances
            active_count = len([
                i for i in instances.values()
                if i["status"] not in [
                    InstanceStatus.COMPLETED.value,
                    InstanceStatus.FAILED.value,
                    InstanceStatus.STALE.value
                ]
            ])

            # CRITICAL FIX: Check resource limits and save queue update BEFORE raising exception
            if active_count >= limits["max_instances"]:
                # Add to queue (this will be saved)
                limits["queue"].append({
                    "task_id": task_id,
                    "requested_at": datetime.utcnow().isoformat()
                })
                limit_reached = True  # Set flag to raise exception after save
                return None  # Return None to save modifications

            # Register instance
            instances[instance_id] = {
                "instance_id": instance_id,
                "session_id": session_id,
                "task_id": task_id,
                "status": InstanceStatus.STARTING.value,
                "start_time": datetime.utcnow().isoformat(),
                "last_heartbeat": datetime.utcnow().isoformat(),
                "worktree_path": str(worktree_path),
                "current_phase": "STARTUP",
                "gates_passed": [],
                "pid": os.getpid(),
                "hostname": socket.gethostname()
            }

            # Update count
            limits["current_count"] = active_count + 1

            return None  # Return None to save in-place modifications

        # Register atomically
        self.lock_manager.atomic_file_operation(
            self.registry_path,
            _register,
            lock_name="instances"
        )

        # CRITICAL FIX: Raise exception AFTER queue has been saved
        if limit_reached:
            raise RuntimeError(
                f"Instance limit reached (3). "
                f"Task {task_id} added to queue."
            )

        self.instance_id = instance_id

        # Start heartbeat thread
        self._start_heartbeat()

        print(f"Instance registered: {instance_id}")
        return instance_id

    def _start_heartbeat(self):
        """Start background thread for heartbeats."""
        # CRITICAL FIX: Capture instance_id in closure to prevent race condition
        # during shutdown when instance_id might be set to None
        instance_id_copy = self.instance_id

        def heartbeat_worker():
            while not self._stop_heartbeat.is_set():
                try:
                    # Use captured instance_id to avoid race condition
                    if instance_id_copy:
                        self._send_heartbeat_for_instance(instance_id_copy)
                    self._stop_heartbeat.wait(self.HEARTBEAT_INTERVAL)
                except Exception as e:
                    print(f"Heartbeat error: {e}")

        self._heartbeat_thread = threading.Thread(
            target=heartbeat_worker,
            daemon=True,
            name="heartbeat"
        )
        self._heartbeat_thread.start()

    def send_heartbeat(self):
        """Update instance heartbeat timestamp (public method)."""
        if not self.instance_id:
            return
        self._send_heartbeat_for_instance(self.instance_id)

    def _send_heartbeat_for_instance(self, instance_id: str):
        """Update heartbeat for specific instance (thread-safe internal method)."""
        def _update_heartbeat(data: Dict) -> Dict:
            instances = data.get("instances", {})
            if instance_id in instances:
                instances[instance_id]["last_heartbeat"] = \
                    datetime.utcnow().isoformat()
            return data

        self.lock_manager.atomic_file_operation(
            self.registry_path,
            _update_heartbeat,
            lock_name="instances"
        )

    def update_status(self, status: InstanceStatus, phase: Optional[str] = None):
        """
        Update instance status and optionally phase.

        Args:
            status: New instance status
            phase: Optional phase name (e.g., "EXECUTING", "VALIDATION")
        """
        if not self.instance_id:
            return

        def _update(data: Dict) -> Dict:
            instances = data.get("instances", {})
            if self.instance_id in instances:
                instances[self.instance_id]["status"] = status.value
                if phase:
                    instances[self.instance_id]["current_phase"] = phase
            return data

        self.lock_manager.atomic_file_operation(
            self.registry_path,
            _update,
            lock_name="instances"
        )

    def cleanup_stale_instances(self) -> List[str]:
        """
        Mark stale instances and clean up old ones.

        Instances are marked stale if no heartbeat received within STALE_THRESHOLD (60s).
        Stale instances are removed after REMOVAL_THRESHOLD (300s).

        Returns:
            List of instance IDs that were marked stale
        """
        stale_instances = []

        def _cleanup(data: Dict) -> Dict:
            nonlocal stale_instances
            instances = data.get("instances", {})
            now = datetime.utcnow()

            instances_to_remove = []

            for instance_id, instance in instances.items():
                last_heartbeat = datetime.fromisoformat(instance["last_heartbeat"])
                time_since_heartbeat = now - last_heartbeat

                # Mark as stale if no heartbeat
                if time_since_heartbeat.total_seconds() > self.STALE_THRESHOLD:
                    if instance["status"] not in [
                        InstanceStatus.STALE.value,
                        InstanceStatus.COMPLETED.value,
                        InstanceStatus.FAILED.value
                    ]:
                        instance["status"] = InstanceStatus.STALE.value
                        stale_instances.append(instance_id)
                        print(f"WARNING: Marked instance {instance_id} as STALE")

                # Remove very old stale instances
                if (instance["status"] == InstanceStatus.STALE.value and
                    time_since_heartbeat.total_seconds() > self.REMOVAL_THRESHOLD):
                    instances_to_remove.append(instance_id)

            # Remove old stale instances
            for instance_id in instances_to_remove:
                del instances[instance_id]
                print(f"Removed stale instance {instance_id}")

            # Update active count
            active_count = len([
                i for i in instances.values()
                if i["status"] not in [
                    InstanceStatus.COMPLETED.value,
                    InstanceStatus.FAILED.value,
                    InstanceStatus.STALE.value
                ]
            ])
            data.setdefault("resource_limits", {})["current_count"] = active_count

            return data

        self.lock_manager.atomic_file_operation(
            self.registry_path,
            _cleanup,
            lock_name="instances"
        )

        return stale_instances

    def shutdown(self):
        """Clean shutdown of instance."""
        # Stop heartbeat
        self._stop_heartbeat.set()
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=2)

        # Mark as completed
        self.update_status(InstanceStatus.COMPLETED)

        # Release resources
        def _release(data: Dict) -> Dict:
            limits = data.get("resource_limits", {})
            limits["current_count"] = max(0, limits.get("current_count", 1) - 1)

            # Process queue if slots available
            if (limits["current_count"] < limits.get("max_instances", 3) and
                limits.get("queue")):
                next_task = limits["queue"].pop(0)
                print(f"Queue processed: Task {next_task['task_id']} can now start")
                # TODO: Trigger spawn for queued task

            return data

        self.lock_manager.atomic_file_operation(
            self.registry_path,
            _release,
            lock_name="instances"
        )


# ==============================================================================
# SECTION 3: Task Queue Coordinator
# ==============================================================================

class TaskCoordinator:
    """
    Manages task assignment and completion without conflicts.

    Ensures atomic operations on task_queue.json to prevent
    double-assignment and maintain consistency across concurrent instances.

    Features:
    - Atomic task claiming (no double-assignment)
    - Atomic task completion (move from in_progress to completed)
    - Task release on failure (return to backlog)
    - Priority-based task selection
    """

    def __init__(self, lock_manager: DistributedLockManager):
        """
        Initialize coordinator with lock manager.

        Args:
            lock_manager: DistributedLockManager instance
        """
        self.lock_manager = lock_manager
        self.task_queue_path = lock_manager.main_repo / "tasks" / "task_queue.json"

    def claim_task(self,
                   instance_id: str,
                   task_id: Optional[str] = None,
                   priority_filter: Optional[str] = None) -> Optional[Dict]:
        """
        Atomically claim a task from the queue.

        Args:
            instance_id: ID of claiming instance
            task_id: Specific task to claim (optional)
            priority_filter: Only claim tasks with this priority (optional, e.g., "HIGH")

        Returns:
            Claimed task dict or None if no suitable task available

        Example:
            # Claim any HIGH priority task
            task = coordinator.claim_task("instance-12345", priority_filter="HIGH")

            # Claim specific task
            task = coordinator.claim_task("instance-12345", task_id="TASK-058")
        """
        claimed_task = None

        def _claim(data: Dict) -> Dict:
            nonlocal claimed_task

            backlog = data.get("backlog", [])
            in_progress = data.setdefault("in_progress", [])

            # CRITICAL FIX: Validate that requested task exists if specific task_id provided
            if task_id:
                task_exists = any(task["id"] == task_id for task in backlog)
                if not task_exists:
                    # Check if task is already in progress or completed
                    in_progress_ids = [task["id"] for task in in_progress]
                    completed_ids = [task["id"] for task in data.get("completed", [])]

                    if task_id in in_progress_ids:
                        print(f"WARNING: Task {task_id} already in progress (cannot claim)")
                    elif task_id in completed_ids:
                        print(f"WARNING: Task {task_id} already completed (cannot claim)")
                    else:
                        print(f"WARNING: Task {task_id} does not exist in task queue")
                    return data  # Return without claiming

            for task in backlog:
                # Check if specific task requested
                if task_id and task["id"] != task_id:
                    continue

                # Check priority filter
                if priority_filter and task.get("priority") != priority_filter:
                    continue

                # Check if task is available
                if task.get("status") == "READY" and not task.get("assigned_to"):
                    # Claim the task
                    task["status"] = "IN_PROGRESS"
                    task["assigned_to"] = instance_id
                    task["assigned_at"] = datetime.utcnow().isoformat()

                    # Move to in_progress
                    backlog.remove(task)
                    in_progress.append(task)

                    claimed_task = task.copy()
                    print(f"Instance {instance_id} claimed task {task['id']}")
                    break

            data["backlog"] = backlog
            data["in_progress"] = in_progress

            return data

        self.lock_manager.atomic_file_operation(
            self.task_queue_path,
            _claim,
            lock_name="task_queue"
        )

        return claimed_task

    def complete_task(self,
                      instance_id: str,
                      task_id: str,
                      pr_number: Optional[int] = None,
                      commit_hash: Optional[str] = None,
                      notes: Optional[str] = None) -> bool:
        """
        Mark task as completed and move to completed array.

        Args:
            instance_id: ID of completing instance
            task_id: Task to complete
            pr_number: Associated PR number (optional)
            commit_hash: Associated commit hash (optional)
            notes: Completion notes (optional)

        Returns:
            True if task was completed successfully, False otherwise
        """
        completed = False

        def _complete(data: Dict) -> Dict:
            nonlocal completed

            in_progress = data.get("in_progress", [])
            completed_list = data.setdefault("completed", [])

            for task in in_progress:
                if (task["id"] == task_id and
                    task.get("assigned_to") == instance_id):

                    # Complete the task
                    task["status"] = "COMPLETED"
                    task["completed_at"] = datetime.utcnow().isoformat()
                    task["completed_by"] = instance_id
                    task["validation_status"] = "APPROVED"  # Can be updated later

                    if pr_number:
                        task["pr_number"] = pr_number
                    if commit_hash:
                        task["commit_hash"] = commit_hash
                    if notes:
                        task["notes"] = notes

                    # Calculate duration
                    if "assigned_at" in task:
                        start = datetime.fromisoformat(task["assigned_at"])
                        end = datetime.utcnow()
                        duration_hours = (end - start).total_seconds() / 3600
                        task["actual_hours"] = round(duration_hours, 2)

                    # Move to completed
                    in_progress.remove(task)
                    completed_list.append(task)

                    completed = True
                    print(f"Task {task_id} completed by {instance_id}")
                    break

            data["in_progress"] = in_progress
            data["completed"] = completed_list

            return data

        self.lock_manager.atomic_file_operation(
            self.task_queue_path,
            _complete,
            lock_name="task_queue"
        )

        return completed

    def release_task(self,
                     instance_id: str,
                     task_id: str,
                     reason: str = "Instance failure") -> bool:
        """
        Release task back to backlog (on failure).

        Args:
            instance_id: ID of releasing instance
            task_id: Task to release
            reason: Why task is being released

        Returns:
            True if task was released successfully, False otherwise
        """
        released = False

        def _release(data: Dict) -> Dict:
            nonlocal released

            in_progress = data.get("in_progress", [])
            backlog = data.get("backlog", [])

            for task in in_progress:
                if (task["id"] == task_id and
                    task.get("assigned_to") == instance_id):

                    # Release the task
                    task["status"] = "READY"
                    task["assigned_to"] = None
                    task["assigned_at"] = None

                    # Track release history
                    task.setdefault("release_history", []).append({
                        "released_by": instance_id,
                        "released_at": datetime.utcnow().isoformat(),
                        "reason": reason
                    })

                    # Move back to backlog (at front for priority)
                    in_progress.remove(task)
                    backlog.insert(0, task)

                    released = True
                    print(f"WARNING: Task {task_id} released: {reason}")
                    break

            data["in_progress"] = in_progress
            data["backlog"] = backlog

            return data

        self.lock_manager.atomic_file_operation(
            self.task_queue_path,
            _release,
            lock_name="task_queue"
        )

        return released

    def get_task_status(self, task_id: str) -> Optional[Dict]:
        """
        Get current status of a task (read-only operation).

        Args:
            task_id: Task ID to query

        Returns:
            Task dict or None if not found
        """
        # CRITICAL FIX: Use read-only lock, don't modify file
        # Previous version used atomic_file_operation which tried to save the returned task dict
        found_task = None

        with self.lock_manager.acquire_lock("task_queue", timeout=self.lock_manager.default_timeout):
            if not self.task_queue_path.exists():
                return None

            with open(self.task_queue_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Check all arrays
            for array_name in ["backlog", "in_progress", "completed", "blocked"]:
                for task in data.get(array_name, []):
                    if task["id"] == task_id:
                        found_task = task.copy()
                        break
                if found_task:
                    break

        return found_task


# ==============================================================================
# SECTION 4: Message Queue for Inter-Instance Communication
# ==============================================================================

class MessageType(Enum):
    """Standard message types for coordination."""
    # Task coordination
    TASK_CLAIMED = "TASK_CLAIMED"
    TASK_COMPLETED = "TASK_COMPLETED"
    TASK_RELEASED = "TASK_RELEASED"

    # Instance status
    INSTANCE_STARTED = "INSTANCE_STARTED"
    INSTANCE_FAILED = "INSTANCE_FAILED"
    PHASE_COMPLETE = "PHASE_COMPLETE"

    # Resource coordination
    RESOURCE_REQUEST = "RESOURCE_REQUEST"
    RESOURCE_GRANTED = "RESOURCE_GRANTED"

    # Help/Collaboration
    REQUEST_HELP = "REQUEST_HELP"
    STATUS_UPDATE = "STATUS_UPDATE"


class MessageQueue:
    """
    Simple file-based inter-instance messaging ("instance mail").

    Allows instances to communicate through a shared message queue
    with polling-based retrieval.

    Features:
    - Broadcast and direct messaging
    - Read tracking (mark messages as read)
    - Automatic message trimming (keep last 100)
    - Timestamp filtering
    """

    def __init__(self, lock_manager: DistributedLockManager):
        """
        Initialize message queue with lock manager.

        Args:
            lock_manager: DistributedLockManager instance
        """
        self.lock_manager = lock_manager
        self.messages_path = lock_manager.main_repo / ".autonomous" / "messages.json"
        self.max_messages = 100  # Keep last N messages

    def send_message(self,
                     from_instance: str,
                     message_type: MessageType,
                     payload: Dict,
                     to_instance: Optional[str] = None) -> str:
        """
        Send a message to instance(s).

        Args:
            from_instance: Sender instance ID
            message_type: Type of message
            payload: Message data (dict)
            to_instance: Target instance ID (None for broadcast)

        Returns:
            Message ID

        Example:
            # Broadcast status update
            msg_id = queue.send_message(
                "instance-12345",
                MessageType.STATUS_UPDATE,
                {"phase": "VALIDATION", "progress": 75},
                to_instance=None  # Broadcast
            )

            # Send direct message
            msg_id = queue.send_message(
                "instance-12345",
                MessageType.REQUEST_HELP,
                {"issue": "Lock timeout"},
                to_instance="instance-67890"
            )
        """
        message_id = f"msg-{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}"

        def _send(data: Dict) -> Dict:
            messages = data.setdefault("messages", [])

            messages.append({
                "id": message_id,
                "from": from_instance,
                "to": to_instance,  # None = broadcast
                "type": message_type.value,
                "payload": payload,
                "timestamp": datetime.utcnow().isoformat(),
                "read_by": []
            })

            # Trim old messages
            if len(messages) > self.max_messages:
                messages = messages[-self.max_messages:]
                data["messages"] = messages

            return data

        self.lock_manager.atomic_file_operation(
            self.messages_path,
            _send,
            lock_name="messages"
        )

        return message_id

    def poll_messages(self,
                      instance_id: str,
                      since: Optional[datetime] = None,
                      mark_read: bool = True) -> List[Dict]:
        """
        Poll for unread messages.

        Args:
            instance_id: Polling instance ID
            since: Only get messages after this timestamp (optional)
            mark_read: Mark messages as read (default: True)

        Returns:
            List of unread messages for this instance
        """
        unread = []

        def _poll(data: Dict) -> Dict:
            nonlocal unread
            messages = data.get("messages", [])

            for message in messages:
                # Check if message is for us (broadcast or direct)
                if message["to"] and message["to"] != instance_id:
                    continue

                # Check if already read
                if instance_id in message.get("read_by", []):
                    continue

                # Check timestamp filter
                if since:
                    msg_time = datetime.fromisoformat(message["timestamp"])
                    if msg_time <= since:
                        continue

                # Add to unread list
                unread.append(message.copy())

                # Mark as read if requested
                if mark_read:
                    message.setdefault("read_by", []).append(instance_id)

            return data

        self.lock_manager.atomic_file_operation(
            self.messages_path,
            _poll,
            lock_name="messages"
        )

        return unread

    def broadcast(self,
                  from_instance: str,
                  message_type: MessageType,
                  payload: Dict) -> str:
        """
        Broadcast message to all instances.

        Args:
            from_instance: Sender instance ID
            message_type: Type of message
            payload: Message data

        Returns:
            Message ID
        """
        return self.send_message(from_instance, message_type, payload, to_instance=None)


# ==============================================================================
# EXPORTS
# ==============================================================================

__all__ = [
    "DistributedLockManager",
    "InstanceRegistry",
    "InstanceStatus",
    "TaskCoordinator",
    "MessageQueue",
    "MessageType",
]
