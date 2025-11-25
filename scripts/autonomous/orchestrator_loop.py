#!/usr/bin/env python3
"""
Orchestrator Monitoring Loop - The Heart of Autonomous Development

Runs forever, monitoring instances and making spawning decisions.
Achieves "spawn once, walk away forever" autonomy.

Usage:
    python -m scripts.autonomous.orchestrator_loop

Architecture:
    - Single-threaded infinite loop (60s intervals)
    - Queries coordination layer (InstanceRegistry, TaskCoordinator, MessageQueue)
    - Makes intelligent spawning decisions via DecisionEngine
    - Detects and recovers from stuck/crashed instances
    - Maintains 24/7 operation with graceful shutdown
"""

import json
import logging
import os
import signal
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from autonomous.coordination import (
    DistributedLockManager,
    InstanceRegistry,
    InstanceStatus,
    TaskCoordinator,
    MessageQueue,
    MessageType
)
from autonomous.monitor import OrchestratorMonitor

logger = logging.getLogger(__name__)


# ============================================================================
# MONITORING-001: Enhanced Monitoring Integration
# ============================================================================


class EventType(Enum):
    """Types of monitoring events."""
    CYCLE_START = "cycle_start"
    CYCLE_END = "cycle_end"
    HEALTH_CHECK = "health_check"
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    GATE_PASSED = "gate_passed"
    GATE_FAILED = "gate_failed"
    INSTANCE_SPAWNED = "instance_spawned"
    INSTANCE_STUCK = "instance_stuck"
    INSTANCE_CRASHED = "instance_crashed"
    INTERVENTION_SENT = "intervention_sent"
    ORCHESTRATOR_STARTED = "orchestrator_started"
    ORCHESTRATOR_SHUTDOWN = "orchestrator_shutdown"


@dataclass
class MonitoringEvent:
    """Structured monitoring event for JSON logging."""
    event_type: EventType
    timestamp: str  # ISO format
    cycle: int
    payload: Dict[str, Any] = field(default_factory=dict)

    # Optional context
    task_id: Optional[str] = None
    instance_id: Optional[str] = None
    gate_name: Optional[str] = None

    def to_json(self) -> str:
        """Convert to JSON string for logging."""
        data = {
            "event": self.event_type.value,
            "timestamp": self.timestamp,
            "cycle": self.cycle,
            **self.payload
        }
        if self.task_id:
            data["task_id"] = self.task_id
        if self.instance_id:
            data["instance_id"] = self.instance_id
        if self.gate_name:
            data["gate_name"] = self.gate_name
        return json.dumps(data)


@dataclass
class MetricsSnapshot:
    """Point-in-time metrics snapshot."""
    timestamp: str
    uptime_seconds: int
    cycle_count: int

    # Task metrics
    tasks_completed: int = 0
    tasks_failed: int = 0
    tasks_in_progress: int = 0
    completion_rate: float = 0.0  # Percentage
    avg_task_duration_seconds: float = 0.0

    # Gate metrics
    gates_passed: int = 0
    gates_failed: int = 0
    gate_failure_rate: float = 0.0

    # Instance metrics
    instances_spawned: int = 0
    instances_crashed: int = 0
    interventions_sent: int = 0

    # Health
    health_status: str = "HEALTHY"
    active_instances: int = 0
    available_tasks: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MonitoringMetrics:
    """
    Aggregates and tracks monitoring metrics for the orchestrator.

    Calculates completion rates, failure counts, and duration statistics.
    Persists metrics to JSON file for crash recovery.
    """

    METRICS_FILE = ".autonomous/monitoring_metrics.json"

    def __init__(self):
        """Initialize metrics tracking."""
        self.start_time = datetime.utcnow()

        # Counters (cumulative since start)
        self.tasks_completed = 0
        self.tasks_failed = 0
        self.gates_passed = 0
        self.gates_failed = 0
        self.instances_spawned = 0
        self.instances_crashed = 0
        self.interventions_sent = 0

        # Duration tracking
        self.task_durations: List[float] = []  # Seconds
        self.max_duration_history = 100  # Keep last 100 for rolling avg

        # Gate failure tracking (for alerting)
        self.recent_gate_failures: List[Dict] = []
        self.gate_failure_window_minutes = 60  # Rolling 1-hour window

        # Load persisted metrics if available
        self._load_persisted_metrics()

    def record_task_completed(self, task_id: str, duration_seconds: float, success: bool):
        """Record task completion."""
        if success:
            self.tasks_completed += 1
        else:
            self.tasks_failed += 1

        self.task_durations.append(duration_seconds)
        if len(self.task_durations) > self.max_duration_history:
            self.task_durations = self.task_durations[-self.max_duration_history:]

        self._persist_metrics()

    def record_gate_result(self, gate_name: str, passed: bool, issues: List[str] = None):
        """Record gate verification result."""
        if passed:
            self.gates_passed += 1
        else:
            self.gates_failed += 1
            self.recent_gate_failures.append({
                "gate_name": gate_name,
                "timestamp": datetime.utcnow().isoformat(),
                "issues": issues or []
            })
            # Clean old failures outside window
            self._clean_old_gate_failures()

        self._persist_metrics()

    def record_instance_spawned(self):
        """Record instance spawn."""
        self.instances_spawned += 1
        self._persist_metrics()

    def record_instance_crashed(self):
        """Record instance crash."""
        self.instances_crashed += 1
        self._persist_metrics()

    def record_intervention(self):
        """Record intervention sent."""
        self.interventions_sent += 1
        self._persist_metrics()

    def get_completion_rate(self) -> float:
        """Calculate task completion success rate (0-100%)."""
        total = self.tasks_completed + self.tasks_failed
        if total == 0:
            return 100.0  # No tasks = 100% success (no failures)
        return (self.tasks_completed / total) * 100.0

    def get_gate_failure_rate(self) -> float:
        """Calculate gate failure rate (0-100%)."""
        total = self.gates_passed + self.gates_failed
        if total == 0:
            return 0.0
        return (self.gates_failed / total) * 100.0

    def get_avg_task_duration(self) -> float:
        """Calculate average task duration in seconds."""
        if not self.task_durations:
            return 0.0
        return sum(self.task_durations) / len(self.task_durations)

    def get_recent_gate_failure_count(self) -> int:
        """Get gate failure count in recent window (for alerting)."""
        self._clean_old_gate_failures()
        return len(self.recent_gate_failures)

    def get_snapshot(self, cycle_count: int, health_status: str,
                     active_instances: int, available_tasks: int) -> MetricsSnapshot:
        """Get current metrics snapshot."""
        uptime = (datetime.utcnow() - self.start_time).total_seconds()

        return MetricsSnapshot(
            timestamp=datetime.utcnow().isoformat(),
            uptime_seconds=int(uptime),
            cycle_count=cycle_count,
            tasks_completed=self.tasks_completed,
            tasks_failed=self.tasks_failed,
            tasks_in_progress=active_instances,  # Approximate
            completion_rate=self.get_completion_rate(),
            avg_task_duration_seconds=self.get_avg_task_duration(),
            gates_passed=self.gates_passed,
            gates_failed=self.gates_failed,
            gate_failure_rate=self.get_gate_failure_rate(),
            instances_spawned=self.instances_spawned,
            instances_crashed=self.instances_crashed,
            interventions_sent=self.interventions_sent,
            health_status=health_status,
            active_instances=active_instances,
            available_tasks=available_tasks
        )

    def _clean_old_gate_failures(self):
        """Remove gate failures outside the rolling window."""
        cutoff = datetime.utcnow() - timedelta(minutes=self.gate_failure_window_minutes)
        self.recent_gate_failures = [
            f for f in self.recent_gate_failures
            if datetime.fromisoformat(f["timestamp"]) > cutoff
        ]

    def _persist_metrics(self):
        """Persist metrics to JSON file for crash recovery."""
        try:
            metrics_path = Path(self.METRICS_FILE)
            metrics_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "start_time": self.start_time.isoformat(),
                "tasks_completed": self.tasks_completed,
                "tasks_failed": self.tasks_failed,
                "gates_passed": self.gates_passed,
                "gates_failed": self.gates_failed,
                "instances_spawned": self.instances_spawned,
                "instances_crashed": self.instances_crashed,
                "interventions_sent": self.interventions_sent,
                "task_durations": self.task_durations[-50:],  # Keep last 50 for space
                "recent_gate_failures": self.recent_gate_failures,
                "persisted_at": datetime.utcnow().isoformat()
            }

            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)

        except (IOError, OSError) as e:
            logger.warning(f"Failed to persist metrics: {e}")

    def _load_persisted_metrics(self):
        """Load metrics from persisted file (crash recovery)."""
        try:
            metrics_path = Path(self.METRICS_FILE)
            if not metrics_path.exists():
                return

            with open(metrics_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Only load if recent (within 1 hour) - otherwise start fresh
            persisted_at = datetime.fromisoformat(data.get("persisted_at", "1970-01-01"))
            if (datetime.utcnow() - persisted_at).total_seconds() > 3600:
                logger.info("Persisted metrics older than 1 hour - starting fresh")
                return

            # Restore counters
            self.tasks_completed = data.get("tasks_completed", 0)
            self.tasks_failed = data.get("tasks_failed", 0)
            self.gates_passed = data.get("gates_passed", 0)
            self.gates_failed = data.get("gates_failed", 0)
            self.instances_spawned = data.get("instances_spawned", 0)
            self.instances_crashed = data.get("instances_crashed", 0)
            self.interventions_sent = data.get("interventions_sent", 0)
            self.task_durations = data.get("task_durations", [])
            self.recent_gate_failures = data.get("recent_gate_failures", [])

            logger.info(f"Loaded persisted metrics from {metrics_path}")

        except (IOError, json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to load persisted metrics: {e}")


class StructuredLogger:
    """
    JSON-formatted event logger for monitoring integration.

    Outputs structured JSON events to a dedicated log file alongside
    human-readable logs. Enables log aggregation and external monitoring.
    """

    def __init__(self, log_dir: Path):
        """
        Initialize structured logger.

        Args:
            log_dir: Directory for log files
        """
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Dedicated JSON events file
        self.events_file = self.log_dir / "orchestrator_events.jsonl"

        # Standard logger for text output (existing behavior)
        self.text_logger = logging.getLogger(__name__)

    def log_event(self, event: MonitoringEvent):
        """
        Log a structured monitoring event.

        Args:
            event: MonitoringEvent to log
        """
        # Write JSON line to events file
        try:
            with open(self.events_file, 'a', encoding='utf-8') as f:
                f.write(event.to_json() + '\n')
        except IOError as e:
            self.text_logger.error(f"Failed to write event log: {e}")

        # Also log human-readable version
        self._log_text(event)

    def _log_text(self, event: MonitoringEvent):
        """Log human-readable version of event."""
        message = self._format_text_message(event)

        if event.event_type in [EventType.GATE_FAILED, EventType.INSTANCE_CRASHED, EventType.TASK_FAILED]:
            self.text_logger.warning(message)
        elif event.event_type in [EventType.HEALTH_CHECK]:
            self.text_logger.debug(message)
        else:
            self.text_logger.info(message)

    def _format_text_message(self, event: MonitoringEvent) -> str:
        """Format event as human-readable text."""
        base = f"[{event.event_type.value}] cycle={event.cycle}"

        if event.task_id:
            base += f" task={event.task_id}"
        if event.instance_id:
            base += f" instance={event.instance_id}"
        if event.gate_name:
            base += f" gate={event.gate_name}"

        # Add key payload items
        payload_items = []
        for key, value in event.payload.items():
            if key not in ['timestamp', 'cycle']:
                payload_items.append(f"{key}={value}")

        if payload_items:
            base += f" {' '.join(payload_items)}"

        return base

    def rotate_logs(self, max_size_mb: int = 100, max_files: int = 10):
        """
        Rotate log files when they exceed max size.

        Args:
            max_size_mb: Maximum file size before rotation
            max_files: Maximum number of rotated files to keep
        """
        try:
            if not self.events_file.exists():
                return

            size_mb = self.events_file.stat().st_size / (1024 * 1024)
            if size_mb < max_size_mb:
                return

            # Rotate: events.jsonl -> events.jsonl.1 -> events.jsonl.2 -> ...
            for i in range(max_files - 1, 0, -1):
                old_file = self.log_dir / f"orchestrator_events.jsonl.{i}"
                new_file = self.log_dir / f"orchestrator_events.jsonl.{i + 1}"
                if old_file.exists():
                    old_file.rename(new_file)

            # Current -> .1
            rotated = self.log_dir / "orchestrator_events.jsonl.1"
            self.events_file.rename(rotated)

            # Delete oldest if over limit
            oldest = self.log_dir / f"orchestrator_events.jsonl.{max_files}"
            if oldest.exists():
                oldest.unlink()

            self.text_logger.info(f"Rotated events log (was {size_mb:.1f}MB)")

        except (IOError, OSError) as e:
            self.text_logger.error(f"Failed to rotate logs: {e}")


class MonitoringHook(ABC):
    """
    Abstract base class for monitoring integration hooks.

    Implement this to integrate with external monitoring systems
    (Prometheus, Datadog, PagerDuty, Slack, etc.)
    """

    @abstractmethod
    def on_task_complete(self, task_id: str, duration_seconds: float,
                         success: bool, details: Dict[str, Any]) -> None:
        """
        Called when a task completes (success or failure).

        Args:
            task_id: Task identifier (e.g., "TASK-001")
            duration_seconds: How long the task took
            success: True if completed successfully
            details: Additional task details
        """
        pass

    @abstractmethod
    def on_gate_failure(self, gate_name: str, issues: List[str],
                        failure_count: int, threshold: int) -> None:
        """
        Called when a gate fails.

        Args:
            gate_name: Name of failed gate
            issues: List of failure issues
            failure_count: Total failures in rolling window
            threshold: Alert threshold for gate failures
        """
        pass

    @abstractmethod
    def on_health_check(self, metrics: MetricsSnapshot) -> None:
        """
        Called on periodic health check.

        Args:
            metrics: Current metrics snapshot
        """
        pass


class LoggingHook(MonitoringHook):
    """
    Default hook that logs to standard logger.

    Serves as reference implementation and fallback.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__ + ".hooks")

    def on_task_complete(self, task_id: str, duration_seconds: float,
                         success: bool, details: Dict[str, Any]) -> None:
        status = "SUCCESS" if success else "FAILED"
        self.logger.info(f"[HOOK] Task {task_id} {status} in {duration_seconds:.1f}s")

    def on_gate_failure(self, gate_name: str, issues: List[str],
                        failure_count: int, threshold: int) -> None:
        self.logger.warning(
            f"[HOOK] Gate {gate_name} failed ({failure_count}/{threshold} in window): {issues}"
        )
        if failure_count >= threshold:
            self.logger.error(f"[ALERT] Gate failure threshold exceeded: {failure_count} >= {threshold}")

    def on_health_check(self, metrics: MetricsSnapshot) -> None:
        self.logger.debug(
            f"[HOOK] Health check: {metrics.health_status} "
            f"completion_rate={metrics.completion_rate:.1f}% "
            f"active={metrics.active_instances}"
        )


class MonitoringHookDispatcher:
    """
    Dispatches monitoring events to registered hooks.

    Supports multiple hooks for different integrations.
    Handles hook failures gracefully (logs and continues).
    """

    def __init__(self):
        """Initialize dispatcher with default logging hook."""
        self.hooks: List[MonitoringHook] = [LoggingHook()]

    def register_hook(self, hook: MonitoringHook):
        """
        Register a monitoring hook.

        Args:
            hook: Hook implementation to register
        """
        self.hooks.append(hook)
        logger.info(f"Registered monitoring hook: {hook.__class__.__name__}")

    def dispatch_task_complete(self, task_id: str, duration_seconds: float,
                               success: bool, details: Dict[str, Any] = None):
        """Dispatch task completion to all hooks."""
        details = details or {}
        for hook in self.hooks:
            try:
                hook.on_task_complete(task_id, duration_seconds, success, details)
            except Exception as e:
                logger.error(f"Hook {hook.__class__.__name__}.on_task_complete failed: {e}")

    def dispatch_gate_failure(self, gate_name: str, issues: List[str],
                              failure_count: int, threshold: int):
        """Dispatch gate failure to all hooks."""
        for hook in self.hooks:
            try:
                hook.on_gate_failure(gate_name, issues, failure_count, threshold)
            except Exception as e:
                logger.error(f"Hook {hook.__class__.__name__}.on_gate_failure failed: {e}")

    def dispatch_health_check(self, metrics: MetricsSnapshot):
        """Dispatch health check to all hooks."""
        for hook in self.hooks:
            try:
                hook.on_health_check(metrics)
            except Exception as e:
                logger.error(f"Hook {hook.__class__.__name__}.on_health_check failed: {e}")


# ============================================================================
# End of MONITORING-001 Enhancement
# ============================================================================


class DecisionEngine:
    """
    Intelligent decision engine for autonomous instance spawning.

    Makes data-driven decisions about when and what to spawn based on:
    - Available tasks in backlog
    - Current resource utilization
    - Task priorities (CRITICAL > HIGH > MEDIUM > LOW)
    - Spawn cooldown periods
    - Model selection based on priority
    """

    # Priority ordering (higher = more important)
    PRIORITY_ORDER = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]

    def __init__(self, config: dict, monitor: OrchestratorMonitor,
                 coordinator: TaskCoordinator, last_spawn_time: Optional[datetime] = None):
        """
        Initialize decision engine.

        Args:
            config: Orchestrator configuration
            monitor: OrchestratorMonitor for querying system state
            coordinator: TaskCoordinator for task queue access
            last_spawn_time: Timestamp of last spawn (for cooldown)
        """
        self.config = config
        self.monitor = monitor
        self.coordinator = coordinator
        self.last_spawn_time = last_spawn_time

    def should_spawn_instance(self) -> tuple[bool, Optional[dict]]:
        """
        Decide whether to spawn a new instance.

        Returns:
            Tuple of (should_spawn: bool, task_details: Optional[dict])
            If should_spawn is True, task_details contains:
                - task_id: str
                - priority: str
                - model: str (opus/sonnet/haiku)
                - description: str
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

        # 4. Get next task from backlog
        next_task = self._get_next_task()
        if next_task is None:
            logger.debug("Cannot spawn: no available tasks")
            return False, None

        # 5. Select model based on priority
        model = self._select_model(next_task["priority"])

        # 6. Build task details
        task_details = {
            "task_id": next_task["id"],
            "priority": next_task["priority"],
            "model": model,
            "description": next_task.get("description", f"Execute {next_task['id']}")
        }

        logger.info(f"Decision: SPAWN instance for {task_details['task_id']} " +
                   f"(priority={task_details['priority']}, model={task_details['model']})")

        return True, task_details

    def _can_spawn(self) -> bool:
        """
        Check if we can spawn another instance (resource limits).

        Returns:
            True if under max_instances limit
        """
        try:
            dashboard = self.monitor.get_instance_dashboard()
            current_instances = dashboard["resource_usage"]["current_instances"]
            max_instances = self.config.get("max_instances", 3)

            return current_instances < max_instances

        except (KeyError, TypeError, RuntimeError, IOError) as e:
            logger.error(f"Failed to check resource limits: {e}")
            return False  # Fail closed (don't spawn if we can't check)

    def _cooldown_elapsed(self) -> bool:
        """
        Check if spawn cooldown period has elapsed.

        Returns:
            True if cooldown elapsed or no last spawn
        """
        if self.last_spawn_time is None:
            return True

        cooldown = self.config.get("spawn_cooldown", 60)
        elapsed = (datetime.utcnow() - self.last_spawn_time).total_seconds()

        return elapsed >= cooldown

    def _get_next_task(self) -> Optional[dict]:
        """
        Get highest priority READY task from backlog.

        Priority order: CRITICAL > HIGH > MEDIUM > LOW

        Returns:
            Task dict with id, priority, status, description
            None if no tasks available
        """
        try:
            # Load task queue
            task_queue_path = Path("tasks/task_queue.json")
            if not task_queue_path.exists():
                logger.warning("Task queue not found")
                return None

            with open(task_queue_path, 'r', encoding='utf-8') as f:
                task_queue = json.load(f)

            # Get backlog tasks
            backlog = task_queue.get("backlog", [])
            if isinstance(backlog, dict):
                # Handle old format (dict of task_id -> task_info)
                backlog = [{"id": task_id, **task_info} for task_id, task_info in backlog.items()]

            # Filter to READY tasks only
            ready_tasks = [t for t in backlog if t.get("status") == "READY"]

            if not ready_tasks:
                return None

            # Sort by priority
            def priority_rank(task):
                priority = task.get("priority", "LOW")
                try:
                    return self.PRIORITY_ORDER.index(priority)
                except ValueError:
                    return len(self.PRIORITY_ORDER)  # Unknown priority = lowest

            ready_tasks.sort(key=priority_rank)

            # Return highest priority task
            return ready_tasks[0]

        except (IOError, json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error(f"Failed to get next task from backlog: {e}", exc_info=True)
            return None

    def _select_model(self, priority: str) -> str:
        """
        Select Claude model based on task priority.

        Args:
            priority: Task priority (CRITICAL/HIGH/MEDIUM/LOW)

        Returns:
            Model name (opus/sonnet/haiku)
        """
        models = self.config.get("models", {
            "CRITICAL": "opus",
            "HIGH": "sonnet",
            "MEDIUM": "sonnet",
            "LOW": "haiku"
        })

        return models.get(priority, "sonnet")


class ProgressTracker:
    """
    Track instance progress beyond heartbeats.

    Monitors:
    - Phase transitions (when instance changes phase)
    - Time spent in each phase
    - Commit activity (optional)

    Enables detection of:
    - Instances stuck in same phase too long
    - Instances with heartbeat but no actual progress
    - Slow progress instances
    """

    # Phase-specific timeout thresholds (minutes)
    PHASE_TIMEOUTS = {
        "INIT": 5,          # 5 minutes - initial setup
        "DATA_CHECK": 15,   # 15 minutes - data validation
        "SEARCH": 120,      # 2 hours - optimization (can be very long)
        "REPORTING": 10,    # 10 minutes - report generation
        "PR_CREATION": 5,   # 5 minutes - PR creation
        "STARTUP": 5,       # 5 minutes - startup phase
        "EXECUTING": 60,    # 1 hour - general execution
        "VALIDATION": 15,   # 15 minutes - validation
        "DEFAULT": 30       # 30 minutes - fallback for unknown phases
    }

    # Intervention strategies by phase
    INTERVENTION_STRATEGIES = {
        "SEARCH": "status_request",      # Don't interrupt long-running optimization
        "EXECUTING": "status_request",   # Don't interrupt execution
        "INIT": "pause",                 # Short phase, pause if stuck
        "STARTUP": "pause",              # Short phase, pause if stuck
        "PR_CREATION": "pause",          # Short phase, pause if stuck
        "DATA_CHECK": "pause",           # Should be quick, pause if stuck
        "REPORTING": "pause",            # Should be quick, pause if stuck
        "VALIDATION": "pause",           # Should be quick, pause if stuck
        "DEFAULT": "status_request"      # Conservative default
    }

    def __init__(self, repo_path: Path):
        """
        Initialize progress tracker.

        Args:
            repo_path: Path to git repository (for commit activity tracking)
        """
        self.repo_path = repo_path
        self.instance_phases: Dict[str, dict] = {}
        # Structure: {instance_id: {
        #     "current_phase": str,
        #     "phase_entered_at": datetime,
        #     "last_commit_time": Optional[datetime],
        #     "phase_history": [(phase, entered_at, exited_at), ...]
        # }}

    def update_phase(self, instance_id: str, phase: str):
        """
        Record phase transition for an instance.

        Args:
            instance_id: Instance identifier
            phase: New phase name
        """
        now = datetime.utcnow()

        if instance_id not in self.instance_phases:
            # First time seeing this instance
            self.instance_phases[instance_id] = {
                "current_phase": phase,
                "phase_entered_at": now,
                "last_commit_time": None,
                "phase_history": []
            }
            logger.debug(f"Progress tracker: {instance_id} entered phase {phase}")
        else:
            # Phase transition
            data = self.instance_phases[instance_id]
            old_phase = data["current_phase"]

            if old_phase != phase:
                # Record phase exit in history
                phase_duration = (now - data["phase_entered_at"]).total_seconds() / 60
                data["phase_history"].append(
                    (old_phase, data["phase_entered_at"], now)
                )

                # Limit history to last 20 transitions to prevent unbounded memory growth
                if len(data["phase_history"]) > 20:
                    data["phase_history"] = data["phase_history"][-20:]

                # Update to new phase
                data["current_phase"] = phase
                data["phase_entered_at"] = now

                logger.info(f"Progress tracker: {instance_id} transitioned from {old_phase} to {phase} " +
                           f"(spent {phase_duration:.1f} minutes in {old_phase})")

    def check_progress(self, instance_id: str, current_phase: str) -> bool:
        """
        Check if instance is making progress.

        Args:
            instance_id: Instance identifier
            current_phase: Current phase from instance registry

        Returns:
            True if instance is making progress, False if stuck
        """
        # Update phase tracking
        self.update_phase(instance_id, current_phase)

        # Check if stuck in current phase too long
        if instance_id not in self.instance_phases:
            return True  # Just started, assume making progress

        data = self.instance_phases[instance_id]
        phase = data["current_phase"]
        entered_at = data["phase_entered_at"]

        # Calculate time in current phase
        time_in_phase = (datetime.utcnow() - entered_at).total_seconds() / 60  # minutes

        # Get timeout threshold for this phase
        timeout = self.PHASE_TIMEOUTS.get(phase, self.PHASE_TIMEOUTS["DEFAULT"])

        # Check if stuck
        if time_in_phase > timeout:
            logger.warning(f"Progress check: {instance_id} stuck in {phase} for {time_in_phase:.1f} minutes " +
                          f"(threshold: {timeout} minutes)")
            return False

        return True

    def get_stuck_instances(self) -> List[tuple[str, str, int]]:
        """
        Identify instances stuck in same phase too long.

        Returns:
            List of (instance_id, phase, minutes_stuck) tuples
        """
        stuck = []
        now = datetime.utcnow()

        for instance_id, data in self.instance_phases.items():
            phase = data["current_phase"]
            entered_at = data["phase_entered_at"]

            # Calculate time in phase
            time_in_phase = (now - entered_at).total_seconds() / 60  # minutes

            # Get timeout threshold
            timeout = self.PHASE_TIMEOUTS.get(phase, self.PHASE_TIMEOUTS["DEFAULT"])

            # Check if stuck
            if time_in_phase > timeout:
                stuck.append((instance_id, phase, int(time_in_phase)))

        return stuck

    def get_intervention_type(self, instance_id: str, phase: str) -> str:
        """
        Determine appropriate intervention based on phase.

        For long-running phases (SEARCH, EXECUTING), use gentle status_request.
        For short phases (INIT, PR_CREATION), use pause to interrupt.

        Args:
            instance_id: Instance identifier
            phase: Current phase

        Returns:
            Intervention type: "status_request" or "pause"
        """
        return self.INTERVENTION_STRATEGIES.get(phase, self.INTERVENTION_STRATEGIES["DEFAULT"])

    def check_commit_activity(self, instance_id: str) -> Optional[datetime]:
        """
        Check git commit activity for this instance (optional enhancement).

        Uses git log to find recent commits by this instance.

        Args:
            instance_id: Instance identifier

        Returns:
            Timestamp of last commit, or None if no commits found or error
        """
        try:
            # Git log command to find commits by this instance
            # NOTE: Limited to 24 hours to avoid performance issues with large repos
            # For very long-running instances (>24h), commit tracking won't detect progress,
            # but phase timeout will still detect stuck instances
            cmd = [
                "git", "-C", str(self.repo_path),
                "log", "--all", "--since=24.hours.ago",
                "--grep", instance_id,
                "--format=%ct", "-1"
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0 and result.stdout.strip():
                # Parse Unix timestamp
                timestamp = int(result.stdout.strip())
                commit_time = datetime.utcfromtimestamp(timestamp)

                # Update tracking
                if instance_id in self.instance_phases:
                    self.instance_phases[instance_id]["last_commit_time"] = commit_time

                logger.debug(f"Commit activity: {instance_id} last commit at {commit_time.isoformat()}")
                return commit_time

            return None

        except (subprocess.TimeoutExpired, subprocess.SubprocessError, ValueError, OSError) as e:
            logger.debug(f"Failed to check commit activity for {instance_id}: {e}")
            return None

    def remove_instance(self, instance_id: str):
        """
        Remove instance from tracking (cleanup after completion/failure).

        Args:
            instance_id: Instance identifier
        """
        if instance_id in self.instance_phases:
            del self.instance_phases[instance_id]
            logger.debug(f"Progress tracker: removed {instance_id}")


class OrchestratorLoop:
    """
    Main orchestrator loop that runs forever.

    Monitors all autonomous workflow activity, makes spawning decisions,
    detects stuck/crashed instances, and maintains system health 24/7.
    """

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize orchestrator loop.

        Args:
            config_path: Path to orchestrator_config.json (defaults to .autonomous/orchestrator_config.json)
        """
        if config_path is None:
            config_path = Path(".autonomous/orchestrator_config.json")

        self.config = self._load_config(config_path)
        self.running = True
        self.cycle_count = 0
        self.start_time = datetime.utcnow()

        # Coordination components (from TASK-058)
        main_repo = Path.cwd()
        lock_manager = DistributedLockManager(main_repo)
        self.registry = InstanceRegistry(lock_manager)
        self.coordinator = TaskCoordinator(lock_manager)
        self.message_queue = MessageQueue(lock_manager)
        self.monitor = OrchestratorMonitor(main_repo)

        # Progress tracking (Phase 4)
        self.progress_tracker = ProgressTracker(main_repo)

        # MONITORING-001: Enhanced monitoring components
        self.metrics = MonitoringMetrics()
        self.structured_logger = StructuredLogger(Path(".autonomous/logs"))
        self.hook_dispatcher = MonitoringHookDispatcher()

        # MONITORING-001: Health check tracking
        self.last_health_check: Optional[datetime] = None
        self.health_check_interval = self.config.get("health_check_interval", 300)  # 5 min default

        # MONITORING-001: Gate failure alerting
        self.gate_failure_threshold = self.config.get("gate_failure_threshold", 5)

        # State tracking
        self.spawned_instances: Dict[int, str] = {}  # pid -> task_id
        self.last_heartbeat: Dict[str, datetime] = {}  # instance_id -> timestamp
        self.last_progress: Dict[str, datetime] = {}  # instance_id -> timestamp
        self.intervention_history: List[dict] = []  # Recent interventions
        self.last_spawn_time: Optional[datetime] = None  # Last successful spawn timestamp

        # Decision engine for spawning intelligence
        self.decision_engine = DecisionEngine(
            config=self.config,
            monitor=self.monitor,
            coordinator=self.coordinator,
            last_spawn_time=self.last_spawn_time
        )

        # Configure signal handlers
        signal.signal(signal.SIGINT, self._handle_shutdown)
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, self._handle_shutdown)

    def _load_config(self, path: Path) -> dict:
        """
        Load orchestrator configuration with defaults.

        Args:
            path: Path to config file

        Returns:
            Configuration dictionary with defaults applied
        """
        default_config = {
            "loop_interval": 60,
            "max_instances": 3,
            "heartbeat_timeout": 300,
            "progress_timeout": 1800,
            "warning_threshold": 900,
            "snapshot_interval": 600,
            "enable_spawning": False,
            "enable_rebalancing": False,
            "enable_interventions": True,
            "models": {
                "CRITICAL": "opus",
                "HIGH": "sonnet",
                "MEDIUM": "sonnet",
                "LOW": "haiku"
            },
            "spawn_cooldown": 60,
            "max_spawn_retries": 3,
            "log_level": "INFO",
            "log_retention_days": 7
        }

        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
                    logger.info(f"Loaded configuration from {path}")
            except (IOError, json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to load config from {path}: {e}, using defaults")
        else:
            logger.info(f"Config file not found at {path}, using defaults")

        return default_config

    def _emit_event(self, event_type: EventType, payload: Dict[str, Any] = None,
                    task_id: str = None, instance_id: str = None, gate_name: str = None):
        """
        Emit a structured monitoring event (MONITORING-001).

        Args:
            event_type: Type of event
            payload: Event-specific data
            task_id: Optional task ID
            instance_id: Optional instance ID
            gate_name: Optional gate name
        """
        event = MonitoringEvent(
            event_type=event_type,
            timestamp=datetime.utcnow().isoformat(),
            cycle=self.cycle_count,
            payload=payload or {},
            task_id=task_id,
            instance_id=instance_id,
            gate_name=gate_name
        )
        self.structured_logger.log_event(event)

    def _maybe_health_check(self, dashboard: Dict, system_health: Dict):
        """
        Perform periodic health check if interval elapsed (MONITORING-001).

        Args:
            dashboard: Current instance dashboard
            system_health: Current system health dict
        """
        now = datetime.utcnow()

        # Check if health check interval elapsed
        if self.last_health_check is not None:
            elapsed = (now - self.last_health_check).total_seconds()
            if elapsed < self.health_check_interval:
                return  # Not time yet

        self.last_health_check = now

        # Get metrics snapshot
        metrics = self.metrics.get_snapshot(
            cycle_count=self.cycle_count,
            health_status=system_health.get("health_status", "UNKNOWN"),
            active_instances=len(dashboard.get("active_instances", [])),
            available_tasks=dashboard.get("available_tasks", 0)
        )

        # Emit health check event
        self._emit_event(
            EventType.HEALTH_CHECK,
            payload=metrics.to_dict()
        )

        # Dispatch to hooks
        self.hook_dispatcher.dispatch_health_check(metrics)

        # Log summary
        logger.info(
            f"[HEALTH CHECK] status={metrics.health_status} "
            f"completion_rate={metrics.completion_rate:.1f}% "
            f"tasks_completed={metrics.tasks_completed} "
            f"gate_failures={metrics.gates_failed} "
            f"active_instances={metrics.active_instances}"
        )

        # Rotate logs if needed
        self.structured_logger.rotate_logs()

    def _check_gate_failure_alert(self, gate_name: str, issues: List[str]):
        """
        Check if gate failure count exceeds threshold and dispatch alert (MONITORING-001).

        Args:
            gate_name: Name of failed gate
            issues: List of failure issues
        """
        failure_count = self.metrics.get_recent_gate_failure_count()

        if failure_count >= self.gate_failure_threshold:
            logger.error(
                f"[ALERT] Gate failure threshold exceeded: "
                f"{failure_count} failures in last {self.metrics.gate_failure_window_minutes} minutes"
            )

        # Always dispatch to hooks (they decide whether to alert)
        self.hook_dispatcher.dispatch_gate_failure(
            gate_name=gate_name,
            issues=issues,
            failure_count=failure_count,
            threshold=self.gate_failure_threshold
        )

    def run(self):
        """
        Main orchestrator loop - runs forever until shutdown signal.

        This is the heart of autonomous development. The loop:
        1. Queries all coordination components for current state
        2. Detects stuck/crashed instances
        3. Makes spawning decisions (if enabled)
        4. Sends interventions (if needed)
        5. Logs all activity
        6. Sleeps until next cycle
        """
        logger.info("=" * 80)
        logger.info("ORCHESTRATOR STARTING - Full Autonomous Mode")
        logger.info(f"Configuration: {self.config}")
        logger.info(f"Start time: {self.start_time.isoformat()}")
        logger.info(f"Loop interval: {self.config['loop_interval']}s")
        logger.info(f"Max instances: {self.config['max_instances']}")
        logger.info(f"Spawning enabled: {self.config['enable_spawning']}")
        logger.info("=" * 80)

        # MONITORING-001: Emit orchestrator started event
        self._emit_event(
            EventType.ORCHESTRATOR_STARTED,
            payload={
                "start_time": self.start_time.isoformat(),
                "max_instances": self.config.get("max_instances", 3),
                "spawning_enabled": self.config.get("enable_spawning", False),
                "interventions_enabled": self.config.get("enable_interventions", True),
                "health_check_interval": self.health_check_interval
            }
        )

        while self.running:
            try:
                cycle_start = datetime.utcnow()
                self.cycle_count += 1

                logger.info(f"\n{'=' * 80}")
                logger.info(f"Cycle {self.cycle_count} starting at {cycle_start.isoformat()}")
                logger.info(f"Uptime: {self._format_duration(datetime.utcnow() - self.start_time)}")
                logger.info(f"{'=' * 80}")

                # MONITORING-001: Emit cycle start event
                self._emit_event(
                    EventType.CYCLE_START,
                    payload={
                        "cycle_start": cycle_start.isoformat(),
                        "uptime": self._format_duration(datetime.utcnow() - self.start_time)
                    }
                )

                # Execute monitoring cycle
                self._execute_monitoring_cycle()

                # Snapshot state periodically
                if self.cycle_count % (self.config["snapshot_interval"] // self.config["loop_interval"]) == 0:
                    self._snapshot_state()

                # Calculate sleep time to maintain interval
                cycle_duration = (datetime.utcnow() - cycle_start).total_seconds()
                sleep_time = max(0, self.config["loop_interval"] - cycle_duration)

                # MONITORING-001: Emit cycle end event
                cycle_end = datetime.utcnow()
                self._emit_event(
                    EventType.CYCLE_END,
                    payload={
                        "cycle_duration_seconds": cycle_duration,
                        "cycle_end": cycle_end.isoformat()
                    }
                )

                if sleep_time > 0:
                    logger.info(f"Cycle completed in {cycle_duration:.1f}s, sleeping {sleep_time:.1f}s")
                    time.sleep(sleep_time)
                else:
                    logger.warning(f"Cycle took {cycle_duration:.1f}s, exceeding {self.config['loop_interval']}s interval")

            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received")
                self.running = False
            except (RuntimeError, ValueError, TypeError, KeyError, IOError) as e:
                logger.error(f"Unexpected error in main loop: {e}", exc_info=True)
                logger.info("Pausing 5 seconds before retry...")
                time.sleep(5)

        self._shutdown()

    def _execute_monitoring_cycle(self):
        """
        Execute one complete monitoring cycle.

        Steps:
        1. Query current state from all coordination components
        2. Update heartbeat tracking
        3. Detect stuck/crashed instances
        4. Make spawning decisions (Phase 2)
        5. Send interventions if needed
        6. Broadcast orchestrator heartbeat
        """

        # 1. Query current state
        logger.info("Querying coordination layer...")
        dashboard = self.monitor.get_instance_dashboard()
        system_health = self.monitor.get_system_health()

        active_instances = dashboard["active_instances"]
        claimed_tasks = dashboard["claimed_tasks"]
        stale_instances = dashboard["stale_instances"]
        available_tasks = dashboard["available_tasks"]
        recent_messages = dashboard["recent_messages"]

        # MONITORING-001: Perform periodic health check
        self._maybe_health_check(dashboard, system_health)

        # 2. Log current state
        logger.info(f"Active instances: {len(active_instances)}")
        for inst in active_instances:
            logger.info(f"  - {inst['id']}: {inst['task']} ({inst['status']} / {inst['phase']}) " +
                       f"[heartbeat: {inst['last_heartbeat']}]")

        logger.info(f"Claimed tasks: {len(claimed_tasks)}")
        for task_id, task_info in claimed_tasks.items():
            logger.info(f"  - {task_id}: claimed by {task_info['claimed_by']} ({task_info['status']})")

        logger.info(f"Stale instances: {len(stale_instances)}")
        for inst in stale_instances:
            logger.warning(f"  - {inst['id']}: {inst['task']} (heartbeat: {inst['last_heartbeat']})")

        logger.info(f"Available tasks: {available_tasks}")
        logger.info(f"Recent messages: {len(recent_messages)}")
        logger.info(f"System health: {system_health['health_status']}")
        if system_health['issues']:
            logger.warning(f"  Issues: {system_health['issues']}")

        # 3. Update heartbeat tracking and progress tracking (Phase 4)
        for instance in active_instances:
            instance_id = instance['id']
            current_phase = instance.get('phase', 'UNKNOWN')

            # Parse heartbeat timestamp (format: "Xs ago" or "Xm Ys ago")
            heartbeat_str = instance['last_heartbeat']
            try:
                # Handle both "Xs ago" and "Xm Ys ago" formats
                if 'm' in heartbeat_str:
                    # Parse "Xm Ys ago" format
                    parts = heartbeat_str.split('m')
                    minutes = int(parts[0])
                    seconds = int(parts[1].split('s ago')[0].strip())
                    seconds_ago = minutes * 60 + seconds
                else:
                    # Parse "Xs ago" format
                    seconds_ago = int(heartbeat_str.split('s ago')[0])

                self.last_heartbeat[instance_id] = datetime.utcnow() - timedelta(seconds=seconds_ago)
            except (ValueError, IndexError, AttributeError) as e:
                logger.warning(f"Failed to parse heartbeat for {instance_id}: '{heartbeat_str}' - {e}")

            # Update progress tracking (Phase 4)
            try:
                self.progress_tracker.update_phase(instance_id, current_phase)
            except (ValueError, TypeError, KeyError) as e:
                logger.warning(f"Failed to update progress tracking for {instance_id}: {e}")

        # 4. Detect stuck/crashed instances (Phase 1 + Phase 4 enhancement)
        stuck_instance_ids = []
        crashed_instance_ids = []
        stuck_instances_with_phase: Dict[str, str] = {}  # instance_id -> phase (for phase-aware intervention)

        # 4a. Heartbeat-based detection (Phase 1)
        for inst in stale_instances:
            instance_id = inst['id']
            heartbeat_str = inst['last_heartbeat']

            try:
                # Parse heartbeat format: "Xs ago" or "Xm Ys ago"
                if 'm' in heartbeat_str:
                    # Parse "Xm Ys ago" format
                    parts = heartbeat_str.split('m')
                    minutes = int(parts[0])
                    seconds = int(parts[1].split('s ago')[0].strip())
                    seconds_ago = minutes * 60 + seconds
                else:
                    # Parse "Xs ago" format
                    seconds_ago = int(heartbeat_str.split('s ago')[0])

                if seconds_ago > self.config["heartbeat_timeout"]:
                    crashed_instance_ids.append(instance_id)
                    logger.error(f"Instance {instance_id} CRASHED (no heartbeat for {seconds_ago}s)")
                elif seconds_ago > self.config["progress_timeout"]:
                    stuck_instance_ids.append(instance_id)
                    stuck_instances_with_phase[instance_id] = inst.get('phase', 'UNKNOWN')
                    logger.warning(f"Instance {instance_id} appears STUCK (no progress for {seconds_ago}s)")
            except (ValueError, IndexError, AttributeError) as e:
                # If heartbeat format is unparseable, treat conservatively as crashed
                # (better to intervene on a false positive than miss a real crash)
                logger.warning(f"Failed to parse heartbeat for {instance_id}: '{heartbeat_str}' - treating as crashed")
                crashed_instance_ids.append(instance_id)

        # 4b. Progress-based detection (Phase 4)
        # Check for instances stuck in same phase too long (even if heartbeat is active)
        progress_stuck_instances = self.progress_tracker.get_stuck_instances()
        for instance_id, phase, minutes_stuck in progress_stuck_instances:
            if instance_id not in stuck_instance_ids and instance_id not in crashed_instance_ids:
                # Found instance stuck by progress tracking but not by heartbeat
                stuck_instance_ids.append(instance_id)
                stuck_instances_with_phase[instance_id] = phase
                logger.warning(f"Instance {instance_id} STUCK in {phase} for {minutes_stuck} minutes " +
                              f"(heartbeat active, but no phase progress)")

        # 5. Handle problematic instances
        if self.config["enable_interventions"]:
            if stuck_instance_ids:
                self._handle_stuck_instances(stuck_instance_ids, stuck_instances_with_phase)

            if crashed_instance_ids:
                self._handle_crashed_instances(crashed_instance_ids)

        # 6. Make spawning decision (Phase 2)
        if self.config["enable_spawning"]:
            # Update decision engine with latest spawn time
            self.decision_engine.last_spawn_time = self.last_spawn_time

            # Query decision engine
            should_spawn, task_details = self.decision_engine.should_spawn_instance()

            if should_spawn and task_details:
                logger.info(f"Decision engine recommends spawning instance for {task_details['task_id']}")

                # Spawn instance (Phase 3)
                spawn_result = self._spawn_instance(task_details)

                if spawn_result and spawn_result.get("success"):
                    logger.info(f"Successfully spawned instance for {task_details['task_id']} (PID {spawn_result['pid']})")

                    # Broadcast spawn notification
                    try:
                        self.message_queue.broadcast(
                            "orchestrator",
                            MessageType.INSTANCE_STARTED,
                            {
                                "spawned_task": task_details['task_id'],
                                "pid": spawn_result['pid'],
                                "instance_id": spawn_result['instance_id'],
                                "model": spawn_result['model'],
                                "priority": spawn_result['priority']
                            }
                        )
                    except (RuntimeError, ValueError, IOError) as e:
                        logger.error(f"Failed to broadcast spawn notification: {e}")
                else:
                    logger.error(f"Failed to spawn instance for {task_details['task_id']}")
            else:
                logger.debug("Decision engine: no spawn recommended")

        # 7. Broadcast orchestrator heartbeat
        try:
            self.message_queue.broadcast(
                "orchestrator",
                MessageType.INSTANCE_STARTED,  # Reuse existing message type
                {
                    "cycle": self.cycle_count,
                    "uptime": self._format_duration(datetime.utcnow() - self.start_time),
                    "active_instances": len(active_instances),
                    "available_tasks": available_tasks,
                    "system_health": system_health['health_status']
                }
            )
            logger.debug("Orchestrator heartbeat broadcast sent")
        except (RuntimeError, ValueError, IOError) as e:
            logger.error(f"Failed to broadcast heartbeat: {e}")

    def _handle_stuck_instances(self, instance_ids: List[str], instances_with_phase: Dict[str, str]):
        """
        Send intervention messages to stuck instances (Phase 1 + Phase 4 enhancement).

        Strategy (Phase 4 - phase-aware):
        - Long-running phases (SEARCH, EXECUTING): Gentle status_request only
        - Short phases: Standard escalation (status_request → pause → force_release)
        - All phases: Force release after 3 interventions

        Args:
            instance_ids: List of stuck instance IDs
            instances_with_phase: Dict mapping instance_id to current phase (for phase-aware intervention)
        """
        for instance_id in instance_ids:
            # Get phase for phase-aware intervention (Phase 4)
            phase = instances_with_phase.get(instance_id, "UNKNOWN")

            # Check intervention history
            previous = [i for i in self.intervention_history
                       if i.get("instance_id") == instance_id]

            if len(previous) == 0:
                # First intervention: Always status_request (gentle)
                logger.info(f"Sending status request to stuck instance: {instance_id} (phase: {phase})")
                try:
                    self.message_queue.send_message(
                        "orchestrator",
                        MessageType.TASK_CLAIMED,  # Reuse existing message type
                        {
                            "type": "status_request",
                            "reason": "no_progress_detected",
                            "timeout": self.config["progress_timeout"],
                            "phase": phase
                        },
                        to_instance=instance_id
                    )
                    self._record_intervention(instance_id, "status_request", phase=phase)

                    # MONITORING-001: Emit event and record metrics
                    self._emit_event(
                        EventType.INSTANCE_STUCK,
                        instance_id=instance_id,
                        payload={
                            "phase": phase,
                            "intervention_type": "status_request",
                            "intervention_count": 1
                        }
                    )
                    self.metrics.record_intervention()
                except (RuntimeError, ValueError, IOError) as e:
                    logger.error(f"Failed to send status request to {instance_id}: {e}")

            elif len(previous) == 1:
                # Second intervention: Phase-aware (Phase 4)
                intervention_type = self.progress_tracker.get_intervention_type(instance_id, phase)

                if intervention_type == "status_request":
                    # Long-running phase: gentle status request again
                    logger.warning(f"Sending 2nd status request to {instance_id} (phase: {phase}, long-running)")
                    try:
                        self.message_queue.send_message(
                            "orchestrator",
                            MessageType.TASK_CLAIMED,
                            {
                                "type": "status_request",
                                "reason": "still_no_progress",
                                "timeout": self.config["progress_timeout"],
                                "phase": phase,
                                "intervention_count": 2
                            },
                            to_instance=instance_id
                        )
                        self._record_intervention(instance_id, "status_request", phase=phase)

                        # MONITORING-001: Emit event and record metrics
                        self._emit_event(
                            EventType.INSTANCE_STUCK,
                            instance_id=instance_id,
                            payload={
                                "phase": phase,
                                "intervention_type": "status_request",
                                "intervention_count": 2
                            }
                        )
                        self.metrics.record_intervention()
                    except (RuntimeError, ValueError, IOError) as e:
                        logger.error(f"Failed to send 2nd status request to {instance_id}: {e}")
                else:
                    # Short phase: escalate to pause
                    logger.warning(f"Sending pause message to {instance_id} (phase: {phase}, 2nd intervention)")
                    try:
                        self.message_queue.send_message(
                            "orchestrator",
                            MessageType.TASK_CLAIMED,
                            {
                                "type": "pause",
                                "duration": 5,
                                "reason": "attempting_recovery",
                                "phase": phase
                            },
                            to_instance=instance_id
                        )
                        self._record_intervention(instance_id, "pause", phase=phase)

                        # MONITORING-001: Emit event and record metrics
                        self._emit_event(
                            EventType.INSTANCE_STUCK,
                            instance_id=instance_id,
                            payload={
                                "phase": phase,
                                "intervention_type": "pause",
                                "intervention_count": 2
                            }
                        )
                        self.metrics.record_intervention()
                    except (RuntimeError, ValueError, IOError) as e:
                        logger.error(f"Failed to send pause to {instance_id}: {e}")

            else:
                # Third intervention: force release (all phases)
                logger.error(f"Force releasing task from stuck instance: {instance_id} (phase: {phase}, 3rd intervention)")
                try:
                    # Find task assigned to this instance
                    dashboard = self.monitor.get_instance_dashboard()
                    for task_id, task_info in dashboard["claimed_tasks"].items():
                        if task_info["claimed_by"] == instance_id:
                            logger.warning(f"Releasing task {task_id} from {instance_id}")
                            self.coordinator.release_task(instance_id, task_id, "forced_release_stuck_instance")

                    # Cleanup progress tracking
                    self.progress_tracker.remove_instance(instance_id)

                    self._record_intervention(instance_id, "force_release", phase=phase)

                    # MONITORING-001: Emit event and record metrics
                    self._emit_event(
                        EventType.INSTANCE_STUCK,
                        instance_id=instance_id,
                        payload={
                            "phase": phase,
                            "intervention_type": "force_release",
                            "intervention_count": 3
                        }
                    )
                    self.metrics.record_intervention()
                except (RuntimeError, ValueError, KeyError, IOError) as e:
                    logger.error(f"Failed to force release from {instance_id}: {e}")

    def _handle_crashed_instances(self, instance_ids: List[str]):
        """
        Clean up after crashed instances.

        Steps:
        1. Release any claimed tasks back to backlog
        2. Record cleanup actions
        3. Log cleanup

        Note: Instance status updates are the responsibility of the instance itself.
        The orchestrator only releases tasks and logs cleanup.

        Args:
            instance_ids: List of crashed instance IDs
        """
        for instance_id in instance_ids:
            logger.error(f"Cleaning up crashed instance: {instance_id}")

            try:
                # 1. Release claimed tasks
                dashboard = self.monitor.get_instance_dashboard()
                for task_id, task_info in dashboard["claimed_tasks"].items():
                    if task_info["claimed_by"] == instance_id:
                        logger.warning(f"Releasing task {task_id} from crashed instance {instance_id}")
                        self.coordinator.release_task(
                            instance_id,
                            task_id,
                            f"Instance crashed (no heartbeat for {self.config['heartbeat_timeout']}s)"
                        )

                # 2. Cleanup progress tracking (Phase 4)
                self.progress_tracker.remove_instance(instance_id)

                # 3. Record cleanup
                self._record_intervention(instance_id, "cleanup_crashed")

                # MONITORING-001: Emit event and record metrics
                self._emit_event(
                    EventType.INSTANCE_CRASHED,
                    instance_id=instance_id,
                    payload={
                        "heartbeat_timeout": self.config["heartbeat_timeout"]
                    }
                )
                self.metrics.record_instance_crashed()

                logger.info(f"Cleanup completed for crashed instance: {instance_id}")

            except (RuntimeError, ValueError, KeyError, IOError) as e:
                logger.error(f"Failed to cleanup crashed instance {instance_id}: {e}", exc_info=True)

    def _record_intervention(self, instance_id: str, intervention_type: str, phase: Optional[str] = None):
        """
        Record an intervention for history tracking (Phase 1 + Phase 4 enhancement).

        Args:
            instance_id: Instance that received intervention
            intervention_type: Type of intervention performed
            phase: Optional phase information (Phase 4)
        """
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "instance_id": instance_id,
            "intervention_type": intervention_type,
            "cycle": self.cycle_count
        }

        if phase:
            record["phase"] = phase

        self.intervention_history.append(record)

        # Keep only last 100 interventions
        self.intervention_history = self.intervention_history[-100:]

    def _spawn_instance(self, task_details: dict, retry_count: int = 0) -> Optional[dict]:
        """
        Spawn a detached subprocess for autonomous task execution.

        Uses cross-platform detachment (Windows: CREATE_NO_WINDOW, Unix: start_new_session)
        with proper resource management and comprehensive error handling.

        Args:
            task_details: Dict from DecisionEngine with:
                - task_id: str (e.g., "TASK-059")
                - priority: str (CRITICAL/HIGH/MEDIUM/LOW)
                - model: str (opus/sonnet/haiku)
                - description: str
            retry_count: Current retry attempt (0-2)

        Returns:
            {
                "success": True,
                "pid": int,
                "task_id": str,
                "instance_id": str,
                "spawned_at": str (ISO format),
                "command": list[str],
                "model": str,
                "logs": {"stdout": str, "stderr": str}
            }
            or None if spawn failed after max retries

        Raises:
            None - All exceptions caught and handled internally
        """
        # Validate task_details
        required_keys = ["task_id", "priority", "model"]
        if not all(key in task_details for key in required_keys):
            logger.error(f"Invalid task_details: missing required keys {required_keys}")
            return None

        task_id = task_details["task_id"]
        model = task_details["model"]
        priority = task_details["priority"]

        # Check retry limit
        if retry_count >= self.config.get("max_spawn_retries", 3):
            logger.error(f"Failed to spawn instance for {task_id} after {retry_count} retries")
            return None

        # Prepare log files
        log_dir = Path(".autonomous/logs")
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        instance_id = f"instance-{timestamp}"
        stdout_log = log_dir / f"{instance_id}_{task_id}.out"
        stderr_log = log_dir / f"{instance_id}_{task_id}.err"

        # Build command
        main_repo = Path.cwd()
        cmd = [
            sys.executable,
            '-m', 'scripts.spawn_orchestrator',
            task_id,
            '--model', model,
            '--priority', priority,
            '--description', task_details.get('description', f'Execute {task_id}')
        ]

        try:
            # Validate executable exists
            if not Path(sys.executable).exists():
                raise FileNotFoundError(f"Python executable not found: {sys.executable}")

            # Ensure log directory exists
            log_dir.mkdir(parents=True, exist_ok=True)

            # Spawn process with proper detachment and resource management
            with open(stdout_log, 'w', encoding='utf-8') as stdout_file, \
                 open(stderr_log, 'w', encoding='utf-8') as stderr_file:

                if sys.platform == 'win32':
                    # Windows: CREATE_NO_WINDOW + CREATE_NEW_PROCESS_GROUP (python-specialist recommendation)
                    creationflags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
                    process = subprocess.Popen(
                        cmd,
                        stdout=stdout_file,
                        stderr=stderr_file,
                        close_fds=True,  # Close inherited FDs (prevent leaks)
                        creationflags=creationflags,
                        cwd=str(main_repo)
                    )
                else:
                    # Unix: start_new_session for detachment (prevents zombies)
                    process = subprocess.Popen(
                        cmd,
                        stdout=stdout_file,
                        stderr=stderr_file,
                        start_new_session=True,  # Detach from terminal
                        close_fds=True,
                        cwd=str(main_repo)
                    )

            pid = process.pid
            spawn_time = datetime.utcnow()

            # Build spawn result
            spawn_result = {
                "success": True,
                "pid": pid,
                "task_id": task_id,
                "instance_id": instance_id,
                "spawned_at": spawn_time.isoformat(),
                "command": cmd,
                "model": model,
                "priority": priority,
                "logs": {
                    "stdout": str(stdout_log),
                    "stderr": str(stderr_log)
                }
            }

            # Update tracking
            self.spawned_instances[pid] = task_id
            self.last_spawn_time = spawn_time

            # Persist PID registry
            self._persist_pid_registry(spawn_result)

            # MONITORING-001: Emit event and record metrics
            self._emit_event(
                EventType.INSTANCE_SPAWNED,
                task_id=task_id,
                instance_id=instance_id,
                payload={
                    "pid": pid,
                    "model": model,
                    "priority": priority
                }
            )
            self.metrics.record_instance_spawned()

            logger.info(f"✅ Spawned instance for {task_id} (PID {pid}, model={model}, priority={priority})")
            logger.info(f"   Logs: stdout={stdout_log.name}, stderr={stderr_log.name}")

            return spawn_result

        except FileNotFoundError as e:
            logger.error(f"❌ Failed to spawn {task_id}: {e}")
            return self._retry_spawn(task_details, retry_count, "executable_not_found")

        except PermissionError as e:
            logger.error(f"❌ Permission denied spawning {task_id}: {e}")
            return self._retry_spawn(task_details, retry_count, "permission_denied")

        except OSError as e:
            # Resource exhaustion (too many open files, out of memory)
            logger.error(f"❌ OS error spawning {task_id}: {e}")
            return self._retry_spawn(task_details, retry_count, "resource_exhaustion")

        except subprocess.SubprocessError as e:
            # Generic subprocess failure
            logger.error(f"❌ Subprocess error spawning {task_id}: {e}")
            return self._retry_spawn(task_details, retry_count, "subprocess_error")

    def _retry_spawn(self, task_details: dict, retry_count: int, error_type: str) -> Optional[dict]:
        """
        Retry spawn with exponential backoff.

        Args:
            task_details: Task details dict
            retry_count: Current retry attempt
            error_type: Type of error that occurred

        Returns:
            Result from retry attempt or None
        """
        retry_count += 1
        if retry_count >= self.config.get("max_spawn_retries", 3):
            logger.error(f"Max retries exceeded for {task_details['task_id']} ({error_type})")
            return None

        # Exponential backoff: 2^retry_count seconds (2s, 4s, 8s)
        backoff_seconds = 2 ** retry_count
        logger.warning(f"Retrying spawn for {task_details['task_id']} in {backoff_seconds}s (attempt {retry_count + 1}/3)")
        time.sleep(backoff_seconds)

        # Recursive retry
        return self._spawn_instance(task_details, retry_count)

    def _persist_pid_registry(self, spawn_result: dict):
        """
        Persist spawned instance info to PID registry file.

        Args:
            spawn_result: Spawn result dict with pid, task_id, etc.
        """
        registry_path = Path(".autonomous/spawned_instances.json")
        registry_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Load existing registry
            if registry_path.exists():
                with open(registry_path, 'r', encoding='utf-8') as f:
                    registry = json.load(f)
            else:
                registry = {"instances": {}}

            # Add new instance
            registry["instances"][str(spawn_result["pid"])] = {
                "pid": spawn_result["pid"],
                "task_id": spawn_result["task_id"],
                "instance_id": spawn_result["instance_id"],
                "spawned_at": spawn_result["spawned_at"],
                "model": spawn_result["model"],
                "priority": spawn_result["priority"],
                "command": spawn_result["command"],
                "logs": spawn_result["logs"]
            }

            # Write updated registry
            with open(registry_path, 'w', encoding='utf-8') as f:
                json.dump(registry, f, indent=2)

            logger.debug(f"PID registry updated: {registry_path}")

        except (IOError, json.JSONDecodeError, json.JSONEncodeError, ValueError) as e:
            logger.error(f"Failed to persist PID registry: {e}")
            # Non-fatal - spawned instance still running

    def _snapshot_state(self):
        """
        Save orchestrator state to disk for crash recovery.

        Snapshots include:
        - Current cycle count and uptime
        - Spawned instances tracking
        - Heartbeat timestamps
        - Last spawn timestamp (for cooldown)
        - Intervention history
        """
        state_path = Path(".autonomous/orchestrator_state.json")
        state_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            state = {
                "timestamp": datetime.utcnow().isoformat(),
                "cycle_count": self.cycle_count,
                "uptime": self._format_duration(datetime.utcnow() - self.start_time),
                "spawned_instances": self.spawned_instances,
                "last_heartbeat": {k: v.isoformat() for k, v in self.last_heartbeat.items()},
                "last_progress": {k: v.isoformat() for k, v in self.last_progress.items()},
                "last_spawn_time": self.last_spawn_time.isoformat() if self.last_spawn_time else None,
                "intervention_history": self.intervention_history[-20:]  # Last 20 interventions
            }

            with open(state_path, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2)

            logger.debug(f"State snapshot saved to {state_path}")

        except (IOError, json.JSONEncodeError, ValueError) as e:
            logger.error(f"Failed to save state snapshot: {e}")

    def _format_duration(self, duration: timedelta) -> str:
        """
        Format timedelta as human-readable string.

        Args:
            duration: Time duration

        Returns:
            Formatted string (e.g., "2h 15m 30s")
        """
        total_seconds = int(duration.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60

        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"

    def _handle_shutdown(self, signum, frame):
        """
        Handle shutdown signals gracefully.

        Args:
            signum: Signal number
            frame: Current stack frame
        """
        logger.info(f"Shutdown signal received: {signum}")
        self.running = False

    def _shutdown(self):
        """
        Clean shutdown of orchestrator.

        Steps:
        1. Final state snapshot
        2. Broadcast shutdown message
        3. Log final statistics
        """
        logger.info("=" * 80)
        logger.info("ORCHESTRATOR SHUTTING DOWN")
        logger.info(f"Total cycles: {self.cycle_count}")
        logger.info(f"Total uptime: {self._format_duration(datetime.utcnow() - self.start_time)}")
        logger.info(f"Interventions sent: {len(self.intervention_history)}")
        logger.info("=" * 80)

        # MONITORING-001: Emit shutdown event with final metrics
        self._emit_event(
            EventType.ORCHESTRATOR_SHUTDOWN,
            payload={
                "total_cycles": self.cycle_count,
                "uptime": self._format_duration(datetime.utcnow() - self.start_time),
                "tasks_completed": self.metrics.tasks_completed,
                "tasks_failed": self.metrics.tasks_failed,
                "final_completion_rate": self.metrics.get_completion_rate(),
                "total_interventions": self.metrics.interventions_sent,
                "instances_spawned": self.metrics.instances_spawned,
                "instances_crashed": self.metrics.instances_crashed
            }
        )

        # Final state snapshot
        try:
            self._snapshot_state()
        except (IOError, json.JSONEncodeError, ValueError) as e:
            logger.error(f"Failed to save final snapshot: {e}")

        # Send shutdown message
        try:
            self.message_queue.broadcast(
                "orchestrator",
                MessageType.STATUS_UPDATE,  # Use STATUS_UPDATE for orchestrator shutdown
                {
                    "type": "orchestrator_shutdown",
                    "cycle_count": self.cycle_count,
                    "uptime": self._format_duration(datetime.utcnow() - self.start_time),
                    "shutdown_time": datetime.utcnow().isoformat()
                }
            )
            logger.info("Shutdown broadcast sent")
        except (RuntimeError, ValueError, IOError) as e:
            logger.error(f"Failed to broadcast shutdown: {e}")

        # Shutdown registry (stops heartbeat)
        try:
            self.registry.shutdown()
            logger.info("Registry shutdown complete")
        except (RuntimeError, ValueError) as e:
            logger.error(f"Failed to shutdown registry: {e}")

        logger.info("Orchestrator shutdown complete")


def main():
    """
    Entry point for orchestrator loop.

    Configures logging and starts the orchestrator.
    """
    # Configure logging
    log_dir = Path(".autonomous/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'orchestrator.log'),
            logging.StreamHandler()
        ]
    )

    logger.info("Starting orchestrator...")

    # Create and run orchestrator
    orchestrator = OrchestratorLoop()
    orchestrator.run()


if __name__ == "__main__":
    main()
