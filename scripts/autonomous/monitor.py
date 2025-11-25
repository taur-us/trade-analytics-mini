#!/usr/bin/env python3
"""
Orchestrator Monitoring Interface for Querying Instance/Task Status

Provides real-time visibility into all autonomous workflow activity:
- Active instances and their current work
- Task assignments and progress
- Inter-instance messages
- System health metrics

This interface enables the orchestrator to autonomously monitor all instances
and make informed decisions about task allocation and resource management.
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

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


class OrchestratorMonitor:
    """
    Query interface for orchestrator to see all instance activity.

    Provides real-time dashboards and status queries for autonomous workflow monitoring.
    """

    def __init__(self, main_repo: Optional[Path] = None):
        """
        Initialize monitoring interface.

        Args:
            main_repo: Path to main repository (defaults to current working directory)
        """
        if main_repo is None:
            # Repo-agnostic: auto-detect current repo or use environment variable
            main_repo = Path(os.getenv('AUTONOMOUS_REPO_PATH', Path.cwd())).resolve()

        self.main_repo = main_repo
        self.lock_manager = DistributedLockManager(main_repo)
        self.registry = InstanceRegistry(self.lock_manager)
        self.coordinator = TaskCoordinator(self.lock_manager)
        self.queue = MessageQueue(self.lock_manager)

    def get_instance_dashboard(self) -> Dict:
        """
        Returns complete dashboard of all active work.

        Returns:
            Dictionary containing:
            - active_instances: List of active instances with details
            - claimed_tasks: Tasks currently being worked on
            - recent_messages: Last 10 messages between instances
            - stale_instances: Instances with no heartbeat >60s
            - available_tasks: Count of tasks in backlog
            - resource_usage: Instance limit and current count
        """
        dashboard = {
            "active_instances": [],
            "claimed_tasks": {},
            "recent_messages": [],
            "stale_instances": [],
            "available_tasks": 0,
            "resource_usage": {
                "max_instances": 3,
                "current_instances": 0,
                "available_slots": 3
            },
            "timestamp": datetime.utcnow().isoformat()
        }

        # Read instances registry
        registry_path = self.lock_manager.main_repo / ".autonomous" / "instances.json"
        if registry_path.exists():
            with self.lock_manager.acquire_lock("instances"):
                with open(registry_path, 'r', encoding='utf-8') as f:
                    registry_data = json.load(f)

                instances = registry_data.get("instances", {})
                limits = registry_data.get("resource_limits", {})

                now = datetime.utcnow()

                for instance_id, instance in instances.items():
                    last_heartbeat_str = instance.get("last_heartbeat")
                    last_heartbeat = datetime.fromisoformat(last_heartbeat_str)
                    time_since_heartbeat = now - last_heartbeat
                    seconds_ago = int(time_since_heartbeat.total_seconds())

                    instance_info = {
                        "id": instance_id,
                        "task": instance.get("task_id"),
                        "status": instance.get("status"),
                        "phase": instance.get("current_phase"),
                        "last_heartbeat": f"{seconds_ago}s ago",
                        "location": "main-repo" if "main" in str(instance.get("worktree_path", "")) else "worktree",
                        "start_time": instance.get("start_time"),
                        "pid": instance.get("pid"),
                        "hostname": instance.get("hostname")
                    }

                    # Categorize as active or stale
                    if instance["status"] in [InstanceStatus.STALE.value, InstanceStatus.COMPLETED.value, InstanceStatus.FAILED.value]:
                        if instance["status"] == InstanceStatus.STALE.value:
                            dashboard["stale_instances"].append(instance_info)
                    else:
                        if seconds_ago > 60:
                            dashboard["stale_instances"].append(instance_info)
                        else:
                            dashboard["active_instances"].append(instance_info)

                # Resource usage
                dashboard["resource_usage"] = {
                    "max_instances": limits.get("max_instances", 3),
                    "current_instances": limits.get("current_count", 0),
                    "available_slots": max(0, limits.get("max_instances", 3) - limits.get("current_count", 0))
                }

        # Read task queue for claimed tasks
        task_queue_path = self.lock_manager.main_repo / "tasks" / "task_queue.json"
        if task_queue_path.exists():
            with self.lock_manager.acquire_lock("task_queue"):
                with open(task_queue_path, 'r', encoding='utf-8') as f:
                    task_data = json.load(f)

                in_progress = task_data.get("in_progress", [])
                backlog = task_data.get("backlog", [])

                for task in in_progress:
                    dashboard["claimed_tasks"][task["id"]] = {
                        "claimed_by": task.get("assigned_to"),
                        "claimed_at": task.get("assigned_at"),
                        "status": task.get("status"),
                        "priority": task.get("priority")
                    }

                dashboard["available_tasks"] = len([t for t in backlog if t.get("status") == "READY"])

        # Read recent messages
        messages_path = self.lock_manager.main_repo / ".autonomous" / "messages.json"
        if messages_path.exists():
            with self.lock_manager.acquire_lock("messages"):
                with open(messages_path, 'r', encoding='utf-8') as f:
                    messages_data = json.load(f)

                messages = messages_data.get("messages", [])
                # Get last 10 messages
                for msg in messages[-10:]:
                    dashboard["recent_messages"].append({
                        "from": msg.get("from"),
                        "to": msg.get("to", "broadcast"),
                        "type": msg.get("type"),
                        "payload": msg.get("payload"),
                        "timestamp": msg.get("timestamp")
                    })

        return dashboard

    def get_instance_status(self, instance_id: str) -> Optional[Dict]:
        """
        Get detailed status for specific instance.

        Args:
            instance_id: Instance ID to query (e.g., "instance-20251112-220000")

        Returns:
            Detailed instance information or None if not found
        """
        registry_path = self.lock_manager.main_repo / ".autonomous" / "instances.json"
        if not registry_path.exists():
            return None

        with self.lock_manager.acquire_lock("instances"):
            with open(registry_path, 'r', encoding='utf-8') as f:
                registry_data = json.load(f)

            instances = registry_data.get("instances", {})
            if instance_id not in instances:
                return None

            instance = instances[instance_id]

            # Calculate heartbeat freshness
            last_heartbeat = datetime.fromisoformat(instance["last_heartbeat"])
            now = datetime.utcnow()
            time_since_heartbeat = now - last_heartbeat

            return {
                "instance_id": instance_id,
                "session_id": instance.get("session_id"),
                "task_id": instance.get("task_id"),
                "status": instance.get("status"),
                "current_phase": instance.get("current_phase"),
                "start_time": instance.get("start_time"),
                "last_heartbeat": instance.get("last_heartbeat"),
                "heartbeat_seconds_ago": int(time_since_heartbeat.total_seconds()),
                "is_stale": time_since_heartbeat.total_seconds() > 60,
                "worktree_path": instance.get("worktree_path"),
                "gates_passed": instance.get("gates_passed", []),
                "pid": instance.get("pid"),
                "hostname": instance.get("hostname")
            }

    def get_task_status(self, task_id: str) -> Optional[Dict]:
        """
        Get status of specific task (who claimed it, when, progress).

        Args:
            task_id: Task ID to query (e.g., "TASK-058")

        Returns:
            Task status information or None if not found
        """
        return self.coordinator.get_task_status(task_id)

    def get_system_health(self) -> Dict:
        """
        Overall system health check.

        Returns:
            System health metrics including:
            - total_instances: Total instances registered
            - active_instances: Currently active instances
            - stale_instances: Instances with no heartbeat
            - tasks_in_progress: Tasks being worked on
            - tasks_available: Tasks ready to claim
            - resource_utilization: Percentage of instance slots used
            - message_queue_size: Number of messages in queue
            - health_status: HEALTHY, DEGRADED, or UNHEALTHY
        """
        health = {
            "total_instances": 0,
            "active_instances": 0,
            "stale_instances": 0,
            "tasks_in_progress": 0,
            "tasks_available": 0,
            "resource_utilization": 0.0,
            "message_queue_size": 0,
            "health_status": "HEALTHY",
            "issues": [],
            "timestamp": datetime.utcnow().isoformat()
        }

        # Check instances
        registry_path = self.lock_manager.main_repo / ".autonomous" / "instances.json"
        if registry_path.exists():
            with self.lock_manager.acquire_lock("instances"):
                with open(registry_path, 'r', encoding='utf-8') as f:
                    registry_data = json.load(f)

                instances = registry_data.get("instances", {})
                limits = registry_data.get("resource_limits", {})
                now = datetime.utcnow()

                health["total_instances"] = len(instances)

                for instance in instances.values():
                    last_heartbeat = datetime.fromisoformat(instance["last_heartbeat"])
                    time_since_heartbeat = now - last_heartbeat

                    if instance["status"] in [InstanceStatus.STALE.value]:
                        health["stale_instances"] += 1
                    elif instance["status"] not in [InstanceStatus.COMPLETED.value, InstanceStatus.FAILED.value]:
                        if time_since_heartbeat.total_seconds() > 60:
                            health["stale_instances"] += 1
                            health["issues"].append(f"Instance {instance['instance_id']} appears stale (no heartbeat for {int(time_since_heartbeat.total_seconds())}s)")
                        else:
                            health["active_instances"] += 1

                max_instances = limits.get("max_instances", 3)
                current_count = limits.get("current_count", 0)
                health["resource_utilization"] = (current_count / max_instances * 100) if max_instances > 0 else 0

        # Check tasks
        task_queue_path = self.lock_manager.main_repo / "tasks" / "task_queue.json"
        if task_queue_path.exists():
            with self.lock_manager.acquire_lock("task_queue"):
                with open(task_queue_path, 'r', encoding='utf-8') as f:
                    task_data = json.load(f)

                health["tasks_in_progress"] = len(task_data.get("in_progress", []))
                health["tasks_available"] = len([t for t in task_data.get("backlog", []) if t.get("status") == "READY"])

        # Check messages
        messages_path = self.lock_manager.main_repo / ".autonomous" / "messages.json"
        if messages_path.exists():
            with self.lock_manager.acquire_lock("messages"):
                with open(messages_path, 'r', encoding='utf-8') as f:
                    messages_data = json.load(f)

                health["message_queue_size"] = len(messages_data.get("messages", []))

        # Determine health status
        if health["stale_instances"] > 0:
            health["health_status"] = "DEGRADED"
            health["issues"].append(f"{health['stale_instances']} stale instance(s) detected")

        if health["stale_instances"] >= 2:
            health["health_status"] = "UNHEALTHY"

        if health["resource_utilization"] >= 100:
            health["health_status"] = "DEGRADED" if health["health_status"] == "HEALTHY" else health["health_status"]
            health["issues"].append("Instance limit reached - tasks may be queued")

        return health


def format_dashboard(dashboard: Dict) -> str:
    """
    Format dashboard as readable text.

    Args:
        dashboard: Dashboard dict from get_instance_dashboard()

    Returns:
        Formatted string for display
    """
    lines = []
    lines.append("=" * 80)
    lines.append("ORCHESTRATOR DASHBOARD")
    lines.append("=" * 80)
    lines.append(f"Timestamp: {dashboard['timestamp']}")
    lines.append("")

    # Resource usage
    usage = dashboard["resource_usage"]
    lines.append(f"Resource Usage: {usage['current_instances']}/{usage['max_instances']} instances ({usage['available_slots']} available)")
    lines.append(f"Available Tasks: {dashboard['available_tasks']}")
    lines.append("")

    # Active instances
    lines.append("Active Instances:")
    if dashboard["active_instances"]:
        for instance in dashboard["active_instances"]:
            lines.append(f"  - {instance['id']}")
            lines.append(f"    Task: {instance['task']}")
            lines.append(f"    Status: {instance['status']} / {instance['phase']}")
            lines.append(f"    Heartbeat: {instance['last_heartbeat']}")
            lines.append(f"    Location: {instance['location']}")
            lines.append("")
    else:
        lines.append("  (none)")
        lines.append("")

    # Claimed tasks
    lines.append("Claimed Tasks:")
    if dashboard["claimed_tasks"]:
        for task_id, task_info in dashboard["claimed_tasks"].items():
            lines.append(f"  - {task_id}: claimed by {task_info['claimed_by']} ({task_info['status']})")
    else:
        lines.append("  (none)")
    lines.append("")

    # Stale instances
    if dashboard["stale_instances"]:
        lines.append("Stale Instances:")
        for instance in dashboard["stale_instances"]:
            lines.append(f"  - {instance['id']} (heartbeat: {instance['last_heartbeat']})")
        lines.append("")

    # Recent messages
    lines.append("Recent Messages (last 10):")
    if dashboard["recent_messages"]:
        for msg in dashboard["recent_messages"][-10:]:
            lines.append(f"  - [{msg['timestamp']}] {msg['from']} -> {msg['to']}: {msg['type']}")
    else:
        lines.append("  (none)")

    lines.append("=" * 80)
    return "\n".join(lines)


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Orchestrator monitoring interface")
    parser.add_argument("--dashboard", action="store_true", help="Show complete dashboard")
    parser.add_argument("--instance", type=str, metavar="ID", help="Show specific instance status")
    parser.add_argument("--task", type=str, metavar="ID", help="Show specific task status")
    parser.add_argument("--health", action="store_true", help="Show system health")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    monitor = OrchestratorMonitor()

    if args.dashboard:
        dashboard = monitor.get_instance_dashboard()
        if args.json:
            print(json.dumps(dashboard, indent=2))
        else:
            print(format_dashboard(dashboard))

    elif args.instance:
        status = monitor.get_instance_status(args.instance)
        if status:
            print(json.dumps(status, indent=2))
        else:
            print(f"Instance not found: {args.instance}")
            sys.exit(1)

    elif args.task:
        status = monitor.get_task_status(args.task)
        if status:
            print(json.dumps(status, indent=2))
        else:
            print(f"Task not found: {args.task}")
            sys.exit(1)

    elif args.health:
        health = monitor.get_system_health()
        if args.json:
            print(json.dumps(health, indent=2))
        else:
            print(f"System Health: {health['health_status']}")
            print(f"Active Instances: {health['active_instances']}/{health['total_instances']}")
            print(f"Stale Instances: {health['stale_instances']}")
            print(f"Tasks In Progress: {health['tasks_in_progress']}")
            print(f"Tasks Available: {health['tasks_available']}")
            print(f"Resource Utilization: {health['resource_utilization']:.1f}%")
            if health["issues"]:
                print("\nIssues:")
                for issue in health["issues"]:
                    print(f"  - {issue}")

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
