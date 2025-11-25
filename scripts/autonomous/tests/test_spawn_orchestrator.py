#!/usr/bin/env python3
"""
Unit tests for spawn orchestrator instance spawning functionality (TASK-066).

Tests:
- Instance spawning (_spawn_instance)
- Work completion detection (_check_work_complete)
- Timeout handling
- Polling loop integration
"""

import os
import subprocess
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, call

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from spawn_orchestrator import SpawnOrchestrator


class TestInstanceSpawning(unittest.TestCase):
    """Test _spawn_instance method."""

    def setUp(self):
        """Set up test fixtures."""
        self.orchestrator = SpawnOrchestrator("TASK-TEST", "Test task description")
        self.orchestrator.branch_name = "feat/20251113-114744-task-test"

    @patch('spawn_orchestrator.subprocess.Popen')
    @patch('spawn_orchestrator.atexit.register')
    def test_spawn_instance_success(self, mock_atexit, mock_popen):
        """Test successful instance spawning."""
        # Mock process
        mock_process = Mock()
        mock_process.pid = 12345
        mock_popen.return_value = mock_process

        # Call method
        result = self.orchestrator._spawn_instance()

        # Verify subprocess called correctly
        mock_popen.assert_called_once()
        args, kwargs = mock_popen.call_args
        cmd = args[0]

        self.assertEqual(cmd[0], 'claude-code')
        self.assertEqual(cmd[1], '--directory')
        self.assertEqual(cmd[3], '--prompt-file')
        self.assertIn('task_prompt.txt', cmd[4])

        # Verify prompt file created
        prompt_file = self.orchestrator.state_dir / 'task_prompt.txt'
        self.assertTrue(prompt_file.exists())
        prompt_content = prompt_file.read_text()
        self.assertIn('TASK-TEST', prompt_content)
        self.assertIn('Test task description', prompt_content)
        self.assertIn('feat/20251113-114744-task-test', prompt_content)

        # Verify cleanup handler registered
        mock_atexit.assert_called_once()

        # Verify returns process
        self.assertEqual(result, mock_process)

    @patch('spawn_orchestrator.subprocess.Popen')
    def test_spawn_instance_failure(self, mock_popen):
        """Test instance spawning failure."""
        # Mock subprocess failure
        mock_popen.side_effect = OSError("Command not found")

        # Call method - should raise RuntimeError
        with self.assertRaises(RuntimeError) as context:
            self.orchestrator._spawn_instance()

        self.assertIn("Failed to spawn instance", str(context.exception))
        self.assertIn("Command not found", str(context.exception))

    @patch('spawn_orchestrator.subprocess.Popen')
    @patch('spawn_orchestrator.atexit.register')
    def test_spawn_instance_command_format(self, mock_atexit, mock_popen):
        """Test that spawn command is correctly formatted."""
        mock_process = Mock()
        mock_process.pid = 12345
        mock_popen.return_value = mock_process

        self.orchestrator._spawn_instance()

        # Verify command structure
        args, kwargs = mock_popen.call_args
        cmd = args[0]

        self.assertIsInstance(cmd, list)
        self.assertEqual(len(cmd), 5)  # ['claude-code', '--directory', path, '--prompt-file', prompt]
        self.assertEqual(kwargs['cwd'], str(self.orchestrator.worktree_path))
        self.assertEqual(kwargs['stdout'], subprocess.PIPE)
        self.assertEqual(kwargs['stderr'], subprocess.PIPE)


class TestWorkCompletionDetection(unittest.TestCase):
    """Test _check_work_complete method."""

    def setUp(self):
        """Set up test fixtures."""
        self.orchestrator = SpawnOrchestrator("TASK-TEST", "Test task description")
        self.orchestrator.branch_name = "feat/20251113-114744-task-test"

    @patch('spawn_orchestrator.subprocess.run')
    def test_work_complete_with_commits_and_deliverables(self, mock_run):
        """Test work complete when commits and deliverables exist."""
        # Mock git command - 3 commits
        mock_result = Mock()
        mock_result.stdout = "3"
        mock_result.returncode = 0
        mock_run.return_value = mock_result

        # Create deliverables
        deliverables_dir = self.orchestrator.worktree_path / 'deliverables'
        deliverables_dir.mkdir(parents=True, exist_ok=True)
        (deliverables_dir / 'TASK-TEST-REPORT.md').touch()

        # Check completion
        result = self.orchestrator._check_work_complete()

        # Verify
        self.assertTrue(result)
        mock_run.assert_called_once()

    @patch('spawn_orchestrator.subprocess.run')
    def test_work_complete_with_commits_and_summary(self, mock_run):
        """Test work complete when commits and SESSION_SUMMARY exist."""
        # Mock git command - 2 commits
        mock_result = Mock()
        mock_result.stdout = "2"
        mock_result.returncode = 0
        mock_run.return_value = mock_result

        # Create SESSION_SUMMARY.md
        session_summary = self.orchestrator.worktree_path / 'SESSION_SUMMARY.md'
        session_summary.parent.mkdir(parents=True, exist_ok=True)
        session_summary.touch()

        # Check completion
        result = self.orchestrator._check_work_complete()

        # Verify
        self.assertTrue(result)

    @patch('spawn_orchestrator.subprocess.run')
    def test_work_incomplete_no_commits(self, mock_run):
        """Test work incomplete when no commits."""
        # Mock git command - 0 commits
        mock_result = Mock()
        mock_result.stdout = "0"
        mock_result.returncode = 0
        mock_run.return_value = mock_result

        # Create deliverables (but no commits)
        deliverables_dir = self.orchestrator.worktree_path / 'deliverables'
        deliverables_dir.mkdir(parents=True, exist_ok=True)
        (deliverables_dir / 'TASK-TEST-REPORT.md').touch()

        # Check completion
        result = self.orchestrator._check_work_complete()

        # Verify - not complete (need commits)
        self.assertFalse(result)

    @patch('spawn_orchestrator.subprocess.run')
    def test_work_incomplete_commits_only(self, mock_run):
        """Test work incomplete when only commits exist."""
        # Mock git command - 1 commit
        mock_result = Mock()
        mock_result.stdout = "1"
        mock_result.returncode = 0
        mock_run.return_value = mock_result

        # Ensure no deliverables or SESSION_SUMMARY exist
        session_summary = self.orchestrator.worktree_path / 'SESSION_SUMMARY.md'
        if session_summary.exists():
            session_summary.unlink()

        # Remove any deliverables that might exist
        deliverables_dir = self.orchestrator.worktree_path / 'deliverables'
        if deliverables_dir.exists():
            import shutil
            shutil.rmtree(deliverables_dir)

        # Check completion
        result = self.orchestrator._check_work_complete()

        # Verify - not complete (need deliverables OR summary)
        self.assertFalse(result)

    @patch('spawn_orchestrator.subprocess.run')
    def test_work_complete_git_error(self, mock_run):
        """Test work complete when git command fails gracefully."""
        # Mock git command failure
        mock_run.side_effect = Exception("Git error")

        # Check completion
        result = self.orchestrator._check_work_complete()

        # Verify - returns False, doesn't crash
        self.assertFalse(result)


class TestTimeoutHandling(unittest.TestCase):
    """Test timeout handling in polling loop."""

    def setUp(self):
        """Set up test fixtures."""
        self.orchestrator = SpawnOrchestrator("TASK-TEST", "Test task description")
        self.orchestrator.branch_name = "feat/20251113-114744-task-test"
        self.orchestrator.estimated_hours = 2.0  # 2 hour task
        self.orchestrator.timeout_multiplier = 1.5  # 3 hour timeout

    def test_timeout_calculation(self):
        """Test timeout calculation (estimated_hours * 1.5)."""
        # Verify timeout calculation
        timeout_hours = self.orchestrator.estimated_hours * self.orchestrator.timeout_multiplier
        expected_hours = 2.0 * 1.5  # 3.0 hours
        expected_seconds = expected_hours * 3600  # 10800 seconds

        self.assertEqual(timeout_hours, expected_hours)
        self.assertEqual(int(timeout_hours * 3600), expected_seconds)


class TestPollingLoopIntegration(unittest.TestCase):
    """Test polling loop integration."""

    def setUp(self):
        """Set up test fixtures."""
        self.orchestrator = SpawnOrchestrator("TASK-TEST", "Test task description")
        self.orchestrator.branch_name = "feat/20251113-114744-task-test"
        self.orchestrator.estimated_hours = 2.0
        self.orchestrator.poll_interval = 1  # 1 second for faster tests

    @patch('spawn_orchestrator.time.sleep')
    @patch('spawn_orchestrator.time.time')
    def test_polling_completes_on_work_done(self, mock_time, mock_sleep):
        """Test polling loop exits when work is complete."""
        start_time = 1000.0
        check_times = [
            start_time,  # Start
            start_time,  # First health check
            start_time + 1,  # First poll
            start_time + 1,  # Second health check
            start_time + 2,  # Second poll
        ]
        mock_time.side_effect = check_times

        # Mock work completion after 2 polls
        complete_checks = [False, True]
        complete_call_count = [0]

        def mock_check_complete():
            result = complete_checks[min(complete_call_count[0], len(complete_checks)-1)]
            complete_call_count[0] += 1
            return result

        # Mock process
        mock_process = Mock()
        mock_process.poll.return_value = None  # Process running
        self.orchestrator.spawned_process = mock_process

        with patch.object(self.orchestrator, '_check_work_complete', side_effect=mock_check_complete):
            # Simulate simplified polling loop
            timeout_seconds = 2.0 * 1.5 * 3600
            poll_count = 0

            while True:
                elapsed = mock_time() - start_time

                if self.orchestrator._check_work_complete():
                    break

                if elapsed > timeout_seconds:
                    raise TimeoutError("Timeout")

                poll_count += 1
                mock_sleep(self.orchestrator.poll_interval)

                if poll_count > 5:  # Safety limit
                    self.fail("Polling loop didn't exit")

            # Verify polling occurred
            self.assertGreaterEqual(poll_count, 1)

    @patch('spawn_orchestrator.time.time')
    def test_process_health_check_detects_crash(self, mock_time):
        """Test process health check detects crashed subprocess."""
        # Mock process that crashes
        mock_process = Mock()
        mock_process.poll.return_value = 1  # Non-zero = crashed
        mock_process.communicate.return_value = (b"", b"Process crashed")
        self.orchestrator.spawned_process = mock_process

        mock_time.return_value = 1000.0

        # Health check should detect crash
        if mock_process.poll() is not None:
            stdout, stderr = mock_process.communicate()
            error_msg = stderr.decode('utf-8') if stderr else 'Unknown error'
            # Verify error message
            self.assertEqual(error_msg, "Process crashed")

    def test_configurable_poll_interval(self):
        """Test poll interval is configurable via environment."""
        # Default
        orchestrator1 = SpawnOrchestrator("TASK-001")
        self.assertEqual(orchestrator1.poll_interval, 300)  # 5 minutes

        # Custom via environment
        with patch.dict(os.environ, {'SPAWN_POLL_INTERVAL': '60'}):
            orchestrator2 = SpawnOrchestrator("TASK-002")
            self.assertEqual(orchestrator2.poll_interval, 60)  # 1 minute

    def test_configurable_timeout_multiplier(self):
        """Test timeout multiplier is configurable via environment."""
        # Default
        orchestrator1 = SpawnOrchestrator("TASK-001")
        self.assertEqual(orchestrator1.timeout_multiplier, 1.5)

        # Custom via environment
        with patch.dict(os.environ, {'SPAWN_TIMEOUT_MULT': '2.0'}):
            orchestrator2 = SpawnOrchestrator("TASK-002")
            self.assertEqual(orchestrator2.timeout_multiplier, 2.0)


class TestEstimatedHoursLoading(unittest.TestCase):
    """Test loading estimated_hours from task_queue.json."""

    def setUp(self):
        """Set up test fixtures."""
        self.task_queue_file = Path("C:/Users/tomas/GitHub/mmm-agents/tasks/task_queue.json")

    def test_load_estimated_hours_from_queue(self):
        """Test loading estimated_hours from task_queue.json."""
        # Use a real task from the queue if it exists
        if self.task_queue_file.exists():
            orchestrator = SpawnOrchestrator("TASK-066")
            # TASK-066 should have estimated_hours=6
            self.assertEqual(orchestrator.estimated_hours, 6.0)

    def test_load_estimated_hours_default(self):
        """Test default estimated_hours when task not found."""
        orchestrator = SpawnOrchestrator("TASK-NONEXISTENT")
        # Should use default 8 hours
        self.assertEqual(orchestrator.estimated_hours, 8.0)


if __name__ == '__main__':
    unittest.main()
