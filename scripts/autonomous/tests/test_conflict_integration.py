#!/usr/bin/env python3
"""
Integration tests for CONFLICT-002: ConflictResolver integration with SpawnOrchestrator.

Tests:
- Conflict detection in _phase_merge()
- ConflictResolver invocation on conflict detection
- Successful resolution continues workflow
- Failed resolution triggers gate skip
- Gate failure recording
- Human review escalation
"""

import json
import os
import subprocess
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, call

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from spawn_orchestrator import SpawnOrchestrator, Phase, GateFailure


class TestMergeConflictDetection(unittest.TestCase):
    """Test _is_merge_conflict_error method."""

    def setUp(self):
        """Set up test fixtures."""
        with patch.object(SpawnOrchestrator, '__init__', lambda x, y, z=None: None):
            self.orchestrator = SpawnOrchestrator("TASK-TEST", "Test task")
            self.orchestrator.task_id = "TASK-TEST"
            self.orchestrator._is_merge_conflict_error = SpawnOrchestrator._is_merge_conflict_error.__get__(
                self.orchestrator, SpawnOrchestrator
            )

    def test_detects_merge_conflict_message(self):
        """Test detection of 'merge conflict' in error message."""
        error = "error: merge conflict in file.py"
        self.assertTrue(self.orchestrator._is_merge_conflict_error(error))

    def test_detects_CONFLICT_keyword(self):
        """Test detection of 'CONFLICT' keyword."""
        error = "CONFLICT (content): Merge conflict in src/main.py"
        self.assertTrue(self.orchestrator._is_merge_conflict_error(error))

    def test_detects_automatic_merge_failed(self):
        """Test detection of 'Automatic merge failed' message."""
        error = "Automatic merge failed; fix conflicts and then commit the result."
        self.assertTrue(self.orchestrator._is_merge_conflict_error(error))

    def test_detects_fix_conflicts(self):
        """Test detection of 'fix conflicts' message."""
        error = "Please fix conflicts and then commit the result."
        self.assertTrue(self.orchestrator._is_merge_conflict_error(error))

    def test_detects_unmerged_files(self):
        """Test detection of 'unmerged files' message."""
        error = "Pull is not possible because you have unmerged files."
        self.assertTrue(self.orchestrator._is_merge_conflict_error(error))

    def test_does_not_detect_unrelated_error(self):
        """Test non-conflict errors are not detected."""
        error = "fatal: remote origin already exists."
        self.assertFalse(self.orchestrator._is_merge_conflict_error(error))

    def test_does_not_detect_network_error(self):
        """Test network errors are not detected as conflicts."""
        error = "fatal: unable to access 'https://github.com/...': Could not resolve host"
        self.assertFalse(self.orchestrator._is_merge_conflict_error(error))

    def test_case_insensitive_detection(self):
        """Test conflict detection is case-insensitive."""
        error = "MERGE CONFLICT in File.Py"
        self.assertTrue(self.orchestrator._is_merge_conflict_error(error))


class TestConflictConfigLoading(unittest.TestCase):
    """Test _load_conflict_config method."""

    def setUp(self):
        """Set up test fixtures with temp directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.main_repo = Path(self.temp_dir)
        self.autonomous_dir = self.main_repo / ".autonomous"
        self.autonomous_dir.mkdir(parents=True)

        with patch.object(SpawnOrchestrator, '__init__', lambda x, y, z=None: None):
            self.orchestrator = SpawnOrchestrator("TASK-TEST", "Test task")
            self.orchestrator.main_repo = self.main_repo
            self.orchestrator._load_conflict_config = SpawnOrchestrator._load_conflict_config.__get__(
                self.orchestrator, SpawnOrchestrator
            )

    def tearDown(self):
        """Clean up temp directory."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_default_config_when_no_file(self):
        """Test default config when orchestrator_config.json doesn't exist."""
        config = self.orchestrator._load_conflict_config()

        self.assertEqual(config['max_resolution_attempts'], 3)
        self.assertTrue(config['conflict_resolution_enabled'])

    def test_loads_max_resolution_attempts(self):
        """Test loading max_resolution_attempts from config file."""
        config_file = self.autonomous_dir / "orchestrator_config.json"
        config_file.write_text(json.dumps({
            'max_resolution_attempts': 5
        }))

        config = self.orchestrator._load_conflict_config()
        self.assertEqual(config['max_resolution_attempts'], 5)

    def test_loads_conflict_resolution_section(self):
        """Test loading conflict_resolution section from config."""
        config_file = self.autonomous_dir / "orchestrator_config.json"
        config_file.write_text(json.dumps({
            'conflict_resolution': {
                'enabled': False,
                'max_attempts': 2,
                'custom_setting': 'value'
            }
        }))

        config = self.orchestrator._load_conflict_config()
        self.assertFalse(config.get('enabled', True))
        self.assertEqual(config.get('max_attempts'), 2)
        self.assertEqual(config.get('custom_setting'), 'value')

    def test_handles_invalid_json(self):
        """Test graceful handling of invalid JSON config file."""
        config_file = self.autonomous_dir / "orchestrator_config.json"
        config_file.write_text("not valid json")

        # Should return defaults without crashing
        config = self.orchestrator._load_conflict_config()
        self.assertEqual(config['max_resolution_attempts'], 3)


class TestHandleMergeConflicts(unittest.TestCase):
    """Test _handle_merge_conflicts method."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

        with patch.object(SpawnOrchestrator, '__init__', lambda x, y, z=None: None):
            self.orchestrator = SpawnOrchestrator("TASK-TEST", "Test task")
            self.orchestrator.task_id = "TASK-TEST"
            self.orchestrator.main_repo = Path(self.temp_dir)
            self.orchestrator.worktree_path = Path(self.temp_dir) / "worktree"
            self.orchestrator.state_dir = Path(self.temp_dir) / ".autonomous"
            self.orchestrator.session_id = "test-session"
            self.orchestrator.state_dir.mkdir(parents=True)
            self.orchestrator.worktree_path.mkdir(parents=True)

    def tearDown(self):
        """Clean up temp directory."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('spawn_orchestrator.ConflictResolver')
    def test_returns_true_when_no_conflicts(self, mock_resolver_class):
        """Test returns True when no conflicts detected."""
        # Mock resolver returning clean status
        mock_resolver = Mock()
        mock_resolver.resolve_all.return_value = {
            'status': 'clean',
            'conflicts_found': 0,
            'resolved': 0,
            'failed': 0,
            'results': []
        }
        mock_resolver_class.return_value = mock_resolver

        result = self.orchestrator._handle_merge_conflicts()

        self.assertTrue(result)
        mock_resolver.resolve_all.assert_called_once()

    @patch('spawn_orchestrator.ConflictResolver')
    @patch.object(SpawnOrchestrator, '_commit_conflict_resolutions')
    def test_returns_true_and_commits_when_resolved(self, mock_commit, mock_resolver_class):
        """Test returns True and commits when conflicts resolved."""
        # Mock resolver returning resolved status
        mock_resolver = Mock()
        mock_resolver.resolve_all.return_value = {
            'status': 'resolved',
            'conflicts_found': 2,
            'resolved': 2,
            'failed': 0,
            'results': [
                {'file_path': 'file1.py', 'resolved': True, 'strategy_used': 'prefer_ours'},
                {'file_path': 'file2.py', 'resolved': True, 'strategy_used': 'combine'}
            ]
        }
        mock_resolver_class.return_value = mock_resolver

        result = self.orchestrator._handle_merge_conflicts()

        self.assertTrue(result)
        mock_commit.assert_called_once()

    @patch('spawn_orchestrator.ConflictResolver')
    @patch('spawn_orchestrator.HumanReviewEscalator')
    def test_returns_false_and_escalates_when_failed(self, mock_escalator_class, mock_resolver_class):
        """Test returns False and escalates when resolution fails."""
        # Mock resolver returning partial/failed status
        mock_resolver = Mock()
        mock_resolver.resolve_all.return_value = {
            'status': 'partial',
            'conflicts_found': 2,
            'resolved': 1,
            'failed': 1,
            'results': [
                {'file_path': 'file1.py', 'resolved': True},
                {
                    'file_path': 'file2.py',
                    'resolved': False,
                    'requires_human_review': True,
                    'review_reason': 'Binary conflict',
                    'attempts': []
                }
            ]
        }
        mock_resolver_class.return_value = mock_resolver

        mock_escalator = Mock()
        mock_escalator_class.return_value = mock_escalator

        result = self.orchestrator._handle_merge_conflicts()

        self.assertFalse(result)
        mock_escalator.escalate.assert_called_once()

    @patch('spawn_orchestrator.ConflictResolver')
    def test_respects_disabled_config(self, mock_resolver_class):
        """Test returns False when conflict resolution is disabled."""
        # Create config with resolution disabled
        config_file = self.orchestrator.main_repo / ".autonomous" / "orchestrator_config.json"
        config_file.parent.mkdir(parents=True, exist_ok=True)
        config_file.write_text(json.dumps({
            'conflict_resolution': {
                'conflict_resolution_enabled': False
            }
        }))

        result = self.orchestrator._handle_merge_conflicts()

        self.assertFalse(result)
        mock_resolver_class.assert_not_called()


class TestHandleUnresolvedMergeConflict(unittest.TestCase):
    """Test _handle_unresolved_merge_conflict method."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

        with patch.object(SpawnOrchestrator, '__init__', lambda x, y, z=None: None):
            self.orchestrator = SpawnOrchestrator("TASK-TEST", "Test task")
            self.orchestrator.task_id = "TASK-TEST"
            self.orchestrator.session_id = "test-session"
            self.orchestrator.main_repo = Path(self.temp_dir)
            self.orchestrator.worktree_path = Path(self.temp_dir) / "worktree"
            self.orchestrator.state_dir = Path(self.temp_dir) / ".autonomous"
            self.orchestrator.state_file = self.orchestrator.state_dir / "state.json"
            self.orchestrator.state_dir.mkdir(parents=True)
            self.orchestrator.worktree_path.mkdir(parents=True)

            # Initialize state with mock
            from spawn_orchestrator import WorkflowState
            self.orchestrator.state = WorkflowState("TASK-TEST", "test-session")
            self.orchestrator.state.pr_number = 123

    def tearDown(self):
        """Clean up temp directory."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('spawn_orchestrator.subprocess.run')
    def test_records_gate_failure(self, mock_run):
        """Test that gate failure is recorded in state."""
        mock_run.return_value = Mock(returncode=0)

        initial_failures = len(self.orchestrator.state.gate_failures)

        self.orchestrator._handle_unresolved_merge_conflict()

        self.assertEqual(len(self.orchestrator.state.gate_failures), initial_failures + 1)

        failure = self.orchestrator.state.gate_failures[-1]
        self.assertEqual(failure.gate_name, "merge_conflict_resolution")
        self.assertEqual(failure.phase, Phase.MERGE.value)
        self.assertIn("auto-resolved", failure.error_message.lower())

    @patch('spawn_orchestrator.subprocess.run')
    def test_adds_pr_comment(self, mock_run):
        """Test that PR comment is added."""
        mock_run.return_value = Mock(returncode=0)

        self.orchestrator._handle_unresolved_merge_conflict()

        # Verify subprocess was called with gh pr comment
        mock_run.assert_called()
        call_args = str(mock_run.call_args)
        self.assertIn('gh pr comment', call_args)
        self.assertIn('123', call_args)  # PR number


class TestPhaseMergeWithConflicts(unittest.TestCase):
    """Test _phase_merge method with conflict handling."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

        with patch.object(SpawnOrchestrator, '__init__', lambda x, y, z=None: None):
            self.orchestrator = SpawnOrchestrator("TASK-TEST", "Test task")
            self.orchestrator.task_id = "TASK-TEST"
            self.orchestrator.session_id = "test-session"
            self.orchestrator.main_repo = Path(self.temp_dir)
            self.orchestrator.worktree_path = Path(self.temp_dir) / "worktree"
            self.orchestrator.state_dir = Path(self.temp_dir) / ".autonomous"
            self.orchestrator.state_file = self.orchestrator.state_dir / "state.json"
            self.orchestrator.state_dir.mkdir(parents=True)
            self.orchestrator.worktree_path.mkdir(parents=True)
            self.orchestrator.task_details = {}

            # Initialize state
            from spawn_orchestrator import WorkflowState
            self.orchestrator.state = WorkflowState("TASK-TEST", "test-session")
            self.orchestrator.state.pr_number = 123

    def tearDown(self):
        """Clean up temp directory."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('spawn_orchestrator.subprocess.run')
    def test_successful_merge_without_conflicts(self, mock_run):
        """Test successful merge path without conflicts."""
        # Mock successful merge
        mock_run.return_value = Mock(returncode=0)

        self.orchestrator._phase_merge()

        # Verify gh pr merge was called
        mock_run.assert_called()
        call_args = str(mock_run.call_args)
        self.assertIn('gh pr merge', call_args)

    @patch('spawn_orchestrator.subprocess.run')
    @patch.object(SpawnOrchestrator, '_handle_merge_conflicts')
    @patch.object(SpawnOrchestrator, '_is_merge_conflict_error')
    def test_conflict_detected_triggers_resolution(self, mock_is_conflict, mock_handle_conflicts, mock_run):
        """Test conflict detection triggers conflict resolution."""
        # First call: merge fails with conflict
        # Second call: merge succeeds after resolution
        merge_results = [
            Mock(returncode=1, stderr="CONFLICT: Merge conflict in file.py"),
            Mock(returncode=0)  # Push succeeds
        ]
        merge_call_count = [0]

        def mock_run_side_effect(*args, **kwargs):
            cmd = str(args[0]) if args else str(kwargs.get('args', ''))
            if 'gh pr merge' in cmd:
                result = merge_results[min(merge_call_count[0], len(merge_results) - 1)]
                merge_call_count[0] += 1
                return result
            return Mock(returncode=0)

        mock_run.side_effect = mock_run_side_effect
        mock_is_conflict.return_value = True
        mock_handle_conflicts.return_value = True  # Resolution succeeds

        self.orchestrator._phase_merge()

        mock_handle_conflicts.assert_called_once()

    @patch('spawn_orchestrator.subprocess.run')
    @patch.object(SpawnOrchestrator, '_handle_merge_conflicts')
    @patch.object(SpawnOrchestrator, '_handle_unresolved_merge_conflict')
    @patch.object(SpawnOrchestrator, '_is_merge_conflict_error')
    def test_failed_resolution_triggers_fallback(self, mock_is_conflict, mock_handle_unresolved,
                                                  mock_handle_conflicts, mock_run):
        """Test failed resolution triggers gate skip fallback."""
        mock_run.return_value = Mock(returncode=1, stderr="CONFLICT: Merge conflict")
        mock_is_conflict.return_value = True
        mock_handle_conflicts.return_value = False  # Resolution fails

        self.orchestrator._phase_merge()

        mock_handle_unresolved.assert_called_once()

    @patch('spawn_orchestrator.subprocess.run')
    def test_auto_merge_disabled_skips_merge(self, mock_run):
        """Test auto_merge_disabled flag skips merge phase."""
        self.orchestrator.task_details = {
            'git_safety': {'auto_merge_disabled': True}
        }
        mock_run.return_value = Mock(returncode=0)

        self.orchestrator._phase_merge()

        # Should only call gh pr comment, not gh pr merge
        call_args = [str(c) for c in mock_run.call_args_list]
        merge_calls = [c for c in call_args if 'gh pr merge' in c]
        self.assertEqual(len(merge_calls), 0)


class TestGateSkipPolicyIntegration(unittest.TestCase):
    """Test merge_conflict_resolution gate skip policy."""

    def test_merge_conflict_resolution_is_skippable(self):
        """Test merge_conflict_resolution gate is SKIPPABLE."""
        from autonomous.gates import GATE_SKIP_POLICIES, GateSkipPolicy

        self.assertIn('merge_conflict_resolution', GATE_SKIP_POLICIES)
        self.assertEqual(
            GATE_SKIP_POLICIES['merge_conflict_resolution'],
            GateSkipPolicy.SKIPPABLE
        )


class TestCommitConflictResolutions(unittest.TestCase):
    """Test _commit_conflict_resolutions method."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

        with patch.object(SpawnOrchestrator, '__init__', lambda x, y, z=None: None):
            self.orchestrator = SpawnOrchestrator("TASK-TEST", "Test task")
            self.orchestrator.task_id = "TASK-TEST"
            self.orchestrator.session_id = "test-session"
            self.orchestrator.worktree_path = Path(self.temp_dir) / "worktree"
            self.orchestrator.worktree_path.mkdir(parents=True)

    def tearDown(self):
        """Clean up temp directory."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('spawn_orchestrator.subprocess.run')
    def test_commits_resolved_files(self, mock_run):
        """Test resolved files are committed."""
        mock_run.return_value = Mock(returncode=0)

        resolution_result = {
            'results': [
                {'file_path': 'file1.py', 'resolved': True, 'strategy_used': 'prefer_ours'},
                {'file_path': 'file2.py', 'resolved': True, 'strategy_used': 'combine'},
                {'file_path': 'file3.py', 'resolved': False}  # Not resolved
            ]
        }

        self.orchestrator._commit_conflict_resolutions(resolution_result)

        # Verify git add was called for each resolved file
        call_args = [str(c) for c in mock_run.call_args_list]
        add_calls = [c for c in call_args if 'git add' in c]
        self.assertEqual(len(add_calls), 2)  # Only 2 resolved files

    @patch('spawn_orchestrator.subprocess.run')
    def test_handles_empty_resolution(self, mock_run):
        """Test handles case with no resolved files."""
        resolution_result = {
            'results': [
                {'file_path': 'file1.py', 'resolved': False}
            ]
        }

        self.orchestrator._commit_conflict_resolutions(resolution_result)

        # Should not call git commands
        mock_run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
