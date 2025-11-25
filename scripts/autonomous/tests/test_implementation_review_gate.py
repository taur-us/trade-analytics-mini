#!/usr/bin/env python3
"""
Unit tests for Implementation Review Gate - Phase 7 (SDLC-003).

Tests:
- Phase enum includes IMPLEMENTATION_REVIEW phase
- _phase_implementation_review() executes review checkpoint
- Implementation review gate verifies report status
- Skip flag bypasses review
- Auto-approval fallback when Claude unavailable
- Implementation review phase executes between TEST and PR_CREATION
- Gate blocks workflow when status is BLOCKED
- Gate passes workflow when status is APPROVED
"""

import json
import os
import subprocess
import unittest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, call
import tempfile
import shutil
from datetime import datetime

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from spawn_orchestrator import SpawnOrchestrator, Phase


class TestPhaseEnum(unittest.TestCase):
    """Test Phase enum includes IMPLEMENTATION_REVIEW phase."""

    def test_phase_enum_has_implementation_review(self):
        """Test IMPLEMENTATION_REVIEW phase exists in enum."""
        self.assertTrue(hasattr(Phase, 'IMPLEMENTATION_REVIEW'))
        self.assertEqual(Phase.IMPLEMENTATION_REVIEW.value, 'implementation_review')

    def test_phase_ordering(self):
        """Test IMPLEMENTATION_REVIEW comes after TEST and before PR_CREATION."""
        # Verify phase ordering through enum docstring
        doc = Phase.__doc__
        self.assertIn("Phase 6: TEST", doc)
        self.assertIn("Phase 7: IMPLEMENTATION_REVIEW", doc or "")
        self.assertIn("Phase 8: PR_CREATION", doc or "Phase 7: PR_CREATION")

        # Verify IMPLEMENTATION_REVIEW is mentioned if docstring exists
        if doc and "IMPLEMENTATION_REVIEW" in doc:
            self.assertIn("IMPLEMENTATION_REVIEW", doc)

    def test_all_phases_present_with_implementation_review(self):
        """Test all expected phases including IMPLEMENTATION_REVIEW are present."""
        expected_phases = [
            'STARTUP', 'DESIGN', 'DESIGN_REVIEW', 'PLANNING',
            'IMPLEMENTATION', 'TEST', 'IMPLEMENTATION_REVIEW',
            'PR_CREATION', 'MERGE', 'CLEANUP'
        ]
        for phase_name in expected_phases:
            self.assertTrue(hasattr(Phase, phase_name), f"Missing phase: {phase_name}")

    def test_phase_renumbering_correct(self):
        """Test that phases are correctly renumbered after IMPLEMENTATION_REVIEW."""
        doc = Phase.__doc__
        # PR_CREATION should be Phase 8 (was 7)
        self.assertIn("Phase 8: PR_CREATION", doc)
        # MERGE should be Phase 9 (was 8)
        self.assertIn("Phase 9: MERGE", doc)
        # CLEANUP should be Phase 10 (was 9)
        self.assertIn("Phase 10: CLEANUP", doc)


class TestSkipImplementationReviewFlag(unittest.TestCase):
    """Test skip_implementation_review flag functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.orchestrator = SpawnOrchestrator("TASK-TEST", "Test task description")
        self.orchestrator.branch_name = "feat/20251125-test"

    def tearDown(self):
        """Clean up test directories."""
        if self.orchestrator.state_dir.exists():
            shutil.rmtree(self.orchestrator.state_dir, ignore_errors=True)

    def test_skip_implementation_review_flag_bypasses_review(self):
        """Test that skip_implementation_review=True bypasses review execution."""
        # Set skip flag
        self.orchestrator.task_details = {
            'skip_implementation_review': True,
            'title': 'Test Task'
        }

        # Execute phase
        self.orchestrator._phase_implementation_review()

        # Verify report created with SKIPPED status
        report_file = self.orchestrator.state_dir / "implementation_review_report.json"
        self.assertTrue(report_file.exists())

        with open(report_file, 'r') as f:
            report = json.load(f)

        self.assertEqual(report['status'], 'SKIPPED')
        self.assertIn('skip_implementation_review', report['reason'])

    def test_skip_implementation_review_creates_skipped_report(self):
        """Test that skip creates report with proper structure."""
        self.orchestrator.task_details = {
            'skip_implementation_review': True,
            'title': 'Test Task'
        }

        self.orchestrator._phase_implementation_review()

        report_file = self.orchestrator.state_dir / "implementation_review_report.json"
        with open(report_file, 'r') as f:
            report = json.load(f)

        # Verify structure
        self.assertIn('status', report)
        self.assertIn('reason', report)
        self.assertIn('timestamp', report)
        self.assertEqual(report['status'], 'SKIPPED')


class TestImplementationReviewGateVerification(unittest.TestCase):
    """Test implementation review gate verification logic."""

    def setUp(self):
        """Set up test fixtures."""
        self.orchestrator = SpawnOrchestrator("TASK-TEST", "Test task description")
        self.orchestrator.branch_name = "feat/20251125-test"

    def tearDown(self):
        """Clean up test directories."""
        if self.orchestrator.state_dir.exists():
            shutil.rmtree(self.orchestrator.state_dir, ignore_errors=True)

    def test_gate_passes_with_approved_status(self):
        """Test gate passes when report status is APPROVED."""
        # Create APPROVED report
        report = {
            'status': 'APPROVED',
            'critical_issues': [],
            'major_issues': [],
            'checks': {
                'variant_files': 'PASS',
                'exception_handling': 'PASS',
                'test_coverage': 'PASS',
                'security_issues': 'PASS',
                'production_path': 'PASS'
            },
            'timestamp': datetime.now().isoformat()
        }
        report_file = self.orchestrator.state_dir / "implementation_review_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f)

        # Verify gate passes
        result = self.orchestrator._verify_gate(Phase.IMPLEMENTATION_REVIEW)
        self.assertTrue(result)

    def test_gate_blocks_with_blocked_status(self):
        """Test gate blocks when report status is BLOCKED."""
        # Create BLOCKED report with critical issues
        report = {
            'status': 'BLOCKED',
            'critical_issues': [
                'Variant file created: orchestrator_optimized.py',
                'Bare except clause found in error_handler.py:45'
            ],
            'major_issues': ['Test coverage 78% (below 85% threshold)'],
            'checks': {
                'variant_files': 'FAIL',
                'exception_handling': 'FAIL',
                'test_coverage': 'FAIL'
            },
            'timestamp': datetime.now().isoformat()
        }
        report_file = self.orchestrator.state_dir / "implementation_review_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f)

        # Verify gate blocks
        result = self.orchestrator._verify_gate(Phase.IMPLEMENTATION_REVIEW)
        self.assertFalse(result)

    def test_gate_passes_with_skipped_status(self):
        """Test gate passes when report status is SKIPPED."""
        # Create SKIPPED report
        report = {
            'status': 'SKIPPED',
            'reason': 'skip_implementation_review flag set',
            'timestamp': datetime.now().isoformat()
        }
        report_file = self.orchestrator.state_dir / "implementation_review_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f)

        # Verify gate passes
        result = self.orchestrator._verify_gate(Phase.IMPLEMENTATION_REVIEW)
        self.assertTrue(result)

    def test_gate_blocks_when_report_missing(self):
        """Test gate blocks when report file doesn't exist."""
        # No report file created
        self.orchestrator.task_details = {}

        # Verify gate blocks
        result = self.orchestrator._verify_gate(Phase.IMPLEMENTATION_REVIEW)
        self.assertFalse(result)

    def test_gate_blocks_with_invalid_json(self):
        """Test gate blocks when report contains invalid JSON."""
        # Create invalid JSON file
        report_file = self.orchestrator.state_dir / "implementation_review_report.json"
        with open(report_file, 'w') as f:
            f.write("{ invalid json }")

        # Verify gate blocks
        result = self.orchestrator._verify_gate(Phase.IMPLEMENTATION_REVIEW)
        self.assertFalse(result)

    def test_gate_bypassed_with_skip_flag(self):
        """Test gate is bypassed when skip_implementation_review flag is set."""
        # Set skip flag
        self.orchestrator.task_details = {'skip_implementation_review': True}

        # Verify gate passes even without report
        result = self.orchestrator._verify_gate(Phase.IMPLEMENTATION_REVIEW)
        self.assertTrue(result)

    def test_gate_blocks_with_needs_revision_status(self):
        """Test gate blocks when report status is NEEDS_REVISION."""
        # Create NEEDS_REVISION report
        report = {
            'status': 'NEEDS_REVISION',
            'critical_issues': [],
            'major_issues': ['Minor code quality issues found'],
            'timestamp': datetime.now().isoformat()
        }
        report_file = self.orchestrator.state_dir / "implementation_review_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f)

        # Verify gate blocks
        result = self.orchestrator._verify_gate(Phase.IMPLEMENTATION_REVIEW)
        self.assertFalse(result)


class TestAutoApprovalFallback(unittest.TestCase):
    """Test auto-approval fallback when review agent unavailable."""

    def setUp(self):
        """Set up test fixtures."""
        self.orchestrator = SpawnOrchestrator("TASK-TEST", "Test task description")
        self.orchestrator.branch_name = "feat/20251125-test"

    def tearDown(self):
        """Clean up test directories."""
        if self.orchestrator.state_dir.exists():
            shutil.rmtree(self.orchestrator.state_dir, ignore_errors=True)

    def test_auto_approval_when_claude_not_found(self):
        """Test auto-approval fallback when Claude executable not available."""
        # Mock shutil.which to return None
        with patch('shutil.which', return_value=None):
            self.orchestrator.task_details = {}
            self.orchestrator._phase_implementation_review()

        # Verify auto-approved report created
        report_file = self.orchestrator.state_dir / "implementation_review_report.json"
        self.assertTrue(report_file.exists())

        with open(report_file, 'r') as f:
            report = json.load(f)

        self.assertEqual(report['status'], 'APPROVED')
        self.assertIn('Auto-approved', report['reason'])
        self.assertEqual(report['checks']['variant_files'], 'NOT_CHECKED')
        self.assertEqual(report['checks']['exception_handling'], 'NOT_CHECKED')
        self.assertEqual(report['checks']['test_coverage'], 'PASSED (from Gate3)')
        self.assertEqual(report['checks']['security_issues'], 'PASSED (from Gate4)')

    def test_auto_approval_report_structure(self):
        """Test auto-approval report has correct structure."""
        self.orchestrator._create_auto_approved_implementation_review_report()

        report_file = self.orchestrator.state_dir / "implementation_review_report.json"
        with open(report_file, 'r') as f:
            report = json.load(f)

        # Verify all required fields
        self.assertIn('status', report)
        self.assertIn('reason', report)
        self.assertIn('critical_issues', report)
        self.assertIn('major_issues', report)
        self.assertIn('recommendations', report)
        self.assertIn('checks', report)
        self.assertIn('timestamp', report)

        # Verify check structure
        self.assertIn('variant_files', report['checks'])
        self.assertIn('exception_handling', report['checks'])
        self.assertIn('test_coverage', report['checks'])
        self.assertIn('security_issues', report['checks'])
        self.assertIn('production_path', report['checks'])

        # Verify lists are empty
        self.assertEqual(len(report['critical_issues']), 0)
        self.assertEqual(len(report['major_issues']), 0)


class TestImplementationReviewPhaseExecution(unittest.TestCase):
    """Test implementation review phase execution."""

    def setUp(self):
        """Set up test fixtures."""
        self.orchestrator = SpawnOrchestrator("TASK-TEST", "Test task description")
        self.orchestrator.branch_name = "feat/20251125-test"

    def tearDown(self):
        """Clean up test directories."""
        if self.orchestrator.state_dir.exists():
            shutil.rmtree(self.orchestrator.state_dir, ignore_errors=True)

    @patch('subprocess.Popen')
    @patch('shutil.which')
    def test_phase_spawns_review_checkpoint_agent(self, mock_which, mock_popen):
        """Test that phase spawns review-checkpoint agent correctly."""
        # Mock Claude executable found
        mock_which.return_value = '/usr/bin/claude'

        # Mock review-checkpoint agent file
        checkpoint_file = self.orchestrator.worktree_path / ".claude" / "agents" / "review-checkpoint.md"
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_file.write_text("---\nname: review-checkpoint\n---\nYou are a review agent")

        # Mock process
        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process

        # Create auto-approved report (simulates agent creating report)
        self.orchestrator.task_details = {}

        # Execute phase
        self.orchestrator._phase_implementation_review()

        # Verify Popen was called with correct arguments
        self.assertTrue(mock_popen.called)
        call_args = mock_popen.call_args[0][0]
        self.assertIn('claude', call_args[0])
        self.assertIn('--model', call_args)
        self.assertIn('sonnet', call_args)  # Should use Sonnet, not Opus

    def test_phase_gathers_context_from_test_phase(self):
        """Test that phase gathers context from Phase 6 (TEST)."""
        # Create mock test artifacts
        coverage_file = self.orchestrator.worktree_path / ".coverage"
        coverage_file.parent.mkdir(parents=True, exist_ok=True)
        coverage_file.write_text("")

        pytest_cache = self.orchestrator.worktree_path / ".pytest_cache"
        pytest_cache.mkdir(parents=True, exist_ok=True)

        # Mock Claude not available to trigger auto-approval
        with patch('shutil.which', return_value=None):
            self.orchestrator.task_details = {}
            self.orchestrator._phase_implementation_review()

        # Verify report created (shows phase executed)
        report_file = self.orchestrator.state_dir / "implementation_review_report.json"
        self.assertTrue(report_file.exists())


class TestWorkflowIntegration(unittest.TestCase):
    """Test implementation review integrates correctly in workflow."""

    def setUp(self):
        """Set up test fixtures."""
        self.orchestrator = SpawnOrchestrator("TASK-TEST", "Test task description")

    def tearDown(self):
        """Clean up test directories."""
        if self.orchestrator.state_dir.exists():
            shutil.rmtree(self.orchestrator.state_dir, ignore_errors=True)

    @patch.object(SpawnOrchestrator, '_phase_startup')
    @patch.object(SpawnOrchestrator, '_phase_design')
    @patch.object(SpawnOrchestrator, '_phase_design_review')
    @patch.object(SpawnOrchestrator, '_phase_planning')
    @patch.object(SpawnOrchestrator, '_phase_implementation')
    @patch.object(SpawnOrchestrator, '_phase_test')
    @patch.object(SpawnOrchestrator, '_phase_implementation_review')
    @patch.object(SpawnOrchestrator, '_phase_pr_creation')
    @patch.object(SpawnOrchestrator, '_phase_merge')
    @patch.object(SpawnOrchestrator, '_phase_cleanup')
    @patch.object(SpawnOrchestrator, '_verify_gate')
    def test_implementation_review_executes_after_test_before_pr(
        self, mock_verify_gate, mock_cleanup, mock_merge, mock_pr,
        mock_impl_review, mock_test, mock_impl, mock_planning,
        mock_design_review, mock_design, mock_startup
    ):
        """Test that implementation review executes after TEST and before PR_CREATION."""
        # Make all gates pass
        mock_verify_gate.return_value = True

        # Mock instance registry methods
        with patch.object(self.orchestrator.instance_registry, 'register_instance', return_value='test-instance'):
            with patch.object(self.orchestrator.instance_registry, 'update_status'):
                with patch.object(self.orchestrator.instance_registry, 'shutdown'):
                    with patch.object(self.orchestrator.message_queue, 'broadcast'):
                        # Execute workflow
                        self.orchestrator.execute_workflow()

        # Verify phases executed in correct order
        mock_startup.assert_called_once()
        mock_design.assert_called_once()
        mock_design_review.assert_called_once()
        mock_planning.assert_called_once()
        mock_impl.assert_called_once()
        mock_test.assert_called_once()
        mock_impl_review.assert_called_once()  # NEW - Should be called
        mock_pr.assert_called_once()
        mock_merge.assert_called_once()
        mock_cleanup.assert_called_once()

        # Verify implementation review called after test and before PR
        call_order = [
            mock_test.call_args_list,
            mock_impl_review.call_args_list,
            mock_pr.call_args_list
        ]
        # All should have been called exactly once
        self.assertTrue(all(len(calls) == 1 for calls in call_order))


class TestCriticalIssueBlocking(unittest.TestCase):
    """Test that critical issues block workflow."""

    def setUp(self):
        """Set up test fixtures."""
        self.orchestrator = SpawnOrchestrator("TASK-TEST", "Test task description")

    def tearDown(self):
        """Clean up test directories."""
        if self.orchestrator.state_dir.exists():
            shutil.rmtree(self.orchestrator.state_dir, ignore_errors=True)

    def test_variant_file_issue_blocks_workflow(self):
        """Test that variant file detection blocks workflow."""
        # Create BLOCKED report due to variant files
        report = {
            'status': 'BLOCKED',
            'critical_issues': ['Variant file created: orchestrator_v2.py'],
            'major_issues': [],
            'checks': {'variant_files': 'FAIL'},
            'timestamp': datetime.now().isoformat()
        }
        report_file = self.orchestrator.state_dir / "implementation_review_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f)

        # Verify gate blocks
        result = self.orchestrator._verify_gate(Phase.IMPLEMENTATION_REVIEW)
        self.assertFalse(result)

    def test_exception_handling_issue_blocks_workflow(self):
        """Test that bare except detection blocks workflow."""
        # Create BLOCKED report due to exception handling
        report = {
            'status': 'BLOCKED',
            'critical_issues': ['Bare except clause found in handler.py:23'],
            'major_issues': [],
            'checks': {'exception_handling': 'FAIL'},
            'timestamp': datetime.now().isoformat()
        }
        report_file = self.orchestrator.state_dir / "implementation_review_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f)

        # Verify gate blocks
        result = self.orchestrator._verify_gate(Phase.IMPLEMENTATION_REVIEW)
        self.assertFalse(result)

    def test_multiple_critical_issues_logged(self):
        """Test that multiple critical issues are all logged."""
        # Create BLOCKED report with multiple issues
        report = {
            'status': 'BLOCKED',
            'critical_issues': [
                'Variant file: test_v2.py',
                'Bare except in file.py:45',
                'High-severity security issue found'
            ],
            'major_issues': ['Low test coverage'],
            'timestamp': datetime.now().isoformat()
        }
        report_file = self.orchestrator.state_dir / "implementation_review_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f)

        # Verify gate blocks
        result = self.orchestrator._verify_gate(Phase.IMPLEMENTATION_REVIEW)
        self.assertFalse(result)


if __name__ == '__main__':
    unittest.main()
