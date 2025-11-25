#!/usr/bin/env python3
"""
Unit tests for Design Phase (SDLC-001).

Tests:
- Phase enum includes DESIGN phase
- _phase_design() creates design documents
- Design gate verifies deliverable exists
- Design phase executes before implementation
"""

import os
import subprocess
import unittest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, call
import tempfile
import shutil

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from spawn_orchestrator import SpawnOrchestrator, Phase


class TestPhaseEnum(unittest.TestCase):
    """Test Phase enum includes new phases."""

    def test_phase_enum_has_design(self):
        """Test DESIGN phase exists in enum."""
        self.assertTrue(hasattr(Phase, 'DESIGN'))
        self.assertEqual(Phase.DESIGN.value, 'design')

    def test_phase_enum_has_planning(self):
        """Test PLANNING phase exists in enum."""
        self.assertTrue(hasattr(Phase, 'PLANNING'))
        self.assertEqual(Phase.PLANNING.value, 'planning')

    def test_phase_enum_has_implementation(self):
        """Test IMPLEMENTATION phase exists in enum."""
        self.assertTrue(hasattr(Phase, 'IMPLEMENTATION'))
        self.assertEqual(Phase.IMPLEMENTATION.value, 'implementation')

    def test_phase_enum_has_test(self):
        """Test TEST phase exists in enum."""
        self.assertTrue(hasattr(Phase, 'TEST'))
        self.assertEqual(Phase.TEST.value, 'test')

    def test_all_phases_present(self):
        """Test all expected phases are present."""
        expected_phases = [
            'STARTUP', 'DESIGN', 'PLANNING', 'IMPLEMENTATION',
            'TEST', 'PR_CREATION', 'MERGE', 'CLEANUP'
        ]
        for phase_name in expected_phases:
            self.assertTrue(hasattr(Phase, phase_name), f"Missing phase: {phase_name}")


class TestDesignPhase(unittest.TestCase):
    """Test _phase_design method."""

    def setUp(self):
        """Set up test fixtures."""
        self.orchestrator = SpawnOrchestrator("TASK-TEST", "Test task description")
        self.orchestrator.branch_name = "feat/20251125-test"
        self.orchestrator.task_details = {
            'title': 'Test Task',
            'description': 'Test description',
            'acceptance_criteria': ['Criterion 1', 'Criterion 2']
        }

    def tearDown(self):
        """Clean up test directories."""
        if self.orchestrator.deliverables_path.exists():
            shutil.rmtree(self.orchestrator.deliverables_path, ignore_errors=True)

    @patch('shutil.which')
    def test_phase_design_creates_minimal_when_no_claude(self, mock_which):
        """Test design phase creates minimal doc when Claude not available."""
        mock_which.return_value = None

        # Execute phase
        self.orchestrator._phase_design()

        # Verify design file created
        design_file = self.orchestrator.deliverables_path / "TASK-TEST-DESIGN.md"
        self.assertTrue(design_file.exists())

        # Verify content
        content = design_file.read_text()
        self.assertIn("Technical Design: TASK-TEST", content)
        self.assertIn("Test description", content)

    @patch('subprocess.Popen')
    @patch('shutil.which')
    def test_phase_design_spawns_tech_lead(self, mock_which, mock_popen):
        """Test design phase spawns tech-lead agent when Claude available."""
        mock_which.return_value = '/usr/bin/claude'

        # Mock process that exits immediately
        mock_process = Mock()
        mock_process.pid = 12345
        mock_process.poll.return_value = 0
        mock_popen.return_value = mock_process

        # Create tech-lead agent file
        tech_lead_dir = self.orchestrator.worktree_path / ".claude" / "agents"
        tech_lead_dir.mkdir(parents=True, exist_ok=True)
        (tech_lead_dir / "tech-lead.md").write_text("You are a tech lead.")

        # Create the design file as if agent created it
        self.orchestrator.deliverables_path.mkdir(parents=True, exist_ok=True)
        design_file = self.orchestrator.deliverables_path / "TASK-TEST-DESIGN.md"
        design_file.write_text("# Design Document")

        # Execute phase
        self.orchestrator._phase_design()

        # Verify Claude was called
        mock_popen.assert_called_once()
        args, kwargs = mock_popen.call_args
        self.assertIn('claude', args[0][0])

    def test_design_deliverables_path_created(self):
        """Test deliverables directory is created during design phase."""
        with patch('shutil.which', return_value=None):
            self.orchestrator._phase_design()

        self.assertTrue(self.orchestrator.deliverables_path.exists())


class TestDesignGate(unittest.TestCase):
    """Test design gate verification."""

    def setUp(self):
        """Set up test fixtures."""
        self.orchestrator = SpawnOrchestrator("TASK-TEST", "Test task description")
        self.orchestrator.task_details = {'title': 'Test'}

    def tearDown(self):
        """Clean up test directories."""
        if self.orchestrator.deliverables_path.exists():
            shutil.rmtree(self.orchestrator.deliverables_path, ignore_errors=True)

    def test_design_gate_fails_without_deliverable(self):
        """Test design gate fails when no design document exists."""
        result = self.orchestrator._verify_gate(Phase.DESIGN)
        self.assertFalse(result)

    def test_design_gate_passes_with_deliverable(self):
        """Test design gate passes when design document exists."""
        # Create deliverable
        self.orchestrator.deliverables_path.mkdir(parents=True, exist_ok=True)
        design_file = self.orchestrator.deliverables_path / "TASK-TEST-DESIGN.md"
        design_file.write_text("# Technical Design")

        result = self.orchestrator._verify_gate(Phase.DESIGN)
        self.assertTrue(result)


class TestPlanningPhase(unittest.TestCase):
    """Test _phase_planning method."""

    def setUp(self):
        """Set up test fixtures."""
        self.orchestrator = SpawnOrchestrator("TASK-TEST", "Test task description")

    def tearDown(self):
        """Clean up test directories."""
        if self.orchestrator.state_dir.exists():
            shutil.rmtree(self.orchestrator.state_dir, ignore_errors=True)

    def test_phase_planning_creates_delegation_plan(self):
        """Test planning phase creates delegation plan."""
        self.orchestrator._phase_planning()

        delegation_file = self.orchestrator.state_dir / "delegation_plan.json"
        self.assertTrue(delegation_file.exists())

    def test_planning_gate_passes_with_plan(self):
        """Test planning gate passes when delegation plan exists."""
        # Create delegation plan
        self.orchestrator.state_dir.mkdir(parents=True, exist_ok=True)
        plan_file = self.orchestrator.state_dir / "delegation_plan.json"
        plan_file.write_text('{"primary_agent": "test"}')

        result = self.orchestrator._verify_gate(Phase.PLANNING)
        self.assertTrue(result)

    def test_planning_gate_fails_without_plan(self):
        """Test planning gate fails when no delegation plan exists."""
        # Ensure no plan file exists
        plan_file = self.orchestrator.state_dir / "delegation_plan.json"
        if plan_file.exists():
            plan_file.unlink()

        result = self.orchestrator._verify_gate(Phase.PLANNING)
        self.assertFalse(result)


class TestPhaseOrderExecution(unittest.TestCase):
    """Test that phases execute in correct order."""

    def setUp(self):
        """Set up test fixtures."""
        self.orchestrator = SpawnOrchestrator("TASK-TEST", "Test task description")

    def test_phase_order_in_execute_workflow(self):
        """Verify design comes before implementation."""
        # Get phase methods from execute_workflow by checking order
        # Design should come after startup and before implementation

        # This tests the logical order by checking the Phase enum docstring
        doc = Phase.__doc__
        self.assertIn("Phase 1: STARTUP", doc)
        self.assertIn("Phase 2: DESIGN", doc)
        self.assertIn("Phase 3: PLANNING", doc)
        self.assertIn("Phase 4: IMPLEMENTATION", doc)
        self.assertIn("Phase 5: TEST", doc)


class TestTechLeadAgentFile(unittest.TestCase):
    """Test tech-lead agent file exists."""

    def test_tech_lead_agent_file_exists(self):
        """Verify tech-lead.md exists in .claude/agents/."""
        agent_file = Path(__file__).parent.parent.parent.parent / ".claude" / "agents" / "tech-lead.md"
        self.assertTrue(agent_file.exists(), f"tech-lead.md not found at {agent_file}")

    def test_tech_lead_agent_file_has_content(self):
        """Verify tech-lead.md has expected content."""
        agent_file = Path(__file__).parent.parent.parent.parent / ".claude" / "agents" / "tech-lead.md"
        if agent_file.exists():
            content = agent_file.read_text()
            self.assertIn("Tech Lead", content)
            self.assertIn("design", content.lower())


if __name__ == '__main__':
    unittest.main()
