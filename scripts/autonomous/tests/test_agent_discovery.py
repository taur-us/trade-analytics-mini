#!/usr/bin/env python3
"""
Unit tests for agent discovery functionality (SDLC-005).

Tests:
- Agent file discovery in .claude/agents/
- YAML frontmatter parsing
- Pattern-based agent selection
- All 17 agents discovered correctly
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from autonomous.agent_selector import AgentSelector


class TestAgentDiscovery(unittest.TestCase):
    """Test agent discovery functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Use the actual repo path
        self.repo_path = Path(__file__).parent.parent.parent.parent
        self.agents_dir = self.repo_path / '.claude' / 'agents'

    def test_discovers_all_17_agents(self):
        """Test that all 17 agents are discovered."""
        selector = AgentSelector(worktree_path=self.repo_path)
        agents = selector.get_discovered_agents()

        self.assertGreaterEqual(len(agents), 17,
            f"Expected >= 17 agents, got {len(agents)}")

    def test_discovers_core_agents(self):
        """Test that core agents are discovered."""
        selector = AgentSelector(worktree_path=self.repo_path)
        agents = selector.get_discovered_agents()

        core_agents = [
            'feature-developer',
            'test-engineer',
            'validation-agent',
            'pr-review-agent',
            'security-reviewer',
            'docs-specialist',
            'tech-lead',
        ]

        for agent in core_agents:
            self.assertIn(agent, agents, f"Core agent '{agent}' not discovered")

    def test_discovers_new_specialist_agents(self):
        """Test that new specialist agents are discovered."""
        selector = AgentSelector(worktree_path=self.repo_path)
        agents = selector.get_discovered_agents()

        new_agents = [
            'review-checkpoint',
            'task-manager',
            'git-manager',
            'database-specialist',
            'api-designer',
            'async-specialist',
            'websocket-specialist',
            'optimization-engineer',
            'refactoring-specialist',
            'optuna-specialist',
        ]

        for agent in new_agents:
            self.assertIn(agent, agents, f"New agent '{agent}' not discovered")

    def test_agent_has_required_fields(self):
        """Test that discovered agents have required fields."""
        selector = AgentSelector(worktree_path=self.repo_path)
        agents = selector.get_discovered_agents()

        required_fields = ['name', 'description']

        for agent_name, config in agents.items():
            for field in required_fields:
                self.assertIn(field, config,
                    f"Agent '{agent_name}' missing required field '{field}'")

    def test_agent_file_paths_exist(self):
        """Test that all agent file paths exist."""
        selector = AgentSelector(worktree_path=self.repo_path)
        agents = selector.get_discovered_agents()

        for agent_name, config in agents.items():
            if 'file_path' in config:
                file_path = Path(config['file_path'])
                self.assertTrue(file_path.exists() or (self.repo_path / config['file_path']).exists(),
                    f"Agent file for '{agent_name}' does not exist: {config['file_path']}")


class TestSpecialistAgentSelection(unittest.TestCase):
    """Test specialist agent selection based on task description."""

    def setUp(self):
        """Set up test fixtures."""
        self.repo_path = Path(__file__).parent.parent.parent.parent
        self.selector = AgentSelector(worktree_path=self.repo_path)

    def test_database_task_selects_database_specialist(self):
        """Test that database tasks select database-specialist."""
        plan = self.selector.analyze_task('TEST-001',
            'Create database migration for users table with PostgreSQL schema')

        self.assertEqual(plan.primary_agent, 'database-specialist',
            f"Expected database-specialist, got {plan.primary_agent}")

    def test_api_task_selects_api_designer(self):
        """Test that API tasks select api-designer."""
        plan = self.selector.analyze_task('TEST-002',
            'Design REST API endpoints for user authentication')

        self.assertEqual(plan.primary_agent, 'api-designer',
            f"Expected api-designer, got {plan.primary_agent}")

    def test_async_task_selects_async_specialist(self):
        """Test that async tasks select async-specialist."""
        plan = self.selector.analyze_task('TEST-003',
            'Implement async webhook delivery with concurrent processing and asyncio')

        self.assertEqual(plan.primary_agent, 'async-specialist',
            f"Expected async-specialist, got {plan.primary_agent}")

    def test_websocket_task_selects_websocket_specialist(self):
        """Test that WebSocket tasks select websocket-specialist."""
        plan = self.selector.analyze_task('TEST-004',
            'Implement WebSocket real-time streaming for live updates')

        self.assertEqual(plan.primary_agent, 'websocket-specialist',
            f"Expected websocket-specialist, got {plan.primary_agent}")

    def test_performance_task_selects_optimization_engineer(self):
        """Test that performance tasks select optimization-engineer."""
        plan = self.selector.analyze_task('TEST-005',
            'Profile and optimize performance bottlenecks with caching')

        self.assertEqual(plan.primary_agent, 'optimization-engineer',
            f"Expected optimization-engineer, got {plan.primary_agent}")

    def test_refactor_task_selects_refactoring_specialist(self):
        """Test that refactoring tasks select refactoring-specialist."""
        plan = self.selector.analyze_task('TEST-006',
            'Refactor the module structure to cleanup and consolidate duplicate functions')

        self.assertEqual(plan.primary_agent, 'refactoring-specialist',
            f"Expected refactoring-specialist, got {plan.primary_agent}")

    def test_optuna_task_selects_optuna_specialist(self):
        """Test that Optuna tasks select optuna-specialist."""
        plan = self.selector.analyze_task('TEST-007',
            'Configure Optuna hyperparameter tuning with multi-objective optimization')

        self.assertEqual(plan.primary_agent, 'optuna-specialist',
            f"Expected optuna-specialist, got {plan.primary_agent}")

    def test_git_task_selects_git_manager(self):
        """Test that Git tasks select git-manager."""
        plan = self.selector.analyze_task('TEST-008',
            'Set up git worktree isolation for branch management')

        self.assertEqual(plan.primary_agent, 'git-manager',
            f"Expected git-manager, got {plan.primary_agent}")

    def test_review_task_selects_review_checkpoint(self):
        """Test that review tasks select review-checkpoint."""
        plan = self.selector.analyze_task('TEST-009',
            'Perform design review checkpoint validation before implementation')

        self.assertEqual(plan.primary_agent, 'review-checkpoint',
            f"Expected review-checkpoint, got {plan.primary_agent}")

    def test_supporting_agents_always_included(self):
        """Test that supporting agents are always included."""
        plan = self.selector.analyze_task('TEST-010',
            'Implement any generic feature')

        # Default supporting agents should be included
        self.assertIn('test-engineer', plan.supporting_agents)
        self.assertIn('validation-agent', plan.supporting_agents)


class TestAgentFrontmatterParsing(unittest.TestCase):
    """Test YAML frontmatter parsing from agent files."""

    def setUp(self):
        """Set up test fixtures."""
        self.repo_path = Path(__file__).parent.parent.parent.parent
        self.agents_dir = self.repo_path / '.claude' / 'agents'

    def test_all_agent_files_have_valid_frontmatter(self):
        """Test that all agent files have valid YAML frontmatter."""
        selector = AgentSelector(worktree_path=self.repo_path)
        agents = selector.get_discovered_agents()

        # All agents should have been successfully parsed
        self.assertGreaterEqual(len(agents), 17)

    def test_agent_names_match_filenames(self):
        """Test that agent names match their filenames."""
        for md_file in self.agents_dir.glob('*.md'):
            expected_name = md_file.stem  # filename without extension

            # Parse file to get name (handle encoding properly)
            try:
                content = md_file.read_text(encoding='utf-8')
            except UnicodeDecodeError:
                content = md_file.read_text(encoding='utf-8', errors='ignore')

            if content.startswith('---'):
                parts = content.split('---', 2)
                if len(parts) >= 3:
                    import yaml
                    try:
                        frontmatter = yaml.safe_load(parts[1])
                        if frontmatter and 'name' in frontmatter:
                            self.assertEqual(frontmatter['name'], expected_name,
                                f"Agent name mismatch in {md_file.name}: "
                                f"'{frontmatter['name']}' != '{expected_name}'")
                    except yaml.YAMLError:
                        pass  # Invalid YAML is handled elsewhere

    def test_agent_models_are_valid(self):
        """Test that agent models are valid (sonnet, opus, haiku)."""
        valid_models = {'sonnet', 'opus', 'haiku'}
        selector = AgentSelector(worktree_path=self.repo_path)
        agents = selector.get_discovered_agents()

        for agent_name, config in agents.items():
            if 'model' in config:
                self.assertIn(config['model'], valid_models,
                    f"Agent '{agent_name}' has invalid model: {config['model']}")


if __name__ == '__main__':
    unittest.main()
