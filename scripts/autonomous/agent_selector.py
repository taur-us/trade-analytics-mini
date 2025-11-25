#!/usr/bin/env python3
"""
Automatic agent selection based on task analysis.

Analyzes task description and determines which agents should be called
based on pattern matching and task complexity.
"""

import json
import logging
import re
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Optional

logger = logging.getLogger(__name__)


@dataclass
class DelegationPlan:
    """Plan for agent delegation."""

    task_id: str
    task_description: str
    primary_agent: str
    supporting_agents: List[str] = field(default_factory=list)
    rationale: str = ""
    agent_file_path: Optional[str] = None  # Path to custom agent file if applicable

    def all_agents(self) -> List[str]:
        """Get all agents in order (primary first, then supporting)."""
        return [self.primary_agent] + self.supporting_agents

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            'task_id': self.task_id,
            'task_description': self.task_description,
            'primary_agent': self.primary_agent,
            'supporting_agents': self.supporting_agents,
            'rationale': self.rationale,
            'agents': self.all_agents()
        }
        if self.agent_file_path:
            result['agent_file_path'] = self.agent_file_path
        return result


class AgentSelector:
    """Select appropriate agents for task based on pattern matching."""

    # Agent delegation matrix - maps patterns to agents
    AGENT_MATRIX = {
        # Database-related tasks
        "database": {
            "primary": "database-specialist",
            "supporting": ["test-engineer"],
            "patterns": ["database", "db", "postgresql", "query", "schema", "migration"]
        },
        # Optimization tasks
        "optimization": {
            "primary": "optuna-specialist",
            "supporting": ["optimization-engineer", "test-engineer"],
            "patterns": ["optun", "hyperparameter", "optimize", "tuning", "trial"]
        },
        # Feature development
        "feature": {
            "primary": "feature-developer",
            "supporting": ["test-engineer", "docs-specialist"],
            "patterns": ["implement", "add", "feature", "create", "build"]
        },
        # Bug fixes
        "bugfix": {
            "primary": "feature-developer",
            "supporting": ["test-engineer", "validation-agent"],
            "patterns": ["fix", "bug", "error", "issue", "problem", "broken"]
        },
        # API development
        "api": {
            "primary": "api-designer",
            "supporting": ["feature-developer", "test-engineer"],
            "patterns": ["api", "endpoint", "rest", "graphql", "webhook"]
        },
        # Security
        "security": {
            "primary": "security-reviewer",
            "supporting": ["test-engineer", "validation-agent"],
            "patterns": ["security", "vulnerability", "audit", "penetration", "exploit"]
        },
        # Refactoring
        "refactor": {
            "primary": "refactoring-specialist",
            "supporting": ["test-engineer", "validation-agent"],
            "patterns": ["refactor", "cleanup", "restructure", "reorganize", "simplify"]
        },
        # Documentation
        "documentation": {
            "primary": "docs-specialist",
            "supporting": [],
            "patterns": ["document", "docs", "readme", "guide", "tutorial"]
        },
        # WebSocket/async
        "websocket": {
            "primary": "websocket-specialist",
            "supporting": ["async-specialist", "test-engineer"],
            "patterns": ["websocket", "ws", "real-time", "streaming"]
        },
        # Async programming
        "async": {
            "primary": "async-specialist",
            "supporting": ["test-engineer"],
            "patterns": ["async", "asyncio", "concurrent", "parallel"]
        },
        # Performance
        "performance": {
            "primary": "performance-analyst",
            "supporting": ["optimization-engineer", "test-engineer"],
            "patterns": ["performance", "speed", "latency", "throughput", "benchmark"]
        },
        # Machine Learning
        "ml": {
            "primary": "ml-engineer",
            "supporting": ["data-analyst", "test-engineer"],
            "patterns": ["machine learning", "ml", "neural", "model", "training", "prediction"]
        },
        # Quantitative Analysis
        "quant": {
            "primary": "quant-analyst",
            "supporting": ["optuna-specialist", "risk-manager"],
            "patterns": ["strategy", "trading", "backtest", "sharpe", "metrics"]
        },
        # Testing
        "testing": {
            "primary": "test-engineer",
            "supporting": ["validation-agent"],
            "patterns": ["test", "coverage", "pytest", "unit test", "integration test"]
        },
        # DevOps/Infrastructure
        "devops": {
            "primary": "devops-engineer",
            "supporting": ["test-engineer"],
            "patterns": ["docker", "ci/cd", "deployment", "pipeline", "infrastructure"]
        },
        # Frontend
        "frontend": {
            "primary": "frontend-developer",
            "supporting": ["ui-designer", "test-engineer"],
            "patterns": ["frontend", "ui", "web", "react", "interface"]
        },
    }

    # Agents that are ALWAYS included
    ALWAYS_INCLUDE = ["validation-agent", "pr-review-agent"]

    def __init__(self, worktree_path: Optional[Path] = None):
        """Initialize agent selector with optional worktree path for agent discovery.

        Args:
            worktree_path: Path to worktree directory for discovering project-level agents.
                          If provided, will scan .claude/agents/ for custom agents.
        """
        self.delegation_history: List[DelegationPlan] = []
        self.worktree_path = worktree_path
        self._project_agents: Dict[str, Dict] = {}
        self._agent_file_paths: Dict[str, str] = {}  # Maps agent name to file path
        self._merged_matrix: Dict[str, Dict] = {}

        # Discover and merge project agents if worktree path provided
        if worktree_path:
            self._discover_project_agents()
            self._merge_agent_matrices()
        else:
            self._merged_matrix = self.AGENT_MATRIX.copy()

    def _discover_project_agents(self) -> None:
        """Discover agents from the project's .claude/agents/ directory.

        Scans for .md files with frontmatter containing agent definitions.
        """
        if not self.worktree_path:
            return

        agents_dir = self.worktree_path / ".claude" / "agents"

        if not agents_dir.exists():
            logger.debug(f"No .claude/agents/ directory found at {agents_dir}")
            return

        logger.info(f"Discovering agents from {agents_dir}")

        agent_files = list(agents_dir.glob("*.md"))

        if not agent_files:
            logger.debug(f"No agent files found in {agents_dir}")
            return

        for agent_file in agent_files:
            try:
                agent_config = self._parse_agent_file(agent_file)
                if agent_config:
                    agent_name = agent_config['name']
                    self._project_agents[agent_name] = agent_config
                    self._agent_file_paths[agent_name] = str(agent_file)
                    logger.info(f"Discovered project agent: {agent_name} from {agent_file.name}")
            except Exception as e:
                logger.warning(f"Failed to parse agent file {agent_file}: {e}")

        logger.info(f"Discovered {len(self._project_agents)} project-level agents")

    def _parse_agent_file(self, file_path: Path) -> Optional[Dict]:
        """Parse an agent file and extract configuration from frontmatter.

        Args:
            file_path: Path to the agent .md file

        Returns:
            Agent configuration dict or None if parsing failed
        """
        try:
            content = file_path.read_text(encoding='utf-8')

            # Extract YAML frontmatter
            if not content.startswith('---'):
                logger.debug(f"No frontmatter found in {file_path}")
                return None

            # Find the closing --- for frontmatter
            end_marker = content.find('---', 3)
            if end_marker == -1:
                logger.debug(f"Malformed frontmatter in {file_path}")
                return None

            frontmatter = content[3:end_marker].strip()
            metadata = yaml.safe_load(frontmatter)

            if not metadata or not isinstance(metadata, dict):
                logger.debug(f"Invalid frontmatter in {file_path}")
                return None

            # Extract required fields
            name = metadata.get('name')
            if not name:
                logger.debug(f"No 'name' field in {file_path}")
                return None

            description = metadata.get('description', '')
            patterns = metadata.get('patterns', [])

            # If no patterns specified, derive from name and description
            if not patterns:
                patterns = self._derive_patterns_from_name(name, description)

            # Convert to AGENT_MATRIX format
            # Create a category key from agent name
            category_key = name.replace('-', '_')

            return {
                'name': name,
                'category': category_key,
                'primary': name,
                'supporting': metadata.get('supporting', ['test-engineer']),
                'patterns': patterns,
                'description': description,
                'file_path': str(file_path)
            }

        except yaml.YAMLError as e:
            logger.warning(f"YAML parsing error in {file_path}: {e}")
            return None
        except Exception as e:
            logger.warning(f"Error reading {file_path}: {e}")
            return None

    def _derive_patterns_from_name(self, name: str, description: str) -> List[str]:
        """Derive search patterns from agent name and description.

        Args:
            name: Agent name (e.g., 'database-specialist')
            description: Agent description text

        Returns:
            List of patterns for matching
        """
        patterns = []

        # Add name parts as patterns
        name_parts = name.replace('-', ' ').replace('_', ' ').split()
        patterns.extend(name_parts)

        # Add common words from description (excluding stop words)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'for', 'to', 'of', 'in', 'on', 'with', 'is', 'are'}
        desc_words = description.lower().split()
        for word in desc_words:
            word = word.strip('.,;:!?()[]{}')
            if len(word) > 3 and word not in stop_words and word not in patterns:
                patterns.append(word)
                if len(patterns) >= 10:  # Limit patterns
                    break

        return patterns

    def _merge_agent_matrices(self) -> None:
        """Merge project agents with default AGENT_MATRIX.

        Project agents take precedence over default agents when names collide.
        """
        # Start with default matrix
        self._merged_matrix = self.AGENT_MATRIX.copy()

        # Add project agents, which override defaults on collision
        for agent_name, agent_config in self._project_agents.items():
            category = agent_config['category']

            # Check if this is overriding a default agent
            existing_category = None
            for cat_name, cat_info in self.AGENT_MATRIX.items():
                if cat_info['primary'] == agent_name:
                    existing_category = cat_name
                    break

            if existing_category:
                logger.info(f"Project agent '{agent_name}' overrides default agent")
                # Remove the old category entry
                del self._merged_matrix[existing_category]

            # Add the project agent
            self._merged_matrix[category] = {
                'primary': agent_config['primary'],
                'supporting': agent_config['supporting'],
                'patterns': agent_config['patterns']
            }

        logger.debug(f"Merged agent matrix has {len(self._merged_matrix)} categories")

    def get_discovered_agents(self) -> Dict[str, Dict]:
        """Get all discovered project-level agents.

        Returns:
            Dictionary mapping agent names to their configurations
        """
        return self._project_agents.copy()

    def get_agent_file_path(self, agent_name: str) -> Optional[str]:
        """Get the file path for a discovered agent.

        Args:
            agent_name: Name of the agent

        Returns:
            File path string or None if not a discovered agent
        """
        return self._agent_file_paths.get(agent_name)

    def analyze_task(self, task_id: str, task_description: str) -> DelegationPlan:
        """Analyze task and return delegation plan.

        Args:
            task_id: Task identifier (e.g., "TASK-057")
            task_description: Human-readable task description

        Returns:
            DelegationPlan with primary and supporting agents
        """
        logger.info(f"Analyzing task: {task_id}")

        # Convert to lowercase for pattern matching
        desc_lower = task_description.lower()

        # Score each category based on pattern matches
        scores = self._score_categories(desc_lower)

        # Get highest scoring category
        if not scores:
            # Default fallback
            logger.warning("No pattern matches, using default agent (feature-developer)")
            return DelegationPlan(
                task_id=task_id,
                task_description=task_description,
                primary_agent="feature-developer",
                supporting_agents=["test-engineer"] + self.ALWAYS_INCLUDE,
                rationale="No specific patterns matched, using general feature developer"
            )

        best_category = max(scores, key=scores.get)
        category_info = self._merged_matrix[best_category]

        # Build rationale
        matched_patterns = [p for p in category_info['patterns'] if p in desc_lower]
        rationale = f"Category: {best_category} (matched patterns: {', '.join(matched_patterns)})"

        # Create delegation plan
        supporting = category_info['supporting'] + self.ALWAYS_INCLUDE
        # Remove duplicates while preserving order
        supporting = list(dict.fromkeys(supporting))

        # Get agent file path if it's a discovered project agent
        agent_file_path = self.get_agent_file_path(category_info['primary'])

        plan = DelegationPlan(
            task_id=task_id,
            task_description=task_description,
            primary_agent=category_info['primary'],
            supporting_agents=supporting,
            rationale=rationale,
            agent_file_path=agent_file_path
        )

        self.delegation_history.append(plan)
        logger.info(f"Selected primary agent: {plan.primary_agent}")
        logger.info(f"Supporting agents: {', '.join(plan.supporting_agents)}")

        return plan

    def _score_categories(self, description: str) -> Dict[str, int]:
        """Score each category based on pattern matches.

        Args:
            description: Task description (lowercase)

        Returns:
            Dictionary mapping category to score
        """
        scores = {}

        for category, info in self._merged_matrix.items():
            score = 0
            for pattern in info['patterns']:
                # Count occurrences of pattern
                score += len(re.findall(r'\b' + re.escape(pattern) + r'\b', description))

            if score > 0:
                scores[category] = score

        return scores

    def suggest_agents(self, task_description: str) -> List[str]:
        """Quick agent suggestion without full delegation plan.

        Args:
            task_description: Task description

        Returns:
            List of suggested agent names
        """
        plan = self.analyze_task("suggested", task_description)
        return plan.all_agents()

    def save_plan(self, plan: DelegationPlan, output_path: Path) -> None:
        """Save delegation plan to file.

        Args:
            plan: Delegation plan to save
            output_path: Path to save JSON file
        """
        with open(output_path, 'w') as f:
            json.dump(plan.to_dict(), f, indent=2)

        logger.info(f"Delegation plan saved to {output_path}")

    def load_plan(self, input_path: Path) -> DelegationPlan:
        """Load delegation plan from file.

        Args:
            input_path: Path to JSON file

        Returns:
            DelegationPlan
        """
        with open(input_path, 'r') as f:
            data = json.load(f)

        return DelegationPlan(
            task_id=data['task_id'],
            task_description=data['task_description'],
            primary_agent=data['primary_agent'],
            supporting_agents=data['supporting_agents'],
            rationale=data.get('rationale', ''),
            agent_file_path=data.get('agent_file_path')
        )


def main():
    """CLI for testing agent selection."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python agent_selector.py <task_description>")
        sys.exit(1)

    task_description = ' '.join(sys.argv[1:])

    selector = AgentSelector()
    plan = selector.analyze_task("CLI-TEST", task_description)

    print(f"\n{'='*60}")
    print(f"Task: {task_description}")
    print(f"{'='*60}")
    print(f"\nPrimary Agent: {plan.primary_agent}")
    print(f"\nSupporting Agents:")
    for agent in plan.supporting_agents:
        print(f"  - {agent}")
    print(f"\nRationale: {plan.rationale}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
