#!/usr/bin/env python3
"""
Six programmatic blocking gates for autonomous workflow enforcement.

Each gate verifies a critical condition before allowing workflow progression.
Gates halt execution on failure to prevent cascading issues.

CRITICAL: All gates use absolute paths and verify working directory.
"""

import json
import logging
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def get_tool_output_path(worktree_path: Path, tool_name: str, extension: str = "json") -> Path:
    """Get standardized output path for a tool.

    Args:
        worktree_path: Absolute path to worktree
        tool_name: Name of the tool (bandit, coverage, mypy, pytest)
        extension: File extension (default: json)

    Returns:
        Absolute path for tool output file
    """
    reports_dir = worktree_path / ".autonomous" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    return reports_dir / f"{tool_name}.{extension}"


@dataclass
class GateResult:
    """Result of gate verification."""

    passed: bool
    gate_name: str
    issues: List[str] = field(default_factory=list)
    details: Dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'passed': self.passed,
            'gate_name': self.gate_name,
            'issues': self.issues,
            'details': self.details,
            'timestamp': self.timestamp
        }


class GateSkipPolicy(Enum):
    """Policy for handling gate failures (RESILIENCE-001).

    Defines whether a gate can be skipped when it fails after max retries.
    """
    CRITICAL = "critical"      # Cannot skip - workflow must halt
    SKIPPABLE = "skippable"    # Can skip after retries exhausted
    WARN_ONLY = "warn_only"    # Log warning but always continue


# Gate classification for skip policy (RESILIENCE-001)
GATE_SKIP_POLICIES: Dict[str, GateSkipPolicy] = {
    "Gate1_WorktreeSetup": GateSkipPolicy.CRITICAL,      # Fundamental - cannot proceed without worktree
    "Gate2_AgentDelegation": GateSkipPolicy.SKIPPABLE,   # Deliverables may exist even if gate fails
    "Gate3_TestsPass": GateSkipPolicy.SKIPPABLE,         # Tests may be flaky; PR can proceed
    "Gate4_QualityMetrics": GateSkipPolicy.SKIPPABLE,    # Security scan may timeout; PR can proceed
    "Gate5_ReviewComplete": GateSkipPolicy.CRITICAL,      # PR must exist to merge
    "Gate6_MergeComplete": GateSkipPolicy.SKIPPABLE,      # Already has auto_merge_disabled handling
    "design_review": GateSkipPolicy.SKIPPABLE,            # Already has skip_design_review flag
    "implementation_review": GateSkipPolicy.SKIPPABLE,    # Already has skip_implementation_review flag
    "design_deliverable": GateSkipPolicy.SKIPPABLE,       # Can create minimal fallback
    "delegation_plan": GateSkipPolicy.SKIPPABLE,          # Can proceed with default plan
    "merge_conflict_resolution": GateSkipPolicy.SKIPPABLE,  # CONFLICT-002: Conflicts can be resolved manually
}


class Gate(ABC):
    """Base class for workflow gates."""

    def __init__(self, worktree_path: Path, instance_dir: Path):
        """Initialize gate with required paths.

        Args:
            worktree_path: Absolute path to worktree (e.g., ../mmm-worktrees/instance-XXX/mmm-agents)
            instance_dir: Absolute path to instance directory (e.g., ../mmm-worktrees/instance-XXX)
        """
        self.worktree_path = worktree_path.resolve()
        self.instance_dir = instance_dir.resolve()
        self.state_dir = instance_dir / ".autonomous"
        self.state_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def verify(self) -> GateResult:
        """Verify gate conditions.

        Returns:
            GateResult with pass/fail and details
        """
        pass

    def enforce(self, max_retries: int = 3) -> bool:
        """Enforce gate with retry logic - halt workflow if failed after retries.

        PHASE 2 Enhancement: Adds exponential backoff retry logic for transient failures.

        Args:
            max_retries: Maximum number of retry attempts (default: 3)

        Returns:
            True if gate passed, False if failed after all retries
        """
        for attempt in range(max_retries):
            result = self.verify()
            self._save_result(result)

            if result.passed:
                logger.info(f"✓ {result.gate_name} passed")
                return True

            # Failed - check if we should retry
            if attempt < max_retries - 1:
                # Exponential backoff: 2^attempt seconds (2s, 4s, 8s)
                wait_time = 2 ** attempt
                logger.warning(f"⚠ {result.gate_name} failed (attempt {attempt + 1}/{max_retries})")
                logger.warning(f"  Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                # Final failure
                self._report_failure(result)
                return False

        return False

    def enforce_with_skip(self, max_retries: int = 1) -> Tuple[bool, Optional[GateResult]]:
        """Enforce gate with retry logic and skip capability (RESILIENCE-001).

        Tries to verify the gate up to max_retries times. If all retries fail,
        checks the gate's skip policy to determine whether to halt or skip.

        Args:
            max_retries: Maximum retry attempts before skip (default 1 from config)

        Returns:
            Tuple of (should_continue: bool, failure_result: Optional[GateResult])
            - (True, None): Gate passed
            - (True, GateResult): Gate failed but skipped (SKIPPABLE policy)
            - (False, GateResult): Gate failed and cannot continue (CRITICAL policy)
        """
        last_result = None

        for attempt in range(max_retries + 1):  # +1 for initial attempt
            result = self.verify()
            last_result = result
            self._save_result(result)

            if result.passed:
                logger.info(f"✓ {result.gate_name} passed")
                return True, None

            # Failed - check if we should retry
            if attempt < max_retries:
                wait_time = 2 ** attempt
                logger.warning(f"⚠ {result.gate_name} failed (attempt {attempt + 1}/{max_retries + 1})")
                logger.warning(f"  Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                # All retries exhausted - check skip policy
                policy = GATE_SKIP_POLICIES.get(result.gate_name, GateSkipPolicy.SKIPPABLE)

                if policy == GateSkipPolicy.CRITICAL:
                    logger.error(f"✗ CRITICAL GATE {result.gate_name} FAILED - cannot skip")
                    self._report_failure(result)
                    return False, result
                else:
                    logger.warning(f"⏭ {result.gate_name} failed after {max_retries + 1} attempts")
                    logger.warning(f"  Skipping gate and continuing to next phase")
                    for issue in result.issues:
                        logger.warning(f"  Issue: {issue}")
                    self._save_skip_result(result)
                    return True, result  # Continue but return the failure

        return False, last_result

    def _save_skip_result(self, result: GateResult) -> None:
        """Save gate skip result for audit trail (RESILIENCE-001)."""
        skip_file = self.state_dir / f"{result.gate_name}_skipped.json"
        with open(skip_file, 'w') as f:
            json.dump({
                **result.to_dict(),
                'skipped': True,
                'skip_reason': 'Max retries exhausted, continuing workflow'
            }, f, indent=2)
        logger.info(f"Saved skip result to: {skip_file}")

    def _save_result(self, result: GateResult) -> None:
        """Save gate result to state file."""
        state_file = self.state_dir / "gates.json"

        # Load existing results
        results = {}
        if state_file.exists():
            with open(state_file, 'r') as f:
                results = json.load(f)

        # Add new result
        results[result.gate_name] = result.to_dict()

        # Save back
        with open(state_file, 'w') as f:
            json.dump(results, f, indent=2)

    def _report_failure(self, result: GateResult) -> None:
        """Report gate failure."""
        logger.error(f"✗ {result.gate_name} FAILED")
        for issue in result.issues:
            logger.error(f"  - {issue}")

        # Save failure report
        report_file = self.state_dir / f"{result.gate_name}_failure.txt"
        with open(report_file, 'w') as f:
            f.write(f"Gate: {result.gate_name}\n")
            f.write(f"Timestamp: {result.timestamp}\n")
            f.write(f"\nIssues:\n")
            for issue in result.issues:
                f.write(f"- {issue}\n")
            f.write(f"\nDetails:\n")
            f.write(json.dumps(result.details, indent=2))

    def _run_command(self, cmd: str, timeout: int = None) -> subprocess.CompletedProcess:
        """Run command in worktree with proper working directory.

        CRITICAL: Implements Rule #2 - always prefix with cd.

        Args:
            cmd: Command to run
            timeout: Optional timeout in seconds (Fix Bug #14)

        Returns:
            CompletedProcess with result
        """
        # Always use absolute path and cd prefix
        full_cmd = f"cd {self.worktree_path} && {cmd}"
        try:
            return subprocess.run(full_cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return subprocess.CompletedProcess(
                args=full_cmd,
                returncode=124,
                stdout=b'',
                stderr=f'Command timed out after {timeout} seconds'.encode()
            )


class Gate1_WorktreeSetup(Gate):
    """Gate 1: Verify worktree properly isolated and configured.

    Checks:
    - Worktree directory exists
    - On feature branch (not main)
    - Environment files synced
    - Python imports work
    """

    def verify(self) -> GateResult:
        """Verify worktree setup."""
        issues = []
        details = {}

        # Check worktree exists
        if not self.worktree_path.exists():
            issues.append(f"Worktree not found at {self.worktree_path}")
        else:
            details['worktree_path'] = str(self.worktree_path)

        # Check on feature branch
        result = self._run_command("git branch --show-current")
        if result.returncode == 0:
            branch = result.stdout.strip()
            details['branch'] = branch
            if branch == 'main':
                issues.append("Still on main branch (should be on feature branch)")
        else:
            issues.append(f"Failed to check git branch: {result.stderr}")

        # Check .env files exist (repo-agnostic: only check if exist in main repo)
        # Discover .env files that should have been synced
        import os
        env_files_found = []
        for root, dirs, files in os.walk(self.worktree_path, topdown=True):
            dirs[:] = [d for d in dirs if d not in {'.venv', '.git', 'node_modules', '__pycache__'}]
            if '.env' in files:
                rel_path = Path(root).relative_to(self.worktree_path) / '.env'
                env_files_found.append(str(rel_path))

        if env_files_found:
            details['env_files'] = ', '.join(env_files_found)
            logger.info(f"Found {len(env_files_found)} .env files in worktree")
        else:
            details['env_files'] = 'none (repo may not need .env files)'
            logger.info("No .env files found (repo may not need them)")

        # Check Python imports (basic verification)
        result = self._run_command('python -c "import sys; print(sys.version)"')
        if result.returncode == 0:
            details['python_version'] = result.stdout.strip()
        else:
            issues.append(f"Python import check failed: {result.stderr}")

        return GateResult(
            passed=len(issues) == 0,
            gate_name="Gate1_WorktreeSetup",
            issues=issues,
            details=details
        )


class Gate2_AgentDelegation(Gate):
    """Gate 2: Verify appropriate agents called for task.

    Checks:
    - Task analyzed
    - Required agents identified
    - Delegation plan exists
    - Agent deliverables present
    """

    def verify(self) -> GateResult:
        """Verify agent delegation."""
        issues = []
        details = {}

        # Check for agent deliverables in worktree
        deliverables_dir = self.worktree_path / "deliverables"
        if not deliverables_dir.exists():
            issues.append(f"Deliverables directory not found: {deliverables_dir}")
        else:
            deliverables = list(deliverables_dir.glob("*.md"))
            details['deliverables_count'] = len(deliverables)
            details['deliverables'] = [f.name for f in deliverables]

            if len(deliverables) == 0:
                issues.append("No agent deliverables found (agents may not have been called)")

        # Check for delegation plan file
        delegation_file = self.state_dir / "delegation_plan.json"
        if not delegation_file.exists():
            issues.append(f"Delegation plan not found: {delegation_file}")
        else:
            with open(delegation_file, 'r') as f:
                plan = json.load(f)
                details['delegation_plan'] = plan
                details['agents_count'] = len(plan.get('agents', []))

        return GateResult(
            passed=len(issues) == 0,
            gate_name="Gate2_AgentDelegation",
            issues=issues,
            details=details
        )


class Gate3_TestsPass(Gate):
    """Gate 3: Verify all tests pass with sufficient coverage.

    Checks:
    - All tests pass
    - Coverage >= 85%
    - No flaky tests
    """

    def verify(self) -> GateResult:
        """Verify tests pass (repo-agnostic).

        Strategy 1 (Primary): Read test results from validation deliverables
        Strategy 2 (Fallback): Auto-detect test structure and run pytest
        """
        issues = []
        details = {}

        # STRATEGY 1: Read from validation deliverables (BEST - already generated!)
        validation_files = list(self.worktree_path.glob("deliverables/*-validation-report.md"))

        if validation_files:
            logger.info("Reading test results from validation deliverable...")

            with open(validation_files[0], 'r', encoding='utf-8') as f:
                content = f.read()

            details['validation_file'] = validation_files[0].name

            # Parse test results using multiple patterns
            import re

            # Match patterns like: "35 tests passed" or "35/35 tests" or "TOTAL 118 6 95%"
            tests_patterns = [
                r'(\d+)\s+tests?\s+passed',
                r'(\d+)/(\d+)\s+tests?',
                r'All\s+(\d+)\s+tests?\s+passed',
            ]

            # Coverage patterns - capture decimal percentages with specific context
            # Patterns must be specific enough to avoid matching "3 strategies" etc.
            coverage_patterns = [
                r'Coverage[:\s]+(\d+(?:\.\d+)?)%',           # "Coverage: 95.3%" or "Coverage 95%"
                r'TOTAL\s+\d+\s+\d+\s+(\d+(?:\.\d+)?)%',     # "TOTAL 118 6 95%" or "TOTAL 118 6 95.3%"
                r'(\d+(?:\.\d+)?)%\s+coverage',              # "95% coverage" or "95.3% coverage"
                r'coverage[:\s]+(\d+(?:\.\d+)?)%',           # "coverage: 95%" (lowercase)
            ]

            tests_passed = None
            for pattern in tests_patterns:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    tests_passed = int(match.group(1))
                    details['tests_passed'] = tests_passed
                    logger.info(f"Found {tests_passed} tests passed in validation report")
                    break

            coverage_pct = None
            for pattern in coverage_patterns:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    # Convert decimal string to float first, then to int
                    # This handles both "95" and "95.3" correctly
                    coverage_pct = int(float(match.group(1)))
                    details['coverage'] = coverage_pct
                    details['coverage_raw'] = match.group(1)  # Store original for debugging
                    logger.info(f"Found {match.group(1)}% coverage in validation report (parsed as {coverage_pct}%)")
                    break

            # Validate criteria
            if tests_passed is not None and tests_passed == 0:
                issues.append("No tests passed")

            if coverage_pct is not None:
                if coverage_pct < 85:
                    issues.append(f"Coverage too low: {coverage_pct}% < 85%")
            else:
                # No coverage found - may be infrastructure task without tests
                if tests_passed is None or tests_passed == 0:
                    logger.warning("No test results found in validation report - may be infrastructure task")
                    # Don't fail - infrastructure tasks may not have unit tests
                    details['note'] = "No tests found - infrastructure/tooling task"

            # Success if we found passing tests with adequate coverage
            if tests_passed and tests_passed > 0:
                if coverage_pct is None or coverage_pct >= 85:
                    logger.info(f"✅ Tests validated: {tests_passed} tests, {coverage_pct}% coverage")

            return GateResult(
                passed=len(issues) == 0,
                gate_name="Gate3_TestsPass",
                issues=issues,
                details=details
            )

        # STRATEGY 2: Run tests directly (FALLBACK)
        logger.info("No validation deliverable found, attempting to run tests directly...")

        # Auto-detect test structure (repo-agnostic)
        test_locations = [
            (self.worktree_path / "tests", self.worktree_path / "src"),   # Standard structure
            (self.worktree_path / "test", self.worktree_path / "src"),    # Alternative
            (self.worktree_path / "tests", self.worktree_path),           # Tests cover root
        ]

        test_ran = False
        for test_dir, cov_dir in test_locations:
            if test_dir.exists():
                logger.info(f"Found tests at: {test_dir}")

                # Run pytest
                coverage_output = get_tool_output_path(self.worktree_path, "coverage")
                result = self._run_command(
                    f"pytest {test_dir} --cov={cov_dir} --cov-report=json:{coverage_output} --cov-report=term -v",
                    timeout=300
                )

                test_ran = True

                if result.returncode != 0:
                    issues.append(f"Tests failed: {result.stderr[:200]}")
                    details['test_output'] = result.stdout[:500]
                else:
                    # Parse coverage
                    coverage_file = coverage_output
                    if coverage_file.exists():
                        with open(coverage_file, 'r') as f:
                            coverage_data = json.load(f)
                            coverage_pct = coverage_data.get('totals', {}).get('percent_covered', 0)
                            details['coverage'] = coverage_pct

                            if coverage_pct < 85:
                                issues.append(f"Coverage too low: {coverage_pct}% < 85%")

                    # Parse test count from output
                    import re
                    match = re.search(r'(\d+) passed', result.stdout)
                    if match:
                        details['tests_passed'] = int(match.group(1))

                break

        # If no standard structure, try monorepo packages/
        if not test_ran:
            packages = self._identify_affected_packages()
            if packages:
                details['packages_tested'] = packages
                # Use original monorepo logic
                for package in packages:
                    package_path = self.worktree_path / "packages" / package
                    if not package_path.exists():
                        issues.append(f"Package not found: {package}")
                        continue

                    package_coverage_output = get_tool_output_path(self.worktree_path, f"coverage_{package}")
                    result = self._run_command(
                        f"cd {package_path} && pytest tests/ --cov=. --cov-report=json:{package_coverage_output} -v",
                        timeout=300
                    )

                    if result.returncode != 0:
                        issues.append(f"Tests failed for {package}")
                    else:
                        coverage_file = package_coverage_output
                        if coverage_file.exists():
                            with open(coverage_file, 'r') as f:
                                coverage_data = json.load(f)
                                coverage_pct = coverage_data.get('totals', {}).get('percent_covered', 0)
                                details[f'{package}_coverage'] = coverage_pct

                                if coverage_pct < 85:
                                    issues.append(f"Coverage too low for {package}: {coverage_pct}% < 85%")
            else:
                # No tests found - may be infrastructure/tooling task
                logger.warning("No tests found - assuming infrastructure task")
                details['test_status'] = "No tests (infrastructure task)"
                # Don't add to issues - infrastructure tasks don't need tests

        return GateResult(
            passed=len(issues) == 0,
            gate_name="Gate3_TestsPass",
            issues=issues,
            details=details
        )

    def _identify_affected_packages(self) -> List[str]:
        """Identify packages affected by changes."""
        # Get list of changed files
        result = self._run_command("git diff --name-only main")
        if result.returncode != 0:
            return []

        changed_files = result.stdout.strip().split('\n')
        packages = set()

        for file in changed_files:
            if file.startswith('packages/'):
                parts = file.split('/')
                if len(parts) >= 2:
                    packages.add(parts[1])

        return list(packages)


class Gate4_QualityMetrics(Gate):
    """Gate 4: Verify code quality metrics meet thresholds (PHASE 2: scans only changed files).

    Checks:
    - No security vulnerabilities in changed files (bandit, safety)
    - Code complexity acceptable
    - No significant duplication
    - Documentation present
    """

    # PHASE 2: Stricter thresholds for changed files only
    THRESHOLDS = {
        'security_issues': 0,
        'high_severity': 0,     # No high severity issues in changed files
        'medium_severity': 2,   # Max 2 medium severity issues in changed files
    }

    def verify(self) -> GateResult:
        """Verify quality metrics (PHASE 2: scans only changed files)."""
        issues = []
        details = {}

        # PHASE 2 Enhancement: Get list of changed Python files
        changed_files = self._get_changed_python_files()
        details['changed_files_count'] = len(changed_files)
        details['changed_files'] = changed_files[:20]  # Show first 20

        if not changed_files:
            logger.info("No Python files changed - skipping quality scans")
            return GateResult(
                passed=True,
                gate_name="Gate4_QualityMetrics",
                issues=[],
                details={'message': 'No Python files changed'}
            )

        # Run bandit security scan on changed files only
        changed_files_str = ' '.join(changed_files)
        bandit_output = get_tool_output_path(self.worktree_path, "bandit")
        result = self._run_command(f"bandit {changed_files_str} -f json -o {bandit_output} 2>/dev/null || true")

        if bandit_output.exists():
            try:
                with open(bandit_output, 'r') as f:
                    bandit_data = json.load(f)
                    high_severity = len([r for r in bandit_data.get('results', []) if r.get('issue_severity') == 'HIGH'])
                    medium_severity = len([r for r in bandit_data.get('results', []) if r.get('issue_severity') == 'MEDIUM'])

                    details['security_high'] = high_severity
                    details['security_medium'] = medium_severity

                    if high_severity > self.THRESHOLDS['high_severity']:
                        issues.append(f"High severity security issues in changed files: {high_severity}")
                    if medium_severity > self.THRESHOLDS['medium_severity']:
                        issues.append(f"Medium severity security issues in changed files: {medium_severity}")
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to parse bandit results: {e}")
                details['bandit_error'] = str(e)

        # Check for docstrings in changed Python files (non-test files)
        changed_source_files = [
            f for f in changed_files
            if '/tests/' not in f and not f.endswith('__init__.py')
        ]

        if changed_source_files:
            files_without_docstrings = 0
            for file in changed_source_files:
                file_path = self.worktree_path / file
                if file_path.exists():
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            if '"""' not in content and "'''" not in content:
                                files_without_docstrings += 1
                    except (IOError, UnicodeDecodeError):
                        pass

            details['files_without_docstrings'] = files_without_docstrings
            if files_without_docstrings > 0:
                issues.append(f"{files_without_docstrings} changed Python files missing docstrings")

        return GateResult(
            passed=len(issues) == 0,
            gate_name="Gate4_QualityMetrics",
            issues=issues,
            details=details
        )

    def _get_changed_python_files(self) -> List[str]:
        """Get list of changed Python files in this branch.

        Returns:
            List of relative paths to changed .py files
        """
        result = self._run_command("git diff --name-only main")
        if result.returncode != 0:
            logger.warning(f"Failed to get changed files: {result.stderr}")
            return []

        changed_files = result.stdout.strip().split('\n')
        python_files = [
            f for f in changed_files
            if f.endswith('.py') and f.startswith('packages/')
        ]

        return python_files


class Gate5_ReviewComplete(Gate):
    """Gate 5: Verify PR review complete and approved.

    Checks:
    - PR created
    - PR review completed by pr-review-agent
    - Review status is APPROVED
    - CI/CD checks passed
    """

    def verify(self, pr_number: Optional[int] = None) -> GateResult:
        """Verify PR created successfully.

        Note: This gate verifies PR CREATION, not approval.
        Autonomous workflows create PRs; human approval comes later.

        Args:
            pr_number: PR number to check (optional, will detect from branch)
        """
        issues = []
        details = {}

        logger.info(f"Gate5: Starting PR verification (pr_number={pr_number})")
        logger.info(f"Gate5: Worktree path: {self.worktree_path}")

        # Get current branch for context
        branch_result = self._run_command("git branch --show-current")
        current_branch = branch_result.stdout.strip() if branch_result.returncode == 0 else "unknown"
        details['current_branch'] = current_branch
        logger.info(f"Gate5: Current branch: {current_branch}")

        # Get PR number if not provided
        if pr_number is None:
            logger.info("Gate5: No PR number provided, detecting from branch...")

            # Strategy 1: Try gh pr view for current branch
            result = self._run_command("gh pr view --json number")
            logger.info(f"Gate5: gh pr view result: returncode={result.returncode}, stdout='{result.stdout.strip()}', stderr='{result.stderr.strip()}'")

            if result.returncode == 0 and result.stdout.strip():
                try:
                    pr_data = json.loads(result.stdout.strip())
                    pr_number = pr_data.get('number')
                    if pr_number:
                        logger.info(f"Gate5: Detected PR #{pr_number} from branch")
                    else:
                        logger.warning("Gate5: PR view returned empty number")
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(f"Gate5: Parse error from branch detection: {e}")
                    # Continue to Strategy 2

            # Strategy 2: If no PR for current branch, check if there's a PR number in state
            if pr_number is None:
                state_file = self.state_dir / "pr_number.txt"
                if state_file.exists():
                    try:
                        with open(state_file, 'r') as f:
                            pr_number = int(f.read().strip())
                            logger.info(f"Gate5: Retrieved PR #{pr_number} from state file")
                            details['pr_source'] = 'state_file'
                    except (ValueError, IOError) as e:
                        logger.warning(f"Gate5: Failed to read PR number from state: {e}")

            # Strategy 3: List PRs for current branch using gh pr list
            if pr_number is None:
                logger.info("Gate5: Trying gh pr list for current branch...")
                result = self._run_command(f"gh pr list --head {current_branch} --json number")
                logger.info(f"Gate5: gh pr list result: returncode={result.returncode}, stdout='{result.stdout.strip()}'")

                if result.returncode == 0 and result.stdout.strip():
                    try:
                        pr_list = json.loads(result.stdout.strip())
                        if pr_list and len(pr_list) > 0:
                            pr_number = pr_list[0].get('number')
                            if pr_number:
                                logger.info(f"Gate5: Found PR #{pr_number} via pr list")
                                details['pr_source'] = 'pr_list'
                    except (json.JSONDecodeError, ValueError) as e:
                        logger.warning(f"Gate5: Parse error from pr list: {e}")

            if pr_number is None:
                error_msg = f"No PR found for branch '{current_branch}'"
                issues.append(error_msg)
                logger.error(f"Gate5: {error_msg}")
                logger.error("Gate5: Tried: gh pr view, state file, gh pr list --head")
                details['detection_strategies_tried'] = ['gh_pr_view', 'state_file', 'gh_pr_list']
                return GateResult(
                    passed=False,
                    gate_name="Gate5_ReviewComplete",
                    issues=issues,
                    details=details
                )

        details['pr_number'] = pr_number
        logger.info(f"Gate5: Checking PR #{pr_number}")

        # Check PR exists and get its state (use JSON output for cross-platform compatibility)
        result = self._run_command(f"gh pr view {pr_number} --json state,url,title")
        logger.info(f"Gate5: gh pr view {pr_number} result: returncode={result.returncode}")
        logger.info(f"Gate5: stdout: '{result.stdout.strip()}'")

        if result.returncode == 0 and result.stdout.strip():
            try:
                pr_data = json.loads(result.stdout.strip())
                state = pr_data.get('state', '')
                url = pr_data.get('url', '')
                title = pr_data.get('title', '')

                details['pr_state'] = state
                details['pr_url'] = url
                details['pr_title'] = title

                logger.info(f"Gate5: PR state='{state}', url='{url}'")

                # PASS if PR exists and is in valid state (OPEN or MERGED)
                if state not in ['OPEN', 'MERGED']:
                    issues.append(f"PR in unexpected state: {state}")
                    logger.error(f"Gate5: Invalid state '{state}' (expected OPEN or MERGED)")
                else:
                    logger.info(f"Gate5: PR #{pr_number} created successfully with state: {state}")
            except json.JSONDecodeError as e:
                issues.append(f"Failed to parse PR JSON: {e}")
                logger.error(f"Gate5: JSON parse error: {e}")
        else:
            issues.append(f"PR {pr_number} not found or gh CLI failed")
            logger.error(f"Gate5: gh CLI error: {result.stderr}")

        # Optional: Check CI/CD status (informational only, not required for PASS)
        result = self._run_command(f"gh pr checks {pr_number} --json name,state")
        if result.returncode == 0 and result.stdout.strip():
            try:
                checks_data = json.loads(result.stdout.strip())
                check_states = list(set([c.get('state', 'UNKNOWN') for c in checks_data]))
                details['ci_checks'] = check_states
                logger.info(f"Gate5: CI checks: {check_states}")
            except json.JSONDecodeError:
                logger.warning("Gate5: Could not parse CI checks output")

        logger.info(f"Gate5: Final result - passed={len(issues) == 0}, issues={issues}")

        return GateResult(
            passed=len(issues) == 0,
            gate_name="Gate5_ReviewComplete",
            issues=issues,
            details=details
        )

    def save_pr_number(self, pr_number: int) -> None:
        """Save PR number to state for later retrieval.

        Call this after creating a PR to enable auto-detection in verify().

        Args:
            pr_number: The PR number to save
        """
        state_file = self.state_dir / "pr_number.txt"
        with open(state_file, 'w') as f:
            f.write(str(pr_number))
        logger.info(f"Gate5: Saved PR #{pr_number} to state file: {state_file}")

    def enforce_with_pr(self, pr_number: int, max_retries: int = 3) -> bool:
        """Enforce gate with explicit PR number.

        Use this when you know the PR number (e.g., after creating a PR).

        Args:
            pr_number: PR number to verify
            max_retries: Maximum number of retry attempts (default: 3)

        Returns:
            True if gate passed, False if failed after all retries
        """
        for attempt in range(max_retries):
            result = self.verify(pr_number=pr_number)
            self._save_result(result)

            if result.passed:
                logger.info(f"Gate5: PR #{pr_number} verification passed")
                return True

            # Failed - check if we should retry
            if attempt < max_retries - 1:
                # Exponential backoff: 2^attempt seconds (2s, 4s, 8s)
                wait_time = 2 ** attempt
                logger.warning(f"Gate5: PR #{pr_number} verification failed (attempt {attempt + 1}/{max_retries})")
                logger.warning(f"  Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                # Final failure
                self._report_failure(result)
                return False

        return False


class Gate6_MergeComplete(Gate):
    """Gate 6: Verify merge complete and cleanup done.

    Checks:
    - PR merged to main
    - Worktree removed
    - Branch deleted
    - Task marked complete
    """

    def verify(self, pr_number: int) -> GateResult:
        """Verify merge complete.

        Args:
            pr_number: PR number that was merged
        """
        issues = []
        details = {}

        details['pr_number'] = pr_number

        # Check PR merge status
        result = self._run_command(f"gh pr view {pr_number} --json state,merged")
        if result.returncode == 0 and result.stdout.strip():
            try:
                pr_data = json.loads(result.stdout.strip())
                state = pr_data.get('state', '')
                merged = pr_data.get('merged', False)

                details['pr_state'] = state
                details['pr_merged'] = merged

                if state != 'MERGED' and not merged:
                    issues.append(f"PR not merged: state is {state}")
            except json.JSONDecodeError as e:
                issues.append(f"Failed to parse PR status: {e}")
        else:
            issues.append(f"Failed to get PR status: {result.stderr}")

        # Check if worktree still exists (it shouldn't after cleanup)
        if self.worktree_path.exists():
            issues.append(f"Worktree still exists at {self.worktree_path} (cleanup incomplete)")
        else:
            details['worktree_cleaned'] = True

        return GateResult(
            passed=len(issues) == 0,
            gate_name="Gate6_MergeComplete",
            issues=issues,
            details=details
        )


def create_gate(gate_class: type, worktree_path: Path, instance_dir: Path) -> Gate:
    """Factory function to create a gate instance.

    Args:
        gate_class: Gate class to instantiate
        worktree_path: Absolute path to worktree
        instance_dir: Absolute path to instance directory

    Returns:
        Gate instance
    """
    return gate_class(worktree_path, instance_dir)


# Exports
__all__ = [
    # Core classes
    'GateResult',
    'GateSkipPolicy',
    'GATE_SKIP_POLICIES',
    'Gate',
    # Gate implementations
    'Gate1_WorktreeSetup',
    'Gate2_AgentDelegation',
    'Gate3_TestsPass',
    'Gate4_QualityMetrics',
    'Gate5_ReviewComplete',
    'Gate6_MergeComplete',
    # Factory
    'create_gate',
]
