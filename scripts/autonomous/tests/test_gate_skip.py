"""Tests for gate skip behavior (RESILIENCE-001).

Tests the self-unblocking feature where gates can fail, log the failure,
update task metadata, and continue to next phase.
"""

import json
import pytest
import time
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch, PropertyMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from autonomous.gates import (
    Gate,
    GateResult,
    GateSkipPolicy,
    GATE_SKIP_POLICIES,
    Gate1_WorktreeSetup,
    Gate2_AgentDelegation,
    Gate3_TestsPass,
    Gate4_QualityMetrics,
    Gate5_ReviewComplete,
    Gate6_MergeComplete,
    create_gate
)


class MockGate(Gate):
    """Mock gate for testing skip behavior."""

    def __init__(self, worktree_path, instance_dir, pass_on_attempt=None, gate_name="MockGate"):
        super().__init__(worktree_path, instance_dir)
        self.pass_on_attempt = pass_on_attempt  # Attempt number (0-indexed) to pass on
        self.attempt_count = 0
        self._gate_name = gate_name

    def verify(self):
        """Return pass/fail based on configured attempt number."""
        should_pass = (
            self.pass_on_attempt is not None and
            self.attempt_count >= self.pass_on_attempt
        )
        self.attempt_count += 1

        return GateResult(
            passed=should_pass,
            gate_name=self._gate_name,
            issues=[] if should_pass else ["Mock failure"],
            details={"attempt": self.attempt_count}
        )


class TestGateSkipPolicy:
    """Test GateSkipPolicy enum and policies mapping."""

    def test_policy_enum_values(self):
        """Test GateSkipPolicy has expected values."""
        assert GateSkipPolicy.CRITICAL.value == "critical"
        assert GateSkipPolicy.SKIPPABLE.value == "skippable"
        assert GateSkipPolicy.WARN_ONLY.value == "warn_only"

    def test_critical_gates_defined(self):
        """Test that critical gates are correctly marked."""
        assert GATE_SKIP_POLICIES["Gate1_WorktreeSetup"] == GateSkipPolicy.CRITICAL
        assert GATE_SKIP_POLICIES["Gate5_ReviewComplete"] == GateSkipPolicy.CRITICAL

    def test_skippable_gates_defined(self):
        """Test that skippable gates are correctly marked."""
        assert GATE_SKIP_POLICIES["Gate2_AgentDelegation"] == GateSkipPolicy.SKIPPABLE
        assert GATE_SKIP_POLICIES["Gate3_TestsPass"] == GateSkipPolicy.SKIPPABLE
        assert GATE_SKIP_POLICIES["Gate4_QualityMetrics"] == GateSkipPolicy.SKIPPABLE
        assert GATE_SKIP_POLICIES["Gate6_MergeComplete"] == GateSkipPolicy.SKIPPABLE


class TestEnforceWithSkip:
    """Test enforce_with_skip() method behavior."""

    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for testing."""
        with TemporaryDirectory() as worktree:
            with TemporaryDirectory() as instance:
                worktree_path = Path(worktree)
                instance_path = Path(instance)
                # Create required subdirectory structure
                state_dir = instance_path / ".autonomous"
                state_dir.mkdir(parents=True)
                yield worktree_path, instance_path

    def test_pass_on_first_attempt(self, temp_dirs):
        """Test gate that passes on first attempt returns (True, None)."""
        worktree_path, instance_path = temp_dirs
        gate = MockGate(worktree_path, instance_path, pass_on_attempt=0)

        # Patch GATE_SKIP_POLICIES to include MockGate
        with patch.dict(GATE_SKIP_POLICIES, {"MockGate": GateSkipPolicy.SKIPPABLE}):
            should_continue, failure_result = gate.enforce_with_skip(max_retries=1)

        assert should_continue is True
        assert failure_result is None
        assert gate.attempt_count == 1

    def test_pass_on_retry(self, temp_dirs):
        """Test gate that fails first then passes on retry."""
        worktree_path, instance_path = temp_dirs
        gate = MockGate(worktree_path, instance_path, pass_on_attempt=1)

        with patch.dict(GATE_SKIP_POLICIES, {"MockGate": GateSkipPolicy.SKIPPABLE}):
            should_continue, failure_result = gate.enforce_with_skip(max_retries=1)

        assert should_continue is True
        assert failure_result is None
        assert gate.attempt_count == 2  # Initial + 1 retry

    def test_skip_on_skippable_gate(self, temp_dirs):
        """Test skippable gate returns (True, GateResult) after max retries."""
        worktree_path, instance_path = temp_dirs
        gate = MockGate(worktree_path, instance_path, pass_on_attempt=None)  # Never passes

        with patch.dict(GATE_SKIP_POLICIES, {"MockGate": GateSkipPolicy.SKIPPABLE}):
            should_continue, failure_result = gate.enforce_with_skip(max_retries=1)

        assert should_continue is True  # Can continue despite failure
        assert failure_result is not None  # But failure is recorded
        assert failure_result.passed is False
        assert failure_result.gate_name == "MockGate"
        assert gate.attempt_count == 2  # Initial + 1 retry

    def test_halt_on_critical_gate(self, temp_dirs):
        """Test critical gate returns (False, GateResult) when it fails."""
        worktree_path, instance_path = temp_dirs
        gate = MockGate(worktree_path, instance_path, pass_on_attempt=None, gate_name="Gate1_WorktreeSetup")

        # Gate1_WorktreeSetup is already CRITICAL in GATE_SKIP_POLICIES
        should_continue, failure_result = gate.enforce_with_skip(max_retries=1)

        assert should_continue is False  # Cannot continue
        assert failure_result is not None
        assert failure_result.passed is False
        assert gate.attempt_count == 2

    def test_skip_result_saved(self, temp_dirs):
        """Test that skip result is saved to file."""
        worktree_path, instance_path = temp_dirs
        gate = MockGate(worktree_path, instance_path, pass_on_attempt=None)

        with patch.dict(GATE_SKIP_POLICIES, {"MockGate": GateSkipPolicy.SKIPPABLE}):
            gate.enforce_with_skip(max_retries=0)

        skip_file = instance_path / ".autonomous" / "MockGate_skipped.json"
        assert skip_file.exists()

        with open(skip_file) as f:
            skip_data = json.load(f)

        assert skip_data["skipped"] is True
        assert skip_data["gate_name"] == "MockGate"
        assert "skip_reason" in skip_data

    def test_retry_count_with_different_max_retries(self, temp_dirs):
        """Test that correct number of retries are attempted."""
        worktree_path, instance_path = temp_dirs

        # Test with 0 retries
        gate0 = MockGate(worktree_path, instance_path, pass_on_attempt=None)
        with patch.dict(GATE_SKIP_POLICIES, {"MockGate": GateSkipPolicy.SKIPPABLE}):
            gate0.enforce_with_skip(max_retries=0)
        assert gate0.attempt_count == 1  # Just initial attempt

        # Test with 2 retries
        gate2 = MockGate(worktree_path, instance_path, pass_on_attempt=None)
        with patch.dict(GATE_SKIP_POLICIES, {"MockGate": GateSkipPolicy.SKIPPABLE}):
            gate2.enforce_with_skip(max_retries=2)
        assert gate2.attempt_count == 3  # Initial + 2 retries


class TestGateFailureDataclass:
    """Test GateFailure dataclass from spawn_orchestrator."""

    def test_gate_failure_creation(self):
        """Test GateFailure can be created and serialized."""
        # Import here to avoid circular dependency issues
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from spawn_orchestrator import GateFailure

        failure = GateFailure(
            gate_name="Gate3_TestsPass",
            phase="test",
            timestamp=datetime.now().isoformat(),
            error_message="Gate failed after 2 attempts",
            retry_attempts=2,
            issues=["Tests failed: 5 failures"],
            details={"tests_passed": 10, "tests_failed": 5}
        )

        assert failure.gate_name == "Gate3_TestsPass"
        assert failure.phase == "test"
        assert failure.retry_attempts == 2

    def test_gate_failure_to_dict(self):
        """Test GateFailure.to_dict() serialization."""
        from spawn_orchestrator import GateFailure

        timestamp = datetime.now().isoformat()
        failure = GateFailure(
            gate_name="Gate4_QualityMetrics",
            phase="test",
            timestamp=timestamp,
            error_message="Security scan failed",
            retry_attempts=1,
            issues=["High severity: SQL injection found"],
            details={"severity": "high"}
        )

        d = failure.to_dict()

        assert d["gate_name"] == "Gate4_QualityMetrics"
        assert d["phase"] == "test"
        assert d["timestamp"] == timestamp
        assert d["error_message"] == "Security scan failed"
        assert d["retry_attempts"] == 1
        assert d["issues"] == ["High severity: SQL injection found"]
        assert d["details"] == {"severity": "high"}


class TestWorkflowStateWithFailures:
    """Test WorkflowState with gate_failures tracking."""

    def test_workflow_state_includes_gate_failures(self):
        """Test WorkflowState tracks gate failures."""
        from spawn_orchestrator import WorkflowState, GateFailure

        state = WorkflowState("TEST-001", "20231125-120000")

        # Add a gate failure
        failure = GateFailure(
            gate_name="Gate3_TestsPass",
            phase="test",
            timestamp=datetime.now().isoformat(),
            error_message="Tests failed",
            retry_attempts=1,
            issues=["Test timeout"],
            details={}
        )
        state.gate_failures.append(failure)

        # Verify serialization
        d = state.to_dict()

        assert "gate_failures" in d
        assert len(d["gate_failures"]) == 1
        assert d["gate_failures"][0]["gate_name"] == "Gate3_TestsPass"


class TestMaxGateRetriesConfig:
    """Test max_gate_retries configuration loading."""

    @pytest.fixture
    def temp_config_dir(self):
        """Create temp directory with config file."""
        with TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / ".autonomous"
            config_dir.mkdir()
            yield Path(tmpdir)

    def test_default_max_retries(self, temp_config_dir):
        """Test default value when no config file exists."""
        from spawn_orchestrator import SpawnOrchestrator

        # Create orchestrator with non-existent config
        with patch.object(SpawnOrchestrator, '__init__', lambda x: None):
            orch = SpawnOrchestrator.__new__(SpawnOrchestrator)
            orch.main_repo = temp_config_dir

            # Remove config file if it exists
            config_file = temp_config_dir / ".autonomous" / "orchestrator_config.json"
            if config_file.exists():
                config_file.unlink()

            assert orch._get_max_gate_retries() == 1

    def test_configured_max_retries(self, temp_config_dir):
        """Test reading max_gate_retries from config file."""
        from spawn_orchestrator import SpawnOrchestrator

        # Create config file with custom value
        config_file = temp_config_dir / ".autonomous" / "orchestrator_config.json"
        with open(config_file, 'w') as f:
            json.dump({"max_gate_retries": 3}, f)

        with patch.object(SpawnOrchestrator, '__init__', lambda x: None):
            orch = SpawnOrchestrator.__new__(SpawnOrchestrator)
            orch.main_repo = temp_config_dir

            assert orch._get_max_gate_retries() == 3


class TestIntegrationScenarios:
    """Integration tests for gate skip scenarios."""

    @pytest.fixture
    def mock_orchestrator(self):
        """Create mock orchestrator for integration testing."""
        from spawn_orchestrator import SpawnOrchestrator, WorkflowState, Phase

        with TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create directory structure
            worktree = tmppath / "worktree"
            worktree.mkdir()
            instance_dir = tmppath / "instance"
            instance_dir.mkdir()
            state_dir = instance_dir / ".autonomous"
            state_dir.mkdir()
            deliverables = worktree / "deliverables"
            deliverables.mkdir()

            with patch.object(SpawnOrchestrator, '__init__', lambda x: None):
                orch = SpawnOrchestrator.__new__(SpawnOrchestrator)
                orch.task_id = "TEST-001"
                orch.main_repo = tmppath
                orch.worktree_path = worktree
                orch.instance_dir = instance_dir
                orch.state_dir = state_dir
                orch.deliverables_path = deliverables
                orch.task_details = {"title": "Test Task", "description": "Test"}
                orch.state = WorkflowState("TEST-001", "20231125-120000")

                yield orch

    def test_simulate_gate_failure_and_skip(self, mock_orchestrator):
        """Simulate a gate failure that gets skipped."""
        from spawn_orchestrator import Phase, GateResult

        # Simulate design review failure
        result = GateResult(
            passed=False,
            gate_name="design_review",
            issues=["Design incomplete"],
            details={}
        )

        # Record the failure
        mock_orchestrator._record_gate_failure(Phase.DESIGN_REVIEW, result)

        # Verify failure was recorded
        assert len(mock_orchestrator.state.gate_failures) == 1
        failure = mock_orchestrator.state.gate_failures[0]
        assert failure.gate_name == "design_review"
        assert failure.phase == "design_review"
        assert "Design incomplete" in failure.issues

    def test_verify_gate_failures_in_state_file(self, mock_orchestrator):
        """Test that gate failures are saved in state file."""
        from spawn_orchestrator import Phase, GateResult
        import json

        # Add mock _save_state method
        state_file = mock_orchestrator.state_dir / "state.json"
        def mock_save_state():
            with open(state_file, 'w') as f:
                json.dump(mock_orchestrator.state.to_dict(), f, indent=2)
        mock_orchestrator._save_state = mock_save_state

        # Simulate failures
        for i, gate_name in enumerate(["Gate3_TestsPass", "Gate4_QualityMetrics"]):
            result = GateResult(
                passed=False,
                gate_name=gate_name,
                issues=[f"Failure {i}"],
                details={}
            )
            mock_orchestrator._record_gate_failure(Phase.TEST, result)

        # Save state
        mock_orchestrator._save_state()

        # Verify file contents
        with open(state_file) as f:
            state_data = json.load(f)

        assert len(state_data["gate_failures"]) == 2
        gate_names = [f["gate_name"] for f in state_data["gate_failures"]]
        assert "Gate3_TestsPass" in gate_names
        assert "Gate4_QualityMetrics" in gate_names


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
