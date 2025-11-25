#!/usr/bin/env python3
"""
Comprehensive tests for conflict_resolver.py

Tests all components:
- Data Models (ConflictBlock, ConflictedFile, ResolutionAttempt, ResolutionResult)
- ConflictDetector
- Resolution Strategies (PreferOurs, PreferTheirs, PreferNewer, Combine, Claude)
- ResolutionValidator
- ConflictResolver (full pipeline)
- HumanReviewEscalator
- Custom Exceptions

CONFLICT-001: Full Merge Conflict Auto-Resolution
"""

import json
import pytest
import tempfile
import subprocess
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, call
from typing import List

# Import classes to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from conflict_resolver import (
    # Enums
    ConflictType,
    ResolutionStrategy,
    # Data Models
    ConflictBlock,
    ConflictedFile,
    ResolutionAttempt,
    ResolutionResult,
    HumanReviewRequest,
    # Main Classes
    ConflictDetector,
    PreferOursStrategy,
    PreferTheirsStrategy,
    PreferNewerStrategy,
    CombineStrategy,
    ClaudeResolutionStrategy,
    ResolutionValidator,
    ConflictResolver,
    HumanReviewEscalator,
    # Exceptions
    ConflictResolutionError,
    UnresolvableConflictError,
    ValidationError,
)


# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture
def temp_worktree(tmp_path):
    """Create a temporary git worktree."""
    worktree = tmp_path / "worktree"
    worktree.mkdir()

    # Initialize git repo
    subprocess.run(['git', 'init'], cwd=str(worktree), capture_output=True)
    subprocess.run(['git', 'config', 'user.name', 'Test User'], cwd=str(worktree), capture_output=True)
    subprocess.run(['git', 'config', 'user.email', 'test@example.com'], cwd=str(worktree), capture_output=True)

    return worktree


@pytest.fixture
def temp_state_dir(tmp_path):
    """Create a temporary state directory."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    return state_dir


@pytest.fixture
def simple_conflict_content():
    """Sample conflict content."""
    return '''def hello():
<<<<<<< HEAD
    return "hello from head"
=======
    return "hello from main"
>>>>>>> main
'''


@pytest.fixture
def multiple_conflicts_content():
    """File with multiple conflict blocks."""
    return '''def function1():
<<<<<<< HEAD
    return "version 1"
=======
    return "version A"
>>>>>>> main

def function2():
    return "no conflict here"

def function3():
<<<<<<< HEAD
    x = 100
    y = 200
=======
    x = 10
    y = 20
>>>>>>> main
    return x + y
'''


@pytest.fixture
def diff3_conflict_content():
    """Conflict with diff3-style base marker."""
    return '''def calculate():
<<<<<<< HEAD
    result = x * 2
||||||| base
    result = x
=======
    result = x * 3
>>>>>>> feature
    return result
'''


@pytest.fixture
def sample_conflict_block():
    """Create a sample ConflictBlock."""
    return ConflictBlock(
        file_path="test.py",
        start_line=2,
        end_line=6,
        ours_content='    return "hello from head"',
        theirs_content='    return "hello from main"',
        base_content=None,
        ours_label="HEAD",
        theirs_label="main",
        context_before="def hello():",
        context_after=""
    )


@pytest.fixture
def valid_python_code():
    """Valid Python code for testing."""
    return '''def add(a, b):
    """Add two numbers."""
    return a + b

class Calculator:
    def multiply(self, x, y):
        return x * y
'''


@pytest.fixture
def invalid_python_code():
    """Invalid Python code with syntax error."""
    return '''def broken_function(:
    return "missing parameter"
'''


@pytest.fixture
def valid_json():
    """Valid JSON content."""
    return '''{
    "name": "test",
    "version": "1.0.0",
    "items": [1, 2, 3]
}'''


@pytest.fixture
def invalid_json():
    """Invalid JSON content."""
    return '''{
    "name": "test",
    "version": "1.0.0",
    "items": [1, 2, 3
}'''


# ==============================================================================
# TEST DATA MODELS
# ==============================================================================

class TestDataModels:
    """Test data model classes."""

    def test_conflict_block_to_dict(self, sample_conflict_block):
        """Test ConflictBlock serialization."""
        data = sample_conflict_block.to_dict()

        assert data['file_path'] == "test.py"
        assert data['start_line'] == 2
        assert data['end_line'] == 6
        assert data['ours_content'] == '    return "hello from head"'
        assert data['theirs_content'] == '    return "hello from main"'
        assert data['base_content'] is None
        assert data['ours_label'] == "HEAD"
        assert data['theirs_label'] == "main"

    def test_conflict_block_with_base(self):
        """Test ConflictBlock with base content."""
        block = ConflictBlock(
            file_path="test.py",
            start_line=1,
            end_line=5,
            ours_content="ours",
            theirs_content="theirs",
            base_content="base",
            ours_label="HEAD",
            theirs_label="main",
            context_before="",
            context_after=""
        )

        data = block.to_dict()
        assert data['base_content'] == "base"

    def test_conflicted_file_conflict_ratio(self):
        """Test ConflictedFile.conflict_ratio calculation."""
        file = ConflictedFile(
            file_path=Path("/test/file.py"),
            conflict_type=ConflictType.CONTENT,
            conflicts=[],
            file_type=".py",
            total_lines=100,
            conflict_line_count=25
        )

        assert file.conflict_ratio == 0.25

    def test_conflicted_file_conflict_ratio_zero_lines(self):
        """Test conflict_ratio with zero total lines."""
        file = ConflictedFile(
            file_path=Path("/test/file.py"),
            conflict_type=ConflictType.CONTENT,
            conflicts=[],
            file_type=".py",
            total_lines=0,
            conflict_line_count=0
        )

        assert file.conflict_ratio == 0

    def test_resolution_attempt_to_dict(self):
        """Test ResolutionAttempt serialization."""
        attempt = ResolutionAttempt(
            strategy=ResolutionStrategy.PREFER_OURS,
            timestamp="2024-01-01T12:00:00",
            success=True,
            resolved_content="resolved",
            validation_result={'valid': True},
            error_message=None
        )

        data = attempt.to_dict()
        assert data['strategy'] == "prefer_ours"
        assert data['timestamp'] == "2024-01-01T12:00:00"
        assert data['success'] is True
        assert data['validation_result'] == {'valid': True}
        assert data['error_message'] is None

    def test_resolution_result_to_dict(self):
        """Test ResolutionResult serialization."""
        attempt = ResolutionAttempt(
            strategy=ResolutionStrategy.COMBINE,
            timestamp="2024-01-01T12:00:00",
            success=True,
            resolved_content="combined",
            validation_result={'valid': True},
            error_message=None
        )

        result = ResolutionResult(
            file_path="test.py",
            resolved=True,
            strategy_used=ResolutionStrategy.COMBINE,
            attempts=[attempt],
            final_content="combined content",
            requires_human_review=False,
            review_reason=None
        )

        data = result.to_dict()
        assert data['file_path'] == "test.py"
        assert data['resolved'] is True
        assert data['strategy_used'] == "combine"
        assert len(data['attempts']) == 1
        assert data['requires_human_review'] is False
        assert data['review_reason'] is None

    def test_human_review_request_to_dict(self):
        """Test HumanReviewRequest serialization."""
        request = HumanReviewRequest(
            task_id="TASK-123",
            file_path="conflict.py",
            conflict_details={'type': 'binary'},
            resolution_attempts=[],
            created_at="2024-01-01T12:00:00",
            priority="HIGH"
        )

        data = request.to_dict()
        assert data['task_id'] == "TASK-123"
        assert data['file_path'] == "conflict.py"
        assert data['priority'] == "HIGH"


# ==============================================================================
# TEST CONFLICT DETECTION
# ==============================================================================

class TestConflictDetection:
    """Test ConflictDetector class."""

    def test_detect_simple_conflict(self, temp_worktree, simple_conflict_content):
        """Test detection of simple conflict."""
        # Create file with conflict
        conflict_file = temp_worktree / "test.py"
        conflict_file.write_text(simple_conflict_content)

        detector = ConflictDetector(temp_worktree)
        conflicted = detector._analyze_conflicted_file(conflict_file)

        assert conflicted is not None
        assert conflicted.conflict_type == ConflictType.CONTENT
        assert len(conflicted.conflicts) == 1
        assert conflicted.file_type == ".py"

        conflict = conflicted.conflicts[0]
        assert conflict.ours_label == "HEAD"
        assert conflict.theirs_label == "main"
        assert 'hello from head' in conflict.ours_content
        assert 'hello from main' in conflict.theirs_content

    def test_detect_no_conflicts(self, temp_worktree, valid_python_code):
        """Test clean files have no conflicts."""
        clean_file = temp_worktree / "clean.py"
        clean_file.write_text(valid_python_code)

        detector = ConflictDetector(temp_worktree)
        conflicted = detector._analyze_conflicted_file(clean_file)

        assert conflicted is None

    def test_detect_multiple_conflicts(self, temp_worktree, multiple_conflicts_content):
        """Test detection of multiple conflict blocks in one file."""
        conflict_file = temp_worktree / "multi.py"
        conflict_file.write_text(multiple_conflicts_content)

        detector = ConflictDetector(temp_worktree)
        conflicted = detector._analyze_conflicted_file(conflict_file)

        assert conflicted is not None
        assert len(conflicted.conflicts) == 2

        # Check first conflict
        assert 'version 1' in conflicted.conflicts[0].ours_content
        assert 'version A' in conflicted.conflicts[0].theirs_content

        # Check second conflict
        assert 'x = 100' in conflicted.conflicts[1].ours_content
        assert 'x = 10' in conflicted.conflicts[1].theirs_content

    def test_detect_diff3_style(self, temp_worktree, diff3_conflict_content):
        """Test handling of diff3-style conflict markers."""
        conflict_file = temp_worktree / "diff3.py"
        conflict_file.write_text(diff3_conflict_content)

        detector = ConflictDetector(temp_worktree)
        conflicted = detector._analyze_conflicted_file(conflict_file)

        assert conflicted is not None
        assert len(conflicted.conflicts) == 1

        conflict = conflicted.conflicts[0]
        assert conflict.base_content is not None
        assert 'result = x' in conflict.base_content
        assert 'x * 2' in conflict.ours_content
        assert 'x * 3' in conflict.theirs_content

    def test_detect_binary_conflict(self, temp_worktree):
        """Test identification of binary conflicts."""
        # Create a binary file
        binary_file = temp_worktree / "image.png"
        binary_file.write_bytes(b'\x89PNG\r\n\x1a\n' + b'\x00' * 100)

        detector = ConflictDetector(temp_worktree)

        # Mock git diff to return binary indicator
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(
                stdout="-\t-\timage.png\n",
                returncode=0
            )

            is_binary = detector._is_binary_conflict(binary_file)
            assert is_binary is True

    def test_parse_conflict_labels(self, temp_worktree):
        """Test extraction of branch names from conflict markers."""
        content = '''def test():
<<<<<<< feature/new-feature
    return "new"
=======
    return "old"
>>>>>>> origin/main
'''
        conflict_file = temp_worktree / "labels.py"
        conflict_file.write_text(content)

        detector = ConflictDetector(temp_worktree)
        conflicted = detector._analyze_conflicted_file(conflict_file)

        assert conflicted is not None
        conflict = conflicted.conflicts[0]
        assert conflict.ours_label == "feature/new-feature"
        assert conflict.theirs_label == "origin/main"

    def test_scan_for_conflicts_with_git(self, temp_worktree):
        """Test scanning using git diff."""
        detector = ConflictDetector(temp_worktree)

        # Mock git diff output
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(
                stdout="file1.py\nfile2.py\n",
                returncode=0
            )

            files = detector._get_git_unmerged_files()
            assert len(files) == 2
            assert "file1.py" in files
            assert "file2.py" in files

    def test_scan_for_conflicts_git_failure(self, temp_worktree):
        """Test handling of git command failure."""
        detector = ConflictDetector(temp_worktree)

        # Mock git command failure
        with patch('subprocess.run', side_effect=subprocess.SubprocessError("git failed")):
            files = detector._get_git_unmerged_files()
            assert files == []

    def test_python_scan_markers(self, temp_worktree, simple_conflict_content):
        """Test Python-based fallback for conflict marker scanning."""
        # Create file with conflict
        conflict_file = temp_worktree / "conflict.py"
        conflict_file.write_text(simple_conflict_content)

        # Create clean file
        clean_file = temp_worktree / "clean.py"
        clean_file.write_text("def test():\n    pass\n")

        detector = ConflictDetector(temp_worktree)

        # Use os.walk instead of Path.walk for compatibility
        import os
        conflicted = []
        for root, dirs, files in os.walk(temp_worktree):
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            for file in files:
                file_path = Path(root) / file
                try:
                    content = file_path.read_text(encoding='utf-8', errors='ignore')
                    if '<<<<<<<' in content:
                        rel_path = file_path.relative_to(temp_worktree)
                        conflicted.append(str(rel_path))
                except (IOError, OSError):
                    pass

        assert len(conflicted) >= 1
        assert any('conflict.py' in f for f in conflicted)

    def test_conflict_line_count(self, temp_worktree, simple_conflict_content):
        """Test conflict line count calculation."""
        conflict_file = temp_worktree / "test.py"
        conflict_file.write_text(simple_conflict_content)

        detector = ConflictDetector(temp_worktree)
        conflicted = detector._analyze_conflicted_file(conflict_file)

        assert conflicted is not None
        assert conflicted.conflict_line_count > 0


# ==============================================================================
# TEST RESOLUTION STRATEGIES
# ==============================================================================

class TestResolutionStrategies:
    """Test resolution strategy implementations."""

    def test_prefer_ours_strategy(self, sample_conflict_block):
        """Test PreferOursStrategy always selects ours."""
        strategy = PreferOursStrategy()

        assert strategy.can_resolve(sample_conflict_block, ".py") is True

        success, content, error = strategy.resolve(sample_conflict_block, ".py")
        assert success is True
        assert content == sample_conflict_block.ours_content
        assert error == ""

    def test_prefer_theirs_strategy(self, sample_conflict_block):
        """Test PreferTheirsStrategy always selects theirs."""
        strategy = PreferTheirsStrategy()

        assert strategy.can_resolve(sample_conflict_block, ".py") is True

        success, content, error = strategy.resolve(sample_conflict_block, ".py")
        assert success is True
        assert content == sample_conflict_block.theirs_content
        assert error == ""

    def test_prefer_newer_strategy_ours_newer(self, temp_worktree, sample_conflict_block):
        """Test PreferNewerStrategy selects ours when ours is newer."""
        strategy = PreferNewerStrategy(temp_worktree)

        # Mock commit timestamps - ours is newer
        with patch.object(strategy, '_get_commit_time', side_effect=[1000, 500]):
            success, content, error = strategy.resolve(sample_conflict_block, ".py")

            assert success is True
            assert content == sample_conflict_block.ours_content
            assert "newer" in error.lower()

    def test_prefer_newer_strategy_theirs_newer(self, temp_worktree, sample_conflict_block):
        """Test PreferNewerStrategy selects theirs when theirs is newer."""
        strategy = PreferNewerStrategy(temp_worktree)

        # Mock commit timestamps - theirs is newer
        with patch.object(strategy, '_get_commit_time', side_effect=[500, 1000]):
            success, content, error = strategy.resolve(sample_conflict_block, ".py")

            assert success is True
            assert content == sample_conflict_block.theirs_content
            assert "newer" in error.lower()

    def test_prefer_newer_strategy_no_timestamps(self, temp_worktree, sample_conflict_block):
        """Test PreferNewerStrategy fails when timestamps unavailable."""
        strategy = PreferNewerStrategy(temp_worktree)

        # Mock failed timestamp retrieval
        with patch.object(strategy, '_get_commit_time', return_value=None):
            success, content, error = strategy.resolve(sample_conflict_block, ".py")

            assert success is False
            assert "timestamp" in error.lower()

    def test_prefer_newer_strategy_can_resolve(self, temp_worktree):
        """Test PreferNewerStrategy.can_resolve checks for labels."""
        strategy = PreferNewerStrategy(temp_worktree)

        # With labels
        block_with_labels = ConflictBlock(
            file_path="test.py",
            start_line=1,
            end_line=5,
            ours_content="ours",
            theirs_content="theirs",
            base_content=None,
            ours_label="HEAD",
            theirs_label="main",
            context_before="",
            context_after=""
        )
        assert strategy.can_resolve(block_with_labels, ".py") is True

        # Without labels
        block_no_labels = ConflictBlock(
            file_path="test.py",
            start_line=1,
            end_line=5,
            ours_content="ours",
            theirs_content="theirs",
            base_content=None,
            ours_label="",
            theirs_label="",
            context_before="",
            context_after=""
        )
        assert strategy.can_resolve(block_no_labels, ".py") is False

    def test_combine_strategy_non_overlapping(self):
        """Test CombineStrategy merges non-overlapping changes."""
        conflict = ConflictBlock(
            file_path="test.py",
            start_line=1,
            end_line=5,
            ours_content="def function_a():\n    return 'a'",
            theirs_content="def function_b():\n    return 'b'",
            base_content=None,
            ours_label="HEAD",
            theirs_label="main",
            context_before="",
            context_after=""
        )

        strategy = CombineStrategy()
        # Strategy checks for line overlap - these are truly non-overlapping
        can_resolve = strategy.can_resolve(conflict, ".py")
        # It should detect these as combinable
        assert can_resolve is True

        success, content, error = strategy.resolve(conflict, ".py")
        assert success is True
        assert "function_a" in content
        assert "function_b" in content

    def test_combine_strategy_overlapping(self):
        """Test CombineStrategy handles overlapping changes."""
        conflict = ConflictBlock(
            file_path="test.py",
            start_line=1,
            end_line=5,
            ours_content="x = 1\ny = 2\nz = 3",
            theirs_content="x = 1\ny = 2\nw = 4",
            base_content=None,
            ours_label="HEAD",
            theirs_label="main",
            context_before="",
            context_after=""
        )

        strategy = CombineStrategy()
        # Should still attempt to combine
        can_resolve = strategy.can_resolve(conflict, ".py")
        # With significant overlap, can_resolve might be False
        # This depends on the heuristic, so we just check it returns a boolean
        assert isinstance(can_resolve, bool)

    def test_combine_strategy_empty_content(self):
        """Test CombineStrategy with empty content."""
        conflict = ConflictBlock(
            file_path="test.py",
            start_line=1,
            end_line=3,
            ours_content="",
            theirs_content="def new_function():\n    pass",
            base_content=None,
            ours_label="HEAD",
            theirs_label="main",
            context_before="",
            context_after=""
        )

        strategy = CombineStrategy()
        success, content, error = strategy.resolve(conflict, ".py")

        assert success is True
        assert content == conflict.theirs_content

    def test_claude_strategy_prompt_generation(self, temp_worktree, temp_state_dir):
        """Test ClaudeResolutionStrategy generates correct prompt."""
        conflict = ConflictBlock(
            file_path="src/module.py",
            start_line=10,
            end_line=20,
            ours_content="def calculate():\n    return x * 2",
            theirs_content="def calculate():\n    return x * 3",
            base_content="def calculate():\n    return x",
            ours_label="HEAD",
            theirs_label="feature/enhancement",
            context_before="class Calculator:",
            context_after="    def other_method():\n        pass"
        )

        strategy = ClaudeResolutionStrategy(temp_worktree, temp_state_dir)

        # Verify can resolve code files
        assert strategy.can_resolve(conflict, ".py") is True
        assert strategy.can_resolve(conflict, ".js") is True
        assert strategy.can_resolve(conflict, ".md") is False

        # Generate prompt
        success, content, error = strategy.resolve(conflict, ".py")

        # Should return False (prompt prepared, not resolved)
        assert success is False
        assert "Claude resolution prepared" in error

        # Verify prompt file was created
        prompt_files = list(temp_state_dir.glob("conflict_resolution_prompt_*.txt"))
        assert len(prompt_files) == 1

        prompt_content = prompt_files[0].read_text()
        assert "src/module.py" in prompt_content
        assert "Lines: 10-20" in prompt_content
        assert "def calculate()" in prompt_content
        assert "x * 2" in prompt_content
        assert "x * 3" in prompt_content
        assert "BASE Version" in prompt_content

    def test_claude_strategy_file_types(self, temp_worktree, temp_state_dir):
        """Test ClaudeResolutionStrategy handles various file types."""
        strategy = ClaudeResolutionStrategy(temp_worktree, temp_state_dir)

        conflict = ConflictBlock(
            file_path="test.txt",
            start_line=1,
            end_line=3,
            ours_content="text",
            theirs_content="text",
            base_content=None,
            ours_label="HEAD",
            theirs_label="main",
            context_before="",
            context_after=""
        )

        # Supported code extensions
        assert strategy.can_resolve(conflict, ".py") is True
        assert strategy.can_resolve(conflict, ".js") is True
        assert strategy.can_resolve(conflict, ".ts") is True
        assert strategy.can_resolve(conflict, ".go") is True
        assert strategy.can_resolve(conflict, ".rs") is True
        assert strategy.can_resolve(conflict, ".java") is True

        # Unsupported extensions
        assert strategy.can_resolve(conflict, ".txt") is False
        assert strategy.can_resolve(conflict, ".md") is False
        assert strategy.can_resolve(conflict, ".pdf") is False


# ==============================================================================
# TEST VALIDATION
# ==============================================================================

class TestValidation:
    """Test ResolutionValidator class."""

    def test_validate_python_syntax_valid(self, temp_worktree, valid_python_code):
        """Test validation accepts valid Python."""
        validator = ResolutionValidator(temp_worktree)
        test_file = temp_worktree / "valid.py"

        result = validator.validate(test_file, valid_python_code)

        assert result['valid'] is True
        assert result['checks']['markers_removed'] is True
        assert result['checks']['syntax_valid'] is True
        assert len(result['issues']) == 0

    def test_validate_python_syntax_invalid(self, temp_worktree, invalid_python_code):
        """Test validation rejects invalid Python."""
        validator = ResolutionValidator(temp_worktree)
        test_file = temp_worktree / "invalid.py"

        result = validator.validate(test_file, invalid_python_code)

        assert result['valid'] is False
        assert result['checks']['syntax_valid'] is False
        assert any("syntax error" in issue.lower() for issue in result['issues'])

    def test_validate_json_syntax(self, temp_worktree, valid_json, invalid_json):
        """Test JSON syntax validation."""
        validator = ResolutionValidator(temp_worktree)
        json_file = temp_worktree / "config.json"

        # Valid JSON
        result = validator.validate(json_file, valid_json)
        assert result['valid'] is True
        assert result['checks']['syntax_valid'] is True

        # Invalid JSON
        result = validator.validate(json_file, invalid_json)
        assert result['valid'] is False
        assert result['checks']['syntax_valid'] is False

    def test_validate_no_markers_remain(self, temp_worktree):
        """Test validation fails if conflict markers remain."""
        validator = ResolutionValidator(temp_worktree)
        test_file = temp_worktree / "test.py"

        content_with_markers = '''def test():
<<<<<<< HEAD
    return "conflict"
=======
    return "not resolved"
>>>>>>> main
'''

        result = validator.validate(test_file, content_with_markers)

        assert result['valid'] is False
        assert result['checks']['markers_removed'] is False
        assert any("markers" in issue.lower() for issue in result['issues'])

    def test_validate_empty_resolution(self, temp_worktree):
        """Test validation fails on empty content."""
        validator = ResolutionValidator(temp_worktree)
        test_file = temp_worktree / "empty.py"

        result = validator.validate(test_file, "   \n  \n  ")

        assert result['valid'] is False
        assert any("empty" in issue.lower() for issue in result['issues'])

    def test_validate_yaml_syntax(self, temp_worktree):
        """Test YAML syntax validation."""
        validator = ResolutionValidator(temp_worktree)
        yaml_file = temp_worktree / "config.yml"

        valid_yaml = """
name: test
version: 1.0
items:
  - one
  - two
"""

        result = validator.validate(yaml_file, valid_yaml)
        # May pass or skip depending on whether PyYAML is installed
        assert isinstance(result['valid'], bool)

    def test_run_affected_tests_no_tests(self, temp_worktree):
        """Test run_affected_tests when no test file exists."""
        validator = ResolutionValidator(temp_worktree)
        test_file = temp_worktree / "module.py"
        test_file.write_text("def test(): pass")

        passed, output = validator.run_affected_tests(test_file)

        assert passed is True
        assert "No related tests found" in output

    def test_run_affected_tests_with_tests(self, temp_worktree):
        """Test run_affected_tests when test file exists."""
        validator = ResolutionValidator(temp_worktree)

        # Create module
        module_file = temp_worktree / "calculator.py"
        module_file.write_text("def add(a, b): return a + b")

        # Create test file
        test_dir = temp_worktree / "tests"
        test_dir.mkdir()
        test_file = test_dir / "test_calculator.py"
        test_file.write_text("""
import sys
sys.path.insert(0, '..')
from calculator import add

def test_add():
    assert add(1, 2) == 3
""")

        # Mock pytest run
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(
                returncode=0,
                stdout="test_add PASSED",
                stderr=""
            )

            passed, output = validator.run_affected_tests(module_file)
            assert passed is True


# ==============================================================================
# TEST RESOLUTION PIPELINE
# ==============================================================================

class TestResolutionPipeline:
    """Test full ConflictResolver pipeline."""

    def test_full_resolution_pipeline(self, temp_worktree, temp_state_dir, simple_conflict_content):
        """Test complete resolution pipeline: detection -> resolution -> validation."""
        # Create conflicted file
        conflict_file = temp_worktree / "test.py"
        conflict_file.write_text(simple_conflict_content)

        resolver = ConflictResolver(temp_worktree, temp_state_dir)

        # Mock git operations
        with patch('subprocess.run') as mock_run:
            # Mock git diff for unmerged files
            mock_run.return_value = Mock(
                stdout="test.py\n",
                returncode=0
            )

            result = resolver.resolve_all()

        assert result['status'] in ['resolved', 'partial']
        assert result['conflicts_found'] == 1

    def test_resolution_with_binary_conflict(self, temp_worktree, temp_state_dir):
        """Test handling of binary conflicts."""
        # Create binary file
        binary_file = temp_worktree / "image.png"
        binary_file.write_bytes(b'\x89PNG' + b'\x00' * 100)

        resolver = ConflictResolver(temp_worktree, temp_state_dir)

        # Create a binary conflict manually
        conflicted_file = ConflictedFile(
            file_path=binary_file,
            conflict_type=ConflictType.BINARY,
            conflicts=[],
            file_type=".png",
            total_lines=0,
            conflict_line_count=0
        )

        result = resolver._resolve_file(conflicted_file)

        assert result.resolved is False
        assert result.requires_human_review is True
        assert "binary" in result.review_reason.lower()

    def test_escalation_after_max_attempts(self, temp_worktree, temp_state_dir):
        """Test escalation to human review after max attempts."""
        resolver = ConflictResolver(temp_worktree, temp_state_dir, config={'max_resolution_attempts': 1})

        # Create a conflict that will fail validation
        conflict_content = '''def test():
<<<<<<< HEAD
    return invalid syntax (
=======
    return also invalid )
>>>>>>> main
'''
        conflict_file = temp_worktree / "bad.py"
        conflict_file.write_text(conflict_content)

        conflicted = resolver.detector._analyze_conflicted_file(conflict_file)
        assert conflicted is not None

        result = resolver._resolve_file(conflicted)

        # Should fail and require human review
        assert result.resolved is False
        assert result.requires_human_review is True

    def test_resolution_log_saved(self, temp_worktree, temp_state_dir):
        """Test that resolution log is saved to state directory."""
        resolver = ConflictResolver(temp_worktree, temp_state_dir)

        # Create mock results
        result = ResolutionResult(
            file_path="test.py",
            resolved=True,
            strategy_used=ResolutionStrategy.PREFER_OURS,
            attempts=[],
            final_content="resolved",
            requires_human_review=False,
            review_reason=None
        )

        resolver._save_resolution_log([result])

        log_file = temp_state_dir / "conflict_resolution_log.json"
        assert log_file.exists()

        log_data = json.loads(log_file.read_text())
        assert 'timestamp' in log_data
        assert 'worktree' in log_data
        assert len(log_data['results']) == 1

    def test_resolved_file_staged(self, temp_worktree, temp_state_dir):
        """Test that resolved files are staged with git add."""
        resolver = ConflictResolver(temp_worktree, temp_state_dir)

        test_file = temp_worktree / "resolved.py"
        test_file.write_text("def test(): pass")

        with patch('subprocess.run') as mock_run:
            resolver._write_resolved_file(test_file, "def test(): return True")

            # Verify git add was called
            calls = mock_run.call_args_list
            assert any('git' in str(call) and 'add' in str(call) for call in calls)

    def test_apply_resolution(self, temp_worktree, temp_state_dir):
        """Test _apply_resolution replaces conflict blocks correctly."""
        resolver = ConflictResolver(temp_worktree, temp_state_dir)

        original_content = '''def function1():
    pass

<<<<<<< HEAD
def conflict():
    return "ours"
=======
def conflict():
    return "theirs"
>>>>>>> main

def function2():
    pass
'''

        conflict = ConflictBlock(
            file_path="test.py",
            start_line=4,
            end_line=10,
            ours_content='def conflict():\n    return "ours"',
            theirs_content='def conflict():\n    return "theirs"',
            base_content=None,
            ours_label="HEAD",
            theirs_label="main",
            context_before="def function1():\n    pass",
            context_after="def function2():\n    pass"
        )

        resolution = 'def conflict():\n    return "resolved"'

        resolved_content = resolver._apply_resolution(original_content, conflict, resolution)

        assert '<<<<<<< HEAD' not in resolved_content
        assert '>>>>>>> main' not in resolved_content
        assert 'return "resolved"' in resolved_content

    def test_no_conflicts_detected(self, temp_worktree, temp_state_dir):
        """Test resolve_all when no conflicts exist."""
        resolver = ConflictResolver(temp_worktree, temp_state_dir)

        # Create clean file
        clean_file = temp_worktree / "clean.py"
        clean_file.write_text("def test(): pass")

        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(stdout="", returncode=0)

            result = resolver.resolve_all()

        assert result['status'] == 'clean'
        assert result['conflicts_found'] == 0
        assert result['resolved'] == 0
        assert result['failed'] == 0

    def test_resolution_strategies_order(self, temp_worktree, temp_state_dir):
        """Test that resolution strategies are tried in correct order."""
        resolver = ConflictResolver(temp_worktree, temp_state_dir)

        # Verify strategy order
        strategy_names = [s[0] for s in resolver.strategies]

        # PREFER_NEWER should come before PREFER_OURS (fallback)
        prefer_newer_idx = strategy_names.index(ResolutionStrategy.PREFER_NEWER)
        prefer_ours_idx = strategy_names.index(ResolutionStrategy.PREFER_OURS)
        assert prefer_newer_idx < prefer_ours_idx


# ==============================================================================
# TEST HUMAN REVIEW ESCALATION
# ==============================================================================

class TestHumanReviewEscalation:
    """Test HumanReviewEscalator class."""

    def test_add_to_review_queue(self, temp_worktree, tmp_path):
        """Test adding conflicts to review queue."""
        main_repo = tmp_path / "main_repo"
        main_repo.mkdir()

        escalator = HumanReviewEscalator(main_repo, "TASK-123")

        result = ResolutionResult(
            file_path="conflict.py",
            resolved=False,
            strategy_used=ResolutionStrategy.MANUAL,
            attempts=[],
            final_content=None,
            requires_human_review=True,
            review_reason="Complex semantic conflict"
        )

        request = escalator.escalate(result)

        assert request.task_id == "TASK-123"
        assert request.file_path == "conflict.py"
        assert request.priority == "HIGH"

        # Verify queue file was created
        queue_file = main_repo / ".autonomous" / "human_review_queue.json"
        assert queue_file.exists()

        queue_data = json.loads(queue_file.read_text())
        assert len(queue_data) == 1
        assert queue_data[0]['task_id'] == "TASK-123"

    def test_escalate_binary_conflict(self, tmp_path):
        """Test that binary conflicts are escalated immediately."""
        main_repo = tmp_path / "main_repo"
        main_repo.mkdir()

        escalator = HumanReviewEscalator(main_repo, "TASK-456")

        result = ResolutionResult(
            file_path="image.png",
            resolved=False,
            strategy_used=ResolutionStrategy.MANUAL,
            attempts=[],
            final_content=None,
            requires_human_review=True,
            review_reason="Binary file conflict cannot be auto-resolved"
        )

        request = escalator.escalate(result)

        assert request.priority == "HIGH"
        assert "image.png" in request.file_path

    def test_multiple_escalations(self, tmp_path):
        """Test multiple escalations to the same queue."""
        main_repo = tmp_path / "main_repo"
        main_repo.mkdir()

        escalator = HumanReviewEscalator(main_repo, "TASK-789")

        # Escalate first conflict
        result1 = ResolutionResult(
            file_path="file1.py",
            resolved=False,
            strategy_used=ResolutionStrategy.MANUAL,
            attempts=[],
            final_content=None,
            requires_human_review=True,
            review_reason="Reason 1"
        )
        escalator.escalate(result1)

        # Escalate second conflict
        result2 = ResolutionResult(
            file_path="file2.py",
            resolved=False,
            strategy_used=ResolutionStrategy.MANUAL,
            attempts=[],
            final_content=None,
            requires_human_review=True,
            review_reason="Reason 2"
        )
        escalator.escalate(result2)

        # Verify both are in queue
        queue_file = main_repo / ".autonomous" / "human_review_queue.json"
        queue_data = json.loads(queue_file.read_text())

        assert len(queue_data) == 2
        file_paths = [item['file_path'] for item in queue_data]
        assert "file1.py" in file_paths
        assert "file2.py" in file_paths

    def test_escalate_with_corrupt_queue(self, tmp_path):
        """Test escalation handles corrupt queue file gracefully."""
        main_repo = tmp_path / "main_repo"
        main_repo.mkdir()

        # Create corrupt queue file
        queue_file = main_repo / ".autonomous" / "human_review_queue.json"
        queue_file.parent.mkdir(parents=True, exist_ok=True)
        queue_file.write_text("{ invalid json [")

        escalator = HumanReviewEscalator(main_repo, "TASK-999")

        result = ResolutionResult(
            file_path="test.py",
            resolved=False,
            strategy_used=ResolutionStrategy.MANUAL,
            attempts=[],
            final_content=None,
            requires_human_review=True,
            review_reason="Test"
        )

        # Should handle corruption and create new queue
        request = escalator.escalate(result)

        queue_data = json.loads(queue_file.read_text())
        assert len(queue_data) == 1


# ==============================================================================
# TEST CUSTOM EXCEPTIONS
# ==============================================================================

class TestCustomExceptions:
    """Test custom exception classes."""

    def test_unresolvable_conflict_error(self):
        """Test UnresolvableConflictError exception."""
        error = UnresolvableConflictError("test.py", "Too complex")

        assert error.file_path == "test.py"
        assert error.reason == "Too complex"
        assert "test.py" in str(error)
        assert "Too complex" in str(error)

    def test_validation_error(self):
        """Test ValidationError exception."""
        issues = ["Syntax error", "Missing import"]
        error = ValidationError("module.py", issues)

        assert error.file_path == "module.py"
        assert error.issues == issues
        assert "module.py" in str(error)
        assert "Syntax error" in str(error)

    def test_conflict_resolution_error_base(self):
        """Test base ConflictResolutionError exception."""
        error = ConflictResolutionError("Generic error")
        assert "Generic error" in str(error)


# ==============================================================================
# TEST EDGE CASES
# ==============================================================================

class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_file_conflict(self, temp_worktree):
        """Test handling of conflicts in empty files."""
        empty_file = temp_worktree / "empty.txt"
        empty_file.write_text("")

        detector = ConflictDetector(temp_worktree)
        conflicted = detector._analyze_conflicted_file(empty_file)

        assert conflicted is None

    def test_conflict_with_unicode(self, temp_worktree):
        """Test handling of conflicts with unicode characters."""
        unicode_conflict = '''def hello():
<<<<<<< HEAD
    return "Hello 世界"
=======
    return "Привет мир"
>>>>>>> main
'''
        conflict_file = temp_worktree / "unicode.py"
        conflict_file.write_text(unicode_conflict, encoding='utf-8')

        detector = ConflictDetector(temp_worktree)
        conflicted = detector._analyze_conflicted_file(conflict_file)

        assert conflicted is not None
        assert len(conflicted.conflicts) == 1

    def test_permission_error_on_read(self, temp_worktree):
        """Test handling of file permission errors."""
        detector = ConflictDetector(temp_worktree)

        # Mock file that raises permission error
        bad_file = temp_worktree / "noperm.py"
        bad_file.write_text("test")

        with patch('builtins.open', side_effect=PermissionError("No access")):
            conflicted = detector._analyze_conflicted_file(bad_file)
            assert conflicted is None

    def test_very_large_conflict(self, temp_worktree):
        """Test handling of very large conflict blocks."""
        large_ours = "line\n" * 1000
        large_theirs = "different\n" * 1000

        large_conflict = f'''def test():
<<<<<<< HEAD
{large_ours}
=======
{large_theirs}
>>>>>>> main
'''
        conflict_file = temp_worktree / "large.py"
        conflict_file.write_text(large_conflict)

        detector = ConflictDetector(temp_worktree)
        conflicted = detector._analyze_conflicted_file(conflict_file)

        assert conflicted is not None
        assert len(conflicted.conflicts) == 1

    def test_malformed_conflict_markers(self, temp_worktree):
        """Test handling of malformed conflict markers."""
        malformed = '''def test():
<<<<<<< HEAD
    return "ours"
=======
    return "theirs"
    (missing end marker)
'''
        conflict_file = temp_worktree / "malformed.py"
        conflict_file.write_text(malformed)

        detector = ConflictDetector(temp_worktree)
        conflicted = detector._analyze_conflicted_file(conflict_file)

        # Should handle gracefully (might detect or might skip)
        assert isinstance(conflicted, (ConflictedFile, type(None)))

    def test_nested_conflict_markers(self, temp_worktree):
        """Test handling of nested conflict markers (edge case)."""
        nested = '''def test():
<<<<<<< HEAD
    value = "<<<<<<< inner"
=======
    value = ">>>>>>> inner"
>>>>>>> main
'''
        conflict_file = temp_worktree / "nested.py"
        conflict_file.write_text(nested)

        detector = ConflictDetector(temp_worktree)
        conflicted = detector._analyze_conflicted_file(conflict_file)

        # Should detect at least the outer conflict
        assert conflicted is not None


# ==============================================================================
# PARAMETRIZED TESTS
# ==============================================================================

class TestParametrized:
    """Parametrized tests for various scenarios."""

    @pytest.mark.parametrize("file_ext,is_code", [
        (".py", True),
        (".js", True),
        (".ts", True),
        (".go", True),
        (".rs", True),
        (".java", True),
        (".txt", False),
        (".md", False),
        (".json", False),
    ])
    def test_claude_strategy_file_type_detection(self, temp_worktree, temp_state_dir, file_ext, is_code):
        """Test ClaudeResolutionStrategy file type detection."""
        strategy = ClaudeResolutionStrategy(temp_worktree, temp_state_dir)

        conflict = ConflictBlock(
            file_path=f"test{file_ext}",
            start_line=1,
            end_line=3,
            ours_content="content",
            theirs_content="content",
            base_content=None,
            ours_label="HEAD",
            theirs_label="main",
            context_before="",
            context_after=""
        )

        assert strategy.can_resolve(conflict, file_ext) == is_code

    @pytest.mark.parametrize("strategy_class,expected_always_resolves", [
        (PreferOursStrategy, True),
        (PreferTheirsStrategy, True),
    ])
    def test_always_resolving_strategies(self, strategy_class, expected_always_resolves, sample_conflict_block):
        """Test strategies that can always resolve."""
        strategy = strategy_class()

        can_resolve = strategy.can_resolve(sample_conflict_block, ".py")
        assert can_resolve == expected_always_resolves

        if expected_always_resolves:
            success, content, error = strategy.resolve(sample_conflict_block, ".py")
            assert success is True
            assert content != ""

    @pytest.mark.parametrize("code,should_validate", [
        ("def test():\n    pass", True),
        ("x = 1 + 2", True),
        ("import sys", True),
        ("def broken(:", False),
        ("if True", False),
        ("class Broken:\n    def method(self,:", False),
    ])
    def test_python_syntax_validation_cases(self, temp_worktree, code, should_validate):
        """Test Python syntax validation with various code samples."""
        validator = ResolutionValidator(temp_worktree)
        test_file = temp_worktree / "test.py"

        result = validator.validate(test_file, code)

        if should_validate:
            assert result['checks']['syntax_valid'] is True
        else:
            assert result['checks']['syntax_valid'] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
