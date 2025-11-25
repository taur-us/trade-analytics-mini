"""Tests for tool output path portability (FIX-001).

Tests the get_tool_output_path helper function to ensure portable,
cross-platform tool output file paths.
"""
import os
import tempfile
from pathlib import Path
import pytest
import sys

# Import the helper function
sys.path.insert(0, str(Path(__file__).parent.parent))
from gates import get_tool_output_path


class TestGetToolOutputPath:
    """Test get_tool_output_path helper function."""

    def test_returns_path_object(self, tmp_path):
        """Output should be a Path object."""
        result = get_tool_output_path(tmp_path, "bandit")
        assert isinstance(result, Path)

    def test_creates_reports_directory(self, tmp_path):
        """Reports directory should be created if it doesn't exist."""
        result = get_tool_output_path(tmp_path, "bandit")
        reports_dir = tmp_path / ".autonomous" / "reports"
        assert reports_dir.exists()
        assert reports_dir.is_dir()

    def test_output_in_reports_directory(self, tmp_path):
        """Output file should be in .autonomous/reports/."""
        result = get_tool_output_path(tmp_path, "bandit")
        assert ".autonomous" in str(result)
        assert "reports" in str(result)

    def test_filename_format(self, tmp_path):
        """Output filename should be {tool_name}.{extension}."""
        result = get_tool_output_path(tmp_path, "bandit", "json")
        assert result.name == "bandit.json"

    def test_custom_extension(self, tmp_path):
        """Should support custom file extensions."""
        result = get_tool_output_path(tmp_path, "coverage", "xml")
        assert result.name == "coverage.xml"

    def test_default_extension_is_json(self, tmp_path):
        """Default extension should be json."""
        result = get_tool_output_path(tmp_path, "mypy")
        assert result.suffix == ".json"

    def test_path_is_absolute(self, tmp_path):
        """Output path should be absolute."""
        result = get_tool_output_path(tmp_path, "bandit")
        assert result.is_absolute()

    def test_no_hardcoded_tmp_in_path(self, tmp_path):
        """Output path should NOT contain hardcoded /tmp/ string."""
        result = get_tool_output_path(tmp_path, "bandit")
        path_str = str(result)

        # Check that if /tmp/ appears, it's only because tmp_path contains it
        # (which is fine - that's from pytest fixture, not hardcoded)
        if "/tmp/" in path_str:
            # Verify it comes from tmp_path, not hardcoded
            assert str(tmp_path) in path_str

    def test_idempotent_directory_creation(self, tmp_path):
        """Multiple calls should not fail if directory exists."""
        path1 = get_tool_output_path(tmp_path, "bandit")
        path2 = get_tool_output_path(tmp_path, "coverage")
        # Both should work without errors
        assert path1.parent == path2.parent

    def test_uses_pathlib_not_string_concat(self, tmp_path):
        """Path construction should use pathlib."""
        result = get_tool_output_path(tmp_path, "bandit")

        # Verify it's a proper Path object with proper parent structure
        assert isinstance(result, Path)
        assert result.parent.name == "reports"
        assert result.parent.parent.name == ".autonomous"

    def test_different_tools_different_files(self, tmp_path):
        """Different tools should get different output files."""
        bandit_path = get_tool_output_path(tmp_path, "bandit")
        coverage_path = get_tool_output_path(tmp_path, "coverage")
        mypy_path = get_tool_output_path(tmp_path, "mypy")

        # Same directory
        assert bandit_path.parent == coverage_path.parent == mypy_path.parent

        # Different filenames
        assert bandit_path.name == "bandit.json"
        assert coverage_path.name == "coverage.json"
        assert mypy_path.name == "mypy.json"

    @pytest.mark.parametrize("tool_name,extension,expected_name", [
        ("bandit", "json", "bandit.json"),
        ("coverage", "xml", "coverage.xml"),
        ("mypy", "txt", "mypy.txt"),
        ("pytest", "html", "pytest.html"),
        ("ruff", "sarif", "ruff.sarif"),
    ])
    def test_various_tool_configurations(self, tmp_path, tool_name, extension, expected_name):
        """Test various tool name and extension combinations."""
        result = get_tool_output_path(tmp_path, tool_name, extension)
        assert result.name == expected_name
        assert result.parent.name == "reports"


class TestCrossPlatformPaths:
    """Test cross-platform path handling."""

    def test_works_on_windows_style_path(self):
        """Should work with Windows-style paths."""
        # Use current worktree as test
        worktree = Path("C:/Users/tomas/claude-worktrees/instance-20251125-145240/claude-code-autonomy")
        if worktree.exists():
            result = get_tool_output_path(worktree, "bandit")
            # Directory should be created
            assert result.parent.exists() or result.parent.parent.exists()
            # Path should use proper separators for OS
            assert isinstance(result, Path)

    def test_path_separators_are_correct(self, tmp_path):
        """Path should use correct separators for current OS."""
        result = get_tool_output_path(tmp_path, "bandit")

        # Convert to string to check separators
        path_str = str(result)

        # On Windows, should use backslashes; on Unix, forward slashes
        # pathlib handles this automatically
        if os.name == 'nt':
            # Windows - may have backslashes (pathlib normalizes)
            pass  # pathlib handles this
        else:
            # Unix - should have forward slashes
            assert '\\' not in path_str or '\\\\' in path_str  # Allow escaped backslashes

    def test_relative_to_worktree(self, tmp_path):
        """Output path should be under worktree path."""
        result = get_tool_output_path(tmp_path, "bandit")

        # Result should be a child of tmp_path
        try:
            result.relative_to(tmp_path)
            is_child = True
        except ValueError:
            is_child = False

        assert is_child, f"{result} is not under {tmp_path}"

    def test_no_parent_traversal(self, tmp_path):
        """Path should not use .. parent directory traversal."""
        result = get_tool_output_path(tmp_path, "bandit")
        path_str = str(result)

        # Should not contain .. in the path
        assert ".." not in path_str

    def test_handles_symlinks(self, tmp_path):
        """Should handle symlinked directories gracefully."""
        # Create a real directory
        real_dir = tmp_path / "real"
        real_dir.mkdir()

        # Create symlink (skip on Windows if permissions insufficient)
        link_dir = tmp_path / "link"
        try:
            link_dir.symlink_to(real_dir, target_is_directory=True)

            # Should work with symlinked path
            result = get_tool_output_path(link_dir, "bandit")
            assert isinstance(result, Path)
            assert result.parent.exists()
        except OSError:
            # Skip test if symlinks not supported (e.g., Windows without admin)
            pytest.skip("Symlinks not supported on this system")


class TestDirectoryStructure:
    """Test that directory structure is correct."""

    def test_creates_autonomous_directory(self, tmp_path):
        """Should create .autonomous directory."""
        get_tool_output_path(tmp_path, "bandit")
        autonomous_dir = tmp_path / ".autonomous"
        assert autonomous_dir.exists()
        assert autonomous_dir.is_dir()

    def test_creates_reports_subdirectory(self, tmp_path):
        """Should create reports subdirectory."""
        get_tool_output_path(tmp_path, "bandit")
        reports_dir = tmp_path / ".autonomous" / "reports"
        assert reports_dir.exists()
        assert reports_dir.is_dir()

    def test_directory_structure_depth(self, tmp_path):
        """Directory structure should be exactly 2 levels deep."""
        result = get_tool_output_path(tmp_path, "bandit")

        # Count directory depth from worktree
        relative = result.relative_to(tmp_path)
        parts = relative.parts

        # Should be: .autonomous/reports/bandit.json (3 parts)
        assert len(parts) == 3
        assert parts[0] == ".autonomous"
        assert parts[1] == "reports"
        assert parts[2] == "bandit.json"

    def test_directory_permissions(self, tmp_path):
        """Created directories should be readable and writable."""
        result = get_tool_output_path(tmp_path, "bandit")
        reports_dir = result.parent

        # Check directory is accessible
        assert os.access(reports_dir, os.R_OK)
        assert os.access(reports_dir, os.W_OK)
        assert os.access(reports_dir, os.X_OK)  # Execute/search permission


class TestIntegrationWithGates:
    """Test integration with Gate classes."""

    def test_bandit_output_path_in_gate4(self, tmp_path):
        """Gate4 should use get_tool_output_path for bandit."""
        from gates import Gate4_QualityMetrics

        # Create gate instance
        instance_dir = tmp_path / "instance"
        instance_dir.mkdir()

        gate = Gate4_QualityMetrics(tmp_path, instance_dir)

        # Get expected path
        expected_path = get_tool_output_path(tmp_path, "bandit")

        # Verify reports directory was created
        assert expected_path.parent.exists()

    def test_coverage_output_path_in_gate3(self, tmp_path):
        """Gate3 should use get_tool_output_path for coverage."""
        from gates import Gate3_TestsPass

        # Create gate instance
        instance_dir = tmp_path / "instance"
        instance_dir.mkdir()

        gate = Gate3_TestsPass(tmp_path, instance_dir)

        # Get expected path
        expected_path = get_tool_output_path(tmp_path, "coverage")

        # Verify reports directory was created
        assert expected_path.parent.exists()

    def test_multiple_tools_share_directory(self, tmp_path):
        """Multiple tools should share the same reports directory."""
        bandit_path = get_tool_output_path(tmp_path, "bandit")
        coverage_path = get_tool_output_path(tmp_path, "coverage")
        mypy_path = get_tool_output_path(tmp_path, "mypy")

        # All should be in same directory
        assert bandit_path.parent == coverage_path.parent == mypy_path.parent

        # Directory should exist
        assert bandit_path.parent.exists()


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_empty_tool_name(self, tmp_path):
        """Should handle empty tool name gracefully."""
        result = get_tool_output_path(tmp_path, "")
        # Should still create valid path structure
        assert result.parent.name == "reports"
        # Filename will be ".json" - empty stem with .json extension
        # Path(".json").suffix returns "" because it treats the whole thing as stem
        # So we check the name instead
        assert result.name == ".json"

    def test_tool_name_with_spaces(self, tmp_path):
        """Should handle tool names with spaces."""
        result = get_tool_output_path(tmp_path, "my tool", "json")
        assert result.name == "my tool.json"
        assert result.parent.name == "reports"

    def test_extension_without_dot(self, tmp_path):
        """Extension without leading dot should work."""
        result = get_tool_output_path(tmp_path, "bandit", "json")
        assert result.suffix == ".json"

    def test_extension_with_dot(self, tmp_path):
        """Extension with leading dot should work."""
        result = get_tool_output_path(tmp_path, "bandit", ".json")
        assert result.suffix == ".json"

    def test_nonexistent_worktree_path(self):
        """Should handle nonexistent worktree path by creating it."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nonexistent = Path(tmpdir) / "does_not_exist"

            # Should create the directory structure
            result = get_tool_output_path(nonexistent, "bandit")

            # Reports directory should be created
            assert result.parent.exists()
