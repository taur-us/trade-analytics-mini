#!/usr/bin/env python3
"""
Merge Conflict Auto-Resolution for Autonomous Workflow.

Implements conflict detection, parsing, resolution strategies,
and validation for automated merge conflict handling.

CONFLICT-001: Full Merge Conflict Auto-Resolution
"""

import json
import logging
import re
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)


# ==============================================================================
# SECTION 1: Data Models
# ==============================================================================

class ConflictType(Enum):
    """Types of merge conflicts."""
    CONTENT = "content"      # Normal content conflict
    BINARY = "binary"        # Binary file conflict (cannot auto-resolve)
    DELETED = "deleted"      # File deleted in one branch, modified in other
    RENAMED = "renamed"      # File renamed differently in both branches
    MODE = "mode"            # File mode/permission conflict


class ResolutionStrategy(Enum):
    """Available conflict resolution strategies."""
    PREFER_OURS = "prefer_ours"       # Keep our version
    PREFER_THEIRS = "prefer_theirs"   # Keep their version
    PREFER_NEWER = "prefer_newer"     # Keep most recently modified version
    COMBINE = "combine"               # Merge both changes (different regions)
    CLAUDE = "claude"                 # AI-powered semantic resolution
    MANUAL = "manual"                 # Requires human intervention


@dataclass
class ConflictBlock:
    """Represents a single conflict block within a file."""
    file_path: str
    start_line: int
    end_line: int
    ours_content: str           # Content between <<<<<<< and =======
    theirs_content: str         # Content between ======= and >>>>>>>
    base_content: Optional[str] # Content from common ancestor (if available)
    ours_label: str             # Label after <<<<<<< (usually branch name)
    theirs_label: str           # Label after >>>>>>> (usually branch name)
    context_before: str         # Lines before conflict for context
    context_after: str          # Lines after conflict for context

    def to_dict(self) -> Dict:
        return {
            'file_path': self.file_path,
            'start_line': self.start_line,
            'end_line': self.end_line,
            'ours_content': self.ours_content,
            'theirs_content': self.theirs_content,
            'base_content': self.base_content,
            'ours_label': self.ours_label,
            'theirs_label': self.theirs_label,
        }


@dataclass
class ConflictedFile:
    """Represents a file with one or more conflicts."""
    file_path: Path
    conflict_type: ConflictType
    conflicts: List[ConflictBlock]
    file_type: str              # Extension (.py, .md, .json, etc.)
    total_lines: int
    conflict_line_count: int    # Total lines in conflict regions

    @property
    def conflict_ratio(self) -> float:
        """Ratio of conflicted lines to total lines."""
        return self.conflict_line_count / self.total_lines if self.total_lines > 0 else 0


@dataclass
class ResolutionAttempt:
    """Record of a resolution attempt."""
    strategy: ResolutionStrategy
    timestamp: str
    success: bool
    resolved_content: Optional[str]
    validation_result: Optional[Dict]
    error_message: Optional[str]

    def to_dict(self) -> Dict:
        return {
            'strategy': self.strategy.value,
            'timestamp': self.timestamp,
            'success': self.success,
            'validation_result': self.validation_result,
            'error_message': self.error_message,
        }


@dataclass
class ResolutionResult:
    """Result of conflict resolution for a file."""
    file_path: str
    resolved: bool
    strategy_used: ResolutionStrategy
    attempts: List[ResolutionAttempt]
    final_content: Optional[str]
    requires_human_review: bool
    review_reason: Optional[str]

    def to_dict(self) -> Dict:
        return {
            'file_path': self.file_path,
            'resolved': self.resolved,
            'strategy_used': self.strategy_used.value,
            'attempts': [a.to_dict() for a in self.attempts],
            'requires_human_review': self.requires_human_review,
            'review_reason': self.review_reason,
        }


# ==============================================================================
# SECTION 2: Conflict Detection
# ==============================================================================

class ConflictDetector:
    """Detects merge conflicts in files and directories."""

    # Conflict marker patterns
    CONFLICT_START = re.compile(r'^<<<<<<<\s*(.*)$', re.MULTILINE)
    CONFLICT_SEPARATOR = re.compile(r'^=======$', re.MULTILINE)
    CONFLICT_END = re.compile(r'^>>>>>>>\s*(.*)$', re.MULTILINE)
    CONFLICT_BASE = re.compile(r'^\|\|\|\|\|\|\|\s*(.*)$', re.MULTILINE)  # diff3 style

    def __init__(self, worktree_path: Path):
        """Initialize detector with worktree path.

        Args:
            worktree_path: Path to git worktree
        """
        self.worktree_path = Path(worktree_path).resolve()

    def scan_for_conflicts(self) -> List[ConflictedFile]:
        """Scan worktree for files with conflict markers.

        Returns:
            List of ConflictedFile objects
        """
        conflicted_files = []

        # Method 1: Use git to find unmerged files
        git_conflicts = self._get_git_unmerged_files()

        # Method 2: Scan for conflict markers (catches cases git might miss)
        marker_conflicts = self._scan_for_markers()

        # Combine results (deduplicate by path)
        all_paths = set(git_conflicts) | set(marker_conflicts)

        for path in all_paths:
            full_path = self.worktree_path / path
            if full_path.exists() and full_path.is_file():
                conflict_file = self._analyze_conflicted_file(full_path)
                if conflict_file:
                    conflicted_files.append(conflict_file)

        logger.info(f"Found {len(conflicted_files)} conflicted files")
        return conflicted_files

    def _get_git_unmerged_files(self) -> List[str]:
        """Get list of unmerged files from git.

        Returns:
            List of relative file paths with conflicts
        """
        try:
            result = subprocess.run(
                ['git', 'diff', '--name-only', '--diff-filter=U'],
                cwd=str(self.worktree_path),
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip().split('\n')
        except (subprocess.TimeoutExpired, subprocess.SubprocessError) as e:
            logger.warning(f"Failed to get unmerged files from git: {e}")

        return []

    def _scan_for_markers(self) -> List[str]:
        """Scan files for conflict markers.

        Returns:
            List of relative file paths containing conflict markers
        """
        conflicted = []

        # Use grep for efficiency
        try:
            result = subprocess.run(
                ['git', 'grep', '-l', '<<<<<<<'],
                cwd=str(self.worktree_path),
                capture_output=True,
                text=True,
                timeout=60
            )
            if result.returncode == 0 and result.stdout.strip():
                conflicted = result.stdout.strip().split('\n')
        except (subprocess.TimeoutExpired, subprocess.SubprocessError) as e:
            logger.warning(f"Git grep failed, falling back to Python scan: {e}")
            # Fallback: Python-based scan (slower but works)
            conflicted = self._python_scan_markers()

        return conflicted

    def _python_scan_markers(self) -> List[str]:
        """Python-based fallback for conflict marker scanning."""
        conflicted = []

        for root, dirs, files in self.worktree_path.walk():
            # Skip hidden directories and common excludes
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in {'node_modules', '__pycache__', 'venv', '.venv'}]

            for file in files:
                file_path = root / file
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read(10000)  # Check first 10KB
                        if '<<<<<<<' in content:
                            rel_path = file_path.relative_to(self.worktree_path)
                            conflicted.append(str(rel_path))
                except (IOError, OSError):
                    pass

        return conflicted

    def _analyze_conflicted_file(self, file_path: Path) -> Optional[ConflictedFile]:
        """Analyze a conflicted file to extract conflict details.

        Args:
            file_path: Absolute path to file

        Returns:
            ConflictedFile object or None if no conflicts found
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
        except (IOError, UnicodeDecodeError) as e:
            logger.warning(f"Cannot read {file_path}: {e}")
            return None

        # Check for binary conflict
        if self._is_binary_conflict(file_path):
            return ConflictedFile(
                file_path=file_path,
                conflict_type=ConflictType.BINARY,
                conflicts=[],
                file_type=file_path.suffix,
                total_lines=len(lines),
                conflict_line_count=0
            )

        # Parse conflict blocks
        conflicts = self._parse_conflict_blocks(file_path, content, lines)

        if not conflicts:
            return None

        conflict_lines = sum(c.end_line - c.start_line + 1 for c in conflicts)

        return ConflictedFile(
            file_path=file_path,
            conflict_type=ConflictType.CONTENT,
            conflicts=conflicts,
            file_type=file_path.suffix,
            total_lines=len(lines),
            conflict_line_count=conflict_lines
        )

    def _is_binary_conflict(self, file_path: Path) -> bool:
        """Check if file is a binary file conflict."""
        try:
            result = subprocess.run(
                ['git', 'diff', '--numstat', str(file_path)],
                cwd=str(self.worktree_path),
                capture_output=True,
                text=True,
                timeout=10
            )
            # Binary files show as "-\t-\t" in numstat
            return result.stdout.strip().startswith('-\t-')
        except Exception:
            return False

    def _parse_conflict_blocks(self, file_path: Path, content: str, lines: List[str]) -> List[ConflictBlock]:
        """Parse all conflict blocks in file content.

        Args:
            file_path: Path to file
            content: Full file content
            lines: Content split into lines

        Returns:
            List of ConflictBlock objects
        """
        conflicts = []
        i = 0

        while i < len(lines):
            line = lines[i]

            # Look for conflict start marker
            match = self.CONFLICT_START.match(line)
            if match:
                ours_label = match.group(1).strip()
                start_line = i + 1  # 1-indexed for display

                # Find separator and end
                ours_content_lines = []
                theirs_content_lines = []
                base_content_lines = []
                separator_found = False
                base_found = False
                end_line = None
                theirs_label = ""

                j = i + 1
                while j < len(lines):
                    if self.CONFLICT_BASE.match(lines[j]):
                        base_found = True
                        j += 1
                        continue
                    elif self.CONFLICT_SEPARATOR.match(lines[j]):
                        separator_found = True
                        j += 1
                        continue
                    elif (end_match := self.CONFLICT_END.match(lines[j])):
                        theirs_label = end_match.group(1).strip()
                        end_line = j + 1  # 1-indexed
                        break
                    else:
                        if not separator_found:
                            if base_found:
                                base_content_lines.append(lines[j])
                            else:
                                ours_content_lines.append(lines[j])
                        else:
                            theirs_content_lines.append(lines[j])
                    j += 1

                if end_line:
                    # Get context (5 lines before and after)
                    context_start = max(0, i - 5)
                    context_end = min(len(lines), j + 6)

                    conflicts.append(ConflictBlock(
                        file_path=str(file_path.relative_to(self.worktree_path)),
                        start_line=start_line,
                        end_line=end_line,
                        ours_content='\n'.join(ours_content_lines),
                        theirs_content='\n'.join(theirs_content_lines),
                        base_content='\n'.join(base_content_lines) if base_content_lines else None,
                        ours_label=ours_label,
                        theirs_label=theirs_label,
                        context_before='\n'.join(lines[context_start:i]),
                        context_after='\n'.join(lines[j+1:context_end])
                    ))
                    i = j + 1
                    continue

            i += 1

        return conflicts


# ==============================================================================
# SECTION 3: Resolution Strategies
# ==============================================================================

class ResolutionStrategyBase(ABC):
    """Base class for resolution strategies."""

    @abstractmethod
    def can_resolve(self, conflict: ConflictBlock, file_type: str) -> bool:
        """Check if this strategy can resolve the conflict.

        Args:
            conflict: The conflict block to resolve
            file_type: File extension (e.g., '.py', '.md')

        Returns:
            True if strategy is applicable
        """
        pass

    @abstractmethod
    def resolve(self, conflict: ConflictBlock, file_type: str) -> Tuple[bool, str, str]:
        """Attempt to resolve the conflict.

        Args:
            conflict: The conflict block to resolve
            file_type: File extension

        Returns:
            Tuple of (success, resolved_content, error_message)
        """
        pass


class PreferOursStrategy(ResolutionStrategyBase):
    """Strategy: Keep our version, discard theirs."""

    def can_resolve(self, conflict: ConflictBlock, file_type: str) -> bool:
        return True  # Always applicable

    def resolve(self, conflict: ConflictBlock, file_type: str) -> Tuple[bool, str, str]:
        return True, conflict.ours_content, ""


class PreferTheirsStrategy(ResolutionStrategyBase):
    """Strategy: Keep their version, discard ours."""

    def can_resolve(self, conflict: ConflictBlock, file_type: str) -> bool:
        return True  # Always applicable

    def resolve(self, conflict: ConflictBlock, file_type: str) -> Tuple[bool, str, str]:
        return True, conflict.theirs_content, ""


class PreferNewerStrategy(ResolutionStrategyBase):
    """Strategy: Keep the most recently modified version."""

    def __init__(self, worktree_path: Path):
        self.worktree_path = worktree_path

    def can_resolve(self, conflict: ConflictBlock, file_type: str) -> bool:
        # Can resolve if we can determine which is newer
        return bool(conflict.ours_label and conflict.theirs_label)

    def resolve(self, conflict: ConflictBlock, file_type: str) -> Tuple[bool, str, str]:
        try:
            # Get commit timestamps for both branches
            ours_time = self._get_commit_time(conflict.ours_label)
            theirs_time = self._get_commit_time(conflict.theirs_label)

            if ours_time is None or theirs_time is None:
                return False, "", "Cannot determine commit timestamps"

            if ours_time >= theirs_time:
                return True, conflict.ours_content, f"Preferred ours (newer: {conflict.ours_label})"
            else:
                return True, conflict.theirs_content, f"Preferred theirs (newer: {conflict.theirs_label})"

        except Exception as e:
            return False, "", str(e)

    def _get_commit_time(self, ref: str) -> Optional[int]:
        """Get commit timestamp for a reference."""
        try:
            result = subprocess.run(
                ['git', 'log', '-1', '--format=%ct', ref],
                cwd=str(self.worktree_path),
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                return int(result.stdout.strip())
        except Exception:
            pass
        return None


class CombineStrategy(ResolutionStrategyBase):
    """Strategy: Combine both changes when they modify different regions."""

    def can_resolve(self, conflict: ConflictBlock, file_type: str) -> bool:
        # Can combine if changes are to different logical regions
        # This is a heuristic based on line similarity
        ours_lines = set(conflict.ours_content.strip().split('\n'))
        theirs_lines = set(conflict.theirs_content.strip().split('\n'))

        # If no overlap, likely different regions
        overlap = ours_lines & theirs_lines
        return len(overlap) == 0 or len(overlap) < min(len(ours_lines), len(theirs_lines)) / 2

    def resolve(self, conflict: ConflictBlock, file_type: str) -> Tuple[bool, str, str]:
        """Combine changes, placing ours before theirs."""
        # Simple combination: ours + theirs
        # For code files, add a blank line separator
        separator = '\n' if file_type in ['.py', '.js', '.ts', '.go', '.rs'] else '\n'

        combined = conflict.ours_content
        if conflict.ours_content and conflict.theirs_content:
            combined = conflict.ours_content + separator + conflict.theirs_content
        elif conflict.theirs_content:
            combined = conflict.theirs_content

        return True, combined, "Combined both changes"


class ClaudeResolutionStrategy(ResolutionStrategyBase):
    """Strategy: Use Claude to semantically resolve complex conflicts."""

    def __init__(self, worktree_path: Path, state_dir: Path):
        self.worktree_path = worktree_path
        self.state_dir = state_dir

    def can_resolve(self, conflict: ConflictBlock, file_type: str) -> bool:
        # Can attempt resolution for code files
        code_extensions = {'.py', '.js', '.ts', '.go', '.rs', '.java', '.c', '.cpp', '.h'}
        return file_type in code_extensions

    def resolve(self, conflict: ConflictBlock, file_type: str) -> Tuple[bool, str, str]:
        """Use Claude to intelligently resolve the conflict.

        This method prepares a prompt and invokes Claude to analyze and resolve
        the conflict semantically, preserving intent from both sides.
        """
        prompt = self._build_resolution_prompt(conflict, file_type)

        # Write prompt to file for agent invocation
        prompt_file = self.state_dir / f"conflict_resolution_prompt_{datetime.now().strftime('%H%M%S')}.txt"
        prompt_file.write_text(prompt)

        # The actual Claude invocation would be done by the orchestrator
        # This returns the prompt for async processing
        return False, "", f"Claude resolution prepared: {prompt_file}"

    def _build_resolution_prompt(self, conflict: ConflictBlock, file_type: str) -> str:
        """Build a detailed prompt for Claude conflict resolution."""
        return f"""You are resolving a merge conflict in a {file_type} file.

## Conflict Location
File: {conflict.file_path}
Lines: {conflict.start_line}-{conflict.end_line}

## Context Before Conflict
```{file_type[1:] if file_type else ''}
{conflict.context_before}
```

## OURS Version ({conflict.ours_label})
```{file_type[1:] if file_type else ''}
{conflict.ours_content}
```

## THEIRS Version ({conflict.theirs_label})
```{file_type[1:] if file_type else ''}
{conflict.theirs_content}
```

{f"## BASE Version (Common Ancestor){chr(10)}```{file_type[1:] if file_type else ''}{chr(10)}{conflict.base_content}{chr(10)}```" if conflict.base_content else ""}

## Context After Conflict
```{file_type[1:] if file_type else ''}
{conflict.context_after}
```

## Instructions
1. Analyze the intent of both changes
2. Determine the best way to combine or select changes
3. Preserve functionality from BOTH versions if possible
4. Ensure the result is syntactically correct
5. Return ONLY the resolved code (no markers, no explanation)

## Resolved Code:
"""


# ==============================================================================
# SECTION 4: Resolution Validator
# ==============================================================================

class ResolutionValidator:
    """Validates that resolved conflicts are correct."""

    def __init__(self, worktree_path: Path):
        self.worktree_path = worktree_path

    def validate(self, file_path: Path, resolved_content: str) -> Dict[str, Any]:
        """Validate the resolved file content.

        Args:
            file_path: Path to the file being resolved
            resolved_content: The proposed resolution content

        Returns:
            Dict with 'valid', 'issues', and 'checks' keys
        """
        file_type = file_path.suffix
        result = {
            'valid': True,
            'issues': [],
            'checks': {}
        }

        # Check 1: No remaining conflict markers
        if '<<<<<<<' in resolved_content or '>>>>>>>' in resolved_content:
            result['valid'] = False
            result['issues'].append("Conflict markers still present")
            result['checks']['markers_removed'] = False
        else:
            result['checks']['markers_removed'] = True

        # Check 2: Syntax validation (language-specific)
        if file_type == '.py':
            syntax_valid, syntax_error = self._validate_python_syntax(resolved_content)
            result['checks']['syntax_valid'] = syntax_valid
            if not syntax_valid:
                result['valid'] = False
                result['issues'].append(f"Python syntax error: {syntax_error}")

        elif file_type == '.json':
            syntax_valid, syntax_error = self._validate_json_syntax(resolved_content)
            result['checks']['syntax_valid'] = syntax_valid
            if not syntax_valid:
                result['valid'] = False
                result['issues'].append(f"JSON syntax error: {syntax_error}")

        elif file_type in ['.yaml', '.yml']:
            syntax_valid, syntax_error = self._validate_yaml_syntax(resolved_content)
            result['checks']['syntax_valid'] = syntax_valid
            if not syntax_valid:
                result['valid'] = False
                result['issues'].append(f"YAML syntax error: {syntax_error}")

        # Check 3: Basic sanity checks
        if not resolved_content.strip():
            result['valid'] = False
            result['issues'].append("Resolution is empty")

        return result

    def _validate_python_syntax(self, content: str) -> Tuple[bool, str]:
        """Validate Python syntax using AST."""
        try:
            import ast
            ast.parse(content)
            return True, ""
        except SyntaxError as e:
            return False, f"Line {e.lineno}: {e.msg}"

    def _validate_json_syntax(self, content: str) -> Tuple[bool, str]:
        """Validate JSON syntax."""
        try:
            json.loads(content)
            return True, ""
        except json.JSONDecodeError as e:
            return False, f"Line {e.lineno}: {e.msg}"

    def _validate_yaml_syntax(self, content: str) -> Tuple[bool, str]:
        """Validate YAML syntax."""
        try:
            import yaml
            yaml.safe_load(content)
            return True, ""
        except yaml.YAMLError as e:
            return False, str(e)
        except ImportError:
            return True, ""  # Skip if PyYAML not installed

    def run_affected_tests(self, file_path: Path) -> Tuple[bool, str]:
        """Run tests affected by the resolved file.

        Args:
            file_path: Path to the resolved file

        Returns:
            Tuple of (tests_passed, output)
        """
        rel_path = file_path.relative_to(self.worktree_path)

        # Find related test file
        test_patterns = [
            f"tests/test_{file_path.stem}.py",
            f"tests/{file_path.stem}_test.py",
            f"test_{file_path.stem}.py",
        ]

        for pattern in test_patterns:
            test_file = self.worktree_path / pattern
            if test_file.exists():
                try:
                    result = subprocess.run(
                        ['pytest', str(test_file), '-v', '--tb=short'],
                        cwd=str(self.worktree_path),
                        capture_output=True,
                        text=True,
                        timeout=120
                    )
                    return result.returncode == 0, result.stdout + result.stderr
                except Exception as e:
                    return False, str(e)

        # No specific tests found, that's OK
        return True, "No related tests found"


# ==============================================================================
# SECTION 5: Main Conflict Resolver
# ==============================================================================

class ConflictResolver:
    """Main conflict resolution orchestrator."""

    def __init__(self, worktree_path: Path, state_dir: Path, config: Dict = None):
        """Initialize resolver.

        Args:
            worktree_path: Path to git worktree
            state_dir: Path to state directory for storing resolution data
            config: Optional configuration overrides
        """
        self.worktree_path = Path(worktree_path).resolve()
        self.state_dir = Path(state_dir).resolve()
        self.state_dir.mkdir(parents=True, exist_ok=True)

        # Configuration
        self.config = config or {}
        self.max_resolution_attempts = self.config.get('max_resolution_attempts', 3)

        # Initialize components
        self.detector = ConflictDetector(self.worktree_path)
        self.validator = ResolutionValidator(self.worktree_path)

        # Initialize strategies in order of preference
        self.strategies = [
            (ResolutionStrategy.PREFER_NEWER, PreferNewerStrategy(self.worktree_path)),
            (ResolutionStrategy.COMBINE, CombineStrategy()),
            (ResolutionStrategy.CLAUDE, ClaudeResolutionStrategy(self.worktree_path, self.state_dir)),
            (ResolutionStrategy.PREFER_OURS, PreferOursStrategy()),  # Fallback
        ]

        # Resolution history
        self.resolution_log: List[ResolutionResult] = []

    def resolve_all(self) -> Dict[str, Any]:
        """Detect and resolve all conflicts in worktree.

        Returns:
            Dict with resolution summary and results
        """
        logger.info("Starting conflict resolution...")

        # Detect conflicts
        conflicted_files = self.detector.scan_for_conflicts()

        if not conflicted_files:
            logger.info("No conflicts detected")
            return {
                'status': 'clean',
                'conflicts_found': 0,
                'resolved': 0,
                'failed': 0,
                'results': []
            }

        logger.info(f"Found {len(conflicted_files)} files with conflicts")

        results = []
        resolved_count = 0
        failed_count = 0

        for conflicted_file in conflicted_files:
            result = self._resolve_file(conflicted_file)
            results.append(result)

            if result.resolved:
                resolved_count += 1
                # Write resolved content to file
                self._write_resolved_file(conflicted_file.file_path, result.final_content)
            else:
                failed_count += 1

        # Save resolution log
        self._save_resolution_log(results)

        return {
            'status': 'resolved' if failed_count == 0 else 'partial',
            'conflicts_found': len(conflicted_files),
            'resolved': resolved_count,
            'failed': failed_count,
            'results': [r.to_dict() for r in results]
        }

    def _resolve_file(self, conflicted_file: ConflictedFile) -> ResolutionResult:
        """Resolve all conflicts in a single file.

        Args:
            conflicted_file: ConflictedFile object to resolve

        Returns:
            ResolutionResult with resolution details
        """
        logger.info(f"Resolving {conflicted_file.file_path}")

        # Handle binary conflicts
        if conflicted_file.conflict_type == ConflictType.BINARY:
            return ResolutionResult(
                file_path=str(conflicted_file.file_path),
                resolved=False,
                strategy_used=ResolutionStrategy.MANUAL,
                attempts=[],
                final_content=None,
                requires_human_review=True,
                review_reason="Binary file conflict cannot be auto-resolved"
            )

        # Read original file content
        try:
            with open(conflicted_file.file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
        except Exception as e:
            return ResolutionResult(
                file_path=str(conflicted_file.file_path),
                resolved=False,
                strategy_used=ResolutionStrategy.MANUAL,
                attempts=[],
                final_content=None,
                requires_human_review=True,
                review_reason=f"Cannot read file: {e}"
            )

        # Resolve each conflict block
        attempts = []
        resolved_content = original_content

        for conflict in conflicted_file.conflicts:
            block_resolved = False

            for attempt_num in range(self.max_resolution_attempts):
                for strategy_enum, strategy in self.strategies:
                    if not strategy.can_resolve(conflict, conflicted_file.file_type):
                        continue

                    success, resolution, error = strategy.resolve(conflict, conflicted_file.file_type)

                    attempt = ResolutionAttempt(
                        strategy=strategy_enum,
                        timestamp=datetime.now().isoformat(),
                        success=success,
                        resolved_content=resolution if success else None,
                        validation_result=None,
                        error_message=error if not success else None
                    )

                    if success:
                        # Validate resolution
                        # Create temporary resolved content for validation
                        temp_resolved = self._apply_resolution(
                            resolved_content, conflict, resolution
                        )

                        validation = self.validator.validate(
                            conflicted_file.file_path, temp_resolved
                        )
                        attempt.validation_result = validation

                        if validation['valid']:
                            resolved_content = temp_resolved
                            block_resolved = True
                            attempts.append(attempt)
                            break
                        else:
                            logger.warning(f"Resolution validation failed: {validation['issues']}")

                    attempts.append(attempt)

                if block_resolved:
                    break

            if not block_resolved:
                # Could not resolve this conflict block
                return ResolutionResult(
                    file_path=str(conflicted_file.file_path),
                    resolved=False,
                    strategy_used=ResolutionStrategy.MANUAL,
                    attempts=attempts,
                    final_content=None,
                    requires_human_review=True,
                    review_reason=f"Could not resolve conflict at lines {conflict.start_line}-{conflict.end_line}"
                )

        # All conflicts resolved
        strategy_used = attempts[-1].strategy if attempts else ResolutionStrategy.MANUAL

        return ResolutionResult(
            file_path=str(conflicted_file.file_path),
            resolved=True,
            strategy_used=strategy_used,
            attempts=attempts,
            final_content=resolved_content,
            requires_human_review=False,
            review_reason=None
        )

    def _apply_resolution(self, content: str, conflict: ConflictBlock, resolution: str) -> str:
        """Apply a resolution to file content.

        Args:
            content: Original file content with conflict markers
            conflict: The conflict block being resolved
            resolution: The resolution content

        Returns:
            Content with conflict block replaced by resolution
        """
        lines = content.split('\n')

        # Find conflict markers in content
        # We need to find and replace the entire conflict block
        result_lines = []
        i = 0

        while i < len(lines):
            if lines[i].startswith('<<<<<<<'):
                # Found conflict start, skip to end
                while i < len(lines) and not lines[i].startswith('>>>>>>>'):
                    i += 1
                # Skip the end marker
                if i < len(lines):
                    i += 1
                # Add resolution
                result_lines.extend(resolution.split('\n'))
            else:
                result_lines.append(lines[i])
                i += 1

        return '\n'.join(result_lines)

    def _write_resolved_file(self, file_path: Path, content: str) -> None:
        """Write resolved content back to file.

        Args:
            file_path: Path to file
            content: Resolved content
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"Wrote resolved content to {file_path}")

            # Stage the resolved file
            subprocess.run(
                ['git', 'add', str(file_path)],
                cwd=str(self.worktree_path),
                capture_output=True,
                timeout=10
            )
        except Exception as e:
            logger.error(f"Failed to write resolved file: {e}")

    def _save_resolution_log(self, results: List[ResolutionResult]) -> None:
        """Save resolution log to state directory.

        Args:
            results: List of resolution results
        """
        log_file = self.state_dir / "conflict_resolution_log.json"

        log_data = {
            'timestamp': datetime.now().isoformat(),
            'worktree': str(self.worktree_path),
            'max_attempts': self.max_resolution_attempts,
            'results': [r.to_dict() for r in results]
        }

        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)

        logger.info(f"Saved resolution log to {log_file}")


# ==============================================================================
# SECTION 6: Human Review Escalation
# ==============================================================================

@dataclass
class HumanReviewRequest:
    """Request for human review of unresolved conflicts."""
    task_id: str
    file_path: str
    conflict_details: Dict
    resolution_attempts: List[Dict]
    created_at: str
    priority: str  # HIGH, MEDIUM, LOW

    def to_dict(self) -> Dict:
        return {
            'task_id': self.task_id,
            'file_path': self.file_path,
            'conflict_details': self.conflict_details,
            'resolution_attempts': self.resolution_attempts,
            'created_at': self.created_at,
            'priority': self.priority
        }


class HumanReviewEscalator:
    """Manages escalation of unresolved conflicts to human review."""

    def __init__(self, main_repo: Path, task_id: str):
        self.main_repo = main_repo
        self.task_id = task_id
        self.review_queue_file = main_repo / ".autonomous" / "human_review_queue.json"

    def escalate(self, result: ResolutionResult) -> HumanReviewRequest:
        """Escalate unresolved conflict to human review queue.

        Args:
            result: ResolutionResult that requires human review

        Returns:
            HumanReviewRequest object
        """
        request = HumanReviewRequest(
            task_id=self.task_id,
            file_path=result.file_path,
            conflict_details={'review_reason': result.review_reason},
            resolution_attempts=[a.to_dict() for a in result.attempts],
            created_at=datetime.now().isoformat(),
            priority='HIGH'  # Conflicts block workflow
        )

        self._add_to_queue(request)

        logger.warning(f"Escalated {result.file_path} to human review queue")
        return request

    def _add_to_queue(self, request: HumanReviewRequest) -> None:
        """Add request to human review queue file."""
        queue = []

        if self.review_queue_file.exists():
            try:
                with open(self.review_queue_file, 'r') as f:
                    queue = json.load(f)
            except (json.JSONDecodeError, IOError):
                queue = []

        queue.append(request.to_dict())

        self.review_queue_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.review_queue_file, 'w') as f:
            json.dump(queue, f, indent=2)


# ==============================================================================
# SECTION 7: Custom Exceptions
# ==============================================================================

class ConflictResolutionError(Exception):
    """Base exception for conflict resolution errors."""
    pass


class UnresolvableConflictError(ConflictResolutionError):
    """Raised when conflict cannot be auto-resolved."""
    def __init__(self, file_path: str, reason: str):
        self.file_path = file_path
        self.reason = reason
        super().__init__(f"Cannot resolve {file_path}: {reason}")


class ValidationError(ConflictResolutionError):
    """Raised when resolution fails validation."""
    def __init__(self, file_path: str, issues: List[str]):
        self.file_path = file_path
        self.issues = issues
        super().__init__(f"Validation failed for {file_path}: {issues}")
