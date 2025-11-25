#!/usr/bin/env python3
"""
File Placement Validation Script

Validates that files are placed in correct locations according to repository rules.
Can be run manually or as a pre-commit hook.

Usage:
    python scripts/validate_file_placement.py              # Check all files
    python scripts/validate_file_placement.py --staged     # Check only staged files
    python scripts/validate_file_placement.py --file FILE  # Check specific file

Exit codes:
    0: All files in correct locations
    1: Validation errors found
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import List, Tuple, Set

# Files allowed at repository root
ALLOWED_ROOT_FILES = {
    'README.md',
    'CONTRIBUTING.md',
    'LICENSE',
    'CHANGELOG.md',
    'SESSION_SUMMARY.md',
    '.gitignore',
    '.gitattributes',
    '.env.example',
    'pyproject.toml',
    'install.sh',
}

# Allowed directories at repository root
ALLOWED_ROOT_DIRS = {
    '.git',
    '.github',
    '.claude',
    '.autonomous',
    'docs',
    'deliverables',
    'tasks',
    'scripts',
    'reports',
}

# File type to correct location mapping
FILE_LOCATION_RULES = {
    'Documentation': ('docs/', '*.md documentation files'),
    'Task docs': ('tasks/completed/', 'TASK-*.md or SDLC-*.md'),
    'Agent deliverables': ('deliverables/', 'Agent output documents'),
    'Reports': ('reports/', 'Status and performance reports'),
    'Test files': ('scripts/autonomous/tests/', 'test_*.py files'),
    'Temporary prompts': ('docs/prompts/', 'Prompt templates'),
    'Implementation notes': ('docs/', 'Implementation documentation'),
    'Scripts': ('scripts/', 'Python scripts and shell scripts'),
}


def get_repo_root() -> Path:
    """Get the repository root directory."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--show-toplevel'],
            capture_output=True,
            text=True,
            check=True
        )
        return Path(result.stdout.strip())
    except subprocess.CalledProcessError:
        # Fallback: assume we're in the repo
        return Path(__file__).parent.parent


def get_staged_files() -> List[str]:
    """Get list of files staged for commit."""
    try:
        result = subprocess.run(
            ['git', 'diff', '--cached', '--name-only', '--diff-filter=ACM'],
            capture_output=True,
            text=True,
            check=True
        )
        return [f for f in result.stdout.strip().split('\n') if f]
    except subprocess.CalledProcessError:
        return []


def get_all_files(repo_root: Path) -> List[str]:
    """Get all tracked files in the repository."""
    try:
        result = subprocess.run(
            ['git', 'ls-files'],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True
        )
        return [f for f in result.stdout.strip().split('\n') if f]
    except subprocess.CalledProcessError:
        return []


def is_root_file(file_path: str) -> bool:
    """Check if file is at repository root."""
    return '/' not in file_path and '\\' not in file_path


def suggest_location(filename: str) -> List[str]:
    """Suggest correct location for a file."""
    suggestions = []

    if filename.endswith('.md'):
        if filename.startswith(('TASK-', 'SDLC-', 'task-', 'sdlc-')):
            suggestions.append('tasks/completed/')
        elif any(word in filename.upper() for word in ['STATUS', 'REPORT', 'SUMMARY', 'UPDATE']):
            suggestions.append('docs/archive/')
            suggestions.append('reports/')
        elif 'DELIVERABLE' in filename.upper() or filename.startswith('SDLC-'):
            suggestions.append('deliverables/')
        else:
            suggestions.append('docs/')

    elif filename.startswith('test_') and filename.endswith('.py'):
        suggestions.append('scripts/autonomous/tests/')

    elif filename.endswith(('.py', '.sh')):
        suggestions.append('scripts/')

    elif filename.endswith(('.txt', '.log')):
        suggestions.append('reports/')

    if not suggestions:
        suggestions.append('docs/')

    return suggestions


def validate_file_placement(files: List[str], repo_root: Path) -> Tuple[bool, List[str]]:
    """
    Validate file placements according to rules.

    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []

    for file_path in files:
        # Skip if file is in a subdirectory (automatically allowed)
        if not is_root_file(file_path):
            continue

        # Check if file is allowed at root
        filename = os.path.basename(file_path)
        if filename not in ALLOWED_ROOT_FILES:
            suggestions = suggest_location(filename)
            error_msg = f"\nERROR: Unauthorized file at repository root: {filename}\n"
            error_msg += "\nAllowed files at root:\n"
            for allowed in sorted(ALLOWED_ROOT_FILES):
                error_msg += f"  - {allowed}\n"
            error_msg += f"\nPlease move this file to the appropriate location:\n"
            for suggestion in suggestions:
                error_msg += f"  {filename} -> {suggestion}{filename}\n"
            errors.append(error_msg)

    return len(errors) == 0, errors


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Validate file placements according to repository rules'
    )
    parser.add_argument(
        '--staged',
        action='store_true',
        help='Check only staged files (for pre-commit hook)'
    )
    parser.add_argument(
        '--file',
        type=str,
        help='Check specific file'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show verbose output'
    )

    args = parser.parse_args()

    repo_root = get_repo_root()

    # Determine which files to check
    if args.file:
        files = [args.file]
    elif args.staged:
        files = get_staged_files()
        if not files:
            if args.verbose:
                print("No staged files to check.")
            sys.exit(0)
    else:
        files = get_all_files(repo_root)

    if args.verbose:
        print(f"Checking {len(files)} file(s)...")

    # Validate
    is_valid, errors = validate_file_placement(files, repo_root)

    if not is_valid:
        print("=" * 70)
        print("FILE PLACEMENT VALIDATION FAILED")
        print("=" * 70)
        for error in errors:
            print(error)
        print("=" * 70)
        print("\nTo bypass this check (not recommended):")
        print("  git commit --no-verify")
        print("\nFor more information, see CONTRIBUTING.md")
        print("=" * 70)
        sys.exit(1)
    else:
        if args.verbose:
            print("All files are in correct locations.")
        sys.exit(0)


if __name__ == '__main__':
    main()
