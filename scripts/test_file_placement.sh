#!/bin/bash
#
# Test script for file placement validation
#
# Tests that the file placement validator correctly identifies and blocks
# unauthorized files at the repository root.

# Don't exit on error - we want to run all tests
set +e

echo "========================================="
echo "File Placement Validation Test"
echo "========================================="
echo ""

# Get repository root
REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counter
TESTS_PASSED=0
TESTS_FAILED=0

# Helper function to run a test
run_test() {
    local test_name="$1"
    local test_command="$2"
    local expected_exit_code="$3"

    echo -n "Test: $test_name ... "

    if eval "$test_command" > /dev/null 2>&1; then
        actual_exit_code=0
    else
        actual_exit_code=$?
    fi

    if [ "$actual_exit_code" -eq "$expected_exit_code" ]; then
        echo -e "${GREEN}PASS${NC}"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}FAIL${NC}"
        echo "  Expected exit code: $expected_exit_code"
        echo "  Actual exit code: $actual_exit_code"
        ((TESTS_FAILED++))
    fi
}

# Test 1: Allowed files should pass validation
echo "=== Test 1: Allowed files ==="
run_test "README.md at root should be allowed" \
    "python scripts/validate_file_placement.py --file README.md" \
    0

run_test "CONTRIBUTING.md at root should be allowed" \
    "python scripts/validate_file_placement.py --file CONTRIBUTING.md" \
    0

run_test "LICENSE at root should be allowed" \
    "python scripts/validate_file_placement.py --file LICENSE" \
    0

echo ""

# Test 2: Create unauthorized file and test validation
echo "=== Test 2: Unauthorized files ==="

# Create test file
TEST_FILE="STATUS_UPDATE.md"
touch "$TEST_FILE"

run_test "Unauthorized .md file should be blocked" \
    "python scripts/validate_file_placement.py --file $TEST_FILE" \
    1

# Clean up
rm -f "$TEST_FILE"

# Create another test file
TEST_FILE2="IMPLEMENTATION_NOTES.md"
touch "$TEST_FILE2"

run_test "Implementation notes at root should be blocked" \
    "python scripts/validate_file_placement.py --file $TEST_FILE2" \
    1

# Clean up
rm -f "$TEST_FILE2"

echo ""

# Test 3: Files in subdirectories should always pass
echo "=== Test 3: Files in subdirectories ==="

run_test "Files in docs/ should be allowed" \
    "python scripts/validate_file_placement.py --file docs/GATES.md" \
    0

run_test "Files in scripts/ should be allowed" \
    "python scripts/validate_file_placement.py --file scripts/validate_file_placement.py" \
    0

echo ""

# Test 4: Test pre-commit hook
echo "=== Test 4: Pre-commit hook ==="

# Get the common git directory (works for both regular repos and worktrees)
GIT_COMMON_DIR=$(git rev-parse --git-common-dir 2>/dev/null || echo "$REPO_ROOT/.git")
PRE_COMMIT_HOOK="$GIT_COMMON_DIR/hooks/pre-commit"

# Check if pre-commit hook exists
if [ -f "$PRE_COMMIT_HOOK" ]; then
    echo -e "${GREEN}Pre-commit hook exists at $PRE_COMMIT_HOOK${NC}"
    ((TESTS_PASSED++))
else
    echo -e "${RED}Pre-commit hook not found at $PRE_COMMIT_HOOK${NC}"
    ((TESTS_FAILED++))
fi

# Check if hook is executable
if [ -x "$PRE_COMMIT_HOOK" ]; then
    echo -e "${GREEN}Pre-commit hook is executable${NC}"
    ((TESTS_PASSED++))
else
    echo -e "${RED}Pre-commit hook is not executable${NC}"
    ((TESTS_FAILED++))
fi

echo ""

# Test 5: Test with staged file (if git is available)
echo "=== Test 5: Integration test with git ==="

# Store current git state
ORIGINAL_BRANCH=$(git branch --show-current 2>/dev/null || echo "unknown")

# Create test file
TEST_FILE3="TEST_UNAUTHORIZED.md"
echo "This is a test file" > "$TEST_FILE3"

# Try to stage it
git add "$TEST_FILE3" 2>/dev/null || true

# Test validator with staged files
if python scripts/validate_file_placement.py --staged > /dev/null 2>&1; then
    echo -e "${RED}FAIL: Validator should have rejected staged unauthorized file${NC}"
    ((TESTS_FAILED++))
else
    echo -e "${GREEN}PASS: Validator correctly rejected staged unauthorized file${NC}"
    ((TESTS_PASSED++))
fi

# Clean up
git reset HEAD "$TEST_FILE3" 2>/dev/null || true
rm -f "$TEST_FILE3"

echo ""

# Summary
echo "========================================="
echo "Test Summary"
echo "========================================="
echo -e "Tests passed: ${GREEN}$TESTS_PASSED${NC}"
echo -e "Tests failed: ${RED}$TESTS_FAILED${NC}"
echo "========================================="

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed.${NC}"
    exit 1
fi
