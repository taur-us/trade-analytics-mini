#!/usr/bin/env bash
# Signal Completion Helper Script
#
# Usage: ./scripts/signal_complete.sh [instance_dir]
#
# Creates completion marker to signal orchestrator that work is done.
# Should be called by spawned Claude instances after completing their task.

set -e

INSTANCE_DIR="${1:-.autonomous}"

# Ensure we're in a worktree instance
if [ ! -d "$INSTANCE_DIR" ]; then
    echo "ERROR: Instance directory not found: $INSTANCE_DIR"
    echo "Usage: $0 [instance_dir]"
    exit 1
fi

# Create completion marker
MARKER_FILE="$INSTANCE_DIR/COMPLETE"
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$MARKER_FILE"

echo "✅ Completion marker written: $MARKER_FILE"
echo "Orchestrator will detect this and proceed to next phase."

exit 0
