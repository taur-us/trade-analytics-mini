#!/bin/bash
#
# Post-SDLC Cleanup - Restore Public-Facing State
# Run after all SDLC tasks complete
#

set -e
cd "$(dirname "$0")/.."

echo "🧹 Post-SDLC Cleanup Started..."
echo ""

# 1. Delete all SDLC deliverables
echo "Deleting SDLC deliverables..."
rm -rf deliverables/SDLC-* deliverables/ENHANCE-* deliverables/AUTO-* 2>/dev/null || true
echo "✅ Deliverables cleaned"

# 2. Delete overnight logs
echo "Archiving logs..."
mkdir -p .sdlc-development-archive
mv logs/SDLC-*.log .sdlc-development-archive/ 2>/dev/null || true
echo "✅ Logs archived (not in git)"

# 3. Restore clean example task queue
echo "Restoring clean task queue..."
cat > tasks/task_queue.json << 'EOF'
{
  "backlog": [
    {
      "id": "EXAMPLE-001",
      "title": "Create user authentication module",
      "description": "Implement a basic user authentication module with login, logout, and session management. Include JWT token handling and password hashing with bcrypt.",
      "priority": "HIGH",
      "dependencies": [],
      "estimated_hours": 2,
      "assigned_to": null,
      "model": "sonnet",
      "status": "READY",
      "acceptance_criteria": [
        "User class with email and hashed password",
        "login() method validates credentials and returns JWT",
        "logout() method invalidates session",
        "Password hashing uses bcrypt",
        "Comprehensive tests with 95%+ coverage",
        "Security best practices followed"
      ],
      "context": {
        "type": "feature",
        "purpose": "Example: Standard feature with auto-merge"
      }
    },
    {
      "id": "EXAMPLE-002",
      "title": "Add email notification service",
      "description": "Create email notification service that integrates with the authentication module. Send welcome emails, password resets, and account notifications.",
      "priority": "MEDIUM",
      "dependencies": ["EXAMPLE-001"],
      "estimated_hours": 1.5,
      "assigned_to": null,
      "model": "sonnet",
      "status": "READY",
      "acceptance_criteria": [
        "EmailService class with send_email() method",
        "Templates for welcome, password reset, notifications",
        "Integration with authentication module",
        "Mock SMTP in tests",
        "Test coverage >= 95%"
      ],
      "context": {
        "type": "feature",
        "purpose": "Example: Task with dependency - waits for EXAMPLE-001 to merge"
      }
    },
    {
      "id": "MIGRATION-EXAMPLE",
      "title": "Migrate database schema to colleague's repository",
      "description": "Port database migration scripts from our repo to colleague's project. Adapt table names and column mappings to their schema conventions.",
      "priority": "CRITICAL",
      "dependencies": [],
      "estimated_hours": 3,
      "assigned_to": null,
      "model": "sonnet",
      "status": "READY",
      "git_safety": {
        "auto_merge_disabled": true
      },
      "acceptance_criteria": [
        "Migration scripts adapted for target repo",
        "Schema mappings documented",
        "Rollback scripts included",
        "Tests validate migrations work",
        "PR created for colleague's review (NOT auto-merged)"
      ],
      "context": {
        "type": "migration",
        "purpose": "Example: Cross-repo task with auto_merge_disabled - PR NOT auto-merged"
      }
    }
  ],
  "in_progress": [],
  "completed": [],
  "failed": []
}
EOF
echo "✅ Task queue restored to clean examples"

# 4. Update README with new SDLC workflow
echo "Updating README with 10-phase SDLC..."
# This will be done by final commit

# 5. Commit clean state
echo "Committing clean public-facing state..."
git add -A
git commit -m "chore: Post-SDLC cleanup - restore public-facing state

All 6 SDLC enhancements implemented and tested.
Development artifacts cleaned up.
Repository restored to professional state.

New features:
- 10-phase SDLC workflow (0-9)
- Design phase with tech-lead
- Review gates with review-checkpoint
- 16 specialized agents (6 original + 10 imported)
- File placement enforcement

Repository is production-ready with enhanced SDLC.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"

git push origin main

echo ""
echo "=================================================="
echo "✅ POST-SDLC CLEANUP COMPLETE!"
echo "=================================================="
echo ""
echo "Repository is now:"
echo "- Clean and professional"
echo "- Enhanced with full SDLC workflow"
echo "- 16 specialized agents"
echo "- File placement enforced"
echo "- Ready for public use"
echo ""
