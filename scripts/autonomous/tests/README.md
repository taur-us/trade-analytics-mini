# Coordination Tests

Comprehensive unit and integration tests for TASK-058 Phase 2 coordination infrastructure.

## Quick Stats

- **43 tests** (all passing)
- **89% coverage** (327 statements, 100 branches)
- **13.0s** execution time
- **0% flaky** (100% deterministic)

## Test Breakdown

| Component | Tests | Coverage |
|-----------|-------|----------|
| DistributedLockManager | 11 | 95% |
| InstanceRegistry | 11 | 88% |
| TaskCoordinator | 10 | 90% |
| MessageQueue | 7 | 85% |
| Integration | 4 | N/A |

## Running Tests

```bash
# All tests
cd /path/to/your-project
python -m pytest scripts/autonomous/tests/test_coordination.py -v

# With coverage
python -m pytest scripts/autonomous/tests/test_coordination.py \
    --cov=autonomous.coordination \
    --cov-report=term-missing \
    --cov-branch
```

## Critical Findings

### 🐛 Bug #1: Queue Not Persisted (HIGH)
**Location**: `coordination.py` lines 303-312
**Impact**: Tasks queued when resource limit reached are LOST
**Status**: Documented in tests, needs fix in coordination.py

### 🐛 Bug #2: get_task_status Corrupts Data (CRITICAL)
**Location**: `coordination.py` lines 757-779
**Impact**: Calling get_task_status OVERWRITES task_queue.json
**Status**: Workaround in test_get_task_status, BLOCKER for production

## Test Files

- `test_coordination.py` - Main test suite (1,015 lines)
- `__init__.py` - Package initialization
- `README.md` - This file

## Full Report

See `reports/TEST-COORDINATION-COVERAGE-REPORT.md` for complete analysis.

## Key Tests

### Concurrency (Critical)
- `test_atomic_file_operation_concurrent` - 30 concurrent ops, no corruption
- `test_concurrent_claim_no_double_assignment` - Prevents double-assignment
- `test_concurrent_instances_same_task` - Race condition handling

### Lifecycle
- `test_full_workflow_single_instance` - Complete instance lifecycle
- `test_heartbeat_thread_starts` - Automatic heartbeat (10s interval)
- `test_cleanup_stale_instances` - Stale detection (60s threshold)

### Error Handling
- `test_acquire_lock_timeout` - Lock timeout with exponential backoff
- `test_register_instance_limit_reached` - 3-instance limit enforcement
- `test_instance_crash_cleanup` - Crashed instance cleanup

---

**Next Steps**: Fix 2 critical bugs before production deployment.
