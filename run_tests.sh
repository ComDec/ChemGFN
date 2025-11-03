#!/bin/bash
# ChemGFN Test Runner Script
# Provides convenient commands to run different test suites

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Help message
show_help() {
    echo "ChemGFN Test Runner"
    echo ""
    echo "Usage: ./run_tests.sh [OPTION]"
    echo ""
    echo "Options:"
    echo "  all              Run all tests"
    echo "  gfn_utils        Run GFN utility tests"
    echo "  reward           Run reward module tests"
    echo "  loss             Run loss function tests"
    echo "  training         Run training flow tests"
    echo "  unit             Run only unit tests"
    echo "  integration      Run only integration tests"
    echo "  fast             Run only fast tests (skip slow ones)"
    echo "  coverage         Run tests with coverage report"
    echo "  parallel         Run tests in parallel"
    echo "  verbose          Run tests with verbose output"
    echo "  debug            Run tests with debugging enabled"
    echo "  failed           Rerun only failed tests from last run"
    echo "  help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./run_tests.sh all"
    echo "  ./run_tests.sh loss"
    echo "  ./run_tests.sh fast parallel"
}

# Initialize variables
TEST_FILE=""
EXTRA_ARGS=""
RUN_COVERAGE=false
RUN_PARALLEL=false
VERBOSE=false
DEBUG=false
FAST=false
FAILED=false

# Parse arguments
if [ $# -eq 0 ]; then
    show_help
    exit 0
fi

for arg in "$@"; do
    case $arg in
        all)
            TEST_FILE="tests/"
            ;;
        gfn_utils)
            TEST_FILE="tests/test_gfn_utils.py"
            ;;
        reward)
            TEST_FILE="tests/test_reward.py"
            ;;
        loss)
            TEST_FILE="tests/test_loss.py"
            ;;
        training)
            TEST_FILE="tests/test_training_flow.py"
            ;;
        unit)
            EXTRA_ARGS="$EXTRA_ARGS -m unit"
            TEST_FILE="tests/"
            ;;
        integration)
            EXTRA_ARGS="$EXTRA_ARGS -m integration"
            TEST_FILE="tests/"
            ;;
        fast)
            EXTRA_ARGS="$EXTRA_ARGS -m 'not slow'"
            FAST=true
            ;;
        coverage)
            RUN_COVERAGE=true
            ;;
        parallel)
            RUN_PARALLEL=true
            ;;
        verbose)
            VERBOSE=true
            ;;
        debug)
            DEBUG=true
            ;;
        failed)
            FAILED=true
            ;;
        help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $arg${NC}"
            show_help
            exit 1
            ;;
    esac
done

# Build pytest command
CMD="pytest"

# Add test file/directory
if [ -n "$TEST_FILE" ]; then
    CMD="$CMD $TEST_FILE"
else
    CMD="$CMD tests/"
fi

# Add extra arguments
if [ -n "$EXTRA_ARGS" ]; then
    CMD="$CMD $EXTRA_ARGS"
fi

# Add coverage
if [ "$RUN_COVERAGE" = true ]; then
    CMD="$CMD --cov=chemgfn --cov-report=html --cov-report=term-missing"
fi

# Add parallel execution
if [ "$RUN_PARALLEL" = true ]; then
    if command -v pytest-xdist &> /dev/null; then
        CMD="$CMD -n auto"
    else
        echo -e "${YELLOW}Warning: pytest-xdist not installed. Install with: pip install pytest-xdist${NC}"
    fi
fi

# Add verbose
if [ "$VERBOSE" = true ]; then
    CMD="$CMD -vv"
else
    CMD="$CMD -v"
fi

# Add debug
if [ "$DEBUG" = true ]; then
    CMD="$CMD --pdb -s"
fi

# Add failed
if [ "$FAILED" = true ]; then
    CMD="$CMD --lf"
fi

# Print command
echo -e "${GREEN}Running: $CMD${NC}"
echo ""

# Run tests
eval $CMD
TEST_EXIT_CODE=$?

# Print summary
echo ""
if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
    if [ "$RUN_COVERAGE" = true ]; then
        echo -e "${GREEN}Coverage report generated in htmlcov/index.html${NC}"
    fi
else
    echo -e "${RED}✗ Some tests failed${NC}"
    echo -e "${YELLOW}Run with 'debug' option to enter debugger on failure${NC}"
    echo -e "${YELLOW}Run with 'failed' option to rerun only failed tests${NC}"
fi

exit $TEST_EXIT_CODE
