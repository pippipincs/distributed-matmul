#!/bin/bash
# =======================================================
# Block Cyclic Sparse Cannon's Algorithm Autograder
# =======================================================

set -e  # stop on any failure

PROGRAM="sparse_cyclic_cannon"  # target source (sparse_cyclic_cannon.c)
HOSTFILE="hosts.txt"       # contains csel-plate001
TEST_DIR="tests"           # location of test_0, test_1, test_2, test_3, test_4 directories
BUILD_DIR="build"          # build output directory

mkdir -p "$BUILD_DIR"

# -------------------------------
# (1) Compile the C program
# -------------------------------
echo "🔧 Compiling $PROGRAM.c ..."
set +e
mpicc -O3 -std=c99 -o "$BUILD_DIR/$PROGRAM" "$PROGRAM.c" -lm
compile_exit_code=$?
set -e

if [ $compile_exit_code -ne 0 ] || [ ! -f "$BUILD_DIR/$PROGRAM" ]; then
    echo "❌ Compilation failed."
    exit 1
fi
echo "✅ Compilation successful."
echo

# -------------------------------
# (2) Helper function to run test
# -------------------------------
run_test() {
    local test_id=$1
    local np=$2
    local cycle=$3

    local input="$TEST_DIR/test_${test_id}/in.txt"
    local ref_output="$TEST_DIR/test_${test_id}/out.txt"
    local tmp_output="$BUILD_DIR/out_test${test_id}_np${np}.txt"
    local log_file="$BUILD_DIR/log_test${test_id}_np${np}.txt"

    if [ ! -f "$input" ] || [ ! -f "$ref_output" ]; then
        echo "⚠️  Missing input or reference output for test_${test_id}"
        return
    fi

    # --------------------------
    # Timeout and thresholds
    # --------------------------
    local timeout_val=5
    declare -A max_times_4=(
        [1]=6 [4]=1.5 [9]=0.8 [16]=0.5 [25]=0.4
    )

    if [ "$test_id" -eq 4 ]; then
        timeout_val=45
    fi

    echo "🚀 Running test_${test_id} with -np ${np} -cycle=${cycle} (timeout=${timeout_val}s) ..."

    # --------------------------
    # Run with timeout + runtime-error detection
    # --------------------------
    set +e
    timeout "${timeout_val}s" mpirun -np "$np" \
        --hostfile "$HOSTFILE" \
        --map-by node \
        "$BUILD_DIR/$PROGRAM" "$input" "$tmp_output" "$cycle" \
        >"$log_file" 2>&1
    local exit_code=$?
    set -e

    if [ $exit_code -eq 124 ]; then
        echo "⏱️  test_${test_id} (np=${np}, cycle=${cycle}) TIMEOUT after ${timeout_val}s"
        echo
        rm -f "$tmp_output" "$log_file"
        return
    elif [ $exit_code -ne 0 ]; then
        echo "💥 test_${test_id} (np=${np}, cycle=${cycle}) RUNTIME ERROR (exit code $exit_code)"
        echo "🔍 See $log_file for details"
        echo
        rm -f "$tmp_output" "$log_file"
        return
    fi

    # --------------------------
    # Compare against reference
    # --------------------------
    echo "🔍 Comparing output with reference..."
    if diff -q "$tmp_output" "$ref_output" > /dev/null; then
        echo "✅ UNIT TEST PASSED: test_${test_id} (np=${np}, cycle=${cycle})"
    else
        echo "❌ UNIT TEST FAILED: test_${test_id} (np=${np}, cycle=${cycle})"
        if [ "$test_id" -eq 4 ]; then
            echo "❌ test_4 (np=${np}, cycle=${cycle}) SPEEDUP FAIL: skipping speedup check due to incorrect output"
            echo
            rm -f "$tmp_output" "$log_file"
            return
        fi
    fi

    # --------------------------
    # Speedup timing check (only for test 4)
    # --------------------------
    if [ "$test_id" -eq 4 ]; then
        local elapsed
        elapsed=$(grep "Time for matrix multiplication:" "$log_file" | awk '{print $5}')
        if [ -z "$elapsed" ]; then
            echo "⚠️  test_4 (np=${np}, cycle=${cycle}) SPEEDUP FAIL: elapsed time not found in log"
        else
            local threshold=${max_times_4[$np]:-0}
            if (( $(awk -v e="$elapsed" -v t="$threshold" 'BEGIN{print (e>t)?1:0}') )); then
                echo "❌ test_4 (np=${np}, cycle=${cycle}) SPEEDUP FAIL: ${elapsed}s (threshold=${threshold}s)"
            else
                echo "✅ test_4 (np=${np}, cycle=${cycle}) SPEEDUP PASS: ${elapsed}s (threshold=${threshold}s)"
            fi
        fi
    fi

    # --------------------------
    # Cleanup temporary files
    # --------------------------
    rm -f "$tmp_output" "$log_file"

    echo
}


# -------------------------------
# (3) Execute Small matrix tests
# -------------------------------
echo "======================================================================================================"
echo "Running Small Matrix Tests"
echo "======================================================================================================"
# A (4x4) x B (4x4) --> Both Sparse
for test in 0 1; do
    run_test "$test" 1 1
    run_test "$test" 4 1
    run_test "$test" 4 2
    run_test "$test" 16 1
done


# A (101x107) x B (107x103) --> Both Dense
for np in 1 4 9 25; do
    for cycle in 1 2 3; do
        run_test 2 "$np" "$cycle"
    done
done


# A (101x107) x B (107x103) --> Both Sparse
for np in 1 4 9 25; do
    for cycle in 1 2 3; do
        run_test 3 "$np" "$cycle"
    done
done


# -------------------------------
# (4) Execute Large matrix tests with speedups
# -------------------------------
echo "======================================================================================================"
echo "Running matrix tests with speedups"
echo "======================================================================================================"


# A (2003x2011) x B (2011x2017) --> both sparse
for np in 1 4 9 16 25; do
    run_test 4 "$np" 2
done

echo "All tests complete."