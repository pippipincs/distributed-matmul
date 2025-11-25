#!/bin/bash
# autograder.sh
# Tests a single student's parallel DTW code for:
#   (1) compilation
#   (2) runtime correctness
#   (3) memory leaks (run separately under Valgrind)
#   (4) unit correctness
#   (5) speedup performance (only for tests > SPEEDUP_START_INDEX)

THREADS=(1 2 4 8 16 32 64)
declare -A relative_time=(
    [1]=1.0
    [2]=0.6
    [4]=0.36
    [8]=0.216
    [16]=0.130
    [32]=0.078
    [64]=0.047
)

# -------------------------------
# CONFIGURATION VARIABLES
# -------------------------------
tol=.0001                 # numeric tolerance for cost differences
TIMEOUT=45               # timeout for each test in seconds

export OMP_PROC_BIND=spread
export OMP_PLACES=cores

mkdir -p result
LOGFILE="result/result.txt"
> "$LOGFILE"

# -------------------------------
# 1. Variable Preparation
# -------------------------------

# Read settings from config.json
CONFIG_FILE="config.json"


# Extract speedup indices as a space-separated list
SPEEDUP_INDICES=($(jq -r '.speedup_indices[]' "$CONFIG_FILE"))

# Extract visible indices (used to mask hidden test feedback)
VISIBLE_INDICES=($(jq -r '.visible_indices[]' "$CONFIG_FILE"))

# -------------------------------
# 2. File checking --> Determine that dtw_parallel.c/pp is present, as well as dtw_serial. Set COMPILER + SRC
# -------------------------------

if [ -f "dtw_parallel.cpp" ]; then
    SRC="dtw_parallel.cpp"
    COMPILER="g++"
elif [ -f "dtw_parallel.c" ]; then
    SRC="dtw_parallel.c"
    COMPILER="gcc"
else
    echo "Exit Error: ❌ No source file found (dtw_parallel.c or dtw_parallel.cpp missing)!" >> "$LOGFILE"
    exit 1
fi

# Check for serial executable
if [ ! -x "./dtw_serial" ]; then
    echo "Exit Error: ❌ Missing dtw_serial executable — cannot run baseline tests." >> "$LOGFILE"
    exit 1
fi


# -------------------------------
# 3. Random Valgrind index selection
# -------------------------------

# Get all visible indices that are NOT speedup indices
# We do this as valgrind takes too long to complete
NON_SPEEDUP_VISIBLE_INDICES=()
for idx in "${VISIBLE_INDICES[@]}"; do
    if [[ ! " ${SPEEDUP_INDICES[@]} " =~ " ${idx} " ]]; then
        NON_SPEEDUP_VISIBLE_INDICES+=("$idx")
    fi
done

# Choose one of these non-speedup indices at random for valgrind testing
if [ ${#NON_SPEEDUP_VISIBLE_INDICES[@]} -gt 0 ]; then
    VALGRIND_INDEX=${NON_SPEEDUP_VISIBLE_INDICES[$((RANDOM % ${#NON_SPEEDUP_VISIBLE_INDICES[@]}))]}
else
    VALGRIND_INDEX=-1
fi

echo "Selected test index ${VALGRIND_INDEX} (visible, non-speedup) for memory leak check" >> "$LOGFILE"



# -------------------------------
# 4. Compilation --> compile the parallel code
# -------------------------------
# Compile the parallel code
$COMPILER -O3 -fopenmp "$SRC" -o dtw_parallel -lm
if [ $? -ne 0 ]; then
    echo "Exit Error: ❌ Compilation failed!" >> "$LOGFILE"
    exit 1
fi
echo "✅ Compilation successful!" >> "$LOGFILE"


# -------------------------------
# 5. Test Loop
# -------------------------------

# Check for test input files in the tests/ directory
shopt -s nullglob
test_files=(tests/testin_*)
if [ ${#test_files[@]} -eq 0 ]; then
    echo "Exit Error: ❌ No test input files found in tests/ directory!" >> "$LOGFILE"
    exit 1
fi

# Run tests
for input in "${test_files[@]}"; do
    i=$(basename "$input" | sed -E 's/[^0-9]*([0-9]+).*/\1/')

    input="tests/testin_${i}.txt"
    expected_output="tests/testout_${i}.txt"
    mkdir -p "result/test_$i"
    echo "=== Testing $input ===" >> "$LOGFILE"

    # -------------------------------
    # Run serial version once per test to get baseline
    # -------------------------------
    serial_out="result/test_$i/result_serial.txt"
    echo "--- Running serial version ---" >> "$LOGFILE"

    ./dtw_serial "$input" > "$serial_out" 2>&1
    if [ $? -ne 0 ]; then
        echo "Exit Error: Runtime error in serial execution on $input" >> "$LOGFILE"
        base_time_ms=""
        base_cost=""
        exit 1
    else
        base_cost=$(grep -i "The Final Cost" "$serial_out" | awk '{print $5}')
        base_time_ms=$(grep -i "The Total Completion time" "$serial_out" | awk '{print $7}')
        echo "Serial baseline: cost=$base_cost, time=${base_time_ms}ms" >> "$LOGFILE"
    fi

    for t in "${THREADS[@]}"; do
        outfile="result/test_$i/result_${t}_threads.txt"
        valfile="result/test_$i/valgrind_${t}.txt"

        echo "--- Running $t threads (main run) ---" >> "$LOGFILE"

        # -------------------
        # Valgrind run (memory leaks) on specific index with number of threads equal to 1 only
        # -------------------
        if [ "$i" -eq "$VALGRIND_INDEX" ] && [ "$t" -eq 1 ]; then
            echo "--- Running $t threads (valgrind) ---" >> "$LOGFILE"
            valgrind --leak-check=full --error-exitcode=99 ./dtw_parallel "$input" "$t" \
                > /dev/null 2> "$valfile"

            if grep -q "definitely lost: [1-9]" "$valfile"; then
                echo "Memory leak on $input with $t threads" >> "$LOGFILE"
            fi
        fi


        # -------------------
        # Normal run (runtime + unit + speedup)
        # -------------------
        timeout ${TIMEOUT}s ./dtw_parallel "$input" "$t" > "$outfile" 2>&1
        exit_code=$?

        # (a) Handle timeout and other runtime errors
        if [ $exit_code -eq 124 ]; then
            echo "Timeout on $input with $t threads (exceeded ${TIMEOUT}s)" >> "$LOGFILE"
            echo "Runtime error on $input with $t threads (timeout causes failure)" >> "$LOGFILE"
            echo "Unit test fails on $input with $t threads (timeout causes failure)" >> "$LOGFILE"
            if [ "$t" -gt 1 ] && [[ " ${SPEEDUP_INDICES[@]} " =~ " ${i} " ]]; then
                echo "Speedup failure on $input with $t threads (runtime error causes failure)" >> "$LOGFILE"
            fi
            continue
        fi

        # (b) Runtime errors with helpful error code information
        if [ $exit_code -ne 0 ]; then
            echo "Runtime error on $input with $t threads (exit code: $exit_code)" >> "$LOGFILE"
            echo "Unit test fails on $input with $t threads (runtime error causes failure)" >> "$LOGFILE"
            if [ "$t" -gt 1 ] && [[ " ${SPEEDUP_INDICES[@]} " =~ " ${i} " ]]; then
                echo "Speedup failure on $input with $t threads (runtime error causes failure)" >> "$LOGFILE"
            fi
            continue
        fi

        # (c) Extract cost/time
        cost=$(grep -i "The Final Cost" "$outfile" | awk '{print $5}')
        time_ms=$(grep -i "The Total Completion time" "$outfile" | awk '{print $7}')

        if [ -z "$cost" ] || [ -z "$time_ms" ]; then
            echo "Runtime error on $input with $t threads: incorrectly printed output" >> "$LOGFILE"
            echo "Unit test fails on $input with $t threads (runtime error causes failure)" >> "$LOGFILE"
            if [ "$t" -gt 1 ] && [[ " ${SPEEDUP_INDICES[@]} " =~ " ${i} " ]]; then
                echo "Speedup failure on $input with $t threads (runtime error causes failure)" >> "$LOGFILE"
            fi
            continue
        fi

        # (d) Unit test correctness
        expected_cost=$(awk '{print $1}' "$expected_output")
        diff=$(awk -v v="$expected_cost" -v t="$cost" 'BEGIN {d=(v>t)?v-t:t-v; print d}')
        if (( $(echo "$diff >= $tol" | bc -l) )); then
            echo "Unit test fails on $input with $t threads" >> "$LOGFILE"
            if [ "$t" -gt 1 ] && [[ " ${SPEEDUP_INDICES[@]} " =~ " ${i} " ]]; then
                echo "Speedup failure on $input with $t threads (unit test error causes failure)" >> "$LOGFILE"
            fi
        fi

        # (e) Speedup checks only for selected indices in config.json, excluding the t=1 condition
        if [ "$t" -gt 1 ] && [[ " ${SPEEDUP_INDICES[@]} " =~ " ${i} " ]]; then
            threshold=${relative_time[$t]}
            rel_current=$(echo "$time_ms / $base_time_ms" | bc -l)
            if [ "$(echo "$rel_current > $threshold" | bc -l)" -eq 1 ]; then
                echo "Speedup failure on $input with $t threads" >> "$LOGFILE"
            fi
        fi
    done
done