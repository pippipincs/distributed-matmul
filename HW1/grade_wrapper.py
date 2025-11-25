#!/usr/bin/env python3
import subprocess, os, re, shutil, argparse
from typing import Tuple
import json

# Determine hidden vs visible feedback
def is_visible_test(test_name: str, visible_indices: list) -> bool:
    # test_name like 'tests/testin_6.txt'
    match = re.search(r'testin_(\d+)\.txt', test_name)
    if match:
        idx = int(match.group(1))
        return idx in visible_indices
    return True


def hidden_feedback_for(test_name: str, visible_indices: list, hidden_messages: list) -> str:
    match = re.search(r'testin_(\d+)\.txt', test_name)
    if match:
        idx = match.group(1)
        message = hidden_messages.get(idx, None)
        if message:
            return message
    return "Hidden test feedback unavailable."

# Deduction helper to add string describing deduction along with actual deduction
def deduct(entries, per_case, label, points, msg, visible_indices, hidden_messages):
    for inp, t in entries:
        deduction = per_case
        points -= deduction
        if is_visible_test(inp, visible_indices):
            msg.append(f"-{deduction:.4f} {label} on {inp} with {t} threads")
        else:
            hint = hidden_feedback_for(inp, visible_indices, hidden_messages)
            msg.append(f"-{deduction:.4f} {label} on hidden test {inp} with {t} threads ({hint})")

    return points


def cleanup(result_dir: str, binary: str = "dtw_parallel"):
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    if os.path.exists(binary):
        os.remove(binary)


def run_autograder() -> Tuple[float, str, str]:
    """Runs the autograder and computes a (score, message)."""
    result_dir = "result"
    log_file = os.path.join(result_dir, "result.txt")

    # Remove old results
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)

    # Load configuration
    with open("config.json") as f:
        config = json.load(f)

    compiler = config.get("compiler", "gcc")
    visible_indices = config.get("visible_indices", [])
    hidden_messages = config.get("hidden_messages", [])
    speedup_indices = config.get("speedup_indices", [])

    # Run the bash autograder
    subprocess.run(["bash", "autograder.sh"], capture_output=True)

    # If no log was generated → compilation failure
    if not os.path.exists(log_file):
        print("Log file does not exist, likely compilation failure.")
        cleanup(result_dir)
        return (0, "Program logs file does not exist", "Log File Not created")

    with open(log_file) as f:
        logs = f.read()

    # Full run check
    if "Exit Error" in logs:
        print("Non-runtime errors. Likely compilation failure.")
        cleanup(result_dir)
        return (0, "Program does not complete for some reason.\n See autograder.sh & attached Autograder Logs for more details.", logs)

    # Scoring
    points = 12.0
    msg = []

    threads = [1, 2, 4, 8, 16, 32, 64]
    num_threads = len(threads)
    num_tests = len(re.findall(r"=== Testing tests/testin_", logs))

    # Extract cases
    rt_errs = re.findall(r"Runtime error on (.*?) with (\d+) threads", logs)
    leaks = re.findall(r"Memory leak on (.*?) with (\d+) threads", logs)
    unit_fails = re.findall(r"Unit test fails on (.*?) with (\d+) threads", logs)
    speed_fails = re.findall(r"Speedup failure on (.*?) with (\d+) threads", logs)

    points = deduct(rt_errs, 0.25 / (num_tests * num_threads), "Runtime error", points, msg, visible_indices, hidden_messages)
    points = deduct(leaks, 0.25, "Memory Leak", points, msg, visible_indices, hidden_messages)
    points = deduct(unit_fails, 3 / (num_tests * num_threads), "Unit test fails", points, msg, visible_indices, hidden_messages)
    points = deduct(speed_fails, 8 / ((num_threads-1) * len(speedup_indices)), "Speedup fails", points, msg, visible_indices, hidden_messages)

    # Cleanup generated files
    cleanup(result_dir)

    points = max(points, 0)
    if points == 12.0:
        msg.append("All tests passed. Full credit given from Autograder.")
    return points, "\n".join(msg), logs


if __name__ == "__main__":
    points, msg, logs = run_autograder()
    print(f"Final Score: {points:.2f}/12.00\n\n")
    
    print(f"**** Autograder Rubric Feedback (output of 'grade_wrapper.py') ****\n{msg}\n\n\n\n")

    print(f"**** Autograder Logs (output of 'autograder.sh') ****\n\n{logs}")
    
