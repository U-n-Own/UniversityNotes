#!/usr/bin/env python3
import subprocess
import argparse
import time
import csv
import re
from pathlib import Path

def run_experiment(program_path, args, iteration):
    """Run one experiment and extract timing"""
    cmd = [program_path] + list(map(str, args))
    
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
    except Exception as e:
        print(f"Error running experiment {iteration}: {str(e)}")
        return None

    # Parse timing from output using regex
    time_pattern = r"softime_avx:\s+(\d+\.\d+)\s+ms"
    match = re.search(time_pattern, result.stderr)
    
    if not match:
        print(f"Failed to parse time from iteration {iteration}")
        print("Program output:")
        print(result.stderr)
        return None

    return float(match.group(1))

def main():
    parser = argparse.ArgumentParser(description='Softmax benchmark runner')
    parser.add_argument('program', help='Path to the compiled softmax program')
    parser.add_argument('k', type=int, help='Input size K')
    parser.add_argument('print', choices=['true', 'false'], 
                       help='Print output flag')
    parser.add_argument('-n', '--num-runs', type=int, default=10,
                       help='Number of experiment repetitions')
    parser.add_argument('-o', '--output', default='results.csv',
                       help='Output CSV file name')

    args = parser.parse_args()

    # Prepare output file
    csv_path = Path(args.output)
    csv_exists = csv_path.exists()
    
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        
        # Write header if file is new
        if not csv_exists:
            writer.writerow(['Run', 'K', 'Print', 'Time(ms)'])

        # Run experiments
        for i in range(1, args.num_runs + 1):
            start_time = time.time()
            
            # Run program with specified arguments
            elapsed = run_experiment(
                args.program,
                [args.k, args.print],
                i
            )

            if elapsed is None:
                continue

            # Write results
            writer.writerow([
                i,
                args.k,
                args.print.lower() == 'true',
                f"{elapsed:.4f}"
            ])
            
            print(f"Run {i}/{args.num_runs} completed: {elapsed:.2f} ms")
            
if __name__ == "__main__":
    main()