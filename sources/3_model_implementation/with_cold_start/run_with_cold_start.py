"""
Runner script to execute all three with-cold-start algorithms sequentially.

This script runs:
1. SVD Matrix Factorization (fastest, ~5-10 minutes)
2. Item-Based CF (medium, ~15-25 minutes)
3. User-Based CF (slowest, ~30-60 minutes)

Total expected runtime: 50-95 minutes

All algorithms evaluate on FULL temporal test set (including cold-start cases).
"""

import subprocess
import sys
import time
from datetime import datetime

def run_script(script_name, description):
    """Run a Python script and display its output."""
    print("\n" + "="*80)
    print(f"RUNNING: {description}")
    print("="*80)
    print(f"Script: {script_name}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")

    start_time = time.time()

    try:
        # Run the script and stream output in real-time
        process = subprocess.Popen(
            [sys.executable, script_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )

        # Print output line by line
        for line in process.stdout:
            print(line, end='')

        process.wait()
        elapsed = time.time() - start_time

        if process.returncode == 0:
            print(f"\n✓ {description} completed successfully in {elapsed/60:.2f} minutes")
            return True
        else:
            print(f"\n✗ {description} failed with return code {process.returncode}")
            return False

    except Exception as e:
        print(f"\n✗ Error running {description}: {str(e)}")
        return False

def main():
    """Main execution function."""
    print("="*80)
    print("TEMPORAL SPLIT WITH COLD-START - ALGORITHM COMPARISON")
    print("="*80)
    print("\nThis script will run:")
    print("  1. SVD Matrix Factorization (~5-10 minutes)")
    print("  2. Item-Based CF (~15-25 minutes)")
    print("  3. User-Based CF (~30-60 minutes)")
    print("\nAll algorithms include cold-start cases (full test set)")
    print("\nTotal expected runtime: 50-95 minutes")
    print("\nPress Ctrl+C to cancel...")
    print("="*80)

    time.sleep(3)  # Give user time to cancel

    overall_start = time.time()
    results = {}

    # Run SVD first (fastest)
    results['SVD'] = run_script(
        '3_svd_temporal.py',
        'SVD Matrix Factorization (With Cold-Start)'
    )

    # Run Item-Based CF second (medium speed)
    results['Item-Based CF'] = run_script(
        '1_item_based_cf.py',
        'Item-Based CF (With Cold-Start)'
    )

    # Run User-Based CF third (slowest)
    results['User-Based CF'] = run_script(
        '2_user_based_cf_temporal.py',
        'User-Based CF (With Cold-Start)'
    )

    # Summary
    overall_elapsed = time.time() - overall_start

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nTotal runtime: {overall_elapsed/60:.2f} minutes")
    print("\nResults:")
    for algo, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  {algo}: {status}")

    all_success = all(results.values())

    if all_success:
        print("\n✓ All algorithms completed successfully!")
        print("\nResults saved to:")
        print("  - datasets/output/model_implementations/svd_with_cold_start.csv")
        print("  - datasets/output/model_implementations/item_based_cf_with_cold_start.csv")
        print("  - datasets/output/model_implementations/user_based_cf_with_cold_start.csv")
        print("\nYou can now compare these with your filtered (99%) results!")
        print("Run: python compare_results.py")
    else:
        print("\n⚠️  Some algorithms failed. Check the output above for errors.")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user. Exiting...")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Unexpected error: {str(e)}")
        sys.exit(1)
