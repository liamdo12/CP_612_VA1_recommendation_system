"""
Compare Warm-Start results with 80-20 Random Split results

This script loads and compares the results from both split strategies.
Saves the comparison to a text file for easy reference.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

def load_results():
    """Load all result files."""
    import os

    # Get project root (3 levels up from this script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))

    results = {}

    base_path = os.path.join(project_root, 'datasets/output/model_implementations/')

    files = {
        'Item-Based CF (80-20)': 'item_based_cf_results_80_20.csv',
        'Item-Based CF (Warm-Start)': 'item_based_cf_warm_start.csv',
        'User-Based CF (80-20)': 'user_based_cf_results_80_20.csv',
        'User-Based CF (Warm-Start)': 'user_based_cf_warm_start.csv',
        'SVD (80-20)': 'svd_results_80_20.csv',
        'SVD (Warm-Start)': 'svd_warm_start.csv'
    }

    print("Loading results...\n")

    for name, filename in files.items():
        path = os.path.join(base_path, filename)
        if os.path.exists(path):
            results[name] = pd.read_csv(path)
            print(f"✓ Loaded: {name}")
        else:
            print(f"✗ Missing: {name} ({filename})")

    return results

def compare_algorithms(results, output_file=None):
    """Compare algorithms across split strategies."""

    # Create a list to store all output lines
    output_lines = []

    def print_and_save(text=""):
        """Print to console and save to output list."""
        print(text)
        output_lines.append(text)

    if len(results) < 6:
        print_and_save(f"\n⚠️  Not all result files found ({len(results)}/6). Run the algorithms first!")
        if len(results) == 0:
            return output_lines
        print_and_save("Comparing available results...\n")

    print_and_save("\n" + "="*80)
    print_and_save("ALGORITHM COMPARISON: TEMPORAL VS RANDOM SPLIT")
    print_and_save("="*80)

    # Extract metrics
    comparison_data = []

    for name, df in results.items():
        comparison_data.append({
            'Algorithm': name,
            'RMSE': df['rmse'].values[0],
            'MAE': df['mae'].values[0],
            'Coverage (%)': df['coverage'].values[0],
            'Test Samples': df['test_samples'].values[0],
            'Training Time (min)': df['training_time_minutes'].values[0],
            'Pred Time (ms)': df['prediction_time_ms'].values[0]
        })

    comparison_df = pd.DataFrame(comparison_data)

    print_and_save("\n" + "-"*80)
    print_and_save("FULL COMPARISON")
    print_and_save("-"*80)
    print_and_save(comparison_df.to_string(index=False))

    # Calculate differences
    print_and_save("\n" + "="*80)
    print_and_save("RMSE COMPARISON: TEMPORAL-99 vs 80-20 RANDOM")
    print_and_save("="*80)

    # Item-Based CF comparison
    if 'Item-Based CF (80-20)' in results and 'Item-Based CF (Warm-Start)' in results:
        item_80 = results['Item-Based CF (80-20)']['rmse'].values[0]
        item_temp = results['Item-Based CF (Warm-Start)']['rmse'].values[0]
        item_diff = item_temp - item_80
        item_pct = (item_diff / item_80) * 100

        print_and_save(f"\nItem-Based CF:")
        print_and_save(f"  80-20 Random:    {item_80:.4f}")
        print_and_save(f"  Warm-Start:     {item_temp:.4f}")
        print_and_save(f"  Difference:      {item_diff:+.4f} ({item_pct:+.2f}%)")
        if item_diff > 0:
            print_and_save(f"  → Temporal split has HIGHER error (temporal drift)")
        else:
            print_and_save(f"  → Temporal split has LOWER error (unexpected!)")

    # User-Based CF comparison
    if 'User-Based CF (80-20)' in results and 'User-Based CF (Warm-Start)' in results:
        user_80 = results['User-Based CF (80-20)']['rmse'].values[0]
        user_temp = results['User-Based CF (Warm-Start)']['rmse'].values[0]
        user_diff = user_temp - user_80
        user_pct = (user_diff / user_80) * 100

        print_and_save(f"\nUser-Based CF:")
        print_and_save(f"  80-20 Random:    {user_80:.4f}")
        print_and_save(f"  Warm-Start:     {user_temp:.4f}")
        print_and_save(f"  Difference:      {user_diff:+.4f} ({user_pct:+.2f}%)")
        if user_diff > 0:
            print_and_save(f"  → Temporal split has HIGHER error (temporal drift)")
        else:
            print_and_save(f"  → Temporal split has LOWER error (unexpected!)")

    # SVD comparison
    if 'SVD (80-20)' in results and 'SVD (Warm-Start)' in results:
        svd_80 = results['SVD (80-20)']['rmse'].values[0]
        svd_temp = results['SVD (Warm-Start)']['rmse'].values[0]
        svd_diff = svd_temp - svd_80
        svd_pct = (svd_diff / svd_80) * 100

        print_and_save(f"\nSVD:")
        print_and_save(f"  80-20 Random:    {svd_80:.4f}")
        print_and_save(f"  Warm-Start:     {svd_temp:.4f}")
        print_and_save(f"  Difference:      {svd_diff:+.4f} ({svd_pct:+.2f}%)")
        if svd_diff > 0:
            print_and_save(f"  → Temporal split has HIGHER error (temporal drift)")
        else:
            print_and_save(f"  → Temporal split has LOWER error (unexpected!)")

    # Analysis
    print_and_save("\n" + "="*80)
    print_and_save("ANALYSIS")
    print_and_save("="*80)

    # Calculate average increase from available algorithms
    pct_changes = []
    if 'Item-Based CF (80-20)' in results and 'Item-Based CF (Warm-Start)' in results:
        pct_changes.append(item_pct)
    if 'User-Based CF (80-20)' in results and 'User-Based CF (Warm-Start)' in results:
        pct_changes.append(user_pct)
    if 'SVD (80-20)' in results and 'SVD (Warm-Start)' in results:
        pct_changes.append(svd_pct)

    if len(pct_changes) > 0:
        avg_increase = sum(pct_changes) / len(pct_changes)
        print_and_save(f"\nAverage RMSE increase with temporal split: {avg_increase:+.2f}%")
    else:
        avg_increase = 0
        print_and_save("\nNot enough data to calculate average RMSE increase")

    if avg_increase > 0:
        print_and_save("\n✓ Expected result: Temporal split shows higher error")
        print_and_save("\nReasons:")
        print_and_save("  1. Temporal drift: User preferences change over time")
        print_and_save("  2. Concept drift: Movie popularity shifts")
        print_and_save("  3. Random split overestimates performance (no time gap)")
        print_and_save(f"\nProduction impact: Expect ~{avg_increase:.1f}% higher RMSE in deployment")
    else:
        print_and_save("\n⚠️  Unexpected result: Temporal split shows LOWER error")
        print_and_save("\nPossible reasons:")
        print_and_save("  1. Random sampling variation")
        print_and_save("  2. Different test set composition")
        print_and_save("  3. Re-run with different random_state to verify")

    # Speed comparison
    print_and_save("\n" + "="*80)
    print_and_save("SPEED COMPARISON")
    print_and_save("="*80)

    print_and_save("\nTraining Time:")
    for algo in ['Item-Based CF', 'User-Based CF', 'SVD']:
        if f'{algo} (80-20)' in results and f'{algo} (Warm-Start)' in results:
            time_80 = results[f'{algo} (80-20)']['training_time_minutes'].values[0]
            time_temp = results[f'{algo} (Warm-Start)']['training_time_minutes'].values[0]
            print_and_save(f"  {algo:20s} (80-20):    {time_80:6.2f} min")
            print_and_save(f"  {algo:20s} (Temp-99):  {time_temp:6.2f} min")

    print_and_save("\nPrediction Speed:")
    for algo in ['Item-Based CF', 'User-Based CF', 'SVD']:
        if f'{algo} (80-20)' in results and f'{algo} (Warm-Start)' in results:
            speed_80 = results[f'{algo} (80-20)']['prediction_time_ms'].values[0]
            speed_temp = results[f'{algo} (Warm-Start)']['prediction_time_ms'].values[0]
            print_and_save(f"  {algo:20s} (80-20):    {speed_80:6.2f} ms/rating")
            print_and_save(f"  {algo:20s} (Temp-99):  {speed_temp:6.2f} ms/rating")

    # Best algorithm
    print_and_save("\n" + "="*80)
    print_and_save("BEST ALGORITHM")
    print_and_save("="*80)

    # Find best for 80-20
    rmse_80 = {}
    if 'Item-Based CF (80-20)' in results:
        rmse_80['Item-Based CF'] = results['Item-Based CF (80-20)']['rmse'].values[0]
    if 'User-Based CF (80-20)' in results:
        rmse_80['User-Based CF'] = results['User-Based CF (80-20)']['rmse'].values[0]
    if 'SVD (80-20)' in results:
        rmse_80['SVD'] = results['SVD (80-20)']['rmse'].values[0]

    # Find best for Warm-Start
    rmse_temp = {}
    if 'Item-Based CF (Warm-Start)' in results:
        rmse_temp['Item-Based CF'] = results['Item-Based CF (Warm-Start)']['rmse'].values[0]
    if 'User-Based CF (Warm-Start)' in results:
        rmse_temp['User-Based CF'] = results['User-Based CF (Warm-Start)']['rmse'].values[0]
    if 'SVD (Warm-Start)' in results:
        rmse_temp['SVD'] = results['SVD (Warm-Start)']['rmse'].values[0]

    if rmse_80 and rmse_temp:
        best_80 = min(rmse_80, key=rmse_80.get)
        best_temp = min(rmse_temp, key=rmse_temp.get)

        print_and_save(f"\nBest on 80-20 Random Split:  {best_80} (RMSE: {rmse_80[best_80]:.4f})")
        print_and_save(f"Best on Warm-Start Split:   {best_temp} (RMSE: {rmse_temp[best_temp]:.4f})")

        if best_80 == best_temp:
            print_and_save(f"\n✓ {best_80} is consistently the best across both splits")
        else:
            print_and_save(f"\n⚠️  Different winners: {best_80} (80-20) vs {best_temp} (Temporal)")
            print_and_save("   → Algorithm ranking depends on evaluation methodology!")
    else:
        print_and_save("\n⚠️  Not enough results to determine best algorithm")

    return output_lines

def main():
    """Main execution."""
    # Get project root for output file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))

    print("="*80)
    print("TEMPORAL VS RANDOM SPLIT COMPARISON")
    print("="*80)

    results = load_results()

    if len(results) > 0:
        # Run comparison and capture output
        output_lines = compare_algorithms(results)

        # Save to text file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"algorithm_comparison_{timestamp}.txt"
        output_path = os.path.join(project_root, 'sources/3_model_implementation/temporal_split', output_filename)

        print("\n" + "="*80)
        print("SAVING RESULTS TO FILE")
        print("="*80)

        with open(output_path, 'w') as f:
            # Write header
            f.write("="*80 + "\n")
            f.write("ALGORITHM COMPARISON: TEMPORAL-99 VS 80-20 RANDOM SPLIT\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")

            # Write all comparison output
            for line in output_lines:
                f.write(line + "\n")

        print(f"\n✓ Comparison saved to: {output_filename}")
        print(f"  Full path: {output_path}")
    else:
        print("\n✗ No results found. Please run the algorithms first:")
        print("   python run_temporal_99_comparison.py")

    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
