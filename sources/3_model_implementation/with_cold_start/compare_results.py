"""
Compare With Cold-Start results with Warm-Start results

This script loads and compares the results from both approaches:
1. With Cold-Start: Full temporal test set (including cold-start cases)
2. Warm-Start: Filtered test set (known user-movie pairs only)

Saves the comparison to a timestamped text file.
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
        'Item-Based CF (Warm-Start)': 'item_based_cf_warm_start.csv',
        'Item-Based CF (With Cold-Start)': 'item_based_cf_with_cold_start.csv',
        'User-Based CF (Warm-Start)': 'user_based_cf_warm_start.csv',
        'User-Based CF (With Cold-Start)': 'user_based_cf_with_cold_start.csv',
        'SVD (Warm-Start)': 'svd_warm_start.csv',
        'SVD (With Cold-Start)': 'svd_with_cold_start.csv'
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

def compare_algorithms(results):
    """Compare algorithms across cold-start strategies."""

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
    print_and_save("ALGORITHM COMPARISON: WITH COLD-START VS WARM-START")
    print_and_save("="*80)

    # Extract metrics
    comparison_data = []

    for name, df in results.items():
        data = {
            'Algorithm': name,
            'RMSE': df['rmse'].values[0],
            'MAE': df['mae'].values[0],
            'Test Samples': df['test_samples'].values[0],
            'Training Time (min)': df['training_time_minutes'].values[0],
            'Pred Time (ms)': df['prediction_time_ms'].values[0]
        }

        # Add cold-start predictions if available
        if 'cold_start_predictions' in df.columns:
            data['Cold-Start Fallback'] = df['cold_start_predictions'].values[0]
        elif 'cf_predictions' in df.columns:
            data['Cold-Start Fallback'] = df['test_samples'].values[0] - df['cf_predictions'].values[0]
        else:
            data['Cold-Start Fallback'] = 0

        comparison_data.append(data)

    comparison_df = pd.DataFrame(comparison_data)

    print_and_save("\n" + "-"*80)
    print_and_save("FULL COMPARISON")
    print_and_save("-"*80)
    print_and_save(comparison_df.to_string(index=False))

    # Calculate RMSE differences
    print_and_save("\n" + "="*80)
    print_and_save("RMSE COMPARISON: WITH COLD-START vs WARM-START")
    print_and_save("="*80)

    # Item-Based CF comparison
    if 'Item-Based CF (Warm-Start)' in results and 'Item-Based CF (With Cold-Start)' in results:
        item_warm = results['Item-Based CF (Warm-Start)']['rmse'].values[0]
        item_cold = results['Item-Based CF (With Cold-Start)']['rmse'].values[0]
        item_diff = item_cold - item_warm
        item_pct = (item_diff / item_warm) * 100

        print_and_save(f"\nItem-Based CF:")
        print_and_save(f"  Warm-Start:           {item_warm:.4f}")
        print_and_save(f"  With Cold-Start:      {item_cold:.4f}")
        print_and_save(f"  Difference:           {item_diff:+.4f} ({item_pct:+.2f}%)")
        if item_diff > 0:
            print_and_save(f"  → Cold-start increases error")
        else:
            print_and_save(f"  → Cold-start decreases error (unexpected!)")

    # User-Based CF comparison
    if 'User-Based CF (Warm-Start)' in results and 'User-Based CF (With Cold-Start)' in results:
        user_warm = results['User-Based CF (Warm-Start)']['rmse'].values[0]
        user_cold = results['User-Based CF (With Cold-Start)']['rmse'].values[0]
        user_diff = user_cold - user_warm
        user_pct = (user_diff / user_warm) * 100

        print_and_save(f"\nUser-Based CF:")
        print_and_save(f"  Warm-Start:           {user_warm:.4f}")
        print_and_save(f"  With Cold-Start:      {user_cold:.4f}")
        print_and_save(f"  Difference:           {user_diff:+.4f} ({user_pct:+.2f}%)")
        if user_diff > 0:
            print_and_save(f"  → Cold-start increases error")
        else:
            print_and_save(f"  → Cold-start decreases error (unexpected!)")

    # SVD comparison
    if 'SVD (Warm-Start)' in results and 'SVD (With Cold-Start)' in results:
        svd_warm = results['SVD (Warm-Start)']['rmse'].values[0]
        svd_cold = results['SVD (With Cold-Start)']['rmse'].values[0]
        svd_diff = svd_cold - svd_warm
        svd_pct = (svd_diff / svd_warm) * 100

        print_and_save(f"\nSVD:")
        print_and_save(f"  Warm-Start:           {svd_warm:.4f}")
        print_and_save(f"  With Cold-Start:      {svd_cold:.4f}")
        print_and_save(f"  Difference:           {svd_diff:+.4f} ({svd_pct:+.2f}%)")
        if svd_diff > 0:
            print_and_save(f"  → Cold-start increases error")
        else:
            print_and_save(f"  → Cold-start decreases error (unexpected!)")

    # Analysis
    print_and_save("\n" + "="*80)
    print_and_save("ANALYSIS")
    print_and_save("="*80)

    # Calculate average increase from available algorithms
    pct_changes = []
    if 'Item-Based CF (Warm-Start)' in results and 'Item-Based CF (With Cold-Start)' in results:
        pct_changes.append(item_pct)
    if 'User-Based CF (Warm-Start)' in results and 'User-Based CF (With Cold-Start)' in results:
        pct_changes.append(user_pct)
    if 'SVD (Warm-Start)' in results and 'SVD (With Cold-Start)' in results:
        pct_changes.append(svd_pct)

    if len(pct_changes) > 0:
        avg_increase = sum(pct_changes) / len(pct_changes)
        print_and_save(f"\nAverage RMSE increase with cold-start: {avg_increase:+.2f}%")
    else:
        avg_increase = 0
        print_and_save("\nNot enough data to calculate average RMSE increase")

    if avg_increase > 0:
        print_and_save("\n✓ Expected result: Cold-start increases error")
        print_and_save("\nReasons:")
        print_and_save("  1. Global mean fallback is less accurate than CF predictions")
        print_and_save("  2. Cold-start users/items have no collaborative signal")
        print_and_save("  3. Warm-start approach only evaluates on 'easier' predictions")
        print_and_save(f"\nProduction impact: Including cold-start adds ~{avg_increase:.1f}% RMSE")
    else:
        print_and_save("\n⚠️  Unexpected result: Cold-start DECREASES error")
        print_and_save("\nPossible reasons:")
        print_and_save("  1. Global mean may be closer to actual ratings than CF predictions")
        print_and_save("  2. Overfitting in CF models")
        print_and_save("  3. Different test set distribution")

    # Cold-start breakdown
    print_and_save("\n" + "="*80)
    print_and_save("COLD-START BREAKDOWN")
    print_and_save("="*80)

    for algo in ['Item-Based CF', 'User-Based CF', 'SVD']:
        if f'{algo} (With Cold-Start)' in results:
            df = results[f'{algo} (With Cold-Start)']
            test_samples = df['test_samples'].values[0]

            if 'cold_start_predictions' in df.columns:
                cold_start = df['cold_start_predictions'].values[0]
            elif 'cf_predictions' in df.columns:
                cold_start = test_samples - df['cf_predictions'].values[0]
            else:
                cold_start = 0

            cf_predictions = test_samples - cold_start

            print_and_save(f"\n{algo}:")
            print_and_save(f"  Total predictions:    {test_samples:,}")
            print_and_save(f"  CF predictions:       {cf_predictions:,} ({cf_predictions/test_samples*100:.2f}%)")
            print_and_save(f"  Cold-start fallback:  {cold_start:,} ({cold_start/test_samples*100:.2f}%)")

    # Speed comparison
    print_and_save("\n" + "="*80)
    print_and_save("SPEED COMPARISON")
    print_and_save("="*80)

    print_and_save("\nTraining Time:")
    for algo in ['Item-Based CF', 'User-Based CF', 'SVD']:
        if f'{algo} (Warm-Start)' in results and f'{algo} (With Cold-Start)' in results:
            time_warm = results[f'{algo} (Warm-Start)']['training_time_minutes'].values[0]
            time_cold = results[f'{algo} (With Cold-Start)']['training_time_minutes'].values[0]
            print_and_save(f"  {algo:20s} (Warm-Start):    {time_warm:6.2f} min")
            print_and_save(f"  {algo:20s} (Cold-Start):    {time_cold:6.2f} min")

    print_and_save("\nPrediction Speed:")
    for algo in ['Item-Based CF', 'User-Based CF', 'SVD']:
        if f'{algo} (Warm-Start)' in results and f'{algo} (With Cold-Start)' in results:
            speed_warm = results[f'{algo} (Warm-Start)']['prediction_time_ms'].values[0]
            speed_cold = results[f'{algo} (With Cold-Start)']['prediction_time_ms'].values[0]
            print_and_save(f"  {algo:20s} (Warm-Start):    {speed_warm:6.2f} ms/rating")
            print_and_save(f"  {algo:20s} (Cold-Start):    {speed_cold:6.2f} ms/rating")

    # Best algorithm
    print_and_save("\n" + "="*80)
    print_and_save("BEST ALGORITHM")
    print_and_save("="*80)

    # Find best for Warm-Start
    rmse_warm = {}
    if 'Item-Based CF (Warm-Start)' in results:
        rmse_warm['Item-Based CF'] = results['Item-Based CF (Warm-Start)']['rmse'].values[0]
    if 'User-Based CF (Warm-Start)' in results:
        rmse_warm['User-Based CF'] = results['User-Based CF (Warm-Start)']['rmse'].values[0]
    if 'SVD (Warm-Start)' in results:
        rmse_warm['SVD'] = results['SVD (Warm-Start)']['rmse'].values[0]

    # Find best for With Cold-Start
    rmse_cold = {}
    if 'Item-Based CF (With Cold-Start)' in results:
        rmse_cold['Item-Based CF'] = results['Item-Based CF (With Cold-Start)']['rmse'].values[0]
    if 'User-Based CF (With Cold-Start)' in results:
        rmse_cold['User-Based CF'] = results['User-Based CF (With Cold-Start)']['rmse'].values[0]
    if 'SVD (With Cold-Start)' in results:
        rmse_cold['SVD'] = results['SVD (With Cold-Start)']['rmse'].values[0]

    if rmse_warm and rmse_cold:
        best_warm = min(rmse_warm, key=rmse_warm.get)
        best_cold = min(rmse_cold, key=rmse_cold.get)

        print_and_save(f"\nBest on Warm-Start:           {best_warm} (RMSE: {rmse_warm[best_warm]:.4f})")
        print_and_save(f"Best on With Cold-Start:      {best_cold} (RMSE: {rmse_cold[best_cold]:.4f})")

        if best_warm == best_cold:
            print_and_save(f"\n✓ {best_warm} is consistently the best across both approaches")
        else:
            print_and_save(f"\n⚠️  Different winners: {best_warm} (Warm-Start) vs {best_cold} (Cold-Start)")
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
    print("WITH COLD-START VS WARM-START COMPARISON")
    print("="*80)

    results = load_results()

    if len(results) > 0:
        # Run comparison and capture output
        output_lines = compare_algorithms(results)

        # Save to text file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"cold_start_comparison_{timestamp}.txt"
        output_path = os.path.join(script_dir, output_filename)

        print("\n" + "="*80)
        print("SAVING RESULTS TO FILE")
        print("="*80)

        with open(output_path, 'w') as f:
            # Write header
            f.write("="*80 + "\n")
            f.write("ALGORITHM COMPARISON: WITH COLD-START VS WARM-START\n")
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
        print("   python run_with_cold_start.py")

    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
