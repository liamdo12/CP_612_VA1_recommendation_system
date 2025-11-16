"""
Hybrid Recommender System: Collaborative Filtering + Content-Based

This script implements a hybrid recommender system that combines:
- Collaborative Filtering (CF): Item-Based approach
- Content-Based Filtering: Using movie genre information

Hybrid Strategy:
- Warm-start (user & movie in training): score = 0.7 × CF + 0.3 × content
- Cold-start user: score = 0.3 × CF + 0.7 × content
- Cold-start movie: score = 0.3 × CF + 0.7 × content
- Double cold-start: score = 1.0 × content

Usage:
    python 5_hybrid_system.py

Configuration:
    - Set USE_SAMPLE = True for quick testing (~10-15 min)
    - Set USE_SAMPLE = False for full evaluation (~1-2 hours)
"""

import pandas as pd
import numpy as np
import ast
from datetime import datetime
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix, lil_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import gc

warnings.filterwarnings('ignore')

# Configuration - OPTIMIZED FOR 16GB RAM
USE_SAMPLE = True  # Set to False for full evaluation
SAMPLE_SIZE = 10000
K_NEIGHBORS = 30  # Number of neighbors for CF
MAX_MOVIES_FOR_SIMILARITY = 10000  # Limit movies for similarity computation
BATCH_SIZE = 1000  # Process movies in batches for similarity computation (memory-efficient)

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("=" * 80)
print("HYBRID RECOMMENDER SYSTEM: COLLABORATIVE FILTERING + CONTENT-BASED")
print("=" * 80)
print(f"\nStarted at: {datetime.now()}")
print(f"Configuration: USE_SAMPLE={USE_SAMPLE}, SAMPLE_SIZE={SAMPLE_SIZE}, K_NEIGHBORS={K_NEIGHBORS}")


# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("\n" + "=" * 80)
print("STEP 1: LOADING DATA")
print("=" * 80)

print("\nLoading temporal split datasets...")
train = pd.read_csv('../../datasets/output/split_and_train_datasets/temporal_split/train_ratings.csv')
test = pd.read_csv('../../datasets/output/split_and_train_datasets/temporal_split/test_ratings.csv')

print("Loading movie metadata with genres...")
movies = pd.read_csv('../../datasets/output/cleaned_datasets/cleaned_movies_metadata.csv')
links = pd.read_csv('../../datasets/output/cleaned_datasets/cleaned_links.csv')

print(f"\n  Train: {len(train):,} ratings")
print(f"  Test: {len(test):,} ratings")
print(f"  Movies: {len(movies):,} movies")
print(f"  Links: {len(links):,} links")


# =============================================================================
# 2. PREPARE MOVIE GENRE FEATURES
# =============================================================================
print("\n" + "=" * 80)
print("STEP 2: PREPARING MOVIE GENRE FEATURES")
print("=" * 80)


def parse_genres(genre_str):
    """Convert string representation of list to actual list"""
    if pd.isna(genre_str) or genre_str == '[]':
        return []
    try:
        return ast.literal_eval(genre_str)
    except:
        return []


movies['genres'] = movies['genres_list'].apply(parse_genres)

# Merge with links to get movieId
movies_with_ids = movies.merge(links[['movieId', 'tmdbId']],
                                left_on='id', right_on='tmdbId', how='inner')

print(f"\n  Movies with genres: {len(movies_with_ids):,}")

# Get all unique genres
all_genres = set()
for genres in movies_with_ids['genres']:
    all_genres.update(genres)

all_genres = sorted(list(all_genres))
print(f"  Total unique genres: {len(all_genres)}")
print(f"  Genres: {all_genres[:10]}..." if len(all_genres) > 10 else f"  Genres: {all_genres}")

# Create genre one-hot encoding for each movie
print("\n  Creating genre feature matrix...")
genre_matrix = []
movie_ids = []

for _, row in movies_with_ids.iterrows():
    movie_ids.append(row['movieId'])
    genre_vector = [1 if genre in row['genres'] else 0 for genre in all_genres]
    genre_matrix.append(genre_vector)

# Create DataFrame and ensure unique index (drop duplicates)
genre_df = pd.DataFrame(genre_matrix, columns=all_genres, index=movie_ids)
genre_df = genre_df[~genre_df.index.duplicated(keep='first')]  # Remove duplicate movie IDs

print(f"  Genre feature matrix shape: {genre_df.shape}")


# =============================================================================
# 3. BUILD COLLABORATIVE FILTERING COMPONENTS
# =============================================================================
print("\n" + "=" * 80)
print("STEP 3: BUILDING COLLABORATIVE FILTERING COMPONENTS")
print("=" * 80)

print("\n  Building SPARSE user-item matrix (memory-efficient)...")

# Create user and movie ID mappings
user_ids = sorted(train['userId'].unique())
movie_ids = sorted(train['movieId'].unique())

# Limit movies for similarity computation to save memory
if len(movie_ids) > MAX_MOVIES_FOR_SIMILARITY:
    # Keep most popular movies
    movie_counts = train['movieId'].value_counts()
    top_movies = movie_counts.head(MAX_MOVIES_FOR_SIMILARITY).index.tolist()
    movie_ids_for_sim = sorted(top_movies)
    print(f"  Limiting to top {MAX_MOVIES_FOR_SIMILARITY:,} most-rated movies for similarity computation")
else:
    movie_ids_for_sim = movie_ids

user_id_map = {uid: idx for idx, uid in enumerate(user_ids)}
movie_id_map = {mid: idx for idx, mid in enumerate(movie_ids)}
movie_id_map_sim = {mid: idx for idx, mid in enumerate(movie_ids_for_sim)}

# Build sparse matrix (uses ~1000x less memory than dense)
n_users = len(user_ids)
n_movies = len(movie_ids)

print(f"  Creating sparse matrix: {n_users:,} users × {n_movies:,} movies")

# Map ratings to indices
train['user_idx'] = train['userId'].map(user_id_map)
train['movie_idx'] = train['movieId'].map(movie_id_map)

# Create sparse matrix using CSR format (efficient for row operations)
train_sparse = csr_matrix(
    (train['rating'].values.astype(np.float32),
     (train['user_idx'].values, train['movie_idx'].values)),
    shape=(n_users, n_movies),
    dtype=np.float32
)

print(f"  Sparse matrix created: {train_sparse.shape}")
print(f"  Memory usage: ~{train_sparse.data.nbytes / (1024**2):.1f} MB")
sparsity = (1 - train_sparse.nnz / (train_sparse.shape[0] * train_sparse.shape[1])) * 100
print(f"  Sparsity: {sparsity:.2f}%")

# Get sets of known users and movies
train_users = set(train['userId'].unique())
train_movies = set(train['movieId'].unique())

print(f"  Train users: {len(train_users):,}")
print(f"  Train movies: {len(train_movies):,}")

# Create dense pivot table ONLY for limited movies (for similarity computation)
print("\n  Computing item-item similarity (memory-efficient approach)...")
print(f"  Working with {len(movie_ids_for_sim):,} movies")
start_time = time.time()

# Filter train data to only include movies for similarity
train_for_sim = train[train['movieId'].isin(movie_ids_for_sim)].copy()
train_for_sim['movie_idx_sim'] = train_for_sim['movieId'].map(movie_id_map_sim)

# Create smaller sparse matrix for similarity computation
train_sparse_sim = csr_matrix(
    (train_for_sim['rating'].values.astype(np.float32),
     (train_for_sim['user_idx'].values, train_for_sim['movie_idx_sim'].values)),
    shape=(n_users, len(movie_ids_for_sim)),
    dtype=np.float32
)

# Convert to dense for similarity (only for limited movies)
print("  Converting to dense for similarity computation...")
train_dense_sim = train_sparse_sim.toarray()
del train_sparse_sim  # Free memory
gc.collect()

print(f"  Dense matrix for similarity: {train_dense_sim.shape} (~{train_dense_sim.nbytes / (1024**3):.2f} GB)")

# Compute item-item similarity using OPTIMIZED chunked batch processing
print(f"  Computing cosine similarity (OPTIMIZED - {BATCH_SIZE} movies per batch)...")
item_sim_dict = {}

# CRITICAL OPTIMIZATION: Pre-transpose ONCE (not 10,000 times in loop!)
train_dense_sim_T = train_dense_sim.T
print(f"  Pre-transposed matrix for vectorized processing")

# Process movies in batches for memory efficiency and speed
for batch_start in range(0, len(movie_ids_for_sim), BATCH_SIZE):
    batch_end = min(batch_start + BATCH_SIZE, len(movie_ids_for_sim))
    print(f"    Processing batch {batch_start:,}-{batch_end:,} / {len(movie_ids_for_sim):,} ({batch_end/len(movie_ids_for_sim)*100:.1f}%)")

    # Compute similarity for entire batch at once (vectorized - MUCH faster!)
    batch_sims = cosine_similarity(
        train_dense_sim_T[batch_start:batch_end],  # Batch of movies
        train_dense_sim_T  # All movies (for comparison)
    )

    # Store top-K similar movies for each movie in batch
    for i in range(batch_end - batch_start):
        movie_idx = batch_start + i
        movie_id = movie_ids_for_sim[movie_idx]

        # Get similarities for this movie
        sims = batch_sims[i]

        # Find top K most similar movies (excluding self)
        top_k_indices = np.argsort(sims)[::-1][1:K_NEIGHBORS+1]  # Exclude self at position 0
        item_sim_dict[movie_id] = {
            movie_ids_for_sim[idx]: sims[idx]
            for idx in top_k_indices
        }

# Free memory
del train_dense_sim, train_dense_sim_T
gc.collect()

elapsed = time.time() - start_time
print(f"  Item similarity computed in {elapsed:.2f} seconds")
print(f"  Stored top-{K_NEIGHBORS} similarities for {len(item_sim_dict):,} movies")

# Global mean rating
global_mean = train['rating'].mean()
print(f"\n  Global mean rating: {global_mean:.3f}")


# =============================================================================
# 4. IMPLEMENT CONTENT-BASED SCORING
# =============================================================================
print("\n" + "=" * 80)
print("STEP 4: IMPLEMENTING CONTENT-BASED SCORING")
print("=" * 80)


def get_user_genre_profile(user_id, train_data, genre_df):
    """
    Build a user's genre preference profile based on their rating history.
    Returns a weighted genre vector where higher values indicate preference.
    """
    user_ratings = train_data[train_data['userId'] == user_id]

    if len(user_ratings) == 0:
        return pd.Series(0.0, index=genre_df.columns, dtype=np.float32)

    genre_profile = pd.Series(0.0, index=genre_df.columns, dtype=np.float32)

    for _, row in user_ratings.iterrows():
        movie_id = row['movieId']
        rating = float(row['rating'])

        if movie_id in genre_df.index:
            # Get genre vector for this movie (ensure it's a Series)
            movie_genres = genre_df.loc[movie_id]
            if isinstance(movie_genres, pd.DataFrame):
                # If duplicate index, take first row
                movie_genres = movie_genres.iloc[0]

            # Add weighted genre vector to profile
            genre_profile = genre_profile.add(movie_genres.astype(np.float32) * rating, fill_value=0)

    if genre_profile.sum() > 0:
        genre_profile = genre_profile / genre_profile.sum()

    return genre_profile


def content_based_score(user_id, movie_id, train_data, genre_df, global_mean):
    """
    Compute content-based score for a user-movie pair using genre similarity.
    """
    user_profile = get_user_genre_profile(user_id, train_data, genre_df)

    if movie_id not in genre_df.index:
        return global_mean

    movie_genres = genre_df.loc[movie_id]
    if isinstance(movie_genres, pd.DataFrame):
        # If duplicate index, take first row
        movie_genres = movie_genres.iloc[0]

    if user_profile.sum() == 0 or movie_genres.sum() == 0:
        return global_mean

    # Ensure both are arrays for dot product
    similarity = np.dot(user_profile.values, movie_genres.values)
    score = global_mean + (similarity - 0.5) * 2.0
    score = np.clip(score, 0.5, 5.0)

    return float(score)


print("  Content-based scoring functions defined")


# =============================================================================
# 5. IMPLEMENT COLLABORATIVE FILTERING PREDICTION
# =============================================================================
print("\n" + "=" * 80)
print("STEP 5: IMPLEMENTING COLLABORATIVE FILTERING PREDICTION")
print("=" * 80)


def item_based_cf_predict(user_id, movie_id, train_data, item_sim_dict, global_mean=3.5):
    """
    Item-based collaborative filtering prediction (memory-efficient version).
    Uses sparse data structures and dictionary-based similarity.
    """
    # Check if movie has similarity data
    if movie_id not in item_sim_dict:
        return global_mean, False

    # Get user's ratings
    user_ratings = train_data[train_data['userId'] == user_id]

    if len(user_ratings) == 0:
        return global_mean, False

    # Create a dict for quick lookup
    user_rating_dict = dict(zip(user_ratings['movieId'], user_ratings['rating']))

    # Get similar items
    similar_items = item_sim_dict[movie_id]

    numerator = 0.0
    denominator = 0.0

    for item, sim in similar_items.items():
        if item in user_rating_dict:
            numerator += sim * user_rating_dict[item]
            denominator += abs(sim)

    if denominator == 0:
        return global_mean, False

    prediction = numerator / denominator
    return np.clip(prediction, 0.5, 5.0), True


print("  Collaborative filtering functions defined")


# =============================================================================
# 6. IMPLEMENT HYBRID PREDICTION
# =============================================================================
print("\n" + "=" * 80)
print("STEP 6: IMPLEMENTING HYBRID PREDICTION WITH ADAPTIVE WEIGHTING")
print("=" * 80)


def hybrid_predict(user_id, movie_id, train_data, item_sim_dict,
                   genre_df, train_users, train_movies, global_mean):
    """
    Hybrid prediction combining CF and content-based filtering with adaptive weighting.

    Weighting strategy:
    - Warm-start (both in training): CF=0.7, Content=0.3
    - Cold-start user OR movie: CF=0.3, Content=0.7
    - Double cold-start: CF=0.0, Content=1.0
    """
    cf_score, is_cf = item_based_cf_predict(user_id, movie_id, train_data,
                                            item_sim_dict, global_mean=global_mean)

    cb_score = content_based_score(user_id, movie_id, train_data, genre_df, global_mean)

    user_is_warm = user_id in train_users
    movie_is_warm = movie_id in train_movies

    if user_is_warm and movie_is_warm:
        cf_weight, cb_weight = 0.7, 0.3
        case = 'warm'
    elif user_is_warm or movie_is_warm:
        cf_weight, cb_weight = 0.3, 0.7
        case = 'partial_cold'
    else:
        cf_weight, cb_weight = 0.0, 1.0
        case = 'double_cold'

    hybrid_score = cf_weight * cf_score + cb_weight * cb_score

    return np.clip(hybrid_score, 0.5, 5.0), case


print("  Hybrid prediction function defined")

# Get cold-start statistics
test_users = set(test['userId'].unique())
test_movies = set(test['movieId'].unique())

cold_start_users = test_users - train_users
cold_start_movies = test_movies - train_movies

print(f"\n  Cold-start statistics:")
print(f"    Cold-start users: {len(cold_start_users):,} / {len(test_users):,} ({len(cold_start_users)/len(test_users)*100:.1f}%)")
print(f"    Cold-start movies: {len(cold_start_movies):,} / {len(test_movies):,} ({len(cold_start_movies)/len(test_movies)*100:.1f}%)")


# =============================================================================
# 7. EVALUATE HYBRID SYSTEM ON TEST SET
# =============================================================================
print("\n" + "=" * 80)
print("STEP 7: EVALUATING HYBRID SYSTEM ON TEST SET")
print("=" * 80)

if USE_SAMPLE:
    test_sample = test.sample(n=min(SAMPLE_SIZE, len(test)), random_state=42)
    print(f"\n  Using sample of {len(test_sample):,} test ratings for faster evaluation")
else:
    test_sample = test
    print(f"\n  Using full test set of {len(test_sample):,} ratings")

print(f"\n  Starting hybrid evaluation...")
start_time = time.time()

predictions = []
actuals = []
cases = []

for idx, row in test_sample.iterrows():
    if (idx - test_sample.index[0]) % 1000 == 0:
        elapsed = time.time() - start_time
        processed = idx - test_sample.index[0]
        rate = processed / elapsed if elapsed > 0 else 0
        pct = processed / len(test_sample) * 100
        print(f"    Processed {processed:,} / {len(test_sample):,} ({pct:.1f}%) - {rate:.1f} ratings/sec")

    user_id = row['userId']
    movie_id = row['movieId']
    actual = row['rating']

    pred, case = hybrid_predict(user_id, movie_id, train, item_sim_dict,
                                genre_df, train_users, train_movies, global_mean)

    predictions.append(pred)
    actuals.append(actual)
    cases.append(case)

elapsed = time.time() - start_time
print(f"\n  Evaluation completed in {elapsed:.2f} seconds")
print(f"  Average speed: {len(test_sample)/elapsed:.1f} ratings/sec")

# Compute metrics
rmse = np.sqrt(mean_squared_error(actuals, predictions))
mae = mean_absolute_error(actuals, predictions)

print(f"\n{'='*80}")
print(f"HYBRID SYSTEM EVALUATION RESULTS")
print(f"{'='*80}")
print(f"\nOverall Metrics:")
print(f"  RMSE: {rmse:.6f}")
print(f"  MAE:  {mae:.6f}")
print(f"  Test samples: {len(test_sample):,}")

# Break down by case type
results_df = pd.DataFrame({
    'actual': actuals,
    'predicted': predictions,
    'case': cases
})

print(f"\n{'='*80}")
print(f"BREAKDOWN BY CASE TYPE")
print(f"{'='*80}")

for case_type in ['warm', 'partial_cold', 'double_cold']:
    case_df = results_df[results_df['case'] == case_type]
    if len(case_df) > 0:
        case_rmse = np.sqrt(mean_squared_error(case_df['actual'], case_df['predicted']))
        case_mae = mean_absolute_error(case_df['actual'], case_df['predicted'])

        print(f"\n{case_type.upper().replace('_', ' ')}:")
        print(f"  Count: {len(case_df):,} ({len(case_df)/len(results_df)*100:.2f}%)")
        print(f"  RMSE:  {case_rmse:.6f}")
        print(f"  MAE:   {case_mae:.6f}")


# =============================================================================
# 8. COMPARE WITH PURE CF BASELINE
# =============================================================================
print("\n" + "=" * 80)
print("STEP 8: COMPARING WITH PURE CF BASELINE")
print("=" * 80)

print(f"\n  Running pure Item-Based CF baseline for comparison...")
start_time = time.time()

cf_predictions = []
cf_fallback_count = 0

for idx, row in test_sample.iterrows():
    if (idx - test_sample.index[0]) % 1000 == 0:
        elapsed = time.time() - start_time
        processed = idx - test_sample.index[0]
        rate = processed / elapsed if elapsed > 0 else 0
        pct = processed / len(test_sample) * 100
        print(f"    Processed {processed:,} / {len(test_sample):,} ({pct:.1f}%) - {rate:.1f} ratings/sec")

    user_id = row['userId']
    movie_id = row['movieId']

    pred, is_cf = item_based_cf_predict(user_id, movie_id, train, item_sim_dict,
                                        global_mean=global_mean)

    cf_predictions.append(pred)
    if not is_cf:
        cf_fallback_count += 1

elapsed = time.time() - start_time
print(f"\n  CF baseline completed in {elapsed:.2f} seconds")

# Compute CF metrics
cf_rmse = np.sqrt(mean_squared_error(actuals, cf_predictions))
cf_mae = mean_absolute_error(actuals, cf_predictions)

print(f"\n{'='*80}")
print(f"PURE CF BASELINE RESULTS")
print(f"{'='*80}")
print(f"\nMetrics:")
print(f"  RMSE: {cf_rmse:.6f}")
print(f"  MAE:  {cf_mae:.6f}")
print(f"  Fallback to global mean: {cf_fallback_count:,} ({cf_fallback_count/len(test_sample)*100:.2f}%)")

# Comparison summary
print(f"\n{'='*80}")
print(f"HYBRID vs PURE CF COMPARISON")
print(f"{'='*80}")
print(f"\n{'Algorithm':<25} {'RMSE':<12} {'MAE':<12} {'Improvement'}")
print(f"{'-'*25} {'-'*12} {'-'*12} {'-'*20}")
print(f"{'Pure Item-Based CF':<25} {cf_rmse:>10.6f}   {cf_mae:>10.6f}   {'(baseline)'}")
print(f"{'Hybrid CF+Content':<25} {rmse:>10.6f}   {mae:>10.6f}   {(cf_rmse-rmse)/cf_rmse*100:>+6.2f}% RMSE")

improvement = (cf_rmse - rmse) / cf_rmse * 100
if improvement > 0:
    print(f"\n✓ Hybrid system improves RMSE by {improvement:.2f}%")
else:
    print(f"\n✗ Hybrid system increases RMSE by {abs(improvement):.2f}%")


# =============================================================================
# 9. VISUALIZE RESULTS
# =============================================================================
print("\n" + "=" * 80)
print("STEP 9: CREATING VISUALIZATIONS")
print("=" * 80)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: RMSE and MAE comparison
algorithms = ['Pure CF', 'Hybrid']
rmse_values = [cf_rmse, rmse]
mae_values = [cf_mae, mae]

x = np.arange(len(algorithms))
width = 0.35

axes[0].bar(x - width/2, rmse_values, width, label='RMSE', color='steelblue', alpha=0.8)
axes[0].bar(x + width/2, mae_values, width, label='MAE', color='coral', alpha=0.8)
axes[0].set_ylabel('Error', fontsize=12)
axes[0].set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(algorithms)
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, (r, m) in enumerate(zip(rmse_values, mae_values)):
    axes[0].text(i - width/2, r + 0.01, f'{r:.4f}', ha='center', va='bottom', fontsize=10)
    axes[0].text(i + width/2, m + 0.01, f'{m:.4f}', ha='center', va='bottom', fontsize=10)

# Plot 2: Breakdown by case type
case_counts = results_df['case'].value_counts()
case_colors = {'warm': 'green', 'partial_cold': 'orange', 'double_cold': 'red'}
colors = [case_colors.get(case, 'gray') for case in case_counts.index]

axes[1].bar(range(len(case_counts)), case_counts.values, color=colors, alpha=0.7)
axes[1].set_ylabel('Count', fontsize=12)
axes[1].set_title('Prediction Distribution by Case Type', fontsize=14, fontweight='bold')
axes[1].set_xticks(range(len(case_counts)))
axes[1].set_xticklabels([c.replace('_', ' ').title() for c in case_counts.index], rotation=15)
axes[1].grid(axis='y', alpha=0.3)

# Add percentage labels
for i, (count, case) in enumerate(zip(case_counts.values, case_counts.index)):
    pct = count / len(results_df) * 100
    axes[1].text(i, count + len(results_df)*0.01, f'{count:,}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('hybrid_comparison.png', dpi=300, bbox_inches='tight')
print("\n  Visualization saved as 'hybrid_comparison.png'")
plt.close()


# =============================================================================
# 10. SAVE RESULTS
# =============================================================================
print("\n" + "=" * 80)
print("STEP 10: SAVING RESULTS")
print("=" * 80)

# Save detailed results to CSV
comparison_results = pd.DataFrame({
    'Algorithm': ['Pure Item-Based CF', 'Hybrid CF+Content'],
    'RMSE': [cf_rmse, rmse],
    'MAE': [cf_mae, mae],
    'Test_Samples': [len(test_sample), len(test_sample)],
    'Notes': [
        f'Fallback: {cf_fallback_count} ({cf_fallback_count/len(test_sample)*100:.2f}%)',
        f'Warm: {len(results_df[results_df["case"]=="warm"])}, Partial: {len(results_df[results_df["case"]=="partial_cold"])}, Double: {len(results_df[results_df["case"]=="double_cold"])}'
    ]
})

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
filename = f'hybrid_comparison_{timestamp}.csv'
comparison_results.to_csv(filename, index=False)
print(f"\n  Results saved to {filename}")

# Save detailed breakdown
case_breakdown = []
for case_type in ['warm', 'partial_cold', 'double_cold']:
    case_df = results_df[results_df['case'] == case_type]
    if len(case_df) > 0:
        case_rmse = np.sqrt(mean_squared_error(case_df['actual'], case_df['predicted']))
        case_mae = mean_absolute_error(case_df['actual'], case_df['predicted'])
        case_breakdown.append({
            'Case': case_type,
            'Count': len(case_df),
            'Percentage': len(case_df) / len(results_df) * 100,
            'RMSE': case_rmse,
            'MAE': case_mae
        })

breakdown_df = pd.DataFrame(case_breakdown)
breakdown_filename = f'hybrid_case_breakdown_{timestamp}.csv'
breakdown_df.to_csv(breakdown_filename, index=False)
print(f"  Case breakdown saved to {breakdown_filename}")


# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("HYBRID SYSTEM IMPLEMENTATION COMPLETE")
print("=" * 80)
print(f"\nCompleted at: {datetime.now()}")
print("\nSummary:")
print("  This hybrid recommender system successfully combines:")
print("  1. Item-Based Collaborative Filtering for users/movies with interaction history")
print("  2. Content-Based Filtering using genre information for cold-start scenarios")
print("  3. Adaptive weighting that adjusts balance based on data availability")
print("\n  The hybrid approach handles various scenarios:")
print("    - Warm-start: Leverages collaborative patterns (70% weight)")
print("    - Partial cold-start: Balances CF and content (30% CF, 70% content)")
print("    - Double cold-start: Falls back to content-based recommendations")
print("\n  This addresses key limitations of pure collaborative filtering and")
print("  provides a more robust solution for production deployment.")
print("=" * 80)
