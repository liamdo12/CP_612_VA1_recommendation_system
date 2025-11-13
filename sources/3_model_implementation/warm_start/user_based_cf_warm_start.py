"""
User-Based Collaborative Filtering (Temporal Split - Warm-Start Evaluation)

This script uses temporal split data but filters the test set to achieve ~99% coverage
by only evaluating on user-movie pairs where both exist in the training set.

Purpose: Compare algorithm performance on temporal data WITHOUT cold-start complications.

Expected runtime: 30-60 minutes
"""

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm
import time
import gc
import psutil
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("USER-BASED CF (WARM-START EVALUATION)")
print("="*60)
print("\nLibraries imported successfully!")

# ============================================================================
# CONFIGURATION
# ============================================================================
SAMPLE_SIZE = 100000  # 100K for fair comparison with 80-20 split
K_NEIGHBORS = 50
MAX_CANDIDATES = 200
BATCH_SIZE = 1000
GC_FREQUENCY = 5

print(f"\nConfiguration:")
print(f"  Sample size: {SAMPLE_SIZE:,}")
print(f"  k-neighbors: {K_NEIGHBORS}")
print(f"  Max candidates: {MAX_CANDIDATES}")

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n" + "="*60)
print("LOADING TEMPORAL SPLIT DATA")
print("="*60)

import os

# Get project root (3 levels up from this script)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))

train_path = os.path.join(project_root, 'datasets/output/split_and_train_datasets/temporal_split/train_ratings.csv')
test_path = os.path.join(project_root, 'datasets/output/split_and_train_datasets/temporal_split/test_ratings.csv')

print("\nLoading training data...")
train = pd.read_csv(train_path)
print(f"Train shape: {train.shape}")
print(f"Train ratings: {len(train):,}")

print("\nLoading test data...")
test = pd.read_csv(test_path)
print(f"Test shape: {test.shape}")
print(f"Test ratings: {len(test):,}")

# Dataset statistics
print("\n" + "="*60)
print("DATASET STATISTICS")
print("="*60)

print("\nTraining Set:")
print(f"  Unique users: {train['userId'].nunique():,}")
print(f"  Unique movies: {train['movieId'].nunique():,}")
print(f"  Total ratings: {len(train):,}")
print(f"  Mean rating: {train['rating'].mean():.2f}")

print("\nTest Set (BEFORE filtering):")
print(f"  Unique users: {test['userId'].nunique():,}")
print(f"  Unique movies: {test['movieId'].nunique():,}")
print(f"  Total ratings: {len(test):,}")
print(f"  Mean rating: {test['rating'].mean():.2f}")

# ============================================================================
# CREATE ID MAPPINGS
# ============================================================================
print("\n" + "="*60)
print("CREATING ID MAPPINGS")
print("="*60)

unique_users = train['userId'].unique()
unique_movies = train['movieId'].unique()

user_id_map = {user_id: idx for idx, user_id in enumerate(unique_users)}
movie_id_map = {movie_id: idx for idx, movie_id in enumerate(unique_movies)}

idx_to_user = {idx: user_id for user_id, idx in user_id_map.items()}
idx_to_movie = {idx: movie_id for movie_id, idx in movie_id_map.items()}

print(f"Training users: {len(user_id_map):,}")
print(f"Training movies: {len(movie_id_map):,}")

# ============================================================================
# CALCULATE USER MEANS (FOR MEAN-CENTERING)
# ============================================================================
print("\n" + "="*60)
print("CALCULATING USER MEANS")
print("="*60)

user_mean_ratings = train.groupby('userId')['rating'].mean().to_dict()
global_mean_rating = train['rating'].mean()

print(f"Global mean rating: {global_mean_rating:.3f}")
print(f"User mean rating range: [{min(user_mean_ratings.values()):.2f}, {max(user_mean_ratings.values()):.2f}]")

# Apply mean-centering to training data
print("\nApplying mean-centering to training data...")
train['rating_centered'] = train.apply(
    lambda x: x['rating'] - user_mean_ratings.get(x['userId'], global_mean_rating),
    axis=1
)

print(f"Centered rating range: [{train['rating_centered'].min():.2f}, {train['rating_centered'].max():.2f}]")
print(f"Centered rating mean: {train['rating_centered'].mean():.4f} (should be ~0)")

# ============================================================================
# MAP TO MATRIX INDICES AND FILTER TEST SET
# ============================================================================
print("\n" + "="*60)
print("MAPPING TO MATRIX INDICES")
print("="*60)

train['user_idx'] = train['userId'].map(user_id_map)
train['movie_idx'] = train['movieId'].map(movie_id_map)

test['user_idx'] = test['userId'].map(user_id_map)
test['movie_idx'] = test['movieId'].map(movie_id_map)

# CRITICAL: Filter test set to only user-movie pairs that exist in training
print("\n" + "="*60)
print("FILTERING TEST SET FOR HIGH COVERAGE")
print("="*60)

test_before_filter = len(test)
testable = test.dropna(subset=['user_idx', 'movie_idx']).copy()
test_after_filter = len(testable)

print(f"\nTest ratings before filter: {test_before_filter:,}")
print(f"Test ratings after filter: {test_after_filter:,}")
print(f"Removed (cold-start): {test_before_filter - test_after_filter:,}")
print(f"Coverage achieved: {test_after_filter/test_before_filter*100:.2f}%")

# Store for later use
true_coverage = test_after_filter / test_before_filter * 100
total_testable = test_after_filter

print(f"\n✓ Test set filtered to {total_testable:,} testable ratings")

# ============================================================================
# CREATE SPARSE MATRICES
# ============================================================================
print("\n" + "="*60)
print("CREATING SPARSE MATRICES")
print("="*60)

start_time = time.time()

# CSR format for row access (users)
user_item_matrix_csr = csr_matrix(
    (train['rating_centered'].values,
     (train['user_idx'].values, train['movie_idx'].values)),
    shape=(len(user_id_map), len(movie_id_map))
)

# CSC format for column access (movies) - MUCH FASTER for getting movie columns
print("Converting to CSC format for fast movie access...")
user_item_matrix_csc = user_item_matrix_csr.tocsc()

elapsed = time.time() - start_time

print(f"\nMatrices created in {elapsed:.2f} seconds")
print(f"Shape: {user_item_matrix_csr.shape}")
print(f"Memory (CSR): {user_item_matrix_csr.data.nbytes / (1024**2):.2f} MB")
print(f"Memory (CSC): {user_item_matrix_csc.data.nbytes / (1024**2):.2f} MB")

# ============================================================================
# PRE-COMPUTE USER NORMS
# ============================================================================
print("\n" + "="*60)
print("PRE-COMPUTING USER NORMS")
print("="*60)

n_users = user_item_matrix_csr.shape[0]
batch_size = 10000
user_norms = np.zeros(n_users)

print(f"Processing {n_users:,} users in batches of {batch_size:,}...")

for batch_start in tqdm(range(0, n_users, batch_size), desc="Computing norms"):
    batch_end = min(batch_start + batch_size, n_users)
    batch_matrix = user_item_matrix_csr[batch_start:batch_end]
    batch_norms = np.sqrt(batch_matrix.multiply(batch_matrix).sum(axis=1).A1)
    user_norms[batch_start:batch_end] = batch_norms
    del batch_matrix, batch_norms
    if (batch_start // batch_size) % 5 == 0:
        gc.collect()

gc.collect()

print(f"\n✓ Pre-computed {len(user_norms):,} user norms")
print(f"  Memory: {user_norms.nbytes / (1024**2):.2f} MB")

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================

def predict_rating(user_idx, movie_idx, k=K_NEIGHBORS, max_candidates=MAX_CANDIDATES):
    """
    User-Based CF prediction with on-demand similarity computation.

    Parameters:
    - user_idx: Index of user
    - movie_idx: Index of movie
    - k: Number of similar users to consider
    - max_candidates: Max users to consider per prediction

    Returns:
    - predicted_rating: Float [0.5, 5.0]
    """
    user_idx = int(user_idx)
    movie_idx = int(movie_idx)

    # Get users who rated this movie (using CSC for efficient column access)
    movie_col = user_item_matrix_csc[:, movie_idx]
    users_who_rated = movie_col.nonzero()[0]

    if len(users_who_rated) == 0:
        user_id = idx_to_user[user_idx]
        return user_mean_ratings.get(user_id, global_mean_rating)

    # Remove target user from candidates
    users_who_rated = users_who_rated[users_who_rated != user_idx]

    if len(users_who_rated) == 0:
        user_id = idx_to_user[user_idx]
        return user_mean_ratings.get(user_id, global_mean_rating)

    # Limit candidates for efficiency
    if len(users_who_rated) > max_candidates:
        users_who_rated = np.random.choice(users_who_rated, max_candidates, replace=False)

    # Get target user's rating vector
    target_user_row = user_item_matrix_csr[user_idx]
    target_norm = user_norms[user_idx]

    if target_norm == 0:
        user_id = idx_to_user[user_idx]
        return user_mean_ratings.get(user_id, global_mean_rating)

    # Compute similarities on-demand
    similarities = []
    ratings_centered = []

    for other_user_idx in users_who_rated:
        other_norm = user_norms[other_user_idx]
        if other_norm == 0:
            continue

        dot_product = target_user_row.dot(user_item_matrix_csr[other_user_idx].T).toarray()[0, 0]
        similarity = dot_product / (target_norm * other_norm)
        rating_centered = movie_col[other_user_idx, 0]

        similarities.append(similarity)
        ratings_centered.append(rating_centered)

    if len(similarities) == 0:
        user_id = idx_to_user[user_idx]
        return user_mean_ratings.get(user_id, global_mean_rating)

    similarities = np.array(similarities)
    ratings_centered = np.array(ratings_centered)

    # Select top-k most similar users
    if len(similarities) > k:
        top_k_indices = np.argpartition(similarities, -k)[-k:]
        top_k_indices = top_k_indices[np.argsort(similarities[top_k_indices])[::-1]]
        top_k_sims = similarities[top_k_indices]
        top_k_ratings = ratings_centered[top_k_indices]
    else:
        top_k_sims = similarities
        top_k_ratings = ratings_centered

    # Filter out near-zero similarities
    valid = np.abs(top_k_sims) > 1e-6
    top_k_sims = top_k_sims[valid]
    top_k_ratings = top_k_ratings[valid]

    if len(top_k_sims) == 0:
        user_id = idx_to_user[user_idx]
        return user_mean_ratings.get(user_id, global_mean_rating)

    # Compute weighted average
    weighted_sum = np.sum(top_k_sims * top_k_ratings)
    sum_of_weights = np.sum(np.abs(top_k_sims))

    if sum_of_weights == 0:
        user_id = idx_to_user[user_idx]
        return user_mean_ratings.get(user_id, global_mean_rating)

    predicted_centered = weighted_sum / sum_of_weights

    # Denormalize: add back user's mean rating
    user_id = idx_to_user[user_idx]
    user_mean = user_mean_ratings.get(user_id, global_mean_rating)
    predicted = predicted_centered + user_mean

    return np.clip(predicted, 0.5, 5.0)

print("\n✓ Prediction function defined!")

# ============================================================================
# GENERATE PREDICTIONS
# ============================================================================
print("\n" + "="*60)
print("GENERATING PREDICTIONS")
print("="*60)

# Sample from testable ratings
if SAMPLE_SIZE and SAMPLE_SIZE < len(testable):
    print(f"Sampling {SAMPLE_SIZE:,} from {len(testable):,} testable ratings")
    test_sample = testable.sample(SAMPLE_SIZE, random_state=42)
else:
    print(f"Using all {len(testable):,} testable ratings")
    test_sample = testable.copy()

print(f"\nGenerating predictions for {len(test_sample):,} ratings...")
print(f"Estimated time: 30-60 minutes")

mem_before = psutil.Process().memory_info().rss / (1024**3)
print(f"Memory before: {mem_before:.2f} GB\n")

start_time = time.time()
predictions = []
n_batches = (len(test_sample) + BATCH_SIZE - 1) // BATCH_SIZE
test_rows = test_sample[['user_idx', 'movie_idx']].values

for batch_idx in tqdm(range(n_batches), desc="Processing batches"):
    batch_start = batch_idx * BATCH_SIZE
    batch_end = min((batch_idx + 1) * BATCH_SIZE, len(test_rows))

    batch_predictions = []
    for i in range(batch_start, batch_end):
        user_idx, movie_idx = test_rows[i]
        pred = predict_rating(user_idx, movie_idx, k=K_NEIGHBORS, max_candidates=MAX_CANDIDATES)
        batch_predictions.append(pred)

    predictions.extend(batch_predictions)

    if batch_idx % GC_FREQUENCY == 0:
        gc.collect()

gc.collect()
elapsed = time.time() - start_time
mem_after = psutil.Process().memory_info().rss / (1024**3)

print(f"\n✓ Predictions completed in {elapsed/60:.2f} minutes")
print(f"  Average: {elapsed/len(test_sample)*1000:.2f} ms per rating")
print(f"  Memory: {mem_after:.2f} GB")

# ============================================================================
# EVALUATION
# ============================================================================
print("\n" + "="*60)
print("EVALUATION")
print("="*60)

test_sample = test_sample.copy()
test_sample['predicted_rating'] = predictions

actual = test_sample['rating'].values
predicted = test_sample['predicted_rating'].values

rmse = np.sqrt(mean_squared_error(actual, predicted))
mae = mean_absolute_error(actual, predicted)

print("\n" + "="*60)
print("USER-BASED CF RESULTS (WARM-START EVALUATION)")
print("="*60)
print(f"\nAlgorithm: User-Based CF")
print(f"Split: Temporal (filtered for known user-movie pairs)")
print(f"Similarity: Cosine on centered ratings")
print(f"k-neighbors: {K_NEIGHBORS}")
print(f"Max candidates: {MAX_CANDIDATES}")

print(f"\nPerformance:")
print(f"  RMSE: {rmse:.4f}")
print(f"  MAE:  {mae:.4f}")

print(f"\nTiming:")
print(f"  Training: {elapsed/60:.2f} minutes")
print(f"  Per rating: {elapsed/len(test_sample)*1000:.2f} ms")

print(f"\nCoverage:")
print(f"  Coverage: {true_coverage:.2f}%")
print(f"  Test samples: {len(test_sample):,}")
print(f"  Total testable: {total_testable:,}")
print("="*60)

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n" + "="*60)
print("SAVING RESULTS")
print("="*60)

results = {
    'algorithm': 'User-Based CF (Warm-Start)',
    'split_strategy': 'temporal (filtered)',
    'similarity_metric': 'Cosine (centered)',
    'normalization': 'Mean-centering per user',
    'k_neighbors': K_NEIGHBORS,
    'max_candidates': MAX_CANDIDATES,
    'rmse': rmse,
    'mae': mae,
    'coverage': true_coverage,
    'training_time_minutes': elapsed/60,
    'prediction_time_ms': elapsed/len(test_sample)*1000,
    'test_samples': len(test_sample),
    'total_testable': total_testable
}

results_df = pd.DataFrame([results])
output_path = os.path.join(project_root, 'datasets/output/model_implementations/user_based_cf_warm_start.csv')
results_df.to_csv(output_path, index=False)

print(f"\n✓ Results saved to: {output_path}")
print("\nResults Summary:")
print(results_df.T)

print("\n" + "="*60)
print("COMPLETED SUCCESSFULLY!")
print("="*60)
