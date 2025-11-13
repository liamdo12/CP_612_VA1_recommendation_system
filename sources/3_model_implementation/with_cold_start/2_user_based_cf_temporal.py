"""
User-Based Collaborative Filtering (Temporal Split - WITH Cold-Start)

This script evaluates on the FULL temporal test set INCLUDING cold-start cases.
Cold-start predictions use global mean rating as fallback.

Purpose: Compare performance when cold-start users/items are included.

Expected runtime: 30-60 minutes
"""

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm
import time
import gc
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("USER-BASED CF (TEMPORAL SPLIT - WITH COLD-START)")
print("="*60)
print("\nLibraries imported successfully!")

# ============================================================================
# CONFIGURATION
# ============================================================================
SAMPLE_SIZE = 100000  # Match warm-start configuration for fair comparison
K_NEIGHBORS = 30  # Number of nearest neighbors to consider
MAX_CANDIDATES = 100  # Reduced from 200 to 100 for memory efficiency
BATCH_SIZE = 500  # Smaller batches for better memory management
GC_FREQUENCY = 2  # More frequent garbage collection

print(f"\nConfiguration:")
print(f"  Sample size: {SAMPLE_SIZE:,}")
print(f"  k-neighbors: {K_NEIGHBORS}")
print(f"  Max candidates: {MAX_CANDIDATES}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  GC frequency: Every {GC_FREQUENCY} batches")

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

print("\nTest Set (FULL - includes cold-start):")
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

# Get unique users and movies from TRAIN set only
unique_users = train['userId'].unique()
unique_movies = train['movieId'].unique()

user_id_map = {user_id: idx for idx, user_id in enumerate(unique_users)}
movie_id_map = {movie_id: idx for idx, movie_id in enumerate(unique_movies)}

print(f"Training users: {len(user_id_map):,}")
print(f"Training movies: {len(movie_id_map):,}")

# Map train data to matrix indices
print("\nMapping train data to matrix indices...")
train['user_idx'] = train['userId'].map(user_id_map)
train['movie_idx'] = train['movieId'].map(movie_id_map)

# Map test data to matrix indices (will have NaN for cold-start)
print("Mapping test data to matrix indices...")
test['user_idx'] = test['userId'].map(user_id_map)
test['movie_idx'] = test['movieId'].map(movie_id_map)

# ============================================================================
# ANALYZE COLD-START
# ============================================================================
print("\n" + "="*60)
print("COLD-START ANALYSIS")
print("="*60)

# Count cold-start cases
cold_start_users = test['user_idx'].isna().sum()
cold_start_movies = test['movie_idx'].isna().sum()
cold_start_both = (test['user_idx'].isna() | test['movie_idx'].isna()).sum()

print(f"\nCold-start cases in test set:")
print(f"  Cold-start users: {cold_start_users:,} ({cold_start_users/len(test)*100:.2f}%)")
print(f"  Cold-start movies: {cold_start_movies:,} ({cold_start_movies/len(test)*100:.2f}%)")
print(f"  Either cold-start: {cold_start_both:,} ({cold_start_both/len(test)*100:.2f}%)")
print(f"  Known pairs: {len(test) - cold_start_both:,} ({(len(test)-cold_start_both)/len(test)*100:.2f}%)")

# Global mean rating for fallback
global_mean_rating = train['rating'].mean()
print(f"\nGlobal mean rating (for cold-start): {global_mean_rating:.3f}")

# ============================================================================
# CREATE SPARSE USER-ITEM MATRIX
# ============================================================================
print("\n" + "="*60)
print("CREATING SPARSE USER-ITEM MATRICES")
print("="*60)

start_time = time.time()

# CSR matrix (efficient for row access - getting user ratings)
user_item_matrix_csr = csr_matrix(
    (train['rating'].values,
     (train['user_idx'].values, train['movie_idx'].values)),
    shape=(len(user_id_map), len(movie_id_map))
)

# CSC matrix (efficient for column access - finding users who rated a movie)
user_item_matrix_csc = csc_matrix(
    (train['rating'].values,
     (train['user_idx'].values, train['movie_idx'].values)),
    shape=(len(user_id_map), len(movie_id_map))
)

elapsed = time.time() - start_time

print(f"\nSparse matrices created in {elapsed:.2f} seconds")
print(f"Matrix shape: {user_item_matrix_csr.shape}")
print(f"CSR memory: {user_item_matrix_csr.data.nbytes / (1024**2):.2f} MB")
print(f"CSC memory: {user_item_matrix_csc.data.nbytes / (1024**2):.2f} MB")
print(f"Density: {user_item_matrix_csr.nnz / (user_item_matrix_csr.shape[0] * user_item_matrix_csr.shape[1]) * 100:.4f}%")

# ============================================================================
# MEAN-CENTER RATINGS
# ============================================================================
print("\n" + "="*60)
print("COMPUTING USER MEAN RATINGS")
print("="*60)

# Compute mean rating per user (for mean-centering)
user_means = np.zeros(len(user_id_map))

print("Computing mean ratings for each user...")
for user_idx in tqdm(range(len(user_id_map)), desc="User means"):
    user_ratings = user_item_matrix_csr[user_idx].data
    if len(user_ratings) > 0:
        user_means[user_idx] = user_ratings.mean()
    else:
        user_means[user_idx] = global_mean_rating

print(f"\n User means computed")
print(f"  Average user mean: {user_means.mean():.3f}")

# ============================================================================
# TRAINING (On-Demand Similarity)
# ============================================================================
print("\n" + "="*60)
print("TRAINING SETUP")
print("="*60)

print("\nUsing on-demand similarity computation (no precomputation)")
print("Similarity will be computed during prediction for efficiency")

training_time = 0.0  # No training time for on-demand approach

# Precompute user norms for Pearson correlation
print("\nPrecomputing user norms for efficient similarity...")
user_norms = np.zeros(len(user_id_map))

# Mean-center and compute norms
for user_idx in tqdm(range(len(user_id_map)), desc="User norms"):
    user_row = user_item_matrix_csr[user_idx]
    if user_row.nnz > 0:
        # Get rated items
        rated_items = user_row.indices
        ratings = user_row.data

        # Mean-center
        centered = ratings - user_means[user_idx]

        # Compute norm
        user_norms[user_idx] = np.linalg.norm(centered)

print(f" User norms computed")

# ============================================================================
# PREDICTION FUNCTION (WITH COLD-START HANDLING)
# ============================================================================

def predict_rating(user_idx, movie_idx, k=K_NEIGHBORS, max_candidates=MAX_CANDIDATES):
    """
    Predict rating using user-based CF with on-demand similarity.

    Cold-start handling:
    - If user not in training � global mean
    - If movie not in training � global mean
    - If no similar users found � global mean

    Uses Pearson correlation (via cosine on mean-centered ratings).

    Parameters:
    - user_idx: Index of user in matrix (can be NaN for cold-start)
    - movie_idx: Index of movie in matrix (can be NaN for cold-start)
    - k: Number of similar users to consider
    - max_candidates: Maximum candidate users to evaluate

    Returns:
    - predicted_rating: Float between 0.5 and 5.0
    """
    # Handle cold-start: user or movie not in training
    if pd.isna(user_idx) or pd.isna(movie_idx):
        return global_mean_rating

    user_idx = int(user_idx)
    movie_idx = int(movie_idx)

    # Get target user's mean
    target_user_mean = user_means[user_idx]

    # Get users who rated this movie (use CSC for efficient column access)
    movie_col = user_item_matrix_csc[:, movie_idx]
    users_who_rated = movie_col.nonzero()[0]

    if len(users_who_rated) == 0:
        # No users rated this movie
        return global_mean_rating

    # Exclude target user if they rated this movie (shouldn't happen in test, but just in case)
    users_who_rated = users_who_rated[users_who_rated != user_idx]

    if len(users_who_rated) == 0:
        return global_mean_rating

    # Limit candidates for efficiency
    if len(users_who_rated) > max_candidates:
        # Randomly sample candidates (for speed)
        np.random.seed(42)
        users_who_rated = np.random.choice(users_who_rated, max_candidates, replace=False)

    # Get target user's ratings (mean-centered)
    target_user_row = user_item_matrix_csr[user_idx]
    target_norm = user_norms[user_idx]

    if target_norm == 0:
        # Target user has no variance in ratings
        return target_user_mean

    # Compute similarities with candidate users
    similarities = []
    other_users = []
    other_ratings = []

    for other_user_idx in users_who_rated:
        other_norm = user_norms[other_user_idx]

        if other_norm == 0:
            continue  # Skip users with no variance

        # Compute dot product of mean-centered ratings (Pearson correlation)
        # This is equivalent to cosine similarity on centered ratings
        dot_product = target_user_row.dot(user_item_matrix_csr[other_user_idx].T).toarray()[0, 0]

        # Adjust for means (Pearson = cosine on centered data)
        # Get common items
        target_items = set(target_user_row.indices)
        other_items = set(user_item_matrix_csr[other_user_idx].indices)
        common_items = target_items.intersection(other_items)

        if len(common_items) == 0:
            continue

        # Compute centered dot product
        centered_dot = 0.0
        for item_idx in common_items:
            target_rating = target_user_row[0, item_idx]
            other_rating = user_item_matrix_csr[other_user_idx, item_idx]
            centered_dot += (target_rating - target_user_mean) * (other_rating - user_means[other_user_idx])

        # Pearson correlation
        similarity = centered_dot / (target_norm * other_norm)

        # Get other user's rating for target movie
        other_user_rating = user_item_matrix_csr[other_user_idx, movie_idx]

        similarities.append(similarity)
        other_users.append(other_user_idx)
        other_ratings.append(other_user_rating)

    if len(similarities) == 0:
        # No valid similarities
        return target_user_mean

    # Convert to arrays
    similarities = np.array(similarities)
    other_ratings = np.array(other_ratings)
    other_users = np.array(other_users)

    # Get top-k most similar users
    if len(similarities) > k:
        top_k_indices = np.argsort(similarities)[::-1][:k]
    else:
        top_k_indices = np.argsort(similarities)[::-1]

    top_k_sims = similarities[top_k_indices]
    top_k_ratings = other_ratings[top_k_indices]
    top_k_users = other_users[top_k_indices]

    # Filter out negative similarities (optional - can keep for more data)
    # pos_mask = top_k_sims > 0
    # if np.sum(pos_mask) == 0:
    #     return target_user_mean
    # top_k_sims = top_k_sims[pos_mask]
    # top_k_ratings = top_k_ratings[pos_mask]
    # top_k_users = top_k_users[pos_mask]

    # Weighted average with mean-centering
    weighted_sum = 0.0
    sum_of_weights = 0.0

    for i in range(len(top_k_sims)):
        sim = top_k_sims[i]
        rating = top_k_ratings[i]
        other_user_idx = int(top_k_users[i])
        other_mean = user_means[other_user_idx]

        # Mean-centered prediction
        weighted_sum += sim * (rating - other_mean)
        sum_of_weights += abs(sim)

    if sum_of_weights == 0:
        return target_user_mean

    # Predicted rating = target user's mean + weighted average of centered ratings
    predicted = target_user_mean + (weighted_sum / sum_of_weights)

    # Clip to valid rating range [0.5, 5.0]
    predicted = np.clip(predicted, 0.5, 5.0)

    return predicted

print("\n Prediction function defined (with cold-start handling)!")

# ============================================================================
# SAMPLE TEST SET (FROM FULL SET)
# ============================================================================
print("\n" + "="*60)
print("SAMPLING TEST SET")
print("="*60)

# Sample from FULL test set (including cold-start)
if SAMPLE_SIZE and SAMPLE_SIZE < len(test):
    print(f"Sampling {SAMPLE_SIZE:,} from {len(test):,} total test ratings")
    test_sample = test.sample(SAMPLE_SIZE, random_state=42)
else:
    print(f"Using all {len(test):,} test ratings")
    test_sample = test.copy()

# Count cold-start in sample
cold_start_in_sample = (test_sample['user_idx'].isna() | test_sample['movie_idx'].isna()).sum()
known_in_sample = len(test_sample) - cold_start_in_sample

print(f"\n Sample size: {len(test_sample):,}")
print(f"  Known pairs: {known_in_sample:,} ({known_in_sample/len(test_sample)*100:.2f}%)")
print(f"  Cold-start: {cold_start_in_sample:,} ({cold_start_in_sample/len(test_sample)*100:.2f}%)")

# ============================================================================
# GENERATE PREDICTIONS
# ============================================================================
print("\n" + "="*60)
print("GENERATING PREDICTIONS")
print("="*60)

print(f"Generating predictions for {len(test_sample):,} ratings...")
print("This may take 30-60 minutes...")

start_time = time.time()

# Generate predictions with progress bar and memory management
predictions = []
batch_count = 0

for idx, row in tqdm(test_sample.iterrows(), total=len(test_sample), desc="Predicting"):
    pred = predict_rating(row['user_idx'], row['movie_idx'], k=K_NEIGHBORS, max_candidates=MAX_CANDIDATES)
    predictions.append(pred)

    # Periodic garbage collection
    if len(predictions) % BATCH_SIZE == 0:
        batch_count += 1
        if batch_count % GC_FREQUENCY == 0:
            gc.collect()

prediction_time = time.time() - start_time

print(f"\n Predictions completed in {prediction_time/60:.2f} minutes")
print(f"  Average: {prediction_time/len(test_sample)*1000:.2f} ms per rating")

# Count how many predictions used global mean (cold-start)
predictions_array = np.array(predictions)
global_mean_predictions = np.sum(np.abs(predictions_array - global_mean_rating) < 0.001)

print(f"\nPrediction breakdown:")
print(f"  CF predictions: {len(predictions) - global_mean_predictions:,}")
print(f"  Cold-start (global mean): {global_mean_predictions:,}")

# ============================================================================
# EVALUATION
# ============================================================================
print("\n" + "="*60)
print("EVALUATION")
print("="*60)

test_sample = test_sample.copy()
test_sample['predicted_rating'] = predictions

actual = test_sample['rating'].values
predicted = np.array(predictions)

rmse = np.sqrt(mean_squared_error(actual, predicted))
mae = mean_absolute_error(actual, predicted)
correlation = np.corrcoef(actual, predicted)[0, 1]

# Calculate coverage (% of test set that could be evaluated)
coverage = len(test_sample) / len(test) * 100

print("\n" + "="*60)
print("USER-BASED CF RESULTS (TEMPORAL SPLIT - WITH COLD-START)")
print("="*60)
print(f"\nAlgorithm: User-Based CF with Pearson Correlation")
print(f"Split: Temporal (FULL test set with cold-start)")
print(f"k-neighbors: {K_NEIGHBORS}")
print(f"Max candidates: {MAX_CANDIDATES}")

print(f"\nPerformance:")
print(f"  RMSE: {rmse:.4f}")
print(f"  MAE:  {mae:.4f}")
print(f"  Correlation: {correlation:.4f}")

print(f"\nTiming:")
print(f"  Training (on-demand): {training_time:.2f} minutes")
print(f"  Prediction: {prediction_time/60:.2f} minutes")
print(f"  Per rating: {prediction_time/len(test_sample)*1000:.2f} ms")

print(f"\nCoverage:")
print(f"  Test samples: {len(test_sample):,}")
print(f"  CF predictions: {len(predictions) - global_mean_predictions:,} ({(len(predictions) - global_mean_predictions)/len(predictions)*100:.2f}%)")
print(f"  Cold-start fallback: {global_mean_predictions:,} ({global_mean_predictions/len(predictions)*100:.2f}%)")
print("="*60)

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n" + "="*60)
print("SAVING RESULTS")
print("="*60)

results = {
    'algorithm': 'User-Based CF (With Cold-Start)',
    'split_strategy': 'temporal (full)',
    'similarity_metric': 'Pearson',
    'k_neighbors': K_NEIGHBORS,
    'max_candidates': MAX_CANDIDATES,
    'rmse': rmse,
    'mae': mae,
    'coverage': coverage,
    'training_time_minutes': training_time,
    'prediction_time_minutes': prediction_time/60,
    'prediction_time_ms': prediction_time/len(test_sample)*1000,
    'test_samples': len(test_sample),
    'cf_predictions': len(predictions) - global_mean_predictions,
    'cold_start_predictions': global_mean_predictions,
    'correlation': correlation
}

results_df = pd.DataFrame([results])
output_path = os.path.join(project_root, 'datasets/output/model_implementations/user_based_cf_with_cold_start.csv')
results_df.to_csv(output_path, index=False)

print(f"\n Results saved to: {output_path}")
print("\nResults Summary:")
print(results_df.T)

print("\n" + "="*60)
print("COMPLETED SUCCESSFULLY!")
print("="*60)
