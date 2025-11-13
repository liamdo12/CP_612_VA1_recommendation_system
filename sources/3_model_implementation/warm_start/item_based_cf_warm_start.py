"""
Item-Based Collaborative Filtering (Warm-Start Evaluation)

This script evaluates on temporal split data using warm-start conditions:
only user-movie pairs where both user AND movie exist in the training set.

Purpose: Measure pure collaborative filtering performance without cold-start effects.

Expected runtime: 15-25 minutes
"""

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm
import time
import gc
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("ITEM-BASED CF (WARM-START EVALUATION)")
print("="*60)
print("\nLibraries imported successfully!")

# ============================================================================
# CONFIGURATION
# ============================================================================
SAMPLE_SIZE = 100000  # 100K for fair comparison with 80-20 split
K_NEIGHBORS = 30  # Number of similar items to consider

print(f"\nConfiguration:")
print(f"  Sample size: {SAMPLE_SIZE:,}")
print(f"  k-neighbors: {K_NEIGHBORS}")

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

# Get unique users and movies from TRAIN set only
unique_users = train['userId'].unique()
unique_movies = train['movieId'].unique()

user_id_map = {user_id: idx for idx, user_id in enumerate(unique_users)}
movie_id_map = {movie_id: idx for idx, movie_id in enumerate(unique_movies)}

# Reverse mappings (for predictions)
idx_to_user = {idx: user_id for user_id, idx in user_id_map.items()}
idx_to_movie = {idx: movie_id for movie_id, idx in movie_id_map.items()}

print(f"Training users: {len(user_id_map):,}")
print(f"Training movies: {len(movie_id_map):,}")

# Map train data to matrix indices
print("\nMapping train data to matrix indices...")
train['user_idx'] = train['userId'].map(user_id_map)
train['movie_idx'] = train['movieId'].map(movie_id_map)

# Map test data to matrix indices
print("Mapping test data to matrix indices...")
test['user_idx'] = test['userId'].map(user_id_map)
test['movie_idx'] = test['movieId'].map(movie_id_map)

# ============================================================================
# FILTER TEST SET FOR HIGH COVERAGE
# ============================================================================
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

# Global mean rating for fallback
global_mean_rating = train['rating'].mean()
print(f"Global mean rating (for fallback): {global_mean_rating:.3f}")

# ============================================================================
# CREATE SPARSE USER-ITEM MATRIX
# ============================================================================
print("\n" + "="*60)
print("CREATING SPARSE USER-ITEM MATRIX")
print("="*60)

start_time = time.time()

user_item_matrix = csr_matrix(
    (train['rating'].values,
     (train['user_idx'].values, train['movie_idx'].values)),
    shape=(len(user_id_map), len(movie_id_map))
)

elapsed = time.time() - start_time

print(f"\nSparse matrix created in {elapsed:.2f} seconds")
print(f"Matrix shape: {user_item_matrix.shape}")
print(f"Memory usage: {user_item_matrix.data.nbytes / (1024**2):.2f} MB")
print(f"Density: {user_item_matrix.nnz / (user_item_matrix.shape[0] * user_item_matrix.shape[1]) * 100:.4f}%")

# ============================================================================
# COMPUTE ITEM-ITEM SIMILARITY MATRIX
# ============================================================================
print("\n" + "="*60)
print("COMPUTING ITEM-ITEM SIMILARITY MATRIX")
print("="*60)

print("Computing cosine similarity between items...")
print("This may take 5-10 minutes for ~25K movies...")

start_time = time.time()

# Transpose to get items x users, then compute similarity
# Use dense_output=False to get sparse similarity matrix (saves memory)
item_similarity = cosine_similarity(user_item_matrix.T, dense_output=False)

training_time = time.time() - start_time

print(f"\n✓ Item similarity matrix computed in {training_time/60:.2f} minutes")
print(f"Similarity matrix shape: {item_similarity.shape}")
print(f"Memory usage: {item_similarity.data.nbytes / (1024**2):.2f} MB")

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================

def predict_rating(user_idx, movie_idx, k=K_NEIGHBORS):
    """
    Predict rating for a user-movie pair using item-based CF.

    Parameters:
    - user_idx: Index of user in matrix
    - movie_idx: Index of movie in matrix
    - k: Number of similar items to consider

    Returns:
    - predicted_rating: Float between 0.5 and 5.0
    """
    user_idx = int(user_idx)
    movie_idx = int(movie_idx)

    # Get user's rating history
    user_ratings = user_item_matrix[user_idx].toarray().flatten()

    # Get similarity scores for target movie
    similarities = item_similarity[movie_idx].toarray().flatten()

    # Find items user has rated
    rated_items = np.where(user_ratings > 0)[0]

    if len(rated_items) == 0:
        # User has no ratings, return global mean
        return global_mean_rating

    # Get similarities and ratings for rated items
    item_sims = similarities[rated_items]
    item_ratings = user_ratings[rated_items]

    # Sort by similarity and take top-k
    top_k_indices = np.argsort(item_sims)[::-1][:k]
    top_k_sims = item_sims[top_k_indices]
    top_k_ratings = item_ratings[top_k_indices]

    # Remove self-similarity (if target movie was rated by user)
    mask = top_k_sims < 1.0
    top_k_sims = top_k_sims[mask]
    top_k_ratings = top_k_ratings[mask]

    if len(top_k_sims) == 0 or np.sum(np.abs(top_k_sims)) == 0:
        # No similar items found
        return global_mean_rating

    # Weighted average (use absolute similarity for weighting)
    weighted_sum = np.sum(top_k_sims * top_k_ratings)
    sum_of_weights = np.sum(np.abs(top_k_sims))

    predicted = weighted_sum / sum_of_weights

    # Clip to valid rating range [0.5, 5.0]
    predicted = np.clip(predicted, 0.5, 5.0)

    return predicted

print("\n✓ Prediction function defined!")

# ============================================================================
# SAMPLE TEST SET
# ============================================================================
print("\n" + "="*60)
print("SAMPLING TEST SET")
print("="*60)

# Sample from testable ratings
if SAMPLE_SIZE and SAMPLE_SIZE < len(testable):
    print(f"Sampling {SAMPLE_SIZE:,} from {len(testable):,} testable ratings")
    test_sample = testable.sample(SAMPLE_SIZE, random_state=42)
else:
    print(f"Using all {len(testable):,} testable ratings")
    test_sample = testable.copy()

print(f"\n✓ Sample size: {len(test_sample):,}")

# ============================================================================
# GENERATE PREDICTIONS
# ============================================================================
print("\n" + "="*60)
print("GENERATING PREDICTIONS")
print("="*60)

print(f"Generating predictions for {len(test_sample):,} ratings...")
print("This may take 15-25 minutes...")

start_time = time.time()

# Generate predictions with progress bar
predictions = []
for idx, row in tqdm(test_sample.iterrows(), total=len(test_sample), desc="Predicting"):
    pred = predict_rating(row['user_idx'], row['movie_idx'], k=K_NEIGHBORS)
    predictions.append(pred)

prediction_time = time.time() - start_time

print(f"\n✓ Predictions completed in {prediction_time/60:.2f} minutes")
print(f"  Average: {prediction_time/len(test_sample)*1000:.2f} ms per rating")

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

print("\n" + "="*60)
print("ITEM-BASED CF RESULTS (WARM-START EVALUATION)")
print("="*60)
print(f"\nAlgorithm: Item-Based CF with Cosine Similarity")
print(f"Split: Temporal (filtered for known user-movie pairs)")
print(f"k-neighbors: {K_NEIGHBORS}")

print(f"\nPerformance:")
print(f"  RMSE: {rmse:.4f}")
print(f"  MAE:  {mae:.4f}")
print(f"  Correlation: {correlation:.4f}")

print(f"\nTiming:")
print(f"  Training (similarity): {training_time/60:.2f} minutes")
print(f"  Prediction: {prediction_time/60:.2f} minutes")
print(f"  Per rating: {prediction_time/len(test_sample)*1000:.2f} ms")

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
    'algorithm': 'Item-Based CF (Warm-Start)',
    'split_strategy': 'temporal (filtered)',
    'similarity_metric': 'Cosine',
    'k_neighbors': K_NEIGHBORS,
    'rmse': rmse,
    'mae': mae,
    'coverage': true_coverage,
    'training_time_minutes': training_time/60,
    'prediction_time_minutes': prediction_time/60,
    'prediction_time_ms': prediction_time/len(test_sample)*1000,
    'test_samples': len(test_sample),
    'total_testable': total_testable,
    'correlation': correlation
}

results_df = pd.DataFrame([results])
output_path = os.path.join(project_root, 'datasets/output/model_implementations/item_based_cf_warm_start.csv')
results_df.to_csv(output_path, index=False)

print(f"\n✓ Results saved to: {output_path}")
print("\nResults Summary:")
print(results_df.T)

print("\n" + "="*60)
print("COMPLETED SUCCESSFULLY!")
print("="*60)
