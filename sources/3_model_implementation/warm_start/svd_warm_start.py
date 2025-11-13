"""
SVD Matrix Factorization (Temporal Split - Warm-Start Evaluation)

This script uses temporal split data but filters the test set to achieve ~99% coverage
by only evaluating on user-movie pairs where both exist in the training set.

Purpose: Compare algorithm performance on temporal data WITHOUT cold-start complications.

Expected runtime: 5-10 minutes
"""

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("SVD MATRIX FACTORIZATION (WARM-START EVALUATION)")
print("="*60)
print("\nLibraries imported successfully!")
print("Using scipy.sparse.linalg.svds for SVD implementation")

# ============================================================================
# CONFIGURATION
# ============================================================================
SAMPLE_SIZE = 100000  # 100K for fair comparison with 80-20 split
N_FACTORS = 50  # Number of latent factors

print(f"\nConfiguration:")
print(f"  Sample size: {SAMPLE_SIZE:,}")
print(f"  Latent factors: {N_FACTORS}")

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

# Map train and test data to indices
train['user_idx'] = train['userId'].map(user_id_map)
train['movie_idx'] = train['movieId'].map(movie_id_map)

test['user_idx'] = test['userId'].map(user_id_map)
test['movie_idx'] = test['movieId'].map(movie_id_map)

# ============================================================================
# FILTER TEST SET
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

# ============================================================================
# CREATE SPARSE MATRIX
# ============================================================================
print("\n" + "="*60)
print("CREATING SPARSE MATRIX")
print("="*60)

user_item_matrix = csr_matrix(
    (train['rating'].values,
     (train['user_idx'].values, train['movie_idx'].values)),
    shape=(len(user_id_map), len(movie_id_map))
)

print(f"Matrix shape: {user_item_matrix.shape}")
print(f"Memory: {user_item_matrix.data.nbytes / (1024**2):.2f} MB")
print(f"Sparsity: {100 * (1 - user_item_matrix.nnz / (user_item_matrix.shape[0] * user_item_matrix.shape[1])):.2f}%")

# Calculate global mean
global_mean = train['rating'].mean()
print(f"\nGlobal mean rating: {global_mean:.3f}")

# ============================================================================
# MEAN-CENTERING
# ============================================================================
print("\n" + "="*60)
print("MEAN-CENTERING THE RATING MATRIX")
print("="*60)

# Compute user means (only over rated items)
print("Computing user means (only over rated items)...")

user_means = np.zeros(user_item_matrix.shape[0])

for user_idx in tqdm(range(user_item_matrix.shape[0]), desc="Computing user means"):
    user_ratings = user_item_matrix.getrow(user_idx).data
    if len(user_ratings) > 0:
        user_means[user_idx] = user_ratings.mean()
    else:
        user_means[user_idx] = global_mean

print(f"\nUser means computed:")
print(f"  Min: {user_means.min():.3f}")
print(f"  Max: {user_means.max():.3f}")
print(f"  Average: {user_means.mean():.3f}")

print("\nMean-centering matrix...")
user_item_matrix_lil = user_item_matrix.tolil()

for user_idx in tqdm(range(user_item_matrix.shape[0]), desc="Centering users"):
    row = user_item_matrix_lil.rows[user_idx]
    data = user_item_matrix_lil.data[user_idx]

    if len(row) > 0:
        user_mean = user_means[user_idx]
        for i in range(len(data)):
            data[i] -= user_mean

user_item_matrix_centered = user_item_matrix_lil.tocsr()

print(f"\n✓ Matrix centered successfully")
print(f"  Centered mean: {user_item_matrix_centered.data.mean():.6f} (should be ~0)")

# ============================================================================
# SVD DECOMPOSITION
# ============================================================================
print("\n" + "="*60)
print("SVD DECOMPOSITION")
print("="*60)

print(f"Latent factors (k): {N_FACTORS}")
print(f"\nDecomposing into:")
print(f"  U: {user_item_matrix.shape[0]:,} users × {N_FACTORS} factors")
print(f"  Sigma: {N_FACTORS} singular values")
print(f"  Vt: {N_FACTORS} factors × {user_item_matrix.shape[1]:,} movies")
print("\nThis may take 2-5 minutes...")

start_time = time.time()

# Use which='LM' to get LARGEST singular values
U, sigma, Vt = svds(user_item_matrix_centered, k=N_FACTORS, which='LM')

training_time = time.time() - start_time

print(f"\n✓ SVD completed in {training_time/60:.2f} minutes")
print(f"\nDecomposed matrices:")
print(f"  U shape: {U.shape}")
print(f"  Sigma shape: {sigma.shape}")
print(f"  Vt shape: {Vt.shape}")

# Prepare sigma as diagonal matrix for predictions
sigma_diag = np.diag(sigma)
print(f"\n✓ Ready for predictions!")

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

start_time = time.time()

test_predictions = []
test_actuals = []

for idx, row in tqdm(test_sample.iterrows(), total=len(test_sample), desc="Predicting"):
    user_idx = int(row['user_idx'])
    movie_idx = int(row['movie_idx'])
    actual = row['rating']

    # Get prediction: U[user] × Sigma × Vt[:,movie]
    user_vector = U[user_idx, :]
    movie_vector = Vt[:, movie_idx]

    # Compute centered prediction
    pred_centered = np.dot(user_vector, np.dot(sigma_diag, movie_vector))

    # Add back user's mean rating (de-centering)
    pred = pred_centered + user_means[user_idx]

    # Clip to valid range
    pred = np.clip(pred, 0.5, 5.0)

    test_predictions.append(pred)
    test_actuals.append(actual)

prediction_time = time.time() - start_time

print(f"\n✓ Predictions completed in {prediction_time:.2f} seconds")
print(f"  Average: {prediction_time/len(test_sample)*1000:.3f} ms per rating")

# Convert to numpy arrays
test_actuals = np.array(test_actuals)
test_predictions = np.array(test_predictions)

# ============================================================================
# EVALUATION
# ============================================================================
print("\n" + "="*60)
print("EVALUATION")
print("="*60)

rmse = np.sqrt(mean_squared_error(test_actuals, test_predictions))
mae = mean_absolute_error(test_actuals, test_predictions)
correlation = np.corrcoef(test_actuals, test_predictions)[0, 1]

print("\n" + "="*60)
print("SVD RESULTS (WARM-START EVALUATION)")
print("="*60)
print(f"\nAlgorithm: SVD Matrix Factorization")
print(f"Split: Temporal (filtered for known user-movie pairs)")
print(f"Implementation: scipy.sparse.linalg.svds")
print(f"Latent factors: {N_FACTORS}")

print(f"\nPerformance:")
print(f"  RMSE: {rmse:.4f}")
print(f"  MAE:  {mae:.4f}")
print(f"  Correlation: {correlation:.4f}")

print(f"\nTiming:")
print(f"  Training: {training_time/60:.2f} minutes")
print(f"  Prediction: {prediction_time:.2f} seconds")
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
    'algorithm': 'SVD (Warm-Start)',
    'split_strategy': 'temporal (filtered)',
    'method': 'Matrix Factorization',
    'implementation': 'scipy.sparse.linalg.svds',
    'n_factors': N_FACTORS,
    'rmse': rmse,
    'mae': mae,
    'coverage': true_coverage,
    'training_time_minutes': training_time/60,
    'prediction_time_ms': prediction_time/len(test_sample)*1000,
    'test_samples': len(test_sample),
    'total_testable': total_testable,
    'correlation': correlation
}

results_df = pd.DataFrame([results])
output_path = os.path.join(project_root, 'datasets/output/model_implementations/svd_warm_start.csv')
results_df.to_csv(output_path, index=False)

print(f"\n✓ Results saved to: {output_path}")
print("\nResults Summary:")
print(results_df.T)

print("\n" + "="*60)
print("COMPLETED SUCCESSFULLY!")
print("="*60)
