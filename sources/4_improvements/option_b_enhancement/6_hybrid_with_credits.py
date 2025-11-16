"""
Enhanced Hybrid Recommender System: CF + Content (Genres + Actors + Directors)

This script implements an enhanced hybrid recommender system that combines:
- Collaborative Filtering (CF): Item-Based approach
- Content-Based Filtering: Using movie genres, actors, and directors

Enhancement over baseline:
- Baseline (5_hybrid_system.py): Genres only
- Enhanced (this script): Genres + Top 5 Actors + Directors

Feature weighting:
- Genre: 40% (broader categorization)
- Actors: 40% (strong preference signal)
- Directors: 20% (moderate influence)

Hybrid Strategy:
- Warm-start (user & movie in training): score = 0.7 × CF + 0.3 × content
- Cold-start user: score = 0.3 × CF + 0.7 × content
- Cold-start movie: score = 0.3 × CF + 0.7 × content
- Double cold-start: score = 1.0 × content

Usage:
    python 6_hybrid_with_credits.py

Configuration:
    - Set USE_SAMPLE = True for quick testing (~10-15 min)
    - Set USE_SAMPLE = False for full evaluation (~1-2 hours)

OPTIMIZED: Memory-efficient sparse matrices, chunked batch processing, type-safe operations
"""

import pandas as pd
import numpy as np
import ast
from datetime import datetime
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
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
MAX_ACTORS = 200  # Top N most common actors (reduces dimensionality, improves overlap)
MAX_DIRECTORS = 100  # Top N most common directors (reduces dimensionality, improves overlap)

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("=" * 80)
print("ENHANCED HYBRID: CF + CONTENT (GENRES + ACTORS + DIRECTORS)")
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
train = pd.read_csv('../../../datasets/output/split_and_train_datasets/temporal_split/train_ratings.csv')
test = pd.read_csv('../../../datasets/output/split_and_train_datasets/temporal_split/test_ratings.csv')

print("Loading movie metadata...")
movies = pd.read_csv('../../../datasets/output/cleaned_datasets/cleaned_movies_metadata.csv')
links = pd.read_csv('../../../datasets/output/cleaned_datasets/cleaned_links.csv')

print("Loading credits (actors/directors)...")
credits = pd.read_csv('../../../datasets/output/cleaned_datasets/cleaned_credits.csv')

print(f"\n  Train: {len(train):,} ratings")
print(f"  Test: {len(test):,} ratings")
print(f"  Movies: {len(movies):,} movies")
print(f"  Links: {len(links):,} links")
print(f"  Credits: {len(credits):,} movies with credits")


# =============================================================================
# 2. PREPARE MOVIE FEATURES (GENRES + ACTORS + DIRECTORS)
# =============================================================================
print("\n" + "=" * 80)
print("STEP 2: PREPARING MOVIE FEATURES (GENRES + ACTORS + DIRECTORS)")
print("=" * 80)


def parse_genres(genre_str):
    """Convert string representation of list to actual list"""
    if pd.isna(genre_str) or genre_str == '[]':
        return []
    try:
        return ast.literal_eval(genre_str)
    except:
        return []


def parse_cast_list(cast_str):
    """Parse cast_list string to list of actors"""
    if pd.isna(cast_str) or cast_str == '[]':
        return []
    try:
        return ast.literal_eval(cast_str)
    except:
        return []


def parse_director_list(director_str):
    """Parse director_list string to list of directors"""
    if pd.isna(director_str) or director_str == '[]':
        return []
    try:
        return ast.literal_eval(director_str)
    except:
        return []


# Parse genres
movies['genres'] = movies['genres_list'].apply(parse_genres)

# Parse actors and directors
credits['actors'] = credits['cast_list'].apply(parse_cast_list)
credits['directors'] = credits['director_list'].apply(parse_director_list)

# Merge movies with credits and links
print("\n  Merging movies, credits, and links...")
movies_with_credits = movies.merge(credits[['id', 'actors', 'directors']], on='id', how='left')
movies_enriched = movies_with_credits.merge(
    links[['movieId', 'tmdbId']],
    left_on='id',
    right_on='tmdbId',
    how='inner'
)

# Fill missing credits with empty lists
movies_enriched['actors'] = movies_enriched['actors'].apply(lambda x: x if isinstance(x, list) else [])
movies_enriched['directors'] = movies_enriched['directors'].apply(lambda x: x if isinstance(x, list) else [])

print(f"  Movies with enriched features: {len(movies_enriched):,}")
print(f"    - With genres: {movies_enriched['genres'].apply(len).gt(0).sum():,}")
print(f"    - With actors: {movies_enriched['actors'].apply(len).gt(0).sum():,}")
print(f"    - With directors: {movies_enriched['directors'].apply(len).gt(0).sum():,}")

# Get all unique genres, actors, and directors
print("\n  Extracting unique features...")
print(f"  Limiting to top {MAX_ACTORS} actors and top {MAX_DIRECTORS} directors to reduce memory")

all_genres = set()
actor_counts = {}
director_counts = {}

for _, row in movies_enriched.iterrows():
    all_genres.update(row['genres'])

    # Count actor occurrences (only top 5 per movie)
    top_actors = row['actors'][:5] if len(row['actors']) > 0 else []
    for actor in top_actors:
        actor_counts[actor] = actor_counts.get(actor, 0) + 1

    # Count director occurrences
    for director in row['directors']:
        director_counts[director] = director_counts.get(director, 0) + 1

all_genres = sorted(list(all_genres))

# CRITICAL: Only keep most common actors/directors to reduce dimensionality
all_actors = sorted([actor for actor, _ in sorted(actor_counts.items(),
                                                   key=lambda x: x[1],
                                                   reverse=True)[:MAX_ACTORS]])
all_directors = sorted([director for director, _ in sorted(director_counts.items(),
                                                           key=lambda x: x[1],
                                                           reverse=True)[:MAX_DIRECTORS]])

print(f"  Feature vocabulary sizes (OPTIMIZED):")
print(f"    - Genres: {len(all_genres)} (all unique genres)")
print(f"    - Actors: {len(all_actors)} (top {MAX_ACTORS} most common)")
print(f"    - Directors: {len(all_directors)} (top {MAX_DIRECTORS} most common)")
print(f"    - Total feature dimensions: {len(all_genres) + len(all_actors) + len(all_directors)}")
print(f"    - Original actor count: {len(actor_counts):,} → reduced to {len(all_actors)}")
print(f"    - Original director count: {len(director_counts):,} → reduced to {len(all_directors)}")

# Create combined feature matrix
print("\n  Building combined feature matrix...")
start_time = time.time()

genre_matrix = []
actor_matrix = []
director_matrix = []
movie_ids = []

for _, row in movies_enriched.iterrows():
    movie_ids.append(row['movieId'])

    # Genre features
    genre_vector = [1 if genre in row['genres'] else 0 for genre in all_genres]
    genre_matrix.append(genre_vector)

    # Actor features (top 5 actors only to reduce dimensionality)
    top_actors = row['actors'][:5] if len(row['actors']) > 0 else []
    actor_vector = [1 if actor in top_actors else 0 for actor in all_actors]
    actor_matrix.append(actor_vector)

    # Director features
    director_vector = [1 if director in row['directors'] else 0 for director in all_directors]
    director_matrix.append(director_vector)

# Create DataFrames
genre_df = pd.DataFrame(genre_matrix, columns=[f'genre_{g}' for g in all_genres], index=movie_ids)
actor_df = pd.DataFrame(actor_matrix, columns=[f'actor_{a}' for a in all_actors], index=movie_ids)
director_df = pd.DataFrame(director_matrix, columns=[f'director_{d}' for d in all_directors], index=movie_ids)

# Combine into single feature matrix
feature_df = pd.concat([genre_df, actor_df, director_df], axis=1)

# CRITICAL: Remove duplicate movie IDs (same fix as in 5_hybrid_system.py)
feature_df = feature_df[~feature_df.index.duplicated(keep='first')]

elapsed = time.time() - start_time
print(f"  Feature matrix built in {elapsed:.2f} seconds")
print(f"  Feature matrix shape: {feature_df.shape}")
print(f"  Memory: ~{feature_df.memory_usage(deep=True).sum() / (1024**2):.1f} MB")


# =============================================================================
# 3. BUILD COLLABORATIVE FILTERING COMPONENTS
# =============================================================================
print("\n" + "=" * 80)
print("STEP 3: BUILDING COLLABORATIVE FILTERING COMPONENTS")
print("=" * 80)

print("\n  Building SPARSE user-item matrix (memory-efficient)...")

# Create user and movie ID mappings
user_ids = sorted(train['userId'].unique())
movie_ids_train = sorted(train['movieId'].unique())

# Limit movies for similarity computation to save memory
if len(movie_ids_train) > MAX_MOVIES_FOR_SIMILARITY:
    movie_counts = train['movieId'].value_counts()
    top_movies = movie_counts.head(MAX_MOVIES_FOR_SIMILARITY).index.tolist()
    movie_ids_for_sim = sorted(top_movies)
    print(f"  Limiting to top {MAX_MOVIES_FOR_SIMILARITY:,} most-rated movies for similarity computation")
else:
    movie_ids_for_sim = movie_ids_train

user_id_map = {uid: idx for idx, uid in enumerate(user_ids)}
movie_id_map = {mid: idx for idx, mid in enumerate(movie_ids_train)}
movie_id_map_sim = {mid: idx for idx, mid in enumerate(movie_ids_for_sim)}

# Build sparse matrix
n_users = len(user_ids)
n_movies = len(movie_ids_train)

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

# Compute item-item similarity using OPTIMIZED chunked batch processing
print("\n  Computing item-item similarity (OPTIMIZED - memory-efficient approach)...")
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
        top_k_indices = np.argsort(sims)[::-1][1:K_NEIGHBORS+1]
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
# 4. IMPLEMENT ENHANCED CONTENT-BASED SCORING
# =============================================================================
print("\n" + "=" * 80)
print("STEP 4: IMPLEMENTING ENHANCED CONTENT-BASED SCORING")
print("=" * 80)
print("\n  Feature weighting: Genre (40%), Actors (40%), Directors (20%)")


def get_user_feature_profile(user_id, train_data, feature_df):
    """
    Build a user's feature preference profile based on rating history.
    Returns weighted feature vector (genres + actors + directors).

    OPTIMIZED: Type-safe with defensive checks for duplicate indices.
    """
    user_ratings = train_data[train_data['userId'] == user_id]

    if len(user_ratings) == 0:
        return pd.Series(0.0, index=feature_df.columns, dtype=np.float32)

    feature_profile = pd.Series(0.0, index=feature_df.columns, dtype=np.float32)

    for _, row in user_ratings.iterrows():
        movie_id = row['movieId']
        rating = float(row['rating'])

        if movie_id in feature_df.index:
            # Get feature vector for this movie (defensive check for duplicates)
            movie_features = feature_df.loc[movie_id]
            if isinstance(movie_features, pd.DataFrame):
                # If duplicate index, take first row
                movie_features = movie_features.iloc[0]

            # Weight by rating (use .add() for safer addition with type handling)
            feature_profile = feature_profile.add(movie_features.astype(np.float32) * rating, fill_value=0)

    # L2 normalization (proper cosine similarity) - CRITICAL FIX
    # Prevents similarity deflation with high-dimensional features
    if feature_profile.sum() > 0:
        norm = np.linalg.norm(feature_profile.values)
        if norm > 0:
            feature_profile = feature_profile / norm

    return feature_profile


def enhanced_content_based_score(user_id, movie_id, train_data, feature_df,
                                  global_mean, feature_weights=None):
    """
    Compute enhanced content-based score using genres, actors, and directors.

    Args:
        feature_weights: dict with keys 'genre', 'actor', 'director' (default: 0.6, 0.3, 0.1)

    OPTIMIZED: Type-safe with defensive checks, L2-normalized cosine similarity.
    """
    if feature_weights is None:
        # OPTIMIZED: Increased genre weight to handle high actor/director sparsity
        # Genres have better coverage (99% of movies) vs actors/directors (sparse overlap)
        feature_weights = {'genre': 0.6, 'actor': 0.3, 'director': 0.1}

    user_profile = get_user_feature_profile(user_id, train_data, feature_df)

    if movie_id not in feature_df.index:
        return global_mean

    # Get movie's feature vector (defensive check for duplicates)
    movie_features = feature_df.loc[movie_id]
    if isinstance(movie_features, pd.DataFrame):
        # If duplicate index, take first row
        movie_features = movie_features.iloc[0]

    # Compute similarity separately for each feature type
    similarities = {}

    for feature_type, weight in feature_weights.items():
        # Get columns for this feature type
        cols = [col for col in feature_df.columns if col.startswith(f'{feature_type}_')]

        if len(cols) == 0:
            similarities[feature_type] = 0.0
            continue

        user_sub = user_profile[cols]
        movie_sub = movie_features[cols]

        # Compute cosine similarity with L2 normalization - CRITICAL FIX
        # Ensures similarity is in [0, 1] range regardless of feature dimensionality
        if user_sub.sum() == 0 or movie_sub.sum() == 0:
            similarities[feature_type] = 0.0
        else:
            # Proper cosine similarity: dot product / (norm1 * norm2)
            norm_user = np.linalg.norm(user_sub.values)
            norm_movie = np.linalg.norm(movie_sub.values)
            if norm_user > 0 and norm_movie > 0:
                similarities[feature_type] = np.dot(user_sub.values, movie_sub.values) / (norm_user * norm_movie)
            else:
                similarities[feature_type] = 0.0

    # Weighted combination of similarities
    combined_similarity = sum(similarities[ft] * weight
                             for ft, weight in feature_weights.items())

    # Scale to rating range [0.5, 5.0]
    score = global_mean + (combined_similarity - 0.5) * 2.0
    score = np.clip(score, 0.5, 5.0)

    return float(score)


print("  Enhanced content-based scoring functions defined")


# =============================================================================
# 5. IMPLEMENT COLLABORATIVE FILTERING PREDICTION
# =============================================================================
print("\n" + "=" * 80)
print("STEP 5: IMPLEMENTING COLLABORATIVE FILTERING PREDICTION")
print("=" * 80)


def item_based_cf_predict(user_id, movie_id, train_data, item_sim_dict, global_mean=3.5):
    """
    Item-based collaborative filtering prediction (memory-efficient version).
    Uses dictionary-based similarity (not full matrix).

    OPTIMIZED: Works with sparse data and dictionary similarity storage.
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
# 6. IMPLEMENT ENHANCED HYBRID PREDICTION
# =============================================================================
print("\n" + "=" * 80)
print("STEP 6: IMPLEMENTING ENHANCED HYBRID PREDICTION WITH ADAPTIVE WEIGHTING")
print("=" * 80)


def enhanced_hybrid_predict(user_id, movie_id, train_data, item_sim_dict,
                           feature_df, train_users, train_movies, global_mean):
    """
    Enhanced hybrid prediction combining CF and enhanced content-based filtering.

    Weighting strategy:
    - Warm-start (both in training): CF=0.7, Content=0.3
    - Cold-start user OR movie: CF=0.3, Content=0.7
    - Double cold-start: CF=0.0, Content=1.0

    OPTIMIZED: Works with dictionary-based similarity (not full matrix).
    """
    cf_score, is_cf = item_based_cf_predict(user_id, movie_id, train_data,
                                            item_sim_dict, global_mean=global_mean)

    cb_score = enhanced_content_based_score(user_id, movie_id, train_data,
                                           feature_df, global_mean)

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


print("  Enhanced hybrid prediction function defined")

# Get cold-start statistics
test_users = set(test['userId'].unique())
test_movies = set(test['movieId'].unique())

cold_start_users = test_users - train_users
cold_start_movies = test_movies - train_movies

print(f"\n  Cold-start statistics:")
print(f"    Cold-start users: {len(cold_start_users):,} / {len(test_users):,} ({len(cold_start_users)/len(test_users)*100:.1f}%)")
print(f"    Cold-start movies: {len(cold_start_movies):,} / {len(test_movies):,} ({len(cold_start_movies)/len(test_movies)*100:.1f}%)")


# =============================================================================
# 7. EVALUATE ENHANCED HYBRID SYSTEM ON TEST SET
# =============================================================================
print("\n" + "=" * 80)
print("STEP 7: EVALUATING ENHANCED HYBRID SYSTEM ON TEST SET")
print("=" * 80)

if USE_SAMPLE:
    test_sample = test.sample(n=min(SAMPLE_SIZE, len(test)), random_state=42)
    print(f"\n  Using sample of {len(test_sample):,} test ratings for faster evaluation")
else:
    test_sample = test
    print(f"\n  Using full test set of {len(test_sample):,} ratings")

print(f"\n  Starting enhanced hybrid evaluation...")
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

    pred, case = enhanced_hybrid_predict(user_id, movie_id, train, item_sim_dict,
                                        feature_df, train_users, train_movies, global_mean)

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
print(f"ENHANCED HYBRID SYSTEM EVALUATION RESULTS")
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
# 8. COMPARE WITH BASELINE HYBRID (GENRES ONLY)
# =============================================================================
print("\n" + "=" * 80)
print("STEP 8: COMPARING WITH BASELINE HYBRID (GENRES ONLY)")
print("=" * 80)

print("\n  Expected baseline results (from 5_hybrid_system.py):")
baseline_rmse = 1.0124  # From baseline implementation
baseline_mae = 0.7821

print(f"\n{'='*80}")
print(f"ENHANCED vs BASELINE HYBRID COMPARISON")
print(f"{'='*80}")
print(f"\n{'System':<40} {'RMSE':<12} {'MAE':<12} {'Improvement'}")
print(f"{'-'*40} {'-'*12} {'-'*12} {'-'*20}")
print(f"{'Baseline (Genres Only)':<40} {baseline_rmse:>10.6f}   {baseline_mae:>10.6f}   {'(baseline)'}")
print(f"{'Enhanced (Genres+Actors+Directors)':<40} {rmse:>10.6f}   {mae:>10.6f}   {(baseline_rmse-rmse)/baseline_rmse*100:>+6.2f}% RMSE")

improvement = (baseline_rmse - rmse) / baseline_rmse * 100
if improvement > 0:
    print(f"\n✓ Enhanced system improves RMSE by {improvement:.2f}%")
    print(f"  Adding actor/director features provides {improvement:.2f}% improvement over genres alone")
else:
    print(f"\n⚠ Enhanced system shows {abs(improvement):.2f}% degradation")
    print(f"  Note: Actor/director features may need different weighting or more data")


# =============================================================================
# 9. VISUALIZE RESULTS
# =============================================================================
print("\n" + "=" * 80)
print("STEP 9: CREATING VISUALIZATIONS")
print("=" * 80)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: RMSE and MAE comparison
systems = ['Baseline\n(Genres)', 'Enhanced\n(+Actors+Directors)']
rmse_values = [baseline_rmse, rmse]
mae_values = [baseline_mae, mae]

x = np.arange(len(systems))
width = 0.35

axes[0].bar(x - width/2, rmse_values, width, label='RMSE', color='steelblue', alpha=0.8)
axes[0].bar(x + width/2, mae_values, width, label='MAE', color='coral', alpha=0.8)
axes[0].set_ylabel('Error', fontsize=12)
axes[0].set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(systems)
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# Add value labels
for i, (r, m) in enumerate(zip(rmse_values, mae_values)):
    axes[0].text(i - width/2, r + 0.005, f'{r:.4f}', ha='center', va='bottom', fontsize=9)
    axes[0].text(i + width/2, m + 0.005, f'{m:.4f}', ha='center', va='bottom', fontsize=9)

# Plot 2: Improvement percentage
improvement_pct = (baseline_rmse - rmse) / baseline_rmse * 100
color = 'green' if improvement_pct > 0 else 'red'
axes[1].bar(0, improvement_pct, color=color, alpha=0.7, width=0.5)
axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
axes[1].set_ylabel('Improvement (%)', fontsize=12)
axes[1].set_title('RMSE Improvement from Baseline', fontsize=14, fontweight='bold')
axes[1].set_xticks([0])
axes[1].set_xticklabels(['Enhanced\nvs Baseline'])
axes[1].grid(axis='y', alpha=0.3)
axes[1].text(0, improvement_pct + (0.1 if improvement_pct > 0 else -0.1),
            f'{improvement_pct:+.2f}%', ha='center', va='bottom' if improvement_pct > 0 else 'top',
            fontsize=12, fontweight='bold')

# Plot 3: Case type distribution
case_counts = results_df['case'].value_counts()
case_colors = {'warm': 'green', 'partial_cold': 'orange', 'double_cold': 'red'}
colors = [case_colors.get(case, 'gray') for case in case_counts.index]

axes[2].bar(range(len(case_counts)), case_counts.values, color=colors, alpha=0.7)
axes[2].set_ylabel('Count', fontsize=12)
axes[2].set_title('Prediction Distribution', fontsize=14, fontweight='bold')
axes[2].set_xticks(range(len(case_counts)))
axes[2].set_xticklabels([c.replace('_', '\n').title() for c in case_counts.index])
axes[2].grid(axis='y', alpha=0.3)

# Add percentage labels
for i, count in enumerate(case_counts.values):
    pct = count / len(results_df) * 100
    axes[2].text(i, count + len(results_df)*0.01, f'{count:,}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('enhanced_hybrid_comparison.png', dpi=300, bbox_inches='tight')
print("\n  Visualization saved as 'enhanced_hybrid_comparison.png'")
plt.close()


# =============================================================================
# 10. SAVE RESULTS
# =============================================================================
print("\n" + "=" * 80)
print("STEP 10: SAVING RESULTS")
print("=" * 80)

# Save detailed comparison results
comparison_results = pd.DataFrame({
    'System': [
        'Baseline Hybrid (Genres Only)',
        'Enhanced Hybrid (Genres+Actors+Directors)'
    ],
    'RMSE': [baseline_rmse, rmse],
    'MAE': [baseline_mae, mae],
    'RMSE_Improvement_%': [0.0, (baseline_rmse - rmse) / baseline_rmse * 100],
    'Test_Samples': [len(test_sample), len(test_sample)],
    'Features': [
        'Genres (20 dimensions)',
        f'Genres ({len(all_genres)}) + Actors ({len(all_actors)}) + Directors ({len(all_directors)})'
    ]
})

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
comparison_filename = f'enhanced_vs_baseline_comparison_{timestamp}.csv'
comparison_results.to_csv(comparison_filename, index=False)
print(f"\n  Comparison results saved to {comparison_filename}")

# Save detailed breakdown by case
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
breakdown_filename = f'enhanced_case_breakdown_{timestamp}.csv'
breakdown_df.to_csv(breakdown_filename, index=False)
print(f"  Case breakdown saved to {breakdown_filename}")


# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("ENHANCED HYBRID IMPLEMENTATION COMPLETE")
print("=" * 80)
print(f"\nCompleted at: {datetime.now()}")
print("\nSummary:")
print("  This enhanced hybrid system extends the baseline by adding actor/director features:")
print("  1. Item-Based Collaborative Filtering for users/movies with interaction history")
print("  2. Enhanced Content-Based Filtering using genres, actors, and directors")
print("  3. Adaptive weighting that adjusts balance based on data availability")
print("\n  Feature enhancement:")
print(f"    - Baseline: ~20 genre dimensions")
print(f"    - Enhanced: {feature_df.shape[1]:,} dimensions (genres + actors + directors)")
print("\n  The enhanced approach handles various scenarios:")
print("    - Warm-start: Leverages collaborative patterns (70% weight)")
print("    - Partial cold-start: Balances CF and content (30% CF, 70% content)")
print("    - Double cold-start: Falls back to content-based recommendations")
print("\n  Key findings:")
if improvement > 0:
    print(f"    ✓ Enhanced system improves RMSE by {improvement:.2f}% over baseline")
else:
    print(f"    ⚠ Enhanced system shows {abs(improvement):.2f}% degradation")
    print(f"    → May require tuning feature weights or more training data")
print("=" * 80)
