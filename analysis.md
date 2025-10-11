# Movie Recommendation System - Analysis & Implementation Plan

---

## Dataset Relationship Analysis

### 1.1 Dataset Overview

The MovieLens dataset consists of **7 interconnected files** forming a star schema with ratings at the center:

| Dataset | Records | Size | Key Fields | Purpose |
|---------|---------|------|------------|---------|
| `ratings.csv` | ~26M | 710MB | userId, movieId, rating, timestamp | **Core**: User-movie interactions |
| `ratings_small.csv` | ~100K | 2.4MB | userId, movieId, rating, timestamp | **Development**: Subset for testing |
| `movies_metadata.csv` | ~45K | 34MB | id (tmdbId), title, genres, budget, etc. | **Enrichment**: Movie details |
| `links.csv` | ~45K | 989KB | movieId, imdbId, tmdbId | **Bridge**: ID mapping between systems |
| `links_small.csv` | ~9K | 183KB | movieId, imdbId, tmdbId | **Development**: Subset of mappings |
| `credits.csv` | ~45K | 190MB | id (tmdbId), cast, crew | **Content**: Actor/director info |
| `keywords.csv` | ~46K | 6MB | id (tmdbId), keywords | **Content**: Movie tags/keywords |

### 1.2 Entity Relationships

```
┌────────────────────────────────────────────────────────────────────┐
│                    STAR SCHEMA ARCHITECTURE                        │
└────────────────────────────────────────────────────────────────────┘

                         RATINGS (Fact Table)
                    ┌─────────────────────────┐
                    │ userId        (FK)      │
                    │ movieId       (FK) ────┼──┐
                    │ rating        (Measure)│  │
                    │ timestamp     (Measure)│  │
                    └─────────────────────────┘  │
                                                 │
                    ┌────────────────────────────┘
                    │
                    ↓
              LINKS (Bridge Table)
         ┌──────────────────────────┐
         │ movieId      (PK)        │
         │ imdbId       (External)  │
         │ tmdbId       (FK) ───────┼──┐
         └──────────────────────────┘  │
                                       │
    ┌──────────────────────────────────┼─────────────────────┐
    │                                  │                     │
    ↓                                  ↓                     ↓
MOVIES_METADATA              CREDITS               KEYWORDS
(Dimension Table)       (Dimension Table)    (Dimension Table)
┌──────────────┐         ┌──────────────┐    ┌──────────────┐
│ id (tmdbId)  │         │ id (tmdbId)  │    │ id (tmdbId)  │
│ title        │         │ cast (JSON)  │    │ keywords     │
│ genres (JSON)│         │ crew (JSON)  │    │   (JSON)     │
│ budget       │         └──────────────┘    └──────────────┘
│ revenue      │
│ popularity   │
│ vote_average │
└──────────────┘
```

### 1.3 ID Systems & Mapping Strategy

**Three ID Systems Coexist**:

1. **movieId** (Internal MovieLens ID)
   - Used in: `ratings.csv`, `links.csv`
   - Format: Integer (1, 2, 3, ...)
   - Purpose: Primary key for ratings

2. **tmdbId** (The Movie Database ID)
   - Used in: `movies_metadata.csv`, `credits.csv`, `keywords.csv`, `links.csv`
   - Format: Integer (862, 8844, 15602, ...)
   - Purpose: Primary key for metadata

3. **imdbId** (Internet Movie Database ID)
   - Used in: `movies_metadata.csv`, `links.csv`
   - Format: String (tt0114709, tt0113497, ...)
   - Purpose: External reference

**Mapping Flow**:
```
ratings.movieId → links.movieId → links.tmdbId → movies_metadata.id
                                               → credits.id
                                               → keywords.id
```

### 1.4 Join Strategies

#### Strategy A: Inner Join (Recommended for CF)
```python
# Keep only ratings with complete metadata
data = (ratings
    .merge(links, on='movieId', how='inner')
    .merge(movies_metadata, left_on='tmdbId', right_on='id', how='inner'))

# Result: Clean dataset with verified movies only
# Trade-off: Lose some ratings (~5-10%) for movies without metadata
```

#### Strategy B: Left Join (Preserve All Ratings)
```python
# Keep all ratings, even without metadata
data = (ratings
    .merge(links, on='movieId', how='left')
    .merge(movies_metadata, left_on='tmdbId', right_on='id', how='left'))

# Result: All user interactions preserved
# Trade-off: Need to handle NaN values in metadata fields
```

**Recommendation**: Use **Inner Join** for cleaner collaborative filtering; use **Left Join** if optimizing for coverage.

---

## Data Quality Assessment

### 2.1 Identified Issues

| Issue | Dataset | Description | Impact | Solution |
|-------|---------|-------------|--------|----------|
| **Malformed Rows** | movies_metadata.csv | Some rows have parsing errors in 'id' field | 34 rows affected | Use `dtype={'id': str}` then convert with `pd.to_numeric(errors='coerce')` |
| **Missing IDs** | movies_metadata.csv | Empty or null 'id' or 'title' fields | ~100 rows | Drop with `dropna(subset=['id', 'title'])` |
| **JSON Strings** | movies_metadata, credits, keywords | Columns stored as string-formatted JSON | All metadata | Parse with `ast.literal_eval()` |
| **Duplicate Ratings** | ratings.csv | Same user rates same movie multiple times | <0.1% of ratings | Keep most recent: `drop_duplicates(subset=['userId', 'movieId'], keep='last')` |
| **Missing Links** | links.csv ↔ ratings.csv | Some movieIds in ratings have no metadata | ~5-10% of movies | Handle with join strategy (inner/left) |
| **Sparse Matrix** | ratings.csv | Only 0.02% of user-movie pairs have ratings | 99.98% sparse | **CRITICAL**: Use `scipy.sparse.csr_matrix` |
| **Rating Scale** | ratings.csv | 0.5 to 5.0 in 0.5 increments (10 values) | Affects normalization | Choose normalization based on algorithm |
| **Skewed Distribution** | ratings.csv | More 3.0-4.0 ratings than extremes | Positive skew | Consider in evaluation metrics |

### 2.2 Sparsity Analysis

```
Full Dataset:
- Users: ~270,000
- Movies: ~45,000
- Possible ratings: 270,000 × 45,000 = 12.15 billion
- Actual ratings: 26 million
- Sparsity: 1 - (26M / 12.15B) = 99.79% sparse

Memory Requirements:
- Dense matrix: 12.15B × 8 bytes = 97.2 GB RAM ❌
- Sparse matrix: 26M × 12 bytes = 312 MB RAM ✅
- Compression ratio: 312x smaller
```

**Conclusion**: Sparse matrices are **MANDATORY** for the full dataset, optional for `ratings_small.csv`.

### 2.3 Rating Distribution

```
Value   | Count    | Percentage
--------|----------|------------
0.5     | 1,101    | 1.1%
1.0     | 3,326    | 3.3%
1.5     | 1,687    | 1.7%
2.0     | 7,271    | 7.3%
2.5     | 4,449    | 4.5%
3.0     | 20,064   | 20.1%     ← Peak
3.5     | 10,538   | 10.6%
4.0     | 28,750   | 28.8%     ← Peak
4.5     | 7,723    | 7.7%
5.0     | 15,095   | 15.1%

Mean: 3.52 | Median: 3.5 | Mode: 4.0
Distribution: Positively skewed (more high ratings)
```

---

## Proposed Implementation Approaches

### 3.1 Approach 1: Item-Based Collaborative Filtering (Baseline)

**Rationale**: Start with most stable CF approach
- Item similarities change slowly (movie characteristics don't change)
- Typically fewer items than users (~45K movies vs ~270K users)
- Better for cold-start users (need few ratings to find similar items)
- Industry standard (Amazon, Netflix use item-based)

**Algorithm**:
```
For each target user:
  1. Identify items (movies) the user has rated
  2. For each unrated item:
     a. Find k most similar items user has rated
     b. Compute similarity scores (cosine, Pearson)
     c. Predict rating = weighted average of user's ratings for similar items
  3. Recommend top-N highest predicted items
```

**Similarity Metric**: Cosine Similarity
```
sim(i, j) = (ratings_i · ratings_j) / (||ratings_i|| × ||ratings_j||)
```

**Prediction Formula**:
```
pred(u, i) = Σ(sim(i, j) × rating(u, j)) / Σ|sim(i, j)|
             for j in k-nearest neighbors of i that u has rated
```

**Expected Performance**:
- RMSE: 0.85 - 0.95
- Training Time: 15-30 minutes (on small dataset)
- Coverage: 85-90% of users

### 3.2 Approach 2: User-Based Collaborative Filtering (Comparison)

**Rationale**: Compare with user-based approach for report

**Algorithm**:
```
For each target user:
  1. Find k most similar users (based on rating patterns)
  2. For each unrated item:
     a. Get ratings from similar users
     b. Predict rating = weighted average of similar users' ratings
  3. Recommend top-N highest predicted items
```

**Similarity Metric**: Pearson Correlation (handles user bias)
```
sim(u, v) = Σ((r_ui - μ_u) × (r_vi - μ_v)) / (σ_u × σ_v)
```

**Normalization**: Mean-centering (remove user bias)
```
r'_ui = r_ui - μ_u  (subtract user's mean rating)
```

**Expected Performance**:
- RMSE: 0.88 - 0.98 (usually slightly worse than item-based)
- Training Time: 45-60 minutes (more users than items)
- Coverage: 80-85% of users
- Challenge: Scalability issues with large user base

### 3.3 Approach 3: SVD Matrix Factorization (Model-Based)

**Rationale**: Achieve best accuracy with dimensionality reduction

**Algorithm**: Singular Value Decomposition
```
R ≈ U × Σ × V^T

Where:
- R: user-item rating matrix (sparse)
- U: user latent factor matrix (users × factors)
- Σ: diagonal matrix of singular values
- V^T: item latent factor matrix (factors × items)

Prediction:
pred(u, i) = μ + b_u + b_i + q_i^T × p_u

Where:
- μ: global mean rating
- b_u: user bias
- b_i: item bias
- p_u: user latent factors
- q_i: item latent factors
```

**Implementation**: Use `surprise` library
```python
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate

# Configure
algo = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02)

# Train with cross-validation
results = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5)
```

**Hyperparameters to Tune**:
- `n_factors`: 50, 100, 150, 200 (latent dimensions)
- `n_epochs`: 20, 30, 40 (training iterations)
- `lr_all`: 0.002, 0.005, 0.01 (learning rate)
- `reg_all`: 0.02, 0.1, 0.4 (regularization)

**Expected Performance**:
- RMSE: 0.80 - 0.85 (best accuracy)
- Training Time: 8-15 minutes
- Coverage: 95%+ (can predict for all user-item pairs)

### 3.4 Approach 4: Hybrid System (Advanced - For Improvements)

**Rationale**: Combine collaborative + content-based for better recommendations

**Strategy**: Weighted Hybrid
```python
# Extract content features
content_features = extract_features(movies_metadata, credits, keywords)
# genres, actors, directors, keywords

# Compute scores
cf_score = collaborative_filtering_score(user, item)
content_score = content_based_score(user, item, content_features)

# Weighted combination
final_score = 0.7 × cf_score + 0.3 × content_score

# Adjust weights based on context
if user_is_new or item_is_new:
    final_score = 0.3 × cf_score + 0.7 × content_score  # Favor content
```

**Benefits**:
- Addresses cold-start problem
- Increases diversity (prevents filter bubble)
- More robust to sparse data

**Expected Performance**:
- RMSE: 0.82 - 0.88
- Coverage: 98%+ (works for new users/items)
- Diversity: Higher than pure CF

---

## Expected Results & Metrics

### 4.1 Performance Benchmarks

Based on similar implementations on MovieLens dataset:

| Metric | Item-Based CF | User-Based CF | SVD | Hybrid |
|--------|--------------|---------------|-----|--------|
| **RMSE** | 0.85 - 0.95 | 0.88 - 0.98 | 0.80 - 0.85 | 0.82 - 0.88 |
| **MAE** | 0.68 - 0.75 | 0.70 - 0.78 | 0.65 - 0.70 | 0.66 - 0.72 |
| **Precision@10** | 0.25 - 0.35 | 0.22 - 0.32 | 0.30 - 0.40 | 0.32 - 0.42 |
| **Recall@10** | 0.15 - 0.25 | 0.12 - 0.22 | 0.18 - 0.28 | 0.20 - 0.30 |
| **Coverage** | 85-90% | 80-85% | 95%+ | 98%+ |
| **Training Time** | 15-30 min | 45-60 min | 8-15 min | 20-35 min |

### 5.2 Evaluation Metrics Explained

#### Accuracy Metrics

**RMSE (Root Mean Squared Error)**:
```
RMSE = sqrt(Σ(actual - predicted)² / n)
```
- Penalizes large errors more heavily
- Range: 0 to ∞ (lower is better)
- Target: < 0.90 for good performance

**MAE (Mean Absolute Error)**:
```
MAE = Σ|actual - predicted| / n
```
- Average absolute difference
- More robust to outliers than RMSE
- Target: < 0.70 for good performance

#### Ranking Metrics

**Precision@K**:
```
Precision@K = (# relevant items in top K) / K
```
- What fraction of recommendations are relevant?
- Example: If 3 out of 10 recommendations are watched → Precision@10 = 0.3

**Recall@K**:
```
Recall@K = (# relevant items in top K) / (total # relevant items)
```
- What fraction of relevant items are recommended?
- Example: If user likes 20 movies total, and 4 are in top-10 → Recall@10 = 0.2

**Coverage**:
```
Coverage = (# users who receive recommendations) / (total # users)
```
- What percentage of users can get recommendations?
- Important for production systems

### 4.2 Success Criteria

**Minimum Requirements**:
- ✅ RMSE < 1.0
- ✅ MAE < 0.8
- ✅ Coverage > 80%
- ✅ Implement at least 2 algorithms
- ✅ Comprehensive report with visualizations

**Target Performance**:
- 🎯 RMSE < 0.90
- 🎯 MAE < 0.70
- 🎯 Precision@10 > 0.25
- 🎯 Coverage > 90%
- 🎯 Compare 3+ algorithms

**Stretch Goals**:
- 🌟 RMSE < 0.85 (competitive with research)
- 🌟 Implement hybrid system
- 🌟 Address cold-start problem
- 🌟 Add diversity metrics

--
## Key Findings Summary

### Dataset Characteristics
1. **Highly Sparse**: 99.98% of user-movie pairs have no rating
   - **Implication**: Sparse matrices are mandatory for full dataset

2. **Three ID Systems**: movieId, tmdbId, imdbId require careful mapping
   - **Implication**: Links table is crucial bridge; inner joins recommended

3. **Positively Skewed Ratings**: More high ratings (3-4) than low
   - **Implication**: Algorithms may overpredict; consider bias correction

4. **JSON-Heavy Metadata**: Genres, cast, keywords stored as strings
   - **Implication**: Parsing required before use; slow without optimization


