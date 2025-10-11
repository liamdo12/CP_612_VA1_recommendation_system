# Movie Recommendation System - Analysis & Implementation Plan

## 1. Dataset Relationship Analysis

### 1.1 Dataset Overview

| Dataset | Records | Size | Key Fields | Purpose |
|---------|---------|------|------------|---------|
| ratings.csv | 26M | 710MB | userId, movieId, rating, timestamp | Core: User-movie interactions |
| ratings_small.csv | 100K | 2.4MB | userId, movieId, rating, timestamp | Development: Subset for testing |
| movies_metadata.csv | 45K | 34MB | id (tmdbId), title, genres, budget | Enrichment: Movie details |
| links.csv | 45K | 989KB | movieId, imdbId, tmdbId | Bridge: ID mapping |
| credits.csv | 45K | 190MB | id (tmdbId), cast, crew | Content: Actor/director info |
| keywords.csv | 46K | 6MB | id (tmdbId), keywords | Content: Movie tags |

### 1.2 Star Schema Architecture

RATINGS is the central fact table containing user-movie interactions.

LINKS acts as a bridge table connecting MovieLens IDs to TMDB IDs.

MOVIES_METADATA, CREDITS, and KEYWORDS are dimension tables providing movie information.

ID Mapping Flow:
- ratings.movieId connects to links.movieId
- links.tmdbId connects to movies_metadata.id
- links.tmdbId connects to credits.id and keywords.id

### 1.3 Data Quality Issues Summary

| Issue | Dataset | Impact | Solution |
|-------|---------|--------|----------|
| Malformed id field | movies_metadata | 34 rows | Convert to numeric, drop errors |
| Missing IDs/titles | movies_metadata | 100 rows | Drop rows with missing values |
| JSON strings | All metadata | Parsing required | Parse JSON strings to extract fields |
| Duplicate ratings | ratings | Less than 0.1% | Keep most recent rating |
| Missing movieIds | links and ratings | 5-10% loss | Use inner join strategy |
| Sparsity | ratings | 99.98% sparse | Use sparse matrices |
| Positive skew | ratings | Mean=3.52, Mode=4.0 | No transformation needed for CF |

---

## 2. Two-Phase Implementation Plan

### PHASE 1: Data Preparation and Cleaning

#### Step 1.1: Handle Missing Values

movies_metadata.csv:
- Load dataset with correct data types
- Convert id field to numeric, handling malformed rows
- Drop rows with missing id or title fields
- Convert id to integer type

links.csv:
- Check for missing values in key fields
- Drop any rows with missing movieId or tmdbId

ratings.csv:
- Verify no missing values exist
- Check for duplicate user-movie pairs
- Keep most recent rating if duplicates exist

#### Step 1.2: Handle Skewness and Outliers in Ratings

Question: Do we need to handle skewness?

Answer: NO transformation needed for Collaborative Filtering

Rationale:
1. Rating scale is intentional: 0.5 to 5.0 represents actual user preferences
2. Skewness is natural: Users tend to rate movies they like, causing positive skew
3. CF algorithms use raw ratings: Transforming ratings would distort user preferences
4. Outlier detection not applicable: All ratings from 0.5 to 5.0 are valid

What TO DO instead:
- Document the distribution for report: mean, median, standard deviation
- Visualize for exploratory data analysis
- Use raw ratings for collaborative filtering
- Normalization (if needed) comes later based on algorithm choice

When skewness DOES matter:
- If doing content-based filtering with movie features like budget or revenue
  - These need log transformation
- If using neural networks that expect normalized inputs
  - Then apply min-max scaling

#### Step 1.3: Parse JSON Columns

- Create function to safely parse JSON strings
- Extract genre names from genres field
- Extract keyword names from keywords field (optional, for hybrid models)
- Store extracted values as lists

#### Step 1.4: Merge Datasets

Strategy: Inner join to keep only complete data
- Merge ratings with links on movieId
- Merge result with movies_metadata on tmdbId
- Select relevant columns: id, title, genres_list
- Track coverage: percentage of original ratings retained
- Save cleaned dataset

#### Step 1.5: Train/Test Split

Method 1: Random split (80/20)
- Split data randomly into training and test sets

Method 2: Temporal split (more realistic)
- Sort data by timestamp
- Split chronologically: first 80% for training, last 20% for testing
- This simulates predicting future ratings

---

### PHASE 2: Model Implementation and Evaluation

#### Step 2.1: Baseline - Item-Based Collaborative Filtering

Algorithm:
For each target user:
1. Create user-item matrix
2. Compute item-item similarity using cosine similarity
3. For each unrated item:
   - Find k similar items user has rated
   - Predict rating as weighted average
4. Recommend top-N items

Key Parameters:
- Similarity metric: Cosine similarity
- Number of neighbors k: 30
- Use sparse matrices for efficiency

Expected Performance: RMSE 0.85 to 0.95

#### Step 2.2: Comparison - User-Based Collaborative Filtering

Key Difference: Find similar users instead of similar items

Algorithm:
For each target user:
1. Compute user-user similarity using Pearson correlation
2. Apply mean-centering to remove user bias
3. For each unrated item:
   - Get ratings from k similar users
   - Predict rating as weighted average
4. Recommend top-N items

Key Parameters:
- Similarity metric: Pearson correlation
- Number of neighbors k: 50
- Apply mean-centering normalization

Normalization Needed:
- Subtract each user's mean rating to remove bias
- This handles users who rate consistently high or low

Expected Performance: RMSE 0.88 to 0.98

#### Step 2.3: Advanced - SVD Matrix Factorization

Method: Singular Value Decomposition for dimensionality reduction

Using Surprise Library:
- Prepare data in required format
- Define hyperparameter grid for tuning
- Perform grid search with cross-validation
- Train final model with best parameters

Hyperparameters to Tune:
- Number of latent factors: 50, 100, 150
- Number of epochs: 20, 30
- Learning rate: 0.005, 0.01
- Regularization: 0.02, 0.1

Expected Performance: RMSE 0.80 to 0.85 (best accuracy)

#### Step 2.4: Evaluation and Comparison

Evaluate all models:
- Calculate RMSE and MAE for each algorithm
- Create comparison table
- Generate visualizations comparing performance
- Analyze strengths and weaknesses

Deliverable: Algorithm comparison chart showing RMSE/MAE

---

## 3. Proposed Algorithms

### 3.1 Item-Based CF (Baseline)
- Similarity: Cosine similarity
- Prediction: Weighted average of k=30 neighbors
- Pros: Stable, works well for cold-start users
- Cons: Slower than SVD

### 3.2 User-Based CF (Comparison)
- Similarity: Pearson correlation with mean-centering
- Prediction: Weighted average of k=50 neighbors
- Pros: Intuitive, easy to explain
- Cons: Scalability issues, less stable

### 3.3 SVD (Best Accuracy)
- Method: Matrix factorization with latent factors
- Optimization: Gradient descent with regularization
- Pros: Best RMSE, handles sparsity well
- Cons: Black box, harder to interpret

---

## 4. Evaluation Metrics

### Accuracy Metrics

RMSE (Root Mean Squared Error):
- Formula: Square root of average squared differences
- Penalizes large errors more heavily
- Range: 0 to infinity (lower is better)
- Target: Less than 0.90 for good performance

MAE (Mean Absolute Error):
- Formula: Average of absolute differences
- Average absolute error between actual and predicted
- More robust to outliers than RMSE
- Target: Less than 0.70 for good performance

### Ranking Metrics

Precision at 10:
- Formula: Number of relevant items in top-10 divided by 10
- What fraction of recommendations are relevant?
- Example: If 3 out of 10 recommendations are watched, Precision at 10 equals 0.3
- Target: Greater than 0.25

Recall at 10:
- Formula: Number of relevant items in top-10 divided by total relevant items
- What fraction of relevant items are recommended?
- Example: If user likes 20 movies total, and 4 are in top-10, Recall at 10 equals 0.2
- Target: Greater than 0.15

Coverage:
- Formula: Number of users with recommendations divided by total users
- What percentage of users can get recommendations?
- Important for production systems
- Target: Greater than 90%

---

## 5. Expected Results

| Metric | Item-Based CF | User-Based CF | SVD | Target |
|--------|--------------|---------------|-----|--------|
| RMSE | 0.85 to 0.95 | 0.88 to 0.98 | 0.80 to 0.85 | Less than 0.90 |
| MAE | 0.68 to 0.75 | 0.70 to 0.78 | 0.65 to 0.70 | Less than 0.70 |
| Training Time | 15-30 min | 45-60 min | 8-15 min | - |
| Coverage | 85-90% | 80-85% | 95% or more | Greater than 90% |

---

## 6. Key Findings

### Dataset Characteristics
1. 99.98% sparse - Sparse matrices mandatory
2. Three ID systems - Use links table as bridge
3. Positive skew in ratings - Natural, do not transform
4. JSON metadata - Parse to extract fields

### Critical Decisions

1. Missing Values:
- Drop malformed rows in movies_metadata
- Keep most recent duplicate ratings
- Use inner join for clean data

2. Skewness:
- Do not transform ratings (distorts preferences)
- Document distribution in report
- Apply normalization per algorithm if needed

3. Outliers:
- No outlier removal (all ratings 0.5 to 5.0 are valid)
- Use robust metrics (MAE alongside RMSE)

4. Feature Engineering:
- Only for hybrid models (Phase 2 extension)
- Parse genres, keywords for content-based scoring

---

## 7. Success Criteria

Minimum Requirements:
- RMSE less than 1.0
- MAE less than 0.8
- Implement 2 or more algorithms
- Complete report with visualizations

Target Performance:
- RMSE less than 0.90
- MAE less than 0.70
- Compare 3 algorithms
- Coverage greater than 90%

Stretch Goals:
- RMSE less than 0.85 (competitive with research)
- Implement hybrid system
- Address cold-start problem

---

