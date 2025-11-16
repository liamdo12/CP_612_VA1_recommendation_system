# Phase 2: Model Implementation & Evaluation

This directory contains the implementation of three collaborative filtering algorithms for the movie recommendation system.

## 📁 Directory Structure

```
3_model_implementation/
├── README.md                              # This file
├── 1_item_based_cf.ipynb                  # Item-Based Collaborative Filtering
├── 2_user_based_cf.ipynb                  # User-Based Collaborative Filtering
├── 3_svd_matrix_factorization.ipynb       # SVD Matrix Factorization
└── 4_evaluation_comparison.ipynb          # Algorithm Comparison & Analysis
```

## 🚀 Quick Start

### 1. Install Dependencies

From the project root directory:

```bash
pip install -r requirements.txt
```

### 2. Run Notebooks in Order

Execute the notebooks sequentially:

1. **1_item_based_cf.ipynb** - Baseline algorithm
2. **2_user_based_cf.ipynb** - Comparison algorithm
3. **3_svd_matrix_factorization.ipynb** - Advanced algorithm
4. **4_evaluation_comparison.ipynb** - Compare all three

### 3. Expected Runtime

- **Item-Based CF**: 30-60 minutes (similarity computation is slow)
- **User-Based CF**: 45-90 minutes (more users than items)
- **SVD**: 15-30 minutes (grid search for hyperparameters)
- **Comparison**: 5 minutes (generates visualizations)

**Note**: Times are for 100K test sample. Full test set (5M ratings) will take 5-10x longer.

## 📊 Algorithms Implemented

### 1. Item-Based Collaborative Filtering

**File**: `1_item_based_cf.ipynb`

**Concept**: Recommend items based on similarity between items (not users).

**Key Features**:
- Cosine similarity between item rating vectors
- k=30 nearest neighbors
- Sparse matrix implementation for memory efficiency
- Handles cold-start with global mean rating

**Expected Performance**:
- RMSE: 0.85-0.95
- MAE: 0.68-0.75
- Coverage: 85-90%

**Best For**:
- E-commerce recommendations
- Stable item catalogs
- Interpretable recommendations

---

### 2. User-Based Collaborative Filtering

**File**: `2_user_based_cf.ipynb`

**Concept**: Users with similar rating patterns have similar preferences.

**Key Features**:
- Mean-centering to remove user rating bias
- Cosine similarity on centered ratings (approximates Pearson correlation)
- k=50 nearest neighbors
- Denormalization after prediction

**Expected Performance**:
- RMSE: 0.88-0.98
- MAE: 0.70-0.78
- Coverage: 80-85%

**Best For**:
- Social recommendations
- Small to medium user bases
- "People like you" style recommendations

---

### 3. SVD Matrix Factorization

**File**: `3_svd_matrix_factorization.ipynb`

**Concept**: Decompose rating matrix into user/item latent factor matrices.

**Key Features**:
- Model-based approach (learns patterns)
- Hyperparameter tuning with grid search
- Fast predictions (dot product)
- 100% coverage (can predict any user-item pair)

**Hyperparameters Tuned**:
- n_factors: [50, 100, 150]
- n_epochs: [20, 30]
- learning_rate: [0.005, 0.01]
- regularization: [0.02, 0.1]

**Expected Performance**:
- RMSE: 0.80-0.85 (best accuracy)
- MAE: 0.65-0.70
- Coverage: 95-100%

**Best For**:
- Production systems (Netflix, Spotify)
- Large-scale applications
- Accuracy is priority

---

### 4. Evaluation & Comparison

**File**: `4_evaluation_comparison.ipynb`

**Purpose**: Compare all three algorithms across multiple dimensions.

**Metrics Evaluated**:
- **Accuracy**: RMSE, MAE
- **Speed**: Training time, prediction time
- **Coverage**: Percentage of testable ratings
- **Correlation**: Actual vs predicted ratings

**Visualizations Generated**:
- Accuracy comparison (bar charts)
- Speed comparison (bar charts)
- Coverage comparison (bar chart)
- Radar chart (multi-dimensional comparison)

**Outputs**:
- `final_algorithm_comparison.csv` - Complete results table
- `algorithm_comparison_summary.txt` - Text summary
- `comparison_accuracy.png` - Accuracy visualization
- `comparison_speed.png` - Speed visualization
- `comparison_coverage.png` - Coverage visualization
- `comparison_radar.png` - Radar chart

## 📈 Expected Results Summary

| Algorithm | RMSE | MAE | Training Time | Prediction Time | Coverage |
|-----------|------|-----|---------------|-----------------|----------|
| Item-Based CF | 0.85-0.95 | 0.68-0.75 | 15-30 min | Medium | 85-90% |
| User-Based CF | 0.88-0.98 | 0.70-0.78 | 45-60 min | Slow | 80-85% |
| SVD | 0.80-0.85 | 0.65-0.70 | 8-15 min | Fast | 95-100% |

**Winner**: SVD typically achieves best RMSE/MAE

## 🔧 Technical Details

### Data Source

All notebooks use the temporal split data:
- **Train**: `datasets/output/split_and_train_datasets/temporal_split/train_ratings.csv`
- **Test**: `datasets/output/split_and_train_datasets/temporal_split/test_ratings.csv`

### Sampling Strategy

For faster development, notebooks use 100K test samples by default:

```python
SAMPLE_SIZE = 100000  # Change to None for full evaluation
```

To evaluate on full test set (5.2M ratings):
- Set `SAMPLE_SIZE = None` in each notebook
- Expect 5-10x longer runtime

### Memory Considerations

**Item-Based CF**:
- Item similarity matrix: ~25K × 25K = 625M entries
- Using sparse matrices: ~500 MB
- Dense matrices would require ~5 GB (not recommended)

**User-Based CF**:
- User similarity matrix: ~227K × 227K = 51B entries
- Using sparse matrices: ~5 GB
- Dense matrices would require ~400 GB (impossible)

**SVD**:
- Latent factor matrices: Users × k + Items × k
- Example: 227K × 100 + 25K × 100 = ~25M floats = 100 MB
- Very memory efficient!

## 🎯 Success Criteria

### Minimum Requirements (Pass)
- ✅ RMSE < 1.0
- ✅ MAE < 0.8
- ✅ Implement 2+ algorithms
- ✅ Complete evaluation report

### Target Performance (Good)
- ✅ RMSE < 0.90
- ✅ MAE < 0.70
- ✅ Implement 3 algorithms
- ✅ Coverage > 90%

### Stretch Goals (Excellent)
- ✅ RMSE < 0.85 (competitive with research)
- ✅ Hybrid system (CF + content-based)
- ✅ Address cold-start problem
- ✅ Production-ready deployment

## 🐛 Troubleshooting

### Memory Errors

**Symptom**: `MemoryError` or system freeze during similarity computation

**Solutions**:
1. Use smaller test sample: `SAMPLE_SIZE = 10000`
2. Reduce k neighbors: `k = 10` instead of `k = 30`
3. Use sparse matrices (already implemented)
4. Close other applications

### Slow Predictions

**Symptom**: Predictions taking > 1 hour

**Solutions**:
1. Reduce test sample size
2. Use SVD instead (much faster predictions)
3. Precompute similarities (for memory-based methods)
4. Use approximate nearest neighbor methods (Annoy, FAISS)

### Poor Accuracy

**Symptom**: RMSE > 1.0 or MAE > 0.8

**Possible Causes**:
1. Data not cleaned properly (check Phase 1)
2. Wrong hyperparameters (tune with grid search)
3. Cold-start users/items (expected for temporal split)
4. Not using mean-centering (for user-based CF)

**Solutions**:
1. Verify cleaned data exists
2. Run hyperparameter tuning
3. Use global/user mean for cold-start
4. Apply normalization

## 📚 Key Learnings

### When to Use Each Algorithm

**Item-Based CF**:
- ✅ Need interpretable recommendations
- ✅ Stable item catalog
- ✅ Cold-start users common
- ❌ Don't use for: New items, very large catalogs

**User-Based CF**:
- ✅ Social proof important
- ✅ Small user base (< 100K)
- ✅ Discovering new genres
- ❌ Don't use for: Large user bases, changing preferences

**SVD**:
- ✅ Accuracy is priority
- ✅ Large-scale data
- ✅ Fast real-time predictions
- ❌ Don't use for: Interpretability critical, small datasets

### Production Recommendations

For a real-world movie recommendation system:

1. **Primary**: SVD for accuracy and speed
2. **Cold-start users**: Item-Based CF (popular items in liked genres)
3. **Cold-start items**: Content-based (genre, actors, keywords)
4. **Hybrid**: Combine all three with weighted blending

## 🔄 Next Steps

After completing these notebooks:

1. **Analyze Results**: Review comparison notebook findings
2. **Report Writing**: Use visualizations and metrics in your report
3. **Optional Improvements**:
   - Implement hybrid recommender (CF + content-based)
   - Add temporal dynamics (time-aware recommendations)
   - Optimize for production (caching, approximate search)
   - Address diversity/serendipity

## 📖 References

### Course Materials
- Module 2: Collaborative Filtering
- Module 3: Content-Based Filtering
- Module 4: Hybrid Recommender Systems

### Libraries Used
- `pandas` - Data manipulation
- `numpy` - Numerical computations
- `scipy` - Sparse matrices, similarity metrics
- `scikit-learn` - Cosine similarity, evaluation metrics
- `scikit-surprise` - SVD implementation, CF algorithms
- `matplotlib`/`seaborn` - Visualizations

### Research Papers
- MovieLens dataset: Harper & Konstan (2015)
- Item-Based CF: Sarwar et al. (2001)
- Matrix Factorization: Koren et al. (2009)

## 💡 Tips

1. **Start Small**: Use `SAMPLE_SIZE = 10000` for initial testing
2. **Iterate**: Run notebooks multiple times with different parameters
3. **Save Results**: Each notebook saves results CSV for comparison
4. **Use Visualizations**: Include plots in your report
5. **Explain Trade-offs**: Discuss accuracy vs speed vs interpretability

## 📝 Report Structure Suggestion

Use this structure for your project report:

1. **Introduction**
   - Problem statement
   - Dataset description
   - Objectives

2. **Data Preparation** (Phase 1)
   - Cleaning process
   - Train/test split strategy
   - Data statistics

3. **Methodology** (Phase 2)
   - Algorithm selection
   - Implementation details
   - Hyperparameter tuning

4. **Results**
   - Performance metrics (use comparison table)
   - Visualizations (use generated plots)
   - Algorithm comparison

5. **Discussion**
   - Best performing algorithm
   - Trade-offs observed
   - Cold-start challenges

6. **Improvement Opportunities**
   - Hybrid systems
   - Deep learning approaches
   - Scalability considerations

7. **Conclusion**
   - Summary of findings
   - Recommendations
   - Future work

---

## ❓ Frequently Asked Questions (FAQ)

### Q1: What are RMSE and MAE? How do I interpret them?

#### RMSE (Root Mean Squared Error)

**Formula**:
```
RMSE = √(Σ(actual - predicted)² / n)
```

**What it means**:
- Measures the average magnitude of prediction errors
- Units: Same as ratings (0.5 to 5.0 in your case)
- Interpretation: "On average, my predictions are off by X rating points"

**Example**:
```python
Actual ratings:    [4.0, 3.5, 5.0, 2.0, 4.5]
Predicted ratings: [3.8, 3.7, 4.5, 2.5, 4.2]
Errors:            [0.2, 0.2, 0.5, 0.5, 0.3]
Squared errors:    [0.04, 0.04, 0.25, 0.25, 0.09]
Mean squared:      0.134
RMSE:              √0.134 = 0.366
```
→ Predictions are off by ~0.37 rating points on average.

**Key Characteristics**:
- **Penalizes large errors heavily** (due to squaring)
- Error of 2.0 → squared error of 4.0 (4x worse than error of 1.0)
- **Sensitive to outliers**: A few bad predictions hurt RMSE significantly

**Benchmarks for MovieLens (0.5 to 5.0 scale)**:
- **RMSE < 1.0**: Minimum acceptable (better than random)
- **RMSE < 0.90**: Good performance ✅ **YOUR TARGET**
- **RMSE < 0.85**: Excellent performance (competitive with research)
- **RMSE < 0.80**: State-of-the-art (Netflix Prize winner: 0.8567)

---

#### MAE (Mean Absolute Error)

**Formula**:
```
MAE = Σ|actual - predicted| / n
```

**What it means**:
- Average absolute difference between actual and predicted ratings
- Units: Same as ratings (0.5 to 5.0)
- Interpretation: "On average, my predictions are X points away from actual ratings"

**Example** (using same data):
```python
Actual ratings:    [4.0, 3.5, 5.0, 2.0, 4.5]
Predicted ratings: [3.8, 3.7, 4.5, 2.5, 4.2]
Absolute errors:   [0.2, 0.2, 0.5, 0.5, 0.3]
MAE:               (0.2 + 0.2 + 0.5 + 0.5 + 0.3) / 5 = 0.34
```
→ Predictions are off by 0.34 rating points on average.

**Key Characteristics**:
- **Treats all errors equally** (no squaring)
- **More robust to outliers** than RMSE
- **Easier to interpret** (simple average of errors)

**Benchmarks for MovieLens**:
- **MAE < 0.80**: Minimum acceptable
- **MAE < 0.70**: Good performance ✅ **YOUR TARGET**
- **MAE < 0.65**: Excellent performance
- **MAE < 0.60**: State-of-the-art

---

#### RMSE vs MAE: Key Differences

| Aspect | RMSE | MAE |
|--------|------|-----|
| **Calculation** | Square errors → mean → square root | Absolute errors → mean |
| **Sensitivity** | Penalizes large errors heavily | Treats all errors equally |
| **Outliers** | Very sensitive | More robust |
| **Interpretation** | "Standard deviation of errors" | "Average error" |
| **Use When** | Avoiding big mistakes is critical | Want overall average accuracy |

**Why Use Both?**
- **RMSE** tells you: "How bad are my worst predictions?" (catches outliers)
- **MAE** tells you: "What's my typical error?" (average accuracy)

**Example Comparison**:
```
Model A: RMSE = 0.90, MAE = 0.70
→ Slightly higher average error, but fewer large mistakes

Model B: RMSE = 1.10, MAE = 0.68
→ Better average accuracy, but has some outlier predictions

Choose Model A if avoiding bad recommendations is critical.
```

---

#### How to Interpret Your Results

**Example: RMSE = 0.87, MAE = 0.69**

**Interpretation**:
1. **MAE = 0.69**: On average, predictions are **0.69 rating points** away from actual
   - If actual = 4.0, you typically predict between 3.3 and 4.7

2. **RMSE = 0.87**: "Standard deviation" of errors is 0.87
   - ~68% of predictions are within ±0.87 of actual rating
   - ~95% are within ±1.74 of actual rating

3. **RMSE > MAE**: You have some larger errors pulling RMSE up
   - If RMSE = MAE, all errors would be identical
   - If RMSE >> MAE, you have significant outliers

**Is this good?**
- ✅ **YES!** RMSE < 0.90 meets "good performance" target
- ✅ MAE < 0.70 also meets target
- ✅ Close to Netflix Prize performance
- ✅ Sufficient for production recommendation system

---

#### Real-World Context

**Netflix Prize Competition (2006-2009)**:
- **Baseline**: Netflix's Cinematch algorithm had RMSE = 0.9525
- **Winning Team**: Achieved RMSE = 0.8567 (10% improvement = $1M prize)
- **Your target**: RMSE < 0.90 is competitive!

**Practical Example**:

Scenario 1: **Low RMSE/MAE (0.70)**
```
Actual:    [5.0, 4.5, 3.0, 4.0, 2.0]
Predicted: [4.7, 4.2, 3.3, 4.1, 2.4]
```
→ ✅ Good! User will be satisfied with recommendations.

Scenario 2: **High RMSE/MAE (1.5)**
```
Actual:    [5.0, 4.5, 3.0, 4.0, 2.0]
Predicted: [3.5, 3.0, 4.5, 2.5, 3.5]
```
→ ❌ Bad! User sees movies they'll hate → dissatisfaction.

---

#### For Your Report

**How to present metrics**:

```markdown
## Results

| Algorithm      | RMSE   | MAE    | Interpretation |
|----------------|--------|--------|----------------|
| Item-Based CF  | 0.8934 | 0.7123 | Predictions typically off by 0.71 points |
| User-Based CF  | 0.9245 | 0.7456 | Predictions typically off by 0.75 points |
| SVD            | 0.8312 | 0.6789 | Predictions typically off by 0.68 points |

**Analysis**: SVD achieved the best performance with RMSE = 0.83, indicating
predictions are typically within 0.83 rating points of actual ratings. This
exceeds our target of RMSE < 0.90 and is competitive with state-of-the-art
research. The lower MAE (0.68) confirms most predictions are accurate, with
RMSE being slightly higher due to occasional larger errors.
```

---

### Q2: Why use Temporal Split instead of Random (80-20) Split?

**Short Answer**: Temporal split is more realistic for evaluating recommendation systems because it simulates real-world deployment—predicting future ratings based on past behavior.

---

#### Detailed Comparison

**Random Split (80-20)**

**How it works**:
```
All ratings shuffled randomly
├── 80% → Training set
└── 20% → Test set
```

**Example**:
```
User 123's ratings (chronological order):
  Jan 2010: Movie A → 4.0 [TRAIN]
  Mar 2012: Movie B → 5.0 [TEST]
  Jun 2014: Movie C → 3.0 [TRAIN]
  Sep 2016: Movie D → 4.5 [TEST]
```

**Problem**: You're using **future ratings** (2014, 2016) to predict **past ratings** (2012)!
- ❌ In real life, you can't know the future to predict the past
- ❌ This artificially inflates your accuracy
- ❌ **Data leakage**: Training data contains information from after test data

---

**Temporal Split (Time-Based)**

**How it works**:
```
Sort all ratings by timestamp
├── First 80% (oldest) → Training set
└── Last 20% (newest) → Test set
```

**Example**:
```
User 123's ratings (chronological order):
  Jan 2010: Movie A → 4.0 [TRAIN]
  Mar 2012: Movie B → 5.0 [TRAIN]
  Jun 2014: Movie C → 3.0 [TRAIN]
  Sep 2016: Movie D → 4.5 [TEST]
```

**Advantage**: You're using **past ratings** (2010-2014) to predict **future ratings** (2016)!
- ✅ This is exactly how production systems work
- ✅ More realistic evaluation
- ✅ No data leakage

---

#### Key Differences

| Aspect | Random Split (80-20) | Temporal Split |
|--------|---------------------|----------------|
| **Realism** | ❌ Unrealistic (uses future to predict past) | ✅ Realistic (predicts future from past) |
| **Accuracy** | ⬆️ Higher (easier problem) | ⬇️ Lower (harder, more realistic) |
| **Cold-Start** | Few new users/items | Many new users/items (90%!) |
| **Production Similarity** | Low | High |
| **Data Leakage** | Risk of temporal leakage | No leakage |
| **Use Case** | Quick testing, academic | Real-world evaluation |

---

#### Your Dataset Statistics

**Random Split (80-20)**:
```
Train: 20.8M ratings, 269.7K users, 43.3K movies
Test:  5.2M ratings, 253.1K users, 31.6K movies

User overlap: ~94% (most test users are in training)
Movie overlap: ~73%
```
→ **Easy**: Most users/movies in test were seen during training!

**Temporal Split**:
```
Train: 20.8M ratings (1995-2015), 227.2K users, 25.6K movies
Test:  5.2M ratings (2015-2017), 48.5K users, 42.7K movies

Cold-start users: 90% (new users in test!)
Cold-start movies: 46% (new movies in test!)
```
→ **Hard**: Many test users/movies weren't in training - this is the **cold-start problem**!

---

#### Real-World Example: Netflix

**Scenario**: It's December 2015. You want to predict what users will watch in 2016.

**Wrong Approach (Random Split)**:
```python
# Train on random 80% of ALL data (including 2016 ratings)
# Test on remaining 20%

Problem: You're using 2016 data to predict 2016 data!
```
→ **Data leakage** - you're "cheating" by seeing the future.

**Correct Approach (Temporal Split)**:
```python
# Train on 1995-2015 data only
# Test on 2016 data

Reality: This is what actually happens in production.
```
→ You only know the past when making predictions.

---

#### Expected Performance Difference

**Random Split** (easier):
```
Item-Based CF:  RMSE ~0.82-0.88
User-Based CF:  RMSE ~0.85-0.92
SVD:            RMSE ~0.75-0.82
```

**Temporal Split** (harder):
```
Item-Based CF:  RMSE ~0.88-0.95
User-Based CF:  RMSE ~0.92-0.98
SVD:            RMSE ~0.82-0.88
```

**Difference**: ~0.05-0.10 points higher RMSE with temporal split.

**Why?** Cold-start users/movies are harder to predict!

---

#### Why Temporal Split is Better for Your Project

1. **Academic Rigor**: Shows you understand real-world ML evaluation and data leakage
2. **Realistic Performance**: RMSE estimates closer to production performance
3. **Cold-Start Analysis**: Can analyze how system handles new users (90%) and new movies (46%)
4. **Industry Standard**: How Netflix, Spotify, Amazon evaluate systems
5. **Your analysis.md recommended it**: "More realistic for predicting future ratings"

---

#### When to Use Each Split

**Use Random Split (80-20) When**:
- ✅ Quick prototyping / initial testing
- ✅ Comparing algorithms on equal footing
- ✅ Temporal information not important
- ✅ You want higher baseline accuracy

**Use Temporal Split When**:
- ✅ Evaluating production readiness
- ✅ Time-series data (ratings have timestamps)
- ✅ Simulating real-world deployment
- ✅ Analyzing cold-start problem
- ✅ Publishing research (more credible)

---

#### For Your Report

**Methodology Section**:

> **Train/Test Split Strategy**
>
> We employed a **temporal split** approach, where the training set comprises
> ratings from 1995-2015 and the test set contains ratings from 2015-2017.
> This method was chosen over random splitting for the following reasons:
>
> 1. **Realistic Evaluation**: Simulates production deployment where future
>    ratings must be predicted from historical data
> 2. **Temporal Validity**: Avoids data leakage by ensuring training data
>    precedes test data chronologically
> 3. **Cold-Start Analysis**: Reveals system performance with new users (90%
>    in test set) and new movies (46% in test set)
>
> While random splitting typically yields 5-10% better RMSE due to higher
> user/movie overlap, temporal splitting provides a more conservative and
> realistic performance estimate suitable for production deployment evaluation.

**When Asked "Why temporal split?"**:

> "We chose temporal split over random split because it simulates real-world
> deployment, where recommendations must be based on historical data to predict
> future behavior. This approach revealed significant cold-start challenges—90%
> of test users were new—which is typical in production systems. While this
> resulted in slightly higher RMSE (~0.85 vs ~0.80), it provides a more honest
> evaluation of our system's production readiness."

---

### Q3: Why These 3 Algorithms? What About Module 5 (Link Analysis)?

#### Algorithm Selection Rationale

**The 3 algorithms chosen cover the required course material (Modules 2-4)**:

1. **Item-Based CF** (Module 2) - Memory-based collaborative filtering
2. **User-Based CF** (Module 2) - Memory-based collaborative filtering (comparison)
3. **SVD** (Module 2-3) - Model-based collaborative filtering

**Coverage**:
- ✅ **Module 2**: Collaborative Filtering (PRIMARY FOCUS) - All 3 algorithms
- ✅ **Module 3**: Content-Based Filtering - Optional for hybrid (mentioned in improvements)
- ✅ **Module 4**: Hybrid Systems - Optional enhancement (mentioned in improvements)
- ❓ **Module 5**: Link Analysis / PageRank - **NOT REQUIRED**

---

#### What is Module 5 (Link Analysis)?

**From course materials**:
- **Title**: "Applications of Link Analysis"
- **Topics**: Social networks, biological networks, web search, recommender systems, fraud detection
- **Key Algorithm**: PageRank (graph-based ranking)

**How it relates to recommenders**:
- Models user-item interactions as a **bipartite graph**
- Users and items are nodes
- Ratings/interactions are edges
- Apply graph algorithms (PageRank, community detection) to find recommendations

**Example**:
```
     User1 --[5.0]--> Movie A
       |               |
    [4.0]          [4.5]
       |               |
       v               v
     Movie B <--[3.5]-- User2
```
→ Use graph traversal/PageRank to recommend movies

---

#### Is Link Analysis Required for This Project?

**NO** - Module 5 is **NOT required** for this project.

**Evidence**:
1. **question.md** (Project Requirements):
   - States only: "Implement a movie recommending system using **Collaborative Filtering algorithm**"
   - No mention of link analysis, PageRank, or graph-based methods

2. **analysis.md** (Implementation Plan):
   - 455 lines covering Modules 2-4 only
   - Focuses on: Item-Based CF, User-Based CF, SVD
   - **Zero mentions** of link analysis, PageRank, or Module 5

3. **CLAUDE.md** (Project Instructions):
   - Lists: Module 2 (CF), Module 3 (Content-based), Module 4 (Hybrid)
   - **No Module 5 mentioned**

4. **Module 5 lesson material**:
   - Lists "Recommender Systems" as ONE application among many
   - Poses discussion question: "Can you think of another example?" (not a requirement)

---

#### When Would You Use Link Analysis?

**Link Analysis / PageRank for Recommendations**:

**Advantages**:
- Can model complex network relationships
- Identifies influential users/items
- Good for social recommendation scenarios
- Can incorporate trust/reputation

**Disadvantages**:
- More complex than collaborative filtering
- Requires graph database/processing
- Less interpretable than CF
- Not standard in industry for pure recommenders

**Real-World Use Cases**:
- **Social networks**: Friend recommendations (LinkedIn "People You May Know")
- **Citation networks**: Paper recommendations (Google Scholar)
- **Web search**: Page ranking (original PageRank application)
- **Fraud detection**: Identify anomalous patterns

**For Movie Recommendations**: Collaborative Filtering (your approach) is industry standard!

---

#### Should You Mention Module 5 in Your Report?

**Optional**: Include in "Improvement Opportunities" section as an advanced alternative:

> **Graph-Based Approaches (Module 5)**
>
> An alternative approach would be to model user-item interactions as a
> bipartite graph and apply link analysis techniques such as PageRank. This
> could identify influential movies or user communities. However, for pure
> collaborative filtering tasks, matrix factorization methods (SVD) are more
> standard in industry and typically achieve better accuracy.
>
> Future work could explore hybrid approaches that combine collaborative
> filtering with graph-based community detection to improve recommendation
> diversity.

**Benefits**:
- ✅ Shows awareness of advanced techniques
- ✅ Demonstrates course material coverage
- ✅ Fills out "Improvement Opportunities" section
- ✅ No implementation required (just mention)

---

### Q4: How Do I Choose the Right Algorithm for Production?

**Decision Framework**:

#### Use Item-Based CF if:
- ✅ Need interpretable recommendations ("You rated Movie X, similar to Movie Y")
- ✅ Item catalog is relatively stable (movies don't change much)
- ✅ Have cold-start users (new users joining frequently)
- ✅ Memory for storing similarity matrix is available (~500 MB)
- ✅ Can tolerate slightly lower accuracy for interpretability

**Don't use if**:
- ❌ Item catalog changes rapidly (new movies daily)
- ❌ Very large item catalog (> 100K items)
- ❌ Need absolute best accuracy

---

#### Use User-Based CF if:
- ✅ Want social-proof style recommendations ("People like you also liked...")
- ✅ User base is small to medium (< 100K users)
- ✅ User preferences are stable over time
- ✅ Discovering new genres/items is important

**Don't use if**:
- ❌ Large user base (> 100K users) - scalability issues
- ❌ User preferences change rapidly
- ❌ Memory constraints (user similarity matrix is huge)
- ❌ Need fast predictions

---

#### Use SVD if:
- ✅ Accuracy is your top priority (lowest RMSE/MAE)
- ✅ Have large-scale data (millions of ratings)
- ✅ Need fast real-time predictions (dot product is fast)
- ✅ Can afford training time and hyperparameter tuning
- ✅ Interpretability is not critical

**Don't use if**:
- ❌ Need to explain recommendations to users
- ❌ Very small dataset (< 10K ratings) - will overfit
- ❌ Can't afford initial training time

---

#### Hybrid Approach (Recommended for Production)

**Real-world systems (Netflix, Spotify, Amazon) use combinations**:

1. **Primary**: SVD for accuracy and speed
2. **Cold-start users**: Item-Based CF (popular items in genres user liked)
3. **Cold-start items**: Content-based (genre, actors, keywords)
4. **Final ranking**: Blend multiple models with learned weights

**Example Blending**:
```python
final_score = (
    0.50 × svd_score +
    0.25 × item_based_score +
    0.15 × content_based_score +
    0.10 × popularity_score
)
```

---

### Q5: What is the Cold-Start Problem?

**Cold-start** is one of the most challenging problems in recommendation systems.

**Definition**: The difficulty of making accurate recommendations when there is **insufficient data** about users or items.

**Core Challenge**:
> "How do you recommend something when you have little or no information to base the recommendation on?"

---

#### Three Types of Cold-Start Problems

#### 1. **Cold-Start Users** (New User Problem)

**Scenario**: A brand-new user signs up for Netflix. They have rated **zero movies**.

**The Problem**:
- Collaborative filtering relies on finding similar users or items
- With no ratings, you can't determine:
  - Which users are similar to this new user
  - Which movies this user might like
  - What genres they prefer

**Example**:
```
New User (just registered):
  Ratings: []  ← Empty! No data!

Question: What movies should we recommend?
Answer: We have no idea what they like!
```

**In Your Dataset (Temporal Split)**:
- **90% of test users are cold-start** (new users not in training)
- This makes temporal split much harder but more realistic

---

#### 2. **Cold-Start Items** (New Item Problem)

**Scenario**: A new movie is released today. **No users have rated it yet**.

**The Problem**:
- Item-based CF relies on item similarity computed from ratings
- With no ratings, you can't:
  - Find similar movies
  - Predict who might like this movie
  - Rank it against other movies

**Example**:
```
New Movie (just released):
  Ratings: []  ← No one has rated it yet!

Question: Should we recommend this to User 123?
Answer: We don't know if it's similar to movies they like!
```

**In Your Dataset (Temporal Split)**:
- **46% of test movies are cold-start** (new movies not in training)

---

#### 3. **Cold-Start System** (New Platform Problem)

**Scenario**: A brand-new recommendation platform launches. **No users, no ratings, nothing**.

**The Problem**:
- Can't use collaborative filtering (no collaboration yet!)
- Must use alternative strategies initially

**Example**: A new streaming service on Day 1 with 100 movies but zero ratings.

---

#### Cold-Start in Your Project

**Your Temporal Split Statistics**:

```
Training Set (1995-2015):
  - 227,200 users
  - 25,600 movies
  - 20.8M ratings

Test Set (2015-2017):
  - 48,500 users (90% NEW! Not in training)
  - 42,700 movies (46% NEW! Not in training)
  - 5.2M ratings
```

**What this means**:
- **Cold-start users (90%)**: Test users never appeared in training
  - Your algorithm has never seen their preferences
  - No historical data to base recommendations on

- **Cold-start movies (46%)**: Test movies never appeared in training
  - Your algorithm has never seen ratings for these movies
  - No similarity data to work with

**This is realistic!** In real-world systems:
- New users sign up every day (cold-start users)
- New movies are released constantly (cold-start items)

---

#### How Each Algorithm Handles Cold-Start

**Item-Based Collaborative Filtering**

✅ **Cold-Start Users: Handles Well**
```python
# Can still recommend based on item similarity
# Doesn't need extensive user history
# Just needs user to rate 1-2 items

Example:
  New user rates "Inception" → 5.0
  Find movies similar to "Inception"
  Recommend: "Interstellar", "The Matrix", "Shutter Island"
```

❌ **Cold-Start Items: Struggles**
```python
# New movie has no ratings
# Can't compute similarity to other movies
# Must fall back to global mean or content-based

# Your implementation:
if movie_idx >= item_similarity.shape[0]:
    return global_mean_rating  # Fallback for new movies
```

---

**User-Based Collaborative Filtering**

❌ **Cold-Start Users: Struggles**
```python
# New user has no ratings
# Can't find similar users
# Must fall back to global mean or popular items

# Your implementation:
if user_idx >= user_similarity.shape[0]:
    return global_mean_rating  # Fallback for new users
```

✅ **Cold-Start Items: Handles Better**
```python
# Can find similar users who rated this item
# Even if item is new, some early adopters may have rated it
```

---

**SVD Matrix Factorization**

❌ **Cold-Start Users: Cannot Predict Directly**
```python
# User has no latent factor vector
# Must use content-based or ask for initial ratings
```

❌ **Cold-Start Items: Cannot Predict Directly**
```python
# Item has no latent factor vector
# Must use content-based features
```

✅ **BUT: Handles Sparse Data Very Well**
- If user has even 1-2 ratings → Can make predictions
- If item has 1-2 ratings → Can learn latent factors
- Surprise library uses global/user/item biases for partial cold-start

---

#### Fallback Strategies

**Strategy 1: Global Statistics**

```python
# For completely new users/items
if user_id not in training_users or movie_id not in training_movies:
    return global_mean_rating  # e.g., 3.52
```

**Strategy 2: Popularity-Based**

```python
# For new users
if user_is_new:
    # Recommend top-rated movies
    return most_popular_movies(n=10)
```

**Strategy 3: Content-Based Filtering**

**For Cold-Start Users**:
```python
# Ask during onboarding
new_user_onboarding = [
    "Pick 3 movies you've seen and rate them",
    "Select your favorite genres",
    "Tell us about your preferences"
]

# After 3 ratings → No longer cold-start!
```

**For Cold-Start Items**:
```python
# Use movie metadata
new_movie = {
    'title': 'Dune 2',
    'genres': ['Sci-Fi', 'Action'],
    'director': 'Denis Villeneuve',
    'actors': ['Timothée Chalamet', 'Zendaya']
}

# Recommend to users who liked:
# - Other Sci-Fi movies
# - Other Denis Villeneuve films
# - Movies with same actors
```

**Strategy 4: Hybrid Approach** (Production-Ready)

```python
if user_has_ratings and movie_has_ratings:
    # Use collaborative filtering (best accuracy)
    score = svd_predict(user, movie)

elif user_has_ratings and movie_is_new:
    # Cold-start item: Use content-based
    score = content_similarity(user_preferences, movie_metadata)

elif user_is_new and movie_has_ratings:
    # Cold-start user: Use popularity + content
    score = popularity_score(movie) + genre_match(user_prefs, movie)

else:
    # Both cold-start: Use global popularity
    score = global_popularity(movie)
```

This is what **Netflix, Spotify, Amazon** do!

---

#### Cold-Start Impact on Performance

**Why Temporal Split Has Higher RMSE**:

**Random Split**:
```
Test users: 94% overlap with training
Test movies: 73% overlap with training

Cold-start: ~6-27% of test data
Expected RMSE: 0.80-0.85 (easier)
```

**Temporal Split**:
```
Test users: 10% overlap with training (90% cold-start!)
Test movies: 54% overlap with training (46% cold-start!)

Cold-start: ~46-90% of test data
Expected RMSE: 0.85-0.95 (harder, more realistic)
```

**Performance Impact**:
```
Random Split RMSE:   0.82
Temporal Split RMSE: 0.88
Difference:          +0.06 (worse due to cold-start!)
```

**This is expected and realistic!** Cold-start predictions are inherently harder.

---

#### Real-World Examples

**Netflix**

**Cold-Start User**:
```
Day 1: New subscriber signs up
  → Show onboarding: "Rate these 5 popular movies"
  → User provides 5 ratings
  → Now can use collaborative filtering

Strategy:
  - Day 1: Content-based + popularity
  - Day 7: Enough data for collaborative filtering
  - Month 1: Full personalization
```

**Cold-Start Movie**:
```
New Release: "Avatar 3" (no ratings yet)
  → Use content: Similar to Avatar 1, Avatar 2
  → Recommend to users who liked Avatar 1, Avatar 2
  → After 100 ratings: Switch to collaborative filtering

Strategy:
  - Week 1: Content-based (genres, cast, director)
  - Week 2: Early adopter ratings available → Hybrid
  - Month 1: Full collaborative filtering
```

---

**Spotify**

**Cold-Start User**:
```
New user signs up:
  1. Ask: "What artists do you like?"
  2. Seed recommendations based on these artists
  3. After listening to 20 songs → Full CF available
```

**Cold-Start Song**:
```
New song released:
  1. Use audio features (tempo, key, energy, mood)
  2. Recommend to users who like similar songs
  3. After 500 plays → Collaborative filtering kicks in
```

---

#### Code Implementation (From Your Notebooks)

**Item-Based CF** (`1_item_based_cf.ipynb`):
```python
def predict_rating(user_idx, movie_idx, k=30):
    # Handle cold-start movies
    if pd.isna(movie_idx) or movie_idx >= item_similarity.shape[0]:
        return global_mean_rating  # Fallback

    # Handle cold-start users
    if pd.isna(user_idx) or user_idx >= user_item_matrix.shape[0]:
        return global_mean_rating  # Fallback

    # Normal prediction for known users/items
    # ... (item similarity computation)
```

**User-Based CF** (`2_user_based_cf.ipynb`):
```python
def predict_rating(user_idx, movie_idx, k=50):
    # Cold-start handling
    if pd.isna(user_idx) or user_idx >= user_similarity.shape[0]:
        return global_mean_rating

    if pd.isna(movie_idx) or movie_idx >= user_item_matrix.shape[1]:
        # Movie not in training, use user's mean
        user_id = idx_to_user.get(int(user_idx))
        return user_mean_ratings.get(user_id, global_mean_rating)

    # ... (user similarity computation)
```

**SVD** (`3_svd_matrix_factorization.ipynb`):
```python
# Surprise library handles cold-start automatically:
# - New users/items get global bias
# - Partial cold-start uses available biases
predictions = svd_model.test(testset)
```

---

#### For Your Report

**In Methodology Section**:
> "To handle cold-start cases (90% of test users and 46% of test movies in
> temporal split), we implemented a fallback strategy using the global mean
> rating (3.52) for completely new users and items. This is a common approach
> in production systems, which typically combine collaborative filtering with
> content-based methods for comprehensive cold-start handling."

**In Results Section**:
> "The temporal split evaluation revealed significant cold-start challenges, with
> 90% of test users not present in the training set. This is representative of
> real-world deployment where new users join continuously. Our algorithms achieved
> RMSE 0.85-0.95, with the higher error rate (compared to random split) attributed
> primarily to cold-start predictions falling back to global statistics."

**In Discussion Section**:
> "Cold-start handling is a fundamental limitation of pure collaborative filtering.
> While our fallback strategy (global mean rating) ensures predictions are always
> available, accuracy suffers for completely new users/items. Production systems
> address this through:
>
> 1. **Active Learning**: Onboarding flows that collect initial user preferences
> 2. **Hybrid Systems**: Content-based filtering using movie metadata
> 3. **Popularity Fallback**: Recommending trending/popular items
> 4. **Progressive Personalization**: Gradually shifting from content to CF as
>    data accumulates"

**In Improvement Opportunities**:
> "**Cold-Start Mitigation**:
> - Implement hybrid CF + content-based filtering
> - Use movie metadata (genres, actors, directors, keywords) for new items
> - Design onboarding flow asking users to rate 3-5 popular movies
> - Implement 'warm start' by using demographic data for new users
> - Progressive personalization: start with popularity, evolve to CF"

---

#### Key Takeaways

**What is Cold-Start?**
- New users with no rating history (90% in your test set)
- New movies with no ratings (46% in your test set)
- Makes collaborative filtering very difficult

**Why Does It Matter?**
- CF requires data to find patterns
- No data = no patterns = poor predictions
- Temporal split reveals this realistic challenge

**How to Handle It?**
1. **Fallback**: Global mean rating (your current implementation)
2. **Popularity**: Recommend trending/popular items
3. **Content-based**: Use metadata (genres, actors, directors)
4. **Hybrid**: Combine CF + content-based
5. **Active learning**: Ask users for initial ratings during onboarding

**In Your Implementation**:
```python
# All your notebooks handle cold-start like this:
if pd.isna(user_idx) or pd.isna(movie_idx):
    return global_mean_rating  # Fallback strategy
```

**Impact on Results**:
- Temporal split: Higher RMSE due to 90% cold-start users
- Random split: Lower RMSE due to 94% user overlap
- Difference (~0.06 RMSE) is the "cold-start penalty"
- This is **expected and realistic**!

---

## 📚 Additional Resources

### Understanding Your Results

**After running the notebooks, you'll have**:
- RMSE and MAE for each algorithm
- Training and prediction times
- Coverage percentages
- Visualizations (bar charts, radar chart)

**How to analyze**:
1. **Compare RMSE/MAE**: Lower is better
2. **Check coverage**: Higher is better (> 90% is good)
3. **Consider speed**: Faster prediction time = better user experience
4. **Balance trade-offs**: Best accuracy vs interpretability vs speed

**Example Analysis**:
```
SVD: RMSE 0.83, MAE 0.67, Fast predictions
→ Best for production accuracy

Item-Based: RMSE 0.91, MAE 0.72, Medium predictions
→ Best for explainability

User-Based: RMSE 0.95, MAE 0.75, Slow predictions
→ Best for social recommendations (small scale)
```

---

### Common Pitfalls to Avoid

1. **Not handling cold-start**: Always provide fallback (global mean, popular items)
2. **Ignoring temporal aspects**: Use temporal split, not random
3. **Overfitting**: Don't tune hyperparameters on test set (use validation set or cross-validation)
4. **Memory issues**: Always use sparse matrices for large datasets
5. **Unrealistic evaluation**: Don't use future data to predict past

---

**Questions?** Review the inline documentation in each notebook for detailed explanations.

**Good luck with your implementation!** 🚀
