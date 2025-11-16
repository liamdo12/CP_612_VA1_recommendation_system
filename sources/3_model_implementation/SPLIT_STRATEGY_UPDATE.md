# Train/Test Split Strategy Update

## Summary of Changes

All three collaborative filtering notebooks have been updated to use the **80-20 Random Split** instead of the temporal split.

### Files Modified:
- ✅ `1_item_based_cf.ipynb` (Cell 3, Cell 4, Cell 17)
- ✅ `2_user_based_cf.ipynb` (Cell 3, Cell 4, Cell 18, Cell 25)
- ✅ `3_svd_matrix_factorization.ipynb` (Cell 3, Cell 4)

---

## Why This Change Was Made

### Problem with Temporal Split:
```
Temporal Split Statistics:
- Train: 80% of ratings (oldest timestamps)
- Test: 20% of ratings (newest timestamps)
- Coverage: Only 8.97% testable
- Reason: 89.3% cold-start users + 7.7% cold-start movies
```

**Issue**: Temporal split creates a realistic but challenging scenario where 90% of test data cannot be predicted by collaborative filtering (users/movies never seen in training). This makes fair algorithm comparison impossible.

### Benefits of 80-20 Random Split:
```
80-20 Random Split Statistics:
- Train: 80% of ratings (randomly selected)
- Test: 20% of ratings (randomly selected)
- Coverage: ~99.93% testable
- Reason: Only 0.03% cold-start users
```

**Benefit**: Nearly all test ratings can be predicted, allowing fair comparison of pure collaborative filtering performance.

---

## Understanding Split Strategies

### When to Use Each Strategy:

| Scenario | Split Type | Purpose | Coverage |
|----------|------------|---------|----------|
| **Academic Comparison** | 80-20 Random | Measure pure CF algorithm quality | ~99.9% |
| **Hyperparameter Tuning** | 80-20 Random | Optimize model parameters fairly | ~99.9% |
| **Algorithm Benchmarking** | 80-20 Random | Compare algorithms apples-to-apples | ~99.9% |
| **Production Testing** | Temporal | Test cold-start handling, robustness | ~9% |
| **Realistic Evaluation** | Temporal | Simulate real-world deployment | ~9% |

### Industry Best Practice (Netflix, Spotify):

**Use BOTH strategies and report separately:**

1. **80-20 Random Split** → Report as "CF Performance" or "Algorithm Quality"
   - Measures: How well does the CF algorithm learn patterns?
   - Answers: Which algorithm is best for collaborative filtering?

2. **Temporal Split** → Report as "Production Readiness" or "Real-World Performance"
   - Measures: How well does the system handle new users/items?
   - Answers: Will this work in production with cold-start problems?

---

## Technical Implementation

### Code Changes (Example from SVD):

**Before (Temporal Split):**
```python
train_path = '../../datasets/output/split_and_train_datasets/temporal_split/train_ratings.csv'
test_path = '../../datasets/output/split_and_train_datasets/temporal_split/test_ratings.csv'
```

**After (80-20 Random Split):**
```python
train_path = '../../datasets/output/split_and_train_datasets/80-20/train_ratings.csv'
test_path = '../../datasets/output/split_and_train_datasets/80-20/test_ratings.csv'
```

### Consistent Test Sample Size:

All algorithms now use **100,000 test samples** for fair comparison:
- Item-Based CF: 100K samples
- User-Based CF: 100K samples (changed from 25K)
- SVD: 100K samples (can process all ~5.2M if needed)

---

## Expected Results After Re-Running

### Before (Temporal Split):
| Algorithm | RMSE | MAE | Coverage | Test Samples |
|-----------|------|-----|----------|--------------|
| User-Based CF | 0.9194 | 0.6798 | 8.97% | 25,000 |
| SVD | 0.9789 | 0.7326 | 8.97% | 467,049 |
| Item-Based CF | 1.0843 | 0.8413 | 8.97% | 100,000 |

**Problem**: Unfair comparison (different sample sizes, only 9% coverage)

### After (80-20 Random Split - Expected):
| Algorithm | RMSE | MAE | Coverage | Test Samples |
|-----------|------|-----|----------|--------------|
| SVD | ~0.85 | ~0.67 | ~99.9% | 100,000 |
| User-Based CF | ~0.91 | ~0.72 | ~99.9% | 100,000 |
| Item-Based CF | ~0.93 | ~0.74 | ~99.9% | 100,000 |

**Benefit**: Fair comparison with consistent sample sizes and full coverage!

---

## How to Re-Run Experiments

### Step 1: Open Jupyter Notebook
```bash
cd /Users/luan/Study/WLU/Data\ Analysis\ \&\ Management/Project/sources/3_model_implementation
jupyter notebook
```

### Step 2: Run Notebooks in Order
1. **`3_svd_matrix_factorization.ipynb`** (fastest: ~5-10 min)
2. **`1_item_based_cf.ipynb`** (medium: ~10-20 min)
3. **`2_user_based_cf.ipynb`** (slowest: ~30-60 min)

### Step 3: Compare Results
All results will be saved to:
- `../../datasets/output/model_implementations/svd_results.csv`
- `../../datasets/output/model_implementations/user_based_cf_results.csv`
- `../../datasets/output/model_implementations/item_based_cf_results.csv`

---

## Addressing Your Questions

### Q: "Could User-Based CF be using optimal memory?"
**A**: No, it's not about memory optimization. User-Based CF appeared better because:
1. It tested on only 25K samples (vs SVD's 467K)
2. Random sampling may have selected easier predictions
3. The comparison was unfair

### Q: "Why all algorithms only coverage 9%?"
**A**: Because you were using **temporal split** which creates massive cold-start:
- 89% of test users were NEW (never in training)
- 8% of test movies were NEW (never in training)
- Only 9% had both user AND movie in training

**Solution**: Switching to 80-20 random split increases coverage to ~99.9%!

### Q: "Is 80-20 split realistic?"
**A**: No, it's not realistic for production. But:
- ✅ **Use for**: Academic comparison, algorithm benchmarking
- ❌ **Don't use for**: Production testing, cold-start evaluation

**Best Practice**: Report BOTH:
1. **80-20 split**: "CF Performance" (measures algorithm quality)
2. **Temporal split**: "Production Performance" (measures robustness)

---

## Next Steps

1. ✅ All notebooks updated to use 80-20 random split
2. ⏳ Re-run all three notebooks to get fair comparison
3. ⏳ Compare results and determine best algorithm
4. 📝 In your report, mention you tested BOTH splits:
   - "For fair CF algorithm comparison, we used 80-20 random split"
   - "For production readiness evaluation, we tested temporal split"

This demonstrates understanding of evaluation methodology! 🎯
