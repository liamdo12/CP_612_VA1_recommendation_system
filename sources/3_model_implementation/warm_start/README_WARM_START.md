# Warm-Start Evaluation - Collaborative Filtering Comparison

## Overview

These scripts evaluate collaborative filtering algorithms under **warm-start conditions**: only testing on user-movie pairs where both user AND movie exist in the training set. This eliminates cold-start effects and measures pure collaborative filtering performance.

---

## 📁 Files

### Python Scripts (Standalone)
| Script | Algorithm | Runtime | Output |
|--------|-----------|---------|--------|
| `item_based_cf_warm_start.py` | Item-Based CF | 15-25 min | `item_based_cf_warm_start.csv` |
| `user_based_cf_warm_start.py` | User-Based CF | 30-60 min | `user_based_cf_warm_start.csv` |
| `svd_warm_start.py` | SVD Matrix Factorization | 5-10 min | `svd_warm_start.csv` |

### Jupyter Notebooks
| Notebook | Description |
|----------|-------------|
| `1_item_based_cf.ipynb` | Item-Based CF (original notebook) |
| `2_user_based_cf.ipynb` | User-Based CF (original notebook) |
| `2_user_based_cf_warm_start.ipynb` | User-Based CF (warm-start version) |
| `3_svd_matrix_factorization.ipynb` | SVD (original notebook) |
| `3_svd_matrix_factorization_warm_start.ipynb` | SVD (warm-start version) |

### Utility Scripts
| Script | Purpose |
|--------|---------|
| `run_warm_start_comparison.py` | Run all 3 algorithms sequentially |
| `compare_results.py` | Compare warm-start vs cold-start performance |

---

## 🎯 Purpose: Warm-Start vs Cold-Start

### Warm-Start Evaluation (This Folder)
- **Test Set**: Filtered to include only known user-movie pairs
- **Cold-Start Cases**: None (0%)
- **Coverage**: ~10.7% of original temporal test set
- **Purpose**: Measure pure collaborative filtering performance
- **Real-World Analog**: Recommending from existing catalog to existing users

### Cold-Start Evaluation (with_cold_start folder)
- **Test Set**: Full temporal test set including new users/movies
- **Cold-Start Cases**: ~89.3%
- **Coverage**: 100% of original test set (with fallback predictions)
- **Purpose**: Measure realistic production performance
- **Real-World Analog**: Real recommendation system with new users/items

---

## 📊 What "Warm-Start" Means

### Evaluation Strategy Comparison

| Strategy | Test Filtering | Cold-Start | Testable Ratings |
|----------|---------------|------------|------------------|
| **Warm-Start** | Only known user-movie pairs | 0% | ~560K (10.7% of full test) |
| **Cold-Start** | Full test set (all pairs) | 89.3% | 5.2M (100% of test) |
| **Random 80-20** | No filtering (random split) | ~1-2% | ~5.2M (99%+ coverage) |

### Visual Explanation

```
WARM-START (This Folder):
┌─────────────────────────────────────────┐
│ Temporal Test Set (2015-2017)          │
│ Total: 5.2M ratings                     │
│                                         │
│ FILTER: Keep only if:                  │
│  ✓ User in training (1995-2015)        │
│  ✓ Movie in training (1995-2015)       │
│                                         │
│ Result: 560K ratings (~10.7%)          │
│ Sample: 100K for evaluation            │
└─────────────────────────────────────────┘

COLD-START (with_cold_start folder):
┌─────────────────────────────────────────┐
│ Temporal Test Set (2015-2017)          │
│ Total: 5.2M ratings                     │
│                                         │
│ NO FILTERING - Keep all ratings        │
│  ⚠️  89.3% new users (not in training)  │
│  ⚠️  Uses global mean fallback          │
│                                         │
│ Result: 5.2M ratings (100%)            │
│ Sample: 50K for evaluation (16GB RAM)  │
└─────────────────────────────────────────┘
```

---

## 📈 Expected Results

### Coverage & Performance

| Metric | Warm-Start | Cold-Start | Difference |
|--------|-----------|------------|------------|
| **Test sample size** | 100,000 | 50,000 | Warm-start uses larger sample |
| **Known pairs** | 100% | ~10.7% | Warm-start eliminates cold-start |
| **CF predictions** | ~99.9% | ~10.7% | Warm-start enables full CF |
| **Global mean fallback** | ~0.1% | ~89.3% | Cold-start requires fallback |

### RMSE Comparison (Expected)

| Algorithm | Warm-Start RMSE | Cold-Start RMSE | RMSE Increase |
|-----------|----------------|-----------------|---------------|
| **Item-Based CF** | 0.85-0.87 | 1.05-1.10 | +20-25% |
| **User-Based CF** | 0.84-0.86 | 1.00-1.05 | +18-22% |
| **SVD** | 0.90-0.93 | 1.05-1.10 | +15-20% |

**Key Insight**: Warm-start shows "ideal case" performance; cold-start reflects production reality.

---

## 🚀 Quick Start

### Option 1: Run All Algorithms
```bash
cd "/Users/luan/Study/WLU/Data Analysis & Management/Project/sources/3_model_implementation/warm_start"
python run_warm_start_comparison.py
```

**Runtime**: 50-95 minutes total

### Option 2: Run Individual Scripts
```bash
# Fastest first
python svd_warm_start.py                # 5-10 min

# Medium speed
python item_based_cf_warm_start.py      # 15-25 min

# Slowest
python user_based_cf_warm_start.py      # 30-60 min
```

### Option 3: Compare Warm-Start vs Cold-Start
```bash
# After running both warm_start and with_cold_start scripts:
python compare_results.py
```

---

## 📝 For Your Report

### How to Explain Warm-Start in Your Report

**Good Terminology:**
```markdown
## Evaluation Strategies

We employed two evaluation approaches:

### 1. Warm-Start Evaluation
- **Test Set**: Filtered temporal test set (2015-2017) containing only user-movie
  pairs where both appear in training data (1995-2015)
- **Sample Size**: 100,000 ratings
- **Cold-Start Cases**: 0%
- **Purpose**: Measure pure collaborative filtering performance without confounding
  cold-start effects

### 2. Cold-Start Evaluation
- **Test Set**: Full temporal test set including new users and movies
- **Sample Size**: 50,000 ratings (reduced for 16GB RAM compatibility)
- **Cold-Start Cases**: 89.3%
- **Purpose**: Measure realistic production performance when facing cold-start challenges
```

**Why Both Evaluations?**
- **Warm-Start**: Shows your algorithm works correctly (upper-bound performance)
- **Cold-Start**: Shows real-world performance (production-realistic)

---

## ⚙️ Configuration

### Current Settings (100K samples)
```python
SAMPLE_SIZE = 100000      # Test sample size
K_NEIGHBORS = 30-50       # Number of neighbors
N_FACTORS = 50            # SVD latent factors (SVD only)
```

### If Memory Issues (Reduce to 50K)
```python
SAMPLE_SIZE = 50000       # Reduce sample
K_NEIGHBORS = 20-30       # Fewer neighbors
N_FACTORS = 30            # Fewer factors (SVD)
```

---

## 🔍 Validation

### How to Verify Results

1. **Check CSV files exist**:
   ```bash
   ls ~/Study/WLU/*/datasets/output/model_implementations/*warm_start.csv
   ```

2. **Verify RMSE values** are reasonable:
   - Item-Based CF: 0.85-0.87
   - User-Based CF: 0.84-0.86
   - SVD: 0.90-0.93

3. **Confirm high coverage** (~99.9% should be predictable)

---

## 📚 Related Documentation

- `COMPARISON_OUTPUT_GUIDE.md` - How to interpret comparison results
- `RUN_INSTRUCTIONS.md` - Detailed execution instructions
- `../with_cold_start/README.md` - Cold-start evaluation approach
- `../with_cold_start/MEMORY_OPTIMIZATIONS.md` - Memory optimization details

---

## 🤔 FAQ

**Q: Why only 10.7% of test set is usable?**
A: Temporal split creates natural cold-start (89.3% of test users are new). Warm-start evaluation filters these out to isolate CF performance.

**Q: Is 10.7% coverage too low?**
A: No! This is the actual testable portion. Warm-start measures what CF can do when it works; cold-start measures overall system performance.

**Q: Which evaluation should I report?**
A: **Both!** Warm-start shows algorithm correctness; cold-start shows production readiness.

**Q: Why larger sample (100K) vs cold-start (50K)?**
A: Warm-start doesn't face memory issues (no full matrix), so we can use larger sample for more statistical power.

---

**Last Updated**: November 8, 2025
**Optimized for**: 16GB RAM systems
**Python Version**: 3.10+
