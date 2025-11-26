# Movie Recommendation System 

## Project Structure

```
Project/
├── README.md                                    # This file
├── analysis.md                                  # Project analysis and implementation plan
├── requirements.txt                             # Python dependencies
├── .gitignore                                   # Git ignore rules
│
├── references/                                  # Reference materials
│   ├── Taylor Dredge–Mushroom_Edibility.pdf    # structure reference
│   └── Taylor Dredge–Mushroom_Edibility.docx
│
├── sources/                                     # All implementation code
│   ├── 1_data_preparation/                     # Phase 1: Data cleaning & preprocessing
│   │   ├── download_data.ipynb                 # Download MovieLens dataset
│   │   ├── clean_movie_metadata.ipynb          # Clean movies_metadata.csv
│   │   ├── clean_links.ipynb                   # Clean links.csv
│   │   ├── clean_ratings.ipynb                 # Clean ratings.csv
│   │   ├── clean_credits.ipynb                 # Parse credits JSON
│   │   └── clean_keywords.ipynb                # Parse keywords JSON
│   │
│   ├── 2_split_data/                           # Phase 2: Train/test splitting
│   │   ├── random_split.ipynb                  # Random 80/20 split
│   │   └── temporal_split.ipynb                # Time-based split (cold-start)
│   │
│   ├── 3_model_implementation/                 # Phase 3: CF algorithms
│   │   ├── README.md                           # Implementation documentation
│   │   ├── 4_evaluation_comparison.ipynb       # Compare all algorithms
│   │   │
│   │   ├── 80-20/                              # 80/20 random split experiments
│   │   │   ├── 1b_item_based_cf_80_20_split.ipynb
│   │   │   ├── 2b_user_based_cf_80_20_split.ipynb
│   │   │   └── 3b_svd_matrix_factorization_80_20_split.ipynb
│   │   │
│   │   ├── warm_start/                         # Warm-start scenarios
│   │   │   ├── README_WARM_START.md
│   │   │   ├── 1_item_based_cf.ipynb
│   │   │   ├── 2_user_based_cf_warm_start.ipynb
│   │   │   ├── 3_svd_matrix_factorization_warm_start.ipynb
│   │   │   ├── item_based_cf_warm_start.py
│   │   │   ├── user_based_cf_warm_start.py
│   │   │   ├── svd_warm_start.py
│   │   │   ├── run_warm_start_comparison.py
│   │   │   └── compare_results.py
│   │   │
│   │   └── with_cold_start/                    # Cold-start scenarios
│   │       ├── README.md
│   │       ├── MEMORY_OPTIMIZATIONS.md
│   │       ├── 1_item_based_cf.py
│   │       ├── 2_user_based_cf_temporal.py
│   │       ├── 3_svd_temporal.py
│   │       ├── run_with_cold_start.py
│   │       └── compare_results.py
│   │
│   ├── 4_improvements/                         # Phase 4: Hybrid & enhancements
│   │   ├── README.md
│   │   ├── HYBRID_SYSTEM_README.md
│   │   ├── OPTIMIZATION_GUIDE.md
│   │   ├── 5_hybrid_system.ipynb
│   │   ├── 5_hybrid_system.py
│   │   ├── 5_hybrid_system_optimized.ipynb
│   │   ├── 5_hybrid_system_optimized.py
│   │   │
│   │   └── option_b_enhancement/               # Enhanced hybrid experiments
│   │       ├── README.md
│   │       ├── 6_hybrid_with_credits.ipynb
│   │       └── 6_hybrid_with_credits.py
│   │
│   └── simple_recommenders/                    # Early experiments/prototypes
│       ├── item-based.ipynb
│       ├── user-based.ipynb
│       └── using-surprise-lib.ipynb
│
└── submissions/                                 # Final deliverables
    ├── README.md                               # Submission documentation
    ├── Collaborative Movie Filter.pdf          # Final report (PDF)
    ├── Collaborative Movie Filter.docx         # Final report (Word)
    ├── implementation.ipynb                    # Main implementation notebook
    ├── recommender_analysis.ipynb              # Analysis & evaluation
    └── summarize.ipynb                         # Results summary
```

**Note:** The MovieLens dataset files are stored externally or in a data directory not tracked by git (see `.gitignore`). Dataset files include:
- `movies_metadata.csv` (34MB, ~45K movies)
- `ratings.csv` (710MB, ~26M ratings)
- `ratings_small.csv` (2.4MB, ~100K ratings - for development)
- `links.csv` (989KB, ~45K MovieLens → TMDB → IMDB mappings)
- `credits.csv` (190MB, ~45K cast & crew)
- `keywords.csv` (6MB, ~46K movie keywords)

## Installation

### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook or JupyterLab

### Setup

1. Clone this repository:
```bash
git clone <repository-url>
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install pandas numpy jupyter matplotlib seaborn scikit-learn
```

Or install from requirements.txt if available:
```bash
pip install -r requirements.txt
```

## Dataset Information

### Input Files

| File | Size | Rows | Description |
|------|------|------|-------------|
| `movies_metadata.csv` | 34MB | ~45K | Movie details from TMDB |
| `ratings.csv` | 710MB | ~26M | Full user-movie ratings |
| `ratings_small.csv` | 2.4MB | ~100K | Subset for development |
| `links.csv` | 989KB | ~45K | ID mappings (MovieLens to TMDB to IMDB) |
| `credits.csv` | 190MB | ~45K | Cast and crew (JSON format) |
| `keywords.csv` | 6MB | ~46K | Movie keywords (JSON format) |


## Project Goals

1. **Implement collaborative filtering recommendation algorithms:**
   - Item-Based CF (memory-based approach)
   - User-Based CF (memory-based approach)
   - SVD Matrix Factorization (model-based approach)
   - Hybrid system (CF + content-based filtering)

2. **Evaluate model performance:**
   - RMSE (Root Mean Squared Error)
   - MAE (Mean Absolute Error)
   - Precision@K, Recall@K
   - Coverage and diversity metrics

3. **Test under different scenarios:**
   - Warm-start: Standard train/test split
   - Cold-start: Temporal split with new users/items
   - Hybrid: Content features for improved recommendations

4. **Deliver comprehensive report:**
   - Design and architecture decisions
   - Implementation details and code
   - Model performance analysis and comparison
   - Improvement opportunities and future work


## Running the Code

### Recommended: Final Submission Notebooks

The `submissions/` folder contains the complete end-to-end implementation. **Run notebooks in this order:**

```bash
cd submissions
jupyter notebook
```

**Execution Order:**

1. **`summarize.ipynb`** - Complete data pipeline (REQUIRED FIRST)
   - Downloads MovieLens dataset from Kaggle
   - Cleans all datasets (ratings, metadata, credits, keywords, links)
   - Creates train/test splits (random 80/20 and temporal 80/20)
   - Outputs processed data to `./data/processed/`
   - **Must run this first** - creates all data dependencies for other notebooks

2. **`implementation.ipynb`** - Core CF algorithms (warm-start scenarios)
   - Implements Item-Based CF, User-Based CF, SVD
   - Evaluates on both random and temporal splits
   - Focuses on warm-start cases (user and movie exist in training)
   - Generates comparison metrics and visualizations

3. **`recommender_analysis.ipynb`** - Exploratory analysis using Surprise library
   - Data filtering and exploration
   - Cross-validation with multiple algorithms
   - Hyperparameter tuning with GridSearchCV
   - Learning curve analysis
   - Top-N recommendation generation

**Note:** All notebooks are self-contained but depend on `summarize.ipynb` running first to generate the processed datasets.



## Documentation

- **`analysis.md`**: Initial project analysis and implementation plan
- **`sources/3_model_implementation/README.md`**: CF implementation details
- **`sources/4_improvements/HYBRID_SYSTEM_README.md`**: Hybrid system documentation
- **`submissions/README.md`**: Final submission guide
- **`submissions/Collaborative Movie Filter.pdf`**: Complete project report


## References

- [MovieLens Dataset (Kaggle)](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)
- Course materials: Modules 2-4 (Collaborative Filtering, Content-Based, Hybrid Systems)
- Taylor Dredge's Mushroom Classification Report (structure reference)


