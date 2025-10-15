# Movie Recommendation System 

## Project Structure

```

 README.md                               # This file
 analysis.md                             # Project analysis and implementation plan
 Taylor Dredge Mushroom_Edibility.pdf   # Reference: Previous student work

 datasets/                               # All dataset files
 input/                             # Raw MovieLens data
     movies_metadata.csv            # 45K movies (34MB)
     ratings.csv                    # 26M ratings (710MB)
     ratings_small.csv              # 100K ratings (2.4MB) - for development
     links.csv                      # MovieLens � TMDB � IMDB mappings
     links_small.csv                # Subset of links
     credits.csv                    # Cast and crew (190MB)
     keywords.csv                   # Movie keywords (6MB)

 output/                            # Output datasets
     cleaned_datasets/              # Additional cleaned files
        ...
 sources/                               # Source code and notebooks
     1_data_preparation/                # Phase 1: Data cleaning
        clean_movie_metadata.ipynb     # Step 1: Clean movies_metadata.csv
        clean_links.ipynb              # Step 2: Clean links.csv
```

## Installation

### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook or JupyterLab

### Setup

1. Clone this repository:
```bash
git clone <repository-url>
cd Project
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

**Prerequisites:** Must run Steps 1 and 2 first.

### Data Cleaning Workflow Summary

```
Step 1: clean_movie_metadata.ipynb
           (generates cleaned_movies_metadata.csv)
Step 2: clean_links.ipynb
           (generates cleaned_links.csv)
Step 3: clean_ratings.ipynb (coming soon)
           (generates cleaned_ratings.csv)
Phase 2: Model Implementation (coming soon)
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

### Data Relationships

```
RATINGS (userId, movieId, rating, timestamp)
    (join on movieId)
LINKS (movieId, tmdbId, imdbId)
    (join on tmdbId)
MOVIES_METADATA (id=tmdbId, title, genres, budget, revenue, etc.)
```

## Project Goals

1. Implement collaborative filtering recommendation algorithms:
   - Item-Based CF (baseline)
   - User-Based CF (comparison)
   - SVD Matrix Factorization (advanced)

2. Evaluate model performance:
   - RMSE (Root Mean Squared Error)
   - MAE (Mean Absolute Error)
   - Precision@K, Recall@K

3. Write comprehensive report covering:
   - Design and architecture
   - Implementation details
   - Model performance analysis
   - Improvement opportunities

## Documentation
- **`analysis.md`**: Detailed project analysis, data cleaning strategy, and implementation plan

## References
- [MovieLens Dataset (Kaggle)](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)
- Course materials: Modules 2-4 (Collaborative Filtering, Content-Based, Hybrid Systems)
- Taylor Dredge's Mushroom Classification Report (structure reference)


