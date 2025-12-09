# README – Exercise 2: UK vs US Parliamentary Debates Analysis

### Students
- **Ethan Sarfati — 345682546**
- **Ruben Bensimon — 346915424**

## 1. Introduction
This project implements the complete workflow required in Exercise 2, analyzing and classifying a corpus of parliamentary debates from:
- UK Parliament (≈330 documents)
- US Congress (≈360 documents)

The objective is to compare:
- Unsupervised clustering methods
- Supervised classification models
- Neural network architectures
- Feature importance across models
- t-SNE / UMAP visualizations

The pipeline includes:
- Text preprocessing
- BM25 matrix construction
- Word2Vec embeddings
- Clustering (KMeans, DBSCAN, HDBSCAN, GMM)
- Supervised models (NB, LR, SVM, RF)
- ANN models
- Feature importance extraction
- Visualizations

## 2. Project Structure
```
exo2/
├─ src/
│  ├─ 01_ex2_prepare_data.py
│  ├─ 02_ex2_build_matrices.py
│  ├─ 03_ex2_clustering.py
│  ├─ 04_ex2_classifiers.py
│  ├─ 05_ex2_ann.py
│  ├─ 06_ex2_visualizations.py
│  ├─ utils.py
│  └─ data/
│     ├─ ex2_raw/
│     ├─ ex2_clean/
│     ├─ ex2_lemm/
│     └─ ex2_matrices/
│
└─ outputs/ex2/
   ├─ ex2_feature_importance.xlsx
   ├─ tsne_bm25_true_labels.png
   ├─ umap_bm25_true_labels.png
   └─ tsne_bm25_kmeans_clusters.png
```

## 3. Data Preparation
Run:
```
python src/01_ex2_prepare_data.py
```
This step tokenizes, lemmatizes, and structures documents.

## 4. Vector Representations
### BM25
Creates BM25.npz and BM25.meta.json.

### Google News Word2Vec
Creates GOOGLENEWS.npz and metadata.

## 5. Clustering
Includes KMeans, DBSCAN, HDBSCAN, and GMM.
KMeans & GMM achieve 100% accuracy.

## 6. Supervised Classification
All classifiers (NB, LR, SVM, RF) achieve 100% accuracy under 10-fold CV.

## 7. Feature Importance
Excel file includes:
- Top 20 features per model
- Shared important features
- Statistical summaries

## 8. ANN Models
Two architectures (ReLU/GELU), both achieve 100% accuracy on test set.

## 9. Visualizations
Generated via:
```
python src/06_ex2_visualizations.py
```

Image references in project:
- tsne_bm25_true_labels.png
- tsne_bm25_kmeans_clusters.png
- umap_bm25_true_labels.png

## 10. Reproducibility
Run full workflow:
```
python src/01_ex2_prepare_data.py
python src/02_ex2_build_matrices.py
python src/03_ex2_clustering.py
python src/04_ex2_classifiers.py
python src/05_ex2_ann.py
python src/06_ex2_visualizations.py
```

## 11. Submission
Submit:
- Python source files
- README
- Feature importance Excel
- Visualizations
- ZIP of full project
