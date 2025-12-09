# Exercise 2 – UK vs US Parliamentary Debates Analysis  

---

## 1. Overview

This project implements a full text‑processing, representation, clustering, and classification pipeline for distinguishing UK Parliamentary debates from US Congressional debates.

The dataset contains:

- ~329 UK debate transcripts  
- ~360 US debate transcripts  

The project reproduces all required steps of Exercise 2, including:

- Text preprocessing (cleaning → tokenization → lemmatization)  
- Construction of BM25 term‑weighting matrices  
- Generation of document embeddings using Google News Word2Vec  
- Dimensionality reduction (TruncatedSVD)  
- Unsupervised clustering (KMeans, DBSCAN, HDBSCAN, Gaussian Mixture Models)  
- Supervised classification (NB, Logistic Regression, Linear SVM, Random Forest)  
- Neural architectures (ANN with ReLU and GELU)  
- Feature importance extraction  
- 2D visualizations (t‑SNE, UMAP)

The goal is to evaluate how different representation and learning methods behave on a strongly separable two‑class textual dataset.

---

## 2. Project Structure

```
exo2/
├─ src/
│  ├─ 01_ex2_prepare_data.py        # preprocessing pipeline
│  ├─ 02_ex2_build_matrices.py      # BM25 + Word2Vec vectorization
│  ├─ 03_ex2_clustering.py          # unsupervised clustering experiments
│  ├─ 04_ex2_classifiers.py         # supervised ML models + feature importance
│  ├─ 05_ex2_ann.py                 # embedding-based neural networks
│  ├─ 06_ex2_visualizations.py      # t-SNE / UMAP projections
│  ├─ utils.py                      # shared helper functions
│  └─ data/
│      ├─ ex2_raw/                  # raw corpus
│      ├─ ex2_clean/                # cleaned files
│      ├─ ex2_lemm/                 # lemmatized files
│      └─ ex2_matrices/             # BM25 and embedding matrices
└─ outputs/ex2/
    ├─ ex2_feature_importance.xlsx
    ├─ tsne_bm25_true_labels.png
    ├─ tsne_bm25_kmeans_clusters.png
    ├─ umap_bm25_true_labels.png
```

---

## 3. Processing Pipeline (Conceptual Description)

### 3.1 Text Preprocessing  
Run:

```
python src/01_ex2_prepare_data.py
```

This script performs:

1. Cleaning  
   - punctuation separation  
   - lowercasing  
   - removal of unnecessary whitespace  

2. Tokenization  

3. Lemmatization  
   - reduces inflected forms (e.g., “running”, “runs”) → “run”

4. Metadata assignment  
   - each document receives a group label: UK = 0, US = 1

Output folders:  
`ex2_clean/`, `ex2_lemm/`

---

## 4. Vector Representations

### 4.1 BM25 Matrix  
Run:

```
python src/02_ex2_build_matrices.py
```

BM25 is a ranking‑based term‑weighting function widely used in IR.  
The script produces:

- `BM25.npz` — sparse matrix (689 × vocabulary size)  
- `BM25.meta.json` — vocabulary + metadata

BM25 is particularly effective when documents have distinctive term frequency patterns.

### 4.2 Google News Word2Vec Embeddings  
The script also loads the **Word2Vec Google News 300‑dimensional model** and computes per‑document embeddings by averaging word vectors.

Output:

- `GOOGLENEWS.npz`  
- metadata describing embedding dimensions  

These embeddings encode semantic similarities beyond term frequency.

---

## 5. Dimensionality Reduction

BM25 is initially very high‑dimensional.  
We apply:

**TruncatedSVD (100 components)**  

This preserves variance while enabling:

- efficient clustering  
- approximate cosine similarity after L2 normalization  

---

## 6. Unsupervised Clustering

Run:

```
python src/03_ex2_clustering.py
```

Four methods are evaluated:

### 6.1 KMeans (k=2)  
Works extremely well here: **100% accuracy**  
Reason: UK vs US vocabulary is linearly separable after BM25 + SVD.

### 6.2 DBSCAN  
Density‑based clustering.  
Results are weaker due to noise sensitivity and uneven document clusters.

### 6.3 HDBSCAN  
Improved hierarchical version.  
Detects stable clusters and handles noise more effectively: **97%+ accuracy**

### 6.4 Gaussian Mixture Model (2 components)  
Soft clustering using probabilistic assignments.  
Matches KMeans performance: **100% accuracy**

---

## 7. Supervised Classification

Run:

```
python src/04_ex2_classifiers.py
```

On the BM25 matrix, the following models are trained with **10‑fold cross‑validation**:

- Multinomial Naive Bayes  
- Logistic Regression  
- Linear SVM  
- Random Forest  

All achieve:

**Accuracy = Precision = Recall = F1 = 1.0**

This confirms that the dataset is linearly separable.

The script also extracts **feature importance**, saved to:

`outputs/ex2/ex2_feature_importance.xlsx`

---

## 8. Neural Architectures (ANN Models)

Run:

```
python src/05_ex2_ann.py
```

The script builds:

1. Embedding layer (vocab_size=10,000)  
2. Dense layers (ReLU or GELU)  
3. Final binary classifier  

Both architectures reach **100% test accuracy**, demonstrating that even simple feedforward networks can solve this classification problem.

---

## 9. Visualizations (t‑SNE, UMAP)

Run:

```
python src/06_ex2_visualizations.py
```

t-SNE and UMAP are applied to BM25 + SVD representations.

Generated images:

- `tsne_bm25_true_labels.png`
- `tsne_bm25_kmeans_clusters.png`
- `umap_bm25_true_labels.png`

These show a clear two‑cluster separation aligned with the UK/US labels.

---

## 10. Reproducibility: Full Pipeline Execution

To reproduce all results:

```
python src/01_ex2_prepare_data.py
python src/02_ex2_build_matrices.py
python src/03_ex2_clustering.py
python src/04_ex2_classifiers.py
python src/05_ex2_ann.py
python src/06_ex2_visualizations.py
```

Ensure the following external models are installed:

- Google News Word2Vec (gensim‑downloader)
- nltk wordnet lemmatizer resources

---

## 11. Deliverables Included

- Python source code  
- Detailed README  
- Feature importance Excel  
- All clustering and embedding visualizations  
- Raw and processed datasets  
- Complete reproducible pipeline  
