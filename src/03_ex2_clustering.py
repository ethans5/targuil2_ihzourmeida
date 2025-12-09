import json
import pathlib
import numpy as np
from scipy import sparse

from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.preprocessing import normalize

import hdbscan


# -------------------------------------------------------------------
#  LOAD BM25 + META
# -------------------------------------------------------------------
def load_bm25_matrix():
    base = pathlib.Path(__file__).resolve().parents[1]
    mat_dir = base / "src" / "data" / "ex2_matrices"

    bm25_path = mat_dir / "BM25.npz"
    meta_path = mat_dir / "BM25.meta.json"

    print("Loading BM25 from:", bm25_path)
    X = sparse.load_npz(bm25_path)

    with open(meta_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    vocab = info["vocab"]
    meta = info["meta"]

    print(f"BM25 shape: {X.shape}")
    print(f"Vocab size: {len(vocab)}")
    print(f"Meta entries: {len(meta)}")

    return X, vocab, meta


def build_labels_from_meta(meta):
    """
    UK -> 0
    US -> 1
    """
    y = []
    for m in meta:
        parl = m.get("parliament", "UK")
        if parl == "US":
            y.append(1)
        else:
            y.append(0)
    y = np.array(y, dtype=int)
    print("Labels built: UK=0, US=1")
    print("Counts:", {c: int(np.sum(y == c)) for c in [0, 1]})
    return y


# -------------------------------------------------------------------
#  EVAL CLUSTERING (MAJORITY MAPPING)
# -------------------------------------------------------------------
def evaluate_clustering(name, labels_cluster, y_true):
    """
    Map clusters -> classes by majority vote, puis calcule les métriques.
    """
    labels_cluster = np.asarray(labels_cluster)
    y_true = np.asarray(y_true)

    unique_clusters = np.unique(labels_cluster)
    mapping = {}

    for c in unique_clusters:
        idx = labels_cluster == c
        if not np.any(idx):
            continue
        # majorité des labels vrais dans ce cluster
        vals, counts = np.unique(y_true[idx], return_counts=True)
        majority_label = vals[np.argmax(counts)]
        mapping[c] = majority_label

    print(f"\nCluster -> label mapping for {name}: {mapping}")

    y_pred = np.array([mapping[c] for c in labels_cluster])

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    print(f"\n=== Metrics ({name}) ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}\n")

    print("Classification report (UK=0, US=1):")
    print(classification_report(y_true, y_pred, digits=4))


# -------------------------------------------------------------------
#  MAIN
# -------------------------------------------------------------------
def main():
    # 1) Load BM25 & labels
    X_bm25, vocab, meta_bm = load_bm25_matrix()
    y = build_labels_from_meta(meta_bm)

    # 2) Dimensionality reduction + L2 normalization (≈ cosine)
    print("\n=== Dimensionality reduction (TruncatedSVD) ===")
    svd = TruncatedSVD(n_components=100, random_state=42)
    X_reduced = svd.fit_transform(X_bm25)
    print("X_reduced shape (before norm):", X_reduced.shape)

    # 🔹 L2 normalization -> euclidean distance ≈ cosine distance
    X_reduced = normalize(X_reduced, norm="l2")
    print("X_reduced shape (after norm):", X_reduced.shape)

    # ----------------------------------------------------------------
    #  KMEANS (k=2)  -- on L2-normalized SVD
    # ----------------------------------------------------------------
    print("\n=== KMeans clustering (k=2) ===")
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
    labels_km = kmeans.fit_predict(X_reduced)

    evaluate_clustering("KMeans on BM25 + SVD (L2-normalized)", labels_km, y)

    # ----------------------------------------------------------------
    #  DBSCAN (metric='cosine')
    # ----------------------------------------------------------------
    print("\n=== DBSCAN clustering ===")
    dbscan = DBSCAN(
        eps=0.3,          # peut être ajusté, mais 0.3 est raisonnable pour cosine
        min_samples=5,
        metric="cosine"
    )
    labels_db = dbscan.fit_predict(X_reduced)

    print("Clusters found (including -1 = noise):",
          {int(c): int(np.sum(labels_db == c)) for c in np.unique(labels_db)})

    evaluate_clustering("DBSCAN (cosine)", labels_db, y)

    # ----------------------------------------------------------------
    #  HDBSCAN (euclidean on L2-normalized vectors ≈ cosine)
    # ----------------------------------------------------------------
    print("\n=== HDBSCAN clustering ===")
    hdb = hdbscan.HDBSCAN(
        min_cluster_size=15,
        min_samples=5,
        metric="euclidean"   # sur des vecteurs L2-normalisés -> ≈ cosine
    )
    labels_hdb = hdb.fit_predict(X_reduced)

    print("Clusters found (including -1 = noise):",
          {int(c): int(np.sum(labels_hdb == c)) for c in np.unique(labels_hdb)})

    evaluate_clustering("HDBSCAN (euclidean on L2-normalized SVD ≈ cosine)", labels_hdb, y)

    # ----------------------------------------------------------------
    #  GMM (2 components) -- on L2-normalized SVD
    # ----------------------------------------------------------------
    print("\n=== Gaussian Mixture (GMM, n_components=2) ===")
    gmm = GaussianMixture(n_components=2, random_state=42)
    labels_gmm = gmm.fit_predict(X_reduced)

    print("Clusters found:",
          {int(c): int(np.sum(labels_gmm == c)) for c in np.unique(labels_gmm)})

    evaluate_clustering("Gaussian Mixture (on L2-normalized SVD)", labels_gmm, y)


if __name__ == "__main__":
    main()
