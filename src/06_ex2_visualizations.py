import json
import pathlib
import numpy as np
from scipy import sparse

import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import umap


def load_bm25_and_labels():
    base = pathlib.Path(__file__).resolve().parents[1]
    mat_dir = base / "src" / "data" / "ex2_matrices"

    bm25_path = mat_dir / "BM25.npz"
    meta_path = mat_dir / "BM25.meta.json"

    X = sparse.load_npz(bm25_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    meta = info["meta"]
    y = np.array([1 if m["parliament"] == "US" else 0 for m in meta], dtype="int32")
    return X, y


def main():
    base = pathlib.Path(__file__).resolve().parents[1]
    out_dir = base / "src" / "outputs" / "ex2"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading BM25...")
    X_sparse, y = load_bm25_and_labels()

    # ---------------------------------------------------------------
    #  SVD + L2 normalization  (≈ cosine space)
    # ---------------------------------------------------------------
    print("SVD reduction...")
    svd = TruncatedSVD(n_components=50, random_state=42)
    X_red = svd.fit_transform(X_sparse)

    # 🔹 L2-normalize to approximate cosine distance
    X_red = normalize(X_red, norm="l2")

    # ---------------------------------------------------------------
    #  t-SNE with cosine metric
    # ---------------------------------------------------------------
    print("t-SNE projection (metric='cosine')...")
    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=30,
        init="pca",
        metric="cosine"
    )
    X_tsne = tsne.fit_transform(X_red)

    plt.figure(figsize=(8, 6))
    for label, color, name in [(0, "blue", "UK"), (1, "red", "US")]:
        idx = (y == label)
        plt.scatter(X_tsne[idx, 0], X_tsne[idx, 1],
                    s=10, alpha=0.6, c=color, label=name)
    plt.legend()
    plt.title("t-SNE of BM25+SVD (true labels, cosine)")
    plt.tight_layout()
    tsne_path = out_dir / "tsne_bm25_true_labels.png"
    plt.savefig(tsne_path, dpi=200)
    plt.close()
    print("Saved:", tsne_path)

    # ---------------------------------------------------------------
    #  UMAP with cosine metric
    # ---------------------------------------------------------------
    print("UMAP projection (metric='cosine')...")
    reducer = umap.UMAP(
        n_components=2,
        random_state=42,
        metric="cosine"
    )
    X_umap = reducer.fit_transform(X_red)

    plt.figure(figsize=(8, 6))
    for label, color, name in [(0, "blue", "UK"), (1, "red", "US")]:
        idx = (y == label)
        plt.scatter(X_umap[idx, 0], X_umap[idx, 1],
                    s=10, alpha=0.6, c=color, label=name)
    plt.legend()
    plt.title("UMAP of BM25+SVD (true labels, cosine)")
    plt.tight_layout()
    umap_path = out_dir / "umap_bm25_true_labels.png"
    plt.savefig(umap_path, dpi=200)
    plt.close()
    print("Saved:", umap_path)

    # ---------------------------------------------------------------
    #  KMeans on t-SNE (for cluster viz)
    # ---------------------------------------------------------------
    print("KMeans on t-SNE (k=2)...")
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
    cluster_labels = kmeans.fit_predict(X_tsne)

    plt.figure(figsize=(8, 6))
    for cl in [0, 1]:
        idx = (cluster_labels == cl)
        plt.scatter(X_tsne[idx, 0], X_tsne[idx, 1],
                    s=10, alpha=0.6, label=f"Cluster {cl}")
    plt.legend()
    plt.title("t-SNE of BM25+SVD (KMeans clusters)")
    plt.tight_layout()
    km_path = out_dir / "tsne_bm25_kmeans_clusters.png"
    plt.savefig(km_path, dpi=200)
    plt.close()
    print("Saved:", km_path)


if __name__ == "__main__":
    main()
