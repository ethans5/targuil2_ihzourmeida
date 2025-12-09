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


# -------------------------------------------------------------------
#  CHARGER BM25 + LABELS
# -------------------------------------------------------------------
def load_bm25_and_labels():
    """
    Charge la matrice BM25 sparse + meta et construit les labels :
      UK -> 0
      US -> 1
    """
    base = pathlib.Path(__file__).resolve().parents[1]
    mat_dir = base / "src" / "data" / "ex2_matrices"

    bm25_path = mat_dir / "BM25.npz"
    meta_path = mat_dir / "BM25.meta.json"

    print("Loading BM25 from:", bm25_path)
    X = sparse.load_npz(bm25_path)

    with open(meta_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    meta = info["meta"]
    y = np.array([1 if m["parliament"] == "US" else 0 for m in meta], dtype="int32")
    print("BM25 shape:", X.shape)
    print("BM25 labels counts:", {c: int(np.sum(y == c)) for c in [0, 1]})
    return X, y


# -------------------------------------------------------------------
#  CHARGER GOOGLE NEWS + LABELS
# -------------------------------------------------------------------
def load_googlenews_and_labels():
    """
    Charge la matrice de vecteurs de documents Google News (denses)
    + meta et construit les labels :
      UK -> 0
      US -> 1

    Fichiers générés par 02_ex2_build_matrices.py :
      - src/data/ex2_matrices/GOOGLENEWS.npz (clé 'mat')
      - src/data/ex2_matrices/GOOGLENEWS.meta.json
    """
    base = pathlib.Path(__file__).resolve().parents[1]
    mat_dir = base / "src" / "data" / "ex2_matrices"

    gn_path = mat_dir / "GOOGLENEWS.npz"
    meta_path = mat_dir / "GOOGLENEWS.meta.json"

    print("Loading GoogleNews from:", gn_path)
    data = np.load(gn_path)
    X_gn = data["mat"]

    with open(meta_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    meta = info["meta"]
    y = np.array([1 if m["parliament"] == "US" else 0 for m in meta], dtype="int32")
    dim = info.get("dim", X_gn.shape[1])

    print("GoogleNews shape:", X_gn.shape, f"(dim={dim})")
    print("GoogleNews labels counts:", {c: int(np.sum(y == c)) for c in [0, 1]})
    return X_gn, y


# -------------------------------------------------------------------
#  FONCTION GÉNÉRIQUE DE VISUALISATION t-SNE + UMAP + KMEANS
# -------------------------------------------------------------------
def visualize_representation(X, y, out_dir, prefix):
    """
    Applique :
      - t-SNE (metric=cosine)
      - UMAP (metric=cosine)
      - KMeans (k=2) sur l'espace t-SNE
    et sauvegarde les figures dans out_dir avec un préfixe donné.
    """
    # ---------------------------------------------------------------
    #  t-SNE avec cosine
    # ---------------------------------------------------------------
    print(f"t-SNE projection for {prefix} (metric='cosine')...")
    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=30,
        init="pca",
        metric="cosine"
    )
    X_tsne = tsne.fit_transform(X)

    plt.figure(figsize=(8, 6))
    for label, color, name in [(0, "blue", "UK"), (1, "red", "US")]:
        idx = (y == label)
        plt.scatter(X_tsne[idx, 0], X_tsne[idx, 1],
                    s=10, alpha=0.6, c=color, label=name)
    plt.legend()
    plt.title(f"t-SNE of {prefix} (true labels, cosine)")
    plt.tight_layout()
    tsne_path = out_dir / f"tsne_{prefix}_true_labels.png"
    plt.savefig(tsne_path, dpi=200)
    plt.close()
    print("Saved:", tsne_path)

    # ---------------------------------------------------------------
    #  UMAP avec cosine
    # ---------------------------------------------------------------
    print(f"UMAP projection for {prefix} (metric='cosine')...")
    reducer = umap.UMAP(
        n_components=2,
        random_state=42,
        metric="cosine"
    )
    X_umap = reducer.fit_transform(X)

    plt.figure(figsize=(8, 6))
    for label, color, name in [(0, "blue", "UK"), (1, "red", "US")]:
        idx = (y == label)
        plt.scatter(X_umap[idx, 0], X_umap[idx, 1],
                    s=10, alpha=0.6, c=color, label=name)
    plt.legend()
    plt.title(f"UMAP of {prefix} (true labels, cosine)")
    plt.tight_layout()
    umap_path = out_dir / f"umap_{prefix}_true_labels.png"
    plt.savefig(umap_path, dpi=200)
    plt.close()
    print("Saved:", umap_path)

    # ---------------------------------------------------------------
    #  KMeans sur t-SNE (k=2) pour la visualisation des clusters
    # ---------------------------------------------------------------
    print(f"KMeans on t-SNE for {prefix} (k=2)...")
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
    cluster_labels = kmeans.fit_predict(X_tsne)

    plt.figure(figsize=(8, 6))
    for cl in [0, 1]:
        idx = (cluster_labels == cl)
        plt.scatter(X_tsne[idx, 0], X_tsne[idx, 1],
                    s=10, alpha=0.6, label=f"Cluster {cl}")
    plt.legend()
    plt.title(f"t-SNE of {prefix} (KMeans clusters)")
    plt.tight_layout()
    km_path = out_dir / f"tsne_{prefix}_kmeans_clusters.png"
    plt.savefig(km_path, dpi=200)
    plt.close()
    print("Saved:", km_path)


# -------------------------------------------------------------------
#  MAIN
# -------------------------------------------------------------------
def main():
    base = pathlib.Path(__file__).resolve().parents[1]
    out_dir = base / "src" / "outputs" / "ex2"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------
    #  1) VISUALISATIONS POUR BM25
    # ---------------------------------------------------------------
    print("Loading BM25...")
    X_sparse, y_bm = load_bm25_and_labels()

    # SVD + normalisation L2 pour approximer la distance cosinus
    print("SVD reduction on BM25...")
    svd = TruncatedSVD(n_components=50, random_state=42)
    X_red = svd.fit_transform(X_sparse)
    X_red = normalize(X_red, norm="l2")  # ≈ espace cosinus

    # Visualisation BM25
    visualize_representation(X_red, y_bm, out_dir, prefix="bm25")

    # ---------------------------------------------------------------
    #  2) VISUALISATIONS POUR GOOGLE NEWS
    # ---------------------------------------------------------------
    print("\nLoading GoogleNews embeddings...")
    X_gn, y_gn = load_googlenews_and_labels()

    # Normalisation L2 (les vecteurs sont déjà denses, dim=300)
    print("L2-normalizing GoogleNews embeddings...")
    X_gn_norm = normalize(X_gn, norm="l2")

    # Visualisation Google News
    visualize_representation(X_gn_norm, y_gn, out_dir, prefix="gn")


if __name__ == "__main__":
    main()
