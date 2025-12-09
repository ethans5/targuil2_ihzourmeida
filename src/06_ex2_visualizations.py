import json
import pathlib
import numpy as np
from scipy import sparse

import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import normalize
from sklearn.mixture import GaussianMixture
import umap
import hdbscan


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
#  FONCTION UTILITAIRE POUR PLOTER LES CLUSTERS SUR t-SNE
# -------------------------------------------------------------------
def plot_clusters_on_tsne(X_tsne, cluster_labels, title, out_path):
    """
    Affiche les points t-SNE colorés par labels de clusters.
    Gère aussi le label -1 (noise DBSCAN/HDBSCAN).
    """
    unique_labels = np.unique(cluster_labels)

    plt.figure(figsize=(8, 6))
    for lab in unique_labels:
        idx = (cluster_labels == lab)
        if lab == -1:
            label_name = "Noise (-1)"
        else:
            label_name = f"Cluster {lab}"
        plt.scatter(X_tsne[idx, 0], X_tsne[idx, 1],
                    s=10, alpha=0.6, label=label_name)
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print("Saved:", out_path)


# -------------------------------------------------------------------
#  FONCTION GÉNÉRIQUE DE VISUALISATION
# -------------------------------------------------------------------
def visualize_representation(X, y, out_dir, prefix):
    """
    X : représentation haute dimension (déjà normalisée L2)
    y : labels vrais (UK=0, US=1)
    prefix : 'bm25' ou 'gn'

    Étapes :
      1) t-SNE + UMAP colorés par labels vrais
      2) Clustering dans l'espace X avec :
           - KMeans (k=2)
           - DBSCAN (cosine)
           - HDBSCAN
           - Gaussian Mixture (n_components=2)
         Puis visualisation des clusters sur la carte t-SNE.
    """
    # ---------------------------------------------------------------
    #  t-SNE avec cosine (pour visualiser en 2D)
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
    #  UMAP avec cosine (pour visualiser en 2D)
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
    #  CLUSTERING DANS L'ESPACE X
    # ---------------------------------------------------------------
    print(f"\n=== Clustering on {prefix} representation ===")

    # 1) KMeans (k=2)
    print("KMeans (k=2)...")
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
    labels_km = kmeans.fit_predict(X)
    km_path = out_dir / f"tsne_{prefix}_kmeans_clusters.png"
    plot_clusters_on_tsne(
        X_tsne, labels_km,
        title=f"t-SNE of {prefix} (KMeans clusters)",
        out_path=km_path
    )

    # 2) DBSCAN (cosine) – paramètres comme dans le fichier de clustering
    print("DBSCAN (cosine)...")
    dbscan = DBSCAN(
        eps=0.3,
        min_samples=5,
        metric="cosine"
    )
    labels_db = dbscan.fit_predict(X)
    print("DBSCAN clusters (incl. -1):",
          {int(c): int(np.sum(labels_db == c)) for c in np.unique(labels_db)})
    db_path = out_dir / f"tsne_{prefix}_dbscan_clusters.png"
    plot_clusters_on_tsne(
        X_tsne, labels_db,
        title=f"t-SNE of {prefix} (DBSCAN clusters)",
        out_path=db_path
    )

    # 3) HDBSCAN (euclidean sur vecteurs L2-normalisés ≈ cosine)
    print("HDBSCAN ...")
    hdb = hdbscan.HDBSCAN(
        min_cluster_size=15,
        min_samples=5,
        metric="euclidean"
    )
    labels_hdb = hdb.fit_predict(X)
    print("HDBSCAN clusters (incl. -1):",
          {int(c): int(np.sum(labels_hdb == c)) for c in np.unique(labels_hdb)})
    hdb_path = out_dir / f"tsne_{prefix}_hdbscan_clusters.png"
    plot_clusters_on_tsne(
        X_tsne, labels_hdb,
        title=f"t-SNE of {prefix} (HDBSCAN clusters)",
        out_path=hdb_path
    )

    # 4) Gaussian Mixture (2 composantes)
    print("Gaussian Mixture (n_components=2)...")
    gmm = GaussianMixture(n_components=2, random_state=42)
    labels_gmm = gmm.fit_predict(X)
    print("GMM clusters:",
          {int(c): int(np.sum(labels_gmm == c)) for c in np.unique(labels_gmm)})
    gmm_path = out_dir / f"tsne_{prefix}_gmm_clusters.png"
    plot_clusters_on_tsne(
        X_tsne, labels_gmm,
        title=f"t-SNE of {prefix} (GMM clusters)",
        out_path=gmm_path
    )


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
    X_bm = svd.fit_transform(X_sparse)
    X_bm = normalize(X_bm, norm="l2")  # ≈ espace cosinus

    visualize_representation(X_bm, y_bm, out_dir, prefix="bm25")

    # ---------------------------------------------------------------
    #  2) VISUALISATIONS POUR GOOGLE NEWS
    # ---------------------------------------------------------------
    print("\nLoading GoogleNews embeddings...")
    X_gn, y_gn = load_googlenews_and_labels()

    # Normalisation L2
    print("L2-normalizing GoogleNews embeddings...")
    X_gn_norm = normalize(X_gn, norm="l2")

    visualize_representation(X_gn_norm, y_gn, out_dir, prefix="gn")


if __name__ == "__main__":
    main()
