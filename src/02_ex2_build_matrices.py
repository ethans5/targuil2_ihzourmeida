import json
import pathlib
import numpy as np
from tqdm import tqdm
from scipy import sparse
from gensim.downloader import load as gload

from utils import ensure_dir, read_token_docs_for_ex2


# ------------------------------------------------------------
# BM25 IMPLEMENTATION (identique à ton exo 1, fiable + rapide)
# ------------------------------------------------------------

def bm25_compute_matrix(docs, vocab):
    """
    Build BM25 matrix (N_docs x vocab_size)
    docs : list of list of tokens
    vocab : list of words
    """
    vocab_index = {w: i for i, w in enumerate(vocab)}
    n_docs = len(docs)
    V = len(vocab)

    # Term frequencies
    rows = []
    cols = []
    data = []

    df = np.zeros(V, dtype=np.int32)   # document frequency

    for doc_id, doc in enumerate(docs):
        counts = {}
        for w in doc:
            if w in vocab_index:
                counts[w] = counts.get(w, 0) + 1
        for w, tf in counts.items():
            j = vocab_index[w]
            df[j] += 1
            rows.append(doc_id)
            cols.append(j)
            data.append(tf)

    tf_matrix = sparse.csr_matrix((data, (rows, cols)), shape=(n_docs, V), dtype=np.float32)

    # BM25 parameters
    k1 = 1.5
    b = 0.75

    # doc length + avg length
    doc_lengths = np.array([len(d) for d in docs], dtype=np.float32)
    avgdl = doc_lengths.mean()

    # idf
    idf = np.log((n_docs - df + 0.5) / (df + 0.5))

    # build BM25 matrix
    bm25_rows = []
    bm25_cols = []
    bm25_data = []

    for i in range(n_docs):
        dl = doc_lengths[i]
        norm = k1 * (1 - b + b * dl / avgdl)

        row = tf_matrix.getrow(i)
        for idx in row.indices:
            tf = row[0, idx]
            score = idf[idx] * ((tf * (k1 + 1)) / (tf + norm))
            bm25_rows.append(i)
            bm25_cols.append(idx)
            bm25_data.append(score)

    bm25 = sparse.csr_matrix((bm25_data, (bm25_rows, bm25_cols)), shape=(n_docs, V), dtype=np.float32)
    return bm25


# ------------------------------------------------------------
# GOOGLE NEWS DOC EMBEDDINGS
# ------------------------------------------------------------

def doc_mean_vector(tokens, kv, dim):
    """Average word vectors of token list."""
    vecs = []
    for w in tokens:
        if w in kv:
            vecs.append(kv[w])
    if not vecs:
        return np.zeros(dim, dtype=np.float32)
    return np.mean(vecs, axis=0).astype(np.float32)


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():
    base = pathlib.Path(__file__).resolve().parents[1]
    out_dir = base / "src" / "data" / "ex2_matrices"
    ensure_dir(out_dir)

    print("=== Loading EX2 documents ===")
    docs, meta = read_token_docs_for_ex2("Lemm")   # Lemm recommended
    print(f"Total docs loaded: {len(docs)}")

    # --------------------------------------------------------
    # BUILD VOCAB
    # --------------------------------------------------------
    print("\n=== Building Vocabulary ===")
    vocab = sorted({w for doc in docs for w in doc})
    print(f"Vocab size: {len(vocab)}")

    # --------------------------------------------------------
    # BM25 MATRIX
    # --------------------------------------------------------
    print("\n=== Computing BM25 ===")
    bm25_mat = bm25_compute_matrix(docs, vocab)

    print("Saving BM25...")
    sparse.save_npz(str(out_dir / "BM25.npz"), bm25_mat)

    with open(out_dir / "BM25.meta.json", "w", encoding="utf-8") as f:
        json.dump({
            "vocab": vocab,
            "meta": meta
        }, f, ensure_ascii=False, indent=2)

    print("[OK] BM25 saved.")

    # --------------------------------------------------------
    # GOOGLE NEWS WORD2VEC
    # --------------------------------------------------------
    print("\n=== Loading Google News Word2Vec (300D) ===")
    kv = gload("word2vec-google-news-300")
    dim = kv.vector_size
    print("Google News loaded.")

    print("\n=== Building Google News Matrix ===")
    mat = np.vstack([doc_mean_vector(doc, kv, dim) for doc in tqdm(docs)])

    print("Saving Google News vectors...")
    np.savez_compressed(str(out_dir / "GOOGLENEWS.npz"), mat=mat.astype(np.float32))

    with open(out_dir / "GOOGLENEWS.meta.json", "w", encoding="utf-8") as f:
        json.dump({
            "dim": dim,
            "meta": meta,
            "type": "GOOGLENEWS"
        }, f, ensure_ascii=False, indent=2)

    print("[OK] Google News vectors saved.")


if __name__ == "__main__":
    main()
