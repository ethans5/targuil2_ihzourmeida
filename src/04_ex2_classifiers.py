import json
import pathlib
import numpy as np
from scipy import sparse

import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone

from utils import read_token_docs_for_ex2, ensure_dir


# -------------------------------------------------------------------
#  CHARGER BM25 + META
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
#  MATRICE DE COMPTES (POUR NB)
# -------------------------------------------------------------------
def build_counts_matrix(vocab, docs):
    """
    Construit une matrice sparse (docs x vocab_size) de comptes de mots.
    Vocab et ordre des docs DOIVENT être les mêmes que pour BM25.
    """
    vocab_index = {w: i for i, w in enumerate(vocab)}
    rows, cols, data = [], [], []

    for doc_id, doc in enumerate(docs):
        counts = {}
        for w in doc:
            j = vocab_index.get(w)
            if j is not None:
                counts[j] = counts.get(j, 0) + 1
        for j, tf in counts.items():
            rows.append(doc_id)
            cols.append(j)
            data.append(tf)

    X_counts = sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(len(docs), len(vocab)),
        dtype=np.float32
    )
    print("Counts matrix shape:", X_counts.shape)
    return X_counts


# -------------------------------------------------------------------
#  CROSS-VALIDATION GÉNÉRIQUE
# -------------------------------------------------------------------
def cross_validate_model(name, clf_template, X, y, n_splits=10):
    print(f"\n=== {name} - {n_splits}-fold cross-validation ===")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs, precs, recs, f1s = [], [], [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        clf = clone(clf_template)

        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)

        acc = accuracy_score(y_te, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_te, y_pred, average="weighted", zero_division=0
        )

        accs.append(acc)
        precs.append(prec)
        recs.append(rec)
        f1s.append(f1)

        print(f"  Fold {fold:2d}: acc={acc:.4f}, f1={f1:.4f}")

    print(f"\n>>> {name} - CV mean scores:")
    print(f"Accuracy : {np.mean(accs):.4f}")
    print(f"Precision: {np.mean(precs):.4f}")
    print(f"Recall   : {np.mean(recs):.4f}")
    print(f"F1-score : {np.mean(f1s):.4f}")


# -------------------------------------------------------------------
#  FEATURE IMPORTANCE
# -------------------------------------------------------------------
def top_features_nb(clf, vocab, k=20):
    """
    MultinomialNB: feature_log_prob_[class, feature]
    On prend les k mots les plus probables par classe.
    """
    results = {}
    log_prob = clf.feature_log_prob_  # shape (2, n_features)
    classes = clf.classes_           # [0, 1]

    for cls_index, cls_label in enumerate(classes):
        scores = log_prob[cls_index]  # log P(w | class)
        idx = np.argsort(-scores)[:k]
        feats = [vocab[i] for i in idx]
        vals = scores[idx]

        df = pd.DataFrame({
            "feature": feats,
            "score": vals,
            "class": cls_label
        })
        results[f"NB_class_{cls_label}"] = df

    return results


def top_features_linear_model(clf, vocab, model_name="LR", k=20):
    """
    Pour LogisticRegression / LinearSVC :
    coef_[0, j] > 0 => plus associé à la classe 1 (US)
    coef_[0, j] < 0 => plus associé à la classe 0 (UK)
    On prend top k pour chaque classe.
    """
    coef = clf.coef_[0]  # shape (n_features,)

    # Classe 1 (US) : plus gros positifs
    idx_pos = np.argsort(-coef)[:k]
    feats_pos = [vocab[i] for i in idx_pos]
    vals_pos = coef[idx_pos]

    df_pos = pd.DataFrame({
        "feature": feats_pos,
        "weight": vals_pos,
        "class": 1
    })

    # Classe 0 (UK) : plus gros négatifs
    idx_neg = np.argsort(coef)[:k]
    feats_neg = [vocab[i] for i in idx_neg]
    vals_neg = coef[idx_neg]

    df_neg = pd.DataFrame({
        "feature": feats_neg,
        "weight": vals_neg,
        "class": 0
    })

    return {
        f"{model_name}_class_1": df_pos,
        f"{model_name}_class_0": df_neg
    }


def top_features_rf(clf, vocab, k=20):
    """
    RandomForest: feature_importances_ globales
    (pas par classe, mais importance globale dans la séparation)
    """
    imp = clf.feature_importances_  # shape (n_features,)
    idx = np.argsort(-imp)[:k]
    feats = [vocab[i] for i in idx]
    vals = imp[idx]

    df = pd.DataFrame({
        "feature": feats,
        "importance": vals
    })
    return {"RF_global": df}


# -------------------------------------------------------------------
#  ANALYSE COMPARATIVE DES FEATURES
# -------------------------------------------------------------------
def analyze_common_features(nb_feats, lr_feats, svm_feats, rf_feats):
    """
    Analyse approfondie des features communes entre les modèles.
    - NB, LR, SVM : par classe (UK=0, US=1)
    - RF : importance globale
    On calcule :
      * features communes aux 3 méthodes (ALL)
      * features communes à au moins 2 méthodes (MOST)
      * fréquence d'apparition dans NB/LR/SVM
      * overlap avec les top features RF
    """
    from collections import Counter

    # --- Features par classe et par méthode ---
    nb_uk = set(nb_feats["NB_class_0"]["feature"].tolist())
    nb_us = set(nb_feats["NB_class_1"]["feature"].tolist())

    lr_uk = set(lr_feats["LR_class_0"]["feature"].tolist())
    lr_us = set(lr_feats["LR_class_1"]["feature"].tolist())

    svm_uk = set(svm_feats["SVM_class_0"]["feature"].tolist())
    svm_us = set(svm_feats["SVM_class_1"]["feature"].tolist())

    rf_global = set(rf_feats["RF_global"]["feature"].tolist())

    # --- 1) Communes aux 3 méthodes (ALL) ---
    common_all_uk = nb_uk & lr_uk & svm_uk
    common_all_us = nb_us & lr_us & svm_us

    # --- 2) Communes à au moins 2 méthodes (MOST) ---
    common_2plus_uk = (nb_uk & lr_uk) | (nb_uk & svm_uk) | (lr_uk & svm_uk)
    common_2plus_us = (nb_us & lr_us) | (nb_us & svm_us) | (lr_us & svm_us)

    # --- 3) Fréquence d'apparition (sur NB/LR/SVM) ---
    uk_counter = Counter()
    for s in (nb_uk, lr_uk, svm_uk):
        for f in s:
            uk_counter[f] += 1

    us_counter = Counter()
    for s in (nb_us, lr_us, svm_us):
        for f in s:
            us_counter[f] += 1

    print("\n" + "=" * 70)
    print("ANALYSE COMPLÈTE DES FEATURES COMMUNES")
    print("=" * 70)

    print("\n--- Features UK ---")
    print(f"Communes à 3/3 méthodes (ALL): {len(common_all_uk)} features")
    print(f"  {sorted(common_all_uk)}")

    print(f"\nCommunes à 2/3 méthodes (MOST): {len(common_2plus_uk)} features")
    print(f"  Features apparaissant 2 fois: {[f for f, c in uk_counter.items() if c == 2]}")

    print("\n--- Features US ---")
    print(f"Communes à 3/3 méthodes (ALL): {len(common_all_us)} features")
    print(f"  {sorted(common_all_us)}")

    print(f"\nCommunes à 2/3 méthodes (MOST): {len(common_2plus_us)} features")
    print(f"  Features apparaissant 2 fois: {[f for f, c in us_counter.items() if c == 2]}")

    print("\n--- Random Forest overlap ---")
    print(f"RF features in UK top features: {rf_global & (nb_uk | lr_uk | svm_uk)}")
    print(f"RF features in US top features: {rf_global & (nb_us | lr_us | svm_us)}")

    # Retourner pour pouvoir exporter dans Excel
    return {
        "common_all_uk": common_all_uk,
        "common_all_us": common_all_us,
        "common_most_uk": common_2plus_uk,
        "common_most_us": common_2plus_us,
        "uk_frequency": uk_counter,
        "us_frequency": us_counter
    }


# -------------------------------------------------------------------
#  MAIN
# -------------------------------------------------------------------
def main():
    base = pathlib.Path(__file__).resolve().parents[1]
    out_dir = base / "src" / "outputs" / "ex2"
    ensure_dir(out_dir)

    # 1) Charger BM25
    X_bm25, vocab, meta_bm = load_bm25_matrix()
    y = build_labels_from_meta(meta_bm)

    # 2) Charger les docs pour la matrice de comptes (pour NB)
    print("\n=== Loading token docs for EX2 (Lemm) ===")
    docs, meta_docs = read_token_docs_for_ex2("Lemm")
    print(f"Docs loaded: {len(docs)}")

    X_counts = build_counts_matrix(vocab, docs)

    # 3) Définir les modèles
    nb_clf = MultinomialNB()
    lr_clf = LogisticRegression(penalty="l2", solver="liblinear", max_iter=1000)
    svm_clf = LinearSVC(C=1.0, max_iter=2000)
    rf_clf = RandomForestClassifier(n_estimators=200, max_depth=None, n_jobs=-1, random_state=42)

    # 4) Cross-validation
    cross_validate_model("Naive Bayes (MultinomialNB)", nb_clf, X_counts, y)
    cross_validate_model("Logistic Regression", lr_clf, X_bm25, y)
    cross_validate_model("Linear SVM", svm_clf, X_bm25, y)
    cross_validate_model("Random Forest", rf_clf, X_bm25, y)

    # 5) Feature importance : entraînement sur TOUTES les données
    print("\n=== Training models on full data for feature importance ===")

    nb_full = MultinomialNB()
    nb_full.fit(X_counts, y)
    nb_feats = top_features_nb(nb_full, vocab, k=20)

    lr_full = LogisticRegression(penalty="l2", solver="liblinear", max_iter=1000)
    lr_full.fit(X_bm25, y)
    lr_feats = top_features_linear_model(lr_full, vocab, model_name="LR", k=20)

    svm_full = LinearSVC(C=1.0, max_iter=2000)
    svm_full.fit(X_bm25, y)
    svm_feats = top_features_linear_model(svm_full, vocab, model_name="SVM", k=20)

    rf_full = RandomForestClassifier(n_estimators=200, max_depth=None, n_jobs=-1, random_state=42)
    rf_full.fit(X_bm25, y)
    rf_feats = top_features_rf(rf_full, vocab, k=20)

    # 5.5) Analyse comparative des features
    common_analysis = analyze_common_features(nb_feats, lr_feats, svm_feats, rf_feats)

    # 6) Sauvegarde Excel
    excel_path = out_dir / "ex2_feature_importance.xlsx"
    print("\nSaving feature importances to:", excel_path)

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:

        # --- SUMMARY SHEET ---
        nb_uk = set(nb_feats["NB_class_0"]["feature"])
        lr_uk = set(lr_feats["LR_class_0"]["feature"])
        svm_uk = set(svm_feats["SVM_class_0"]["feature"])
        nb_us = set(nb_feats["NB_class_1"]["feature"])
        lr_us = set(lr_feats["LR_class_1"]["feature"])
        svm_us = set(svm_feats["SVM_class_1"]["feature"])
        rf_global = set(rf_feats["RF_global"]["feature"])

        summary_data = {
            "Metric": [
                "Total Vocabulary Size",
                "UK - Common to ALL (3/3 methods)",
                "US - Common to ALL (3/3 methods)",
                "UK - Common to MOST (2+/3 methods)",
                "US - Common to MOST (2+/3 methods)",
                "RF overlap with UK features",
                "RF overlap with US features"
            ],
            "Count": [
                len(vocab),
                len(common_analysis["common_all_uk"]),
                len(common_analysis["common_all_us"]),
                len(common_analysis["common_most_uk"]),
                len(common_analysis["common_most_us"]),
                len(rf_global & (nb_uk | lr_uk | svm_uk)),
                len(rf_global & (nb_us | lr_us | svm_us))
            ]
        }
        pd.DataFrame(summary_data).to_excel(writer, sheet_name="SUMMARY", index=False)

        # --- FEATURES PAR MÉTHODE ---
        for sheet, df in nb_feats.items():
            df.to_excel(writer, sheet_name=sheet[:31], index=False)

        for sheet, df in lr_feats.items():
            df.to_excel(writer, sheet_name=sheet[:31], index=False)

        for sheet, df in svm_feats.items():
            df.to_excel(writer, sheet_name=sheet[:31], index=False)

        for sheet, df in rf_feats.items():
            df.to_excel(writer, sheet_name="RF_global", index=False)

        # --- ANALYSE COMPARATIVE ---
        pd.DataFrame({
            "Feature": sorted(common_analysis["common_all_uk"]),
            "Type": "UK - All 3 methods"
        }).to_excel(writer, sheet_name="Common_ALL_UK", index=False)

        pd.DataFrame({
            "Feature": sorted(common_analysis["common_all_us"]),
            "Type": "US - All 3 methods"
        }).to_excel(writer, sheet_name="Common_ALL_US", index=False)

        pd.DataFrame({
            "Feature": sorted(common_analysis["common_most_uk"]),
            "Type": "UK - At least 2 methods"
        }).to_excel(writer, sheet_name="Common_MOST_UK", index=False)

        pd.DataFrame({
            "Feature": sorted(common_analysis["common_most_us"]),
            "Type": "US - At least 2 methods"
        }).to_excel(writer, sheet_name="Common_MOST_US", index=False)

        # --- FRÉQUENCES ---
        uk_freq_df = pd.DataFrame([
            {"Feature": f, "Occurrences": c, "Percentage": f"{c/3*100:.1f}%"}
            for f, c in common_analysis["uk_frequency"].most_common(30)
        ])
        uk_freq_df.to_excel(writer, sheet_name="UK_Freq_Analysis", index=False)

        us_freq_df = pd.DataFrame([
            {"Feature": f, "Occurrences": c, "Percentage": f"{c/3*100:.1f}%"}
            for f, c in common_analysis["us_frequency"].most_common(30)
        ])
        us_freq_df.to_excel(writer, sheet_name="US_Freq_Analysis", index=False)

    print("Done.")


if __name__ == "__main__":
    main()
