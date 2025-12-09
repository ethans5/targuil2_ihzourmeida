import pathlib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.sequence import pad_sequences

from utils import read_token_docs_for_ex2, extract_date_from_filename


# -------------------------------------------------------------------
#  1. Charger docs + labels
# -------------------------------------------------------------------
def load_docs_and_labels():
    # On utilise les docs lemmatisés pour limiter le vocabulaire
    docs, meta = read_token_docs_for_ex2("Lemm")

    # y : UK = 0, US = 1
    y = []
    for m in meta:
        parl = m.get("parliament", "UK")
        y.append(1 if parl == "US" else 0)
    y = np.array(y, dtype="int32")

    return docs, y


# -------------------------------------------------------------------
#  2. Construire vocab + séquences
# -------------------------------------------------------------------
def build_vocab_and_sequences(docs, vocab_size=10000, min_freq=2):
    """
    docs : liste de listes de tokens
    On construit :
    - un vocabulaire limité à vocab_size mots plus fréquents (freq >= min_freq)
    - des séquences d'indices pour chaque doc
    """

    # Compter les fréquences
    from collections import Counter
    counter = Counter()
    for doc in docs:
        counter.update(doc)

    # Trier par fréquence
    most_common = [w for (w, c) in counter.items() if c >= min_freq]
    most_common = sorted(most_common, key=lambda w: counter[w], reverse=True)
    most_common = most_common[:vocab_size - 2]  # -2 pour réserver 0=PAD, 1=OOV

    # Construire dictionnaire mot -> index
    word_index = {}
    word_index["<PAD>"] = 0
    word_index["<OOV>"] = 1
    for i, w in enumerate(most_common, start=2):
        word_index[w] = i

    # Construire séquences
    sequences = []
    for doc in docs:
        seq = []
        for w in doc:
            idx = word_index.get(w, 1)  # 1 = OOV
            seq.append(idx)
        sequences.append(seq)

    return word_index, sequences


# -------------------------------------------------------------------
#  3. Padding
# -------------------------------------------------------------------
def pad_sequences_docs(sequences, max_len=300):
    """
    Tronque / pad les séquences à max_len
    """
    X = pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")
    return X


# -------------------------------------------------------------------
#  4. Modèle ANN avec Embedding
# -------------------------------------------------------------------
def build_ann_with_embedding(vocab_size, max_len, embedding_dim=64, activation="relu"):
    model = models.Sequential([
        layers.Embedding(input_dim=vocab_size,
                         output_dim=embedding_dim,
                         input_length=max_len),
        layers.GlobalAveragePooling1D(),
        layers.Dense(10, activation=activation),
        layers.Dense(10, activation=activation),
        layers.Dense(7, activation=activation),
        layers.Dense(2, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def train_and_eval(model, X_train, X_val, X_test, y_train, y_val, y_test, name="ANN_Embedding"):
    print(f"\n===== TRAINING {name} =====")

    es = callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=3,
        restore_best_weights=True
    )
    ckpt = callbacks.ModelCheckpoint(
        filepath=f"{name}_best.keras",
        monitor="val_accuracy",
        save_best_only=True
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=15,
        batch_size=16,
        callbacks=[es, ckpt],
        verbose=1
    )

    print(f"\n===== EVALUATION {name} =====")
    y_pred = np.argmax(model.predict(X_test), axis=1)

    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))

    return history


# -------------------------------------------------------------------
#  MAIN
# -------------------------------------------------------------------
def main():
    print("=== Loading docs and labels ===")
    docs, y = load_docs_and_labels()

    print("Building vocab and sequences...")
    word_index, sequences = build_vocab_and_sequences(docs, vocab_size=10000, min_freq=2)
    vocab_size = len(word_index)
    print("Vocab size (with PAD/OOV):", vocab_size)

    max_len = 300  # tu peux tester 200, 400, etc.
    print("Padding sequences to length:", max_len)
    X = pad_sequences_docs(sequences, max_len=max_len)

    # Split test 20%
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Sur les 80% restants, 10% val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.10, random_state=42, stratify=y_train_full
    )

    print("Train:", X_train.shape)
    print("Val:  ", X_val.shape)
    print("Test: ", X_test.shape)

    # Modèle ReLU
    model_relu = build_ann_with_embedding(vocab_size, max_len, embedding_dim=64, activation="relu")
    train_and_eval(model_relu, X_train, X_val, X_test, y_train, y_val, y_test, name="ANN_Embed_ReLU")

    # Modèle GELU
    model_gelu = build_ann_with_embedding(vocab_size, max_len, embedding_dim=64, activation="gelu")
    train_and_eval(model_gelu, X_train, X_val, X_test, y_train, y_val, y_test, name="ANN_Embed_GELU")


if __name__ == "__main__":
    main()
