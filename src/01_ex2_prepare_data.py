import os
import pathlib
from bs4 import BeautifulSoup
import spacy

from utils import ensure_dir, tokenize_with_punct_separated, iter_files, extract_date_from_filename
# utils.py contient déjà ensure_dir(), tokenize_with_punct_separated(), iter_files() et extract_date_from_filename()


def read_raw_text(in_path: str) -> str:
    """
    Lit un fichier brut et renvoie le texte.
    - Si le fichier est XML → extraction texte avec BeautifulSoup
    - Sinon → lecture texte normal
    """
    in_path = str(in_path)
    if in_path.lower().endswith(".xml"):
        with open(in_path, "rb") as f:
            soup = BeautifulSoup(f, "lxml-xml")
        return soup.get_text(" ", strip=True)
    else:
        with open(in_path, "r", encoding="utf-8") as f:
            return f.read()


def write_tokens(tokens, out_path: pathlib.Path):
    """
    Écrit les tokens séparés par des espaces dans un fichier .txt
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(" ".join(tokens))


def process_parliament(raw_dir: pathlib.Path,
                       out_word_dir: pathlib.Path,
                       out_lemm_dir: pathlib.Path,
                       nlp):
    """
    Traite tous les fichiers UK/US :
    - Tokenisation avec ponctuation séparée (mode Word)
    - Lemmatisation avec spaCy (mode Lemm)
    """
    raw_dir = pathlib.Path(raw_dir)
    out_word_dir = pathlib.Path(out_word_dir)
    out_lemm_dir = pathlib.Path(out_lemm_dir)

    ensure_dir(out_word_dir)
    ensure_dir(out_lemm_dir)

    count = 0
    for in_path in iter_files(str(raw_dir), ext=(".txt", ".xml")):
        in_path = pathlib.Path(in_path)
        fn = in_path.name
        base = fn.rsplit(".", 1)[0]

        # sorties
        out_word = out_word_dir / f"{base}.txt"
        out_lemm = out_lemm_dir / f"{base}.txt"

        # lire texte brut
        text = read_raw_text(str(in_path))

        # --- Version Word : ponctuation séparée ---
        tokens = tokenize_with_punct_separated(text)
        write_tokens(tokens, out_word)

        # --- Version Lemma ---
        doc = nlp(text)
        lemmas = []
        for tok in doc:
            if tok.is_punct:
                continue
            elif tok.like_num:
                continue
            else:
                lemmas.append(tok.lemma_.lower())
        write_tokens(lemmas, out_lemm)

        print(f"processed: {fn}")
        count += 1

    print(f"Total processed in {raw_dir}: {count}")


def main():
    """
    Prépare les données EX2 :
    - src/data/ex2_raw/UK → src/data/ex2_clean/UK + src/data/ex2_lemm/UK
    - src/data/ex2_raw/US → src/data/ex2_clean/US + src/data/ex2_lemm/US
    """
    base_dir = pathlib.Path(__file__).resolve().parents[1]
    data_dir = base_dir / "src" / "data"

    raw_root = data_dir / "ex2_raw"
    clean_root = data_dir / "ex2_clean"
    lemm_root = data_dir / "ex2_lemm"

    uk_raw = raw_root / "UK"
    us_raw = raw_root / "US"

    uk_clean = clean_root / "UK"
    us_clean = clean_root / "US"

    uk_lemm = lemm_root / "UK"
    us_lemm = lemm_root / "US"

    print("RAW_ROOT   ->", raw_root)
    print("CLEAN_ROOT ->", clean_root)
    print("LEMM_ROOT  ->", lemm_root)

    # spaCy (comme dans ton Exo 1)
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "textcat"])

    print("\n=== Processing UK ===")
    process_parliament(uk_raw, uk_clean, uk_lemm, nlp)

    print("\n=== Processing US ===")
    process_parliament(us_raw, us_clean, us_lemm, nlp)


if __name__ == "__main__":
    main()
