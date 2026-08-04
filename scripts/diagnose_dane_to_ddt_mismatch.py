"""
CLAUDE

Diagnose text mismatches between DDT and DaNE sentences that are keyed by
the same sent_id. Lives in scripts/, same as combine.py, and resolves
assets/ and corpus/ relative to the project root (one level up). Run it
from the project root with da_ddt / dane already converted to
corpus/da_ddt/{split}.spacy and corpus/dane/{split}.spacy.

For each mismatch it prints:
  - sent_id
  - the two raw texts (repr'd, so whitespace/invisible chars are visible)
  - whether they match after normalizing whitespace
  - a char-level diff if they still don't match after normalizing
"""

import difflib
from pathlib import Path

import spacy
from conllu import parse
from spacy.tokens import Doc
from spacy.training.corpus import Corpus

file_path = Path(__file__)
assets_path = file_path.parent.parent / "assets"
corpus_path = file_path.parent.parent / "corpus"

Doc.set_extension("sent_id", default=None, force=True)
Doc.set_extension("conllu", default=None, force=True)


def normalize(s: str) -> str:
    """Collapse all whitespace runs to single spaces and strip."""
    return " ".join(s.split())


def _add_sent_id(docs, split, dataset):
    path = assets_path / dataset / f"{split}.conllu"
    with path.open(encoding="utf-8") as f:
        text = f.read()
    sentences = parse(text)
    for sent, doc in zip(sentences, docs):
        doc._.sent_id = sent.metadata["sent_id"]
    return docs


def load(dataset_dir_name, split):
    nlp = spacy.blank("da")
    path = corpus_path / dataset_dir_name / f"{split}.spacy"
    corpus = Corpus(path, shuffle=False)
    examples = list(corpus(nlp))
    docs = [e.reference for e in examples]
    return _add_sent_id(docs, split, dataset=dataset_dir_name)


def diagnose(split):
    print(f"\n{'=' * 70}\nsplit: {split}\n{'=' * 70}")
    ddt_docs = load("da_ddt", split)
    dane_docs = load("dane", split)

    if len(ddt_docs) != len(dane_docs):
        print(f"!! length mismatch: ddt={len(ddt_docs)}, dane={len(dane_docs)}")
        print("   (sent_id-based alignment may be broken upstream; stopping here)")
        return

    total = 0
    raw_mismatch = 0
    normalized_still_mismatch = 0
    token_safe_mismatches = [0]  # mutable so the inner loop can bump it

    for ddt_doc, dane_doc in zip(ddt_docs, dane_docs):
        total += 1
        t_ddt = ddt_doc.text
        t_dane = dane_doc.text

        if t_ddt.strip() == t_dane.strip():
            continue  # matches under the original assertion, nothing to report

        raw_mismatch += 1
        norm_ddt = normalize(t_ddt)
        norm_dane = normalize(t_dane)
        sid = ddt_doc._.sent_id or dane_doc._.sent_id or "?"

        # What actually matters for add_dane_to_ddt: do the token
        # sequences line up? Entity spans are transferred by token
        # index, so text mismatches are harmless if tokens match.
        toks_ddt = [t.text for t in ddt_doc]
        toks_dane = [t.text for t in dane_doc]
        tokens_match = toks_ddt == toks_dane

        if norm_ddt == norm_dane:
            print(f"\n[{sid}] whitespace-only text mismatch (safe to normalize)")
            print(f"  ddt : {t_ddt!r}")
            print(f"  dane: {t_dane!r}")
        else:
            normalized_still_mismatch += 1
            status = "TOKENS MATCH (safe for ent transfer)" if tokens_match else "TOKENS DIFFER (unsafe!)"
            print(f"\n[{sid}] CONTENT mismatch in text -- {status}")
            print(f"  ddt : {t_ddt!r}")
            print(f"  dane: {t_dane!r}")
            if not tokens_match:
                print(f"  ddt tokens : {toks_ddt}")
                print(f"  dane tokens: {toks_dane}")
                diff = difflib.ndiff(norm_ddt, norm_dane)
                changed = "".join(c for c in diff if c.startswith(("+", "-")))
                print(f"  char-level diff (added/removed only): {changed!r}")
            if tokens_match:
                token_safe_mismatches[0] += 1

    print(f"\n--- {split} summary ---")
    print(f"total sentence pairs checked: {total}")
    print(f"raw text mismatches (fail current assert): {raw_mismatch}")
    print(f"  of which whitespace-only (safe to normalize): {raw_mismatch - normalized_still_mismatch}")
    print(f"  of which real content mismatches: {normalized_still_mismatch}")
    print(f"    of which token sequences still match (safe for ent transfer): {token_safe_mismatches[0]}")
    print(f"    of which token sequences differ (unsafe, needs manual review): {normalized_still_mismatch - token_safe_mismatches[0]}")


def main():
    for split in ["train", "dev", "test"]:
        diagnose(split)


if __name__ == "__main__":
    main()