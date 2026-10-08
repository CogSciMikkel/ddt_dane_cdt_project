import spacy
from spacy.tokens import DocBin

vocab = spacy.blank("da").vocab
for split in ["train", "dev", "test"]:
    old = list(DocBin().from_disk(f"/tmp/old_out/{split}.spacy").get_docs(vocab))
    new = list(DocBin().from_disk(f"corpus/cdt_ddt/{split}.spacy").get_docs(vocab))
    same = len(old) == len(new) and all(a.to_bytes() == b.to_bytes() for a, b in zip(old, new))
    print(split, len(old), len(new), same)