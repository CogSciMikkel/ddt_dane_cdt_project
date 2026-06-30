import spacy
from spacy.tokens import DocBin, Doc

nlp = Doc(vocab=spacy.blank("da").vocab)

files = [
    "/work/training_practice/ddt_dane_cdt_project/corpus/cdt_ddt/train.spacy",
    "/work/training_practice/ddt_dane_cdt_project/corpus/cdt_ddt/dev.spacy",
    "/work/training_practice/ddt_dane_cdt_project/corpus/cdt_ddt/test.spacy"
    ]

for file in files:
    db = DocBin().from_disk(file)
    docs = list(db.get_docs(nlp.vocab))
    print(f"{file.split("/")[-1]}: {len(docs)}")