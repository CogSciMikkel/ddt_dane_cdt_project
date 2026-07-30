import spacy
import dacy

nlp = dacy.load("da_dacy_large_trf-0.2.0")
pipes_to_remove = ["trainable_lemmatizer", "coref", "span_resolver", "span_cleaner", "entity_linker"]
for pipe in pipes_to_remove:
    nlp.remove_pipe(pipe)
nlp.to_disk("./training/dacy_large_old/model-best")