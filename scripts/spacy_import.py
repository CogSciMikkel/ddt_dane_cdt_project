import spacy

nlp = spacy.load("da_core_news_lg")
import da_core_news_lg
nlp = da_core_news_lg.load()
#print(nlp.pipe_names)
pipes_to_remove = ["lemmatizer", "attribute_ruler"]
for pipe in pipes_to_remove:
    nlp.remove_pipe(pipe)
nlp.to_disk("./training/spacy_da_core_news_lg/model-best")