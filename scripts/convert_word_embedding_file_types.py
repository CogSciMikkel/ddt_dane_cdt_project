from gensim.models import KeyedVectors

kv = KeyedVectors.load_word2vec_format(
    "assets/dsl_skipgram_2020_m5_f500_epoch2_w5.model.w2v.bin",
    binary=True,
)
kv.save_word2vec_format("assets/dsl_skipgram_2020.txt", binary=False)

kv = KeyedVectors.load_word2vec_format(
    "assets/CoNLL2017_model.bin",
    binary=True,
)
kv.save_word2vec_format("assets/CoNLL2017.txt", binary=False)