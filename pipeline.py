from time import time

import gensim

from utils import load_tokens, load_game_names


def read_corpus(steam_tokens):
    for app_id, tokens in steam_tokens.items():
        yield gensim.models.doc2vec.TaggedDocument(tokens, [int(app_id)])


def main():
    game_names = load_game_names()

    steam_tokens = load_tokens()

    corpus = list(read_corpus(steam_tokens))

    model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=40)

    model.build_vocab(corpus)

    start = time()
    model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs)
    print('Cell took %.2f seconds to run.' % (time() - start))

    doc_id = '583950'  # Artifact
    inferred_vector = model.infer_vector(steam_tokens[doc_id])
    sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))

    return True


if __name__ == '__main__':
    main()
