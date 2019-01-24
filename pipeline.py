import multiprocessing
from time import time

import gensim

from sentence_models import print_most_similar_sentences
from utils import load_tokens, load_game_names


def read_corpus(steam_tokens):
    for app_id, tokens in steam_tokens.items():
        yield gensim.models.doc2vec.TaggedDocument(tokens, [int(app_id)])


def main():
    num_workers = multiprocessing.cpu_count()

    game_names = load_game_names()

    steam_tokens = load_tokens()

    corpus = list(read_corpus(steam_tokens))

    doc_model_file_name = 'data/doc2vec.model'
    load_model = True
    train_model = False

    if load_model:
        model = gensim.models.doc2vec.Doc2Vec.load(doc_model_file_name)

    else:
        model = gensim.models.doc2vec.Doc2Vec(workers=num_workers, min_count=2, epochs=40)

        model.build_vocab(corpus)

    if train_model:
        start = time()
        model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs)
        print('Cell took %.2f seconds to run.' % (time() - start))

        model.save(doc_model_file_name)

    doc_id = '583950'  # Artifact
    inferred_vector = model.infer_vector(steam_tokens[doc_id])
    sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))

    similarity_scores = dict()
    for app_id, similarity_value in sims:
        # Remove dummy appIDs
        if str(app_id) in game_names:
            similarity_scores[str(app_id)] = similarity_value

    print_most_similar_sentences(similarity_scores)

    return True


if __name__ == '__main__':
    main()
