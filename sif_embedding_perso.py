import logging
from time import time

import numpy as np
from gensim.corpora import Dictionary
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

from SIF_embedding import remove_pc
from sentence_models import filter_out_words_not_in_vocabulary
from sentence_models import get_store_url_as_bb_code
from utils import load_tokens, load_game_names


def main(compute_from_scratch=True, use_unit_vectors=False, alpha=1e-3, num_removed_components=1):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    game_names, _ = load_game_names(include_genres=False, include_categories=False)

    steam_tokens = load_tokens()

    documents = list(steam_tokens.values())

    if compute_from_scratch:

        dct = Dictionary(documents)
        print(len(dct))
        dct.filter_extremes(no_below=5, no_above=0.5)  # TODO choose parameters
        print(len(dct))

        model = Word2Vec(documents)

        wv = model.wv

        if use_unit_vectors:
            wv.init_sims(replace=True)  # TODO IMPORTANT choose whether to normalize vectors

        index2word_set = set(wv.index2word)

        num_games = len(steam_tokens)

        word_counter = {}

        counter = 0
        for app_id in steam_tokens:
            counter += 1

            if (counter % 1000) == 0:
                print('[{}/{}] appID = {} ({})'.format(counter, num_games, app_id, game_names[app_id]))

            reference_sentence = steam_tokens[app_id]
            reference_sentence = filter_out_words_not_in_vocabulary(reference_sentence, index2word_set)

            for word in reference_sentence:
                if word in wv.vocab:
                    try:
                        word_counter[word] += 1
                    except KeyError:
                        word_counter[word] = 1

        total_counter = sum(word_counter.values())

        word_frequency = dict()
        for word in word_counter:
            word_frequency[word] = word_counter[word] / total_counter

        sentence_vector = {}
        X = np.zeros([num_games, wv.vector_size])

        counter = 0
        for (i, app_id) in enumerate(steam_tokens.keys()):
            counter += 1

            if (counter % 1000) == 0:
                print('[{}/{}] appID = {} ({})'.format(counter, num_games, app_id, game_names[app_id]))

            reference_sentence = steam_tokens[app_id]
            reference_sentence = filter_out_words_not_in_vocabulary(reference_sentence, index2word_set)

            weighted_vector = np.zeros(wv.vector_size)

            for word in reference_sentence:
                weight = (alpha / (alpha + word_frequency[word]))

                # TODO IMPORTANT Why use the normalized word vectors instead of the raw word vectors?
                if use_unit_vectors:
                    weighted_vector += weight * wv.vectors_norm[wv.vocab[word].index]
                else:
                    weighted_vector += weight * wv.vectors[wv.vocab[word].index]

            sentence_vector[app_id] = weighted_vector / len(reference_sentence)
            X[i, :] = sentence_vector[app_id]

        # Reference: https://stackoverflow.com/a/11620982
        X = np.where(np.isfinite(X), X, 0)

        X = remove_pc(X, npc=num_removed_components)

        np.save('data/X.npy', X)

    else:
        X = np.load('data/X.npy', mmap_mode='r')

    app_ids = list(steam_tokens.keys())

    query_app_ids = ['620', '364470', '504230', '583950', '646570', '863550', '794600']
    query_indices = [app_ids.index(query_app_id) for query_app_id in query_app_ids]

    start = time()
    sim = cosine_similarity(X[query_indices, :], X)
    print('Elapsed time: {%.2f}' % (time() - start))

    for (i, query_app_id) in enumerate(query_app_ids):
        print('\nQuery appID: {} ({})'.format(query_app_id, game_names[query_app_id]))
        v = sim[i, :]

        # Reference: https://stackoverflow.com/a/23734295
        ind = np.argpartition(v, -10)[-10:]
        sorted_ind = reversed(ind[np.argsort(v[ind])])

        for j in sorted_ind:
            app_id = app_ids[j]
            store_url = get_store_url_as_bb_code(app_id)
            print('{:2}) similarity: {:.1%} ; appID: {} ({})'.format(j, v[j], store_url, game_names[app_id]))

    return


if __name__ == '__main__':
    main()
