# Objective: learn a Word2Vec model, then build a sentence embedding based on a weighted average of word embeddings.
# References:
# [1] Sanjeev Arora, Yingyu Liang, Tengyu Ma, "A Simple but Tough-to-Beat Baseline for Sentence Embeddings", 2016.
# [2] Jiaqi Mu, Pramod Viswanath, All-but-the-Top: Simple and Effective Postprocessing for Word Representations, 2018.

import logging
import math
import multiprocessing
import random
from time import time

import numpy as np
from gensim.corpora import Dictionary
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

from SIF_embedding import remove_pc
from sentence_models import filter_out_words_not_in_vocabulary
from sentence_models import get_store_url_as_bb_code
from utils import load_tokens, load_game_names


def main(compute_from_scratch=True,
         use_unit_vectors=False,
         alpha=1e-3,  # the parameter in the SIF weighting scheme, usually in the range [3e-5, 3e-3]
         num_removed_components_for_sentence_vectors=0,  # num of principal components to remove in SIF weighting scheme
         pre_process_word_vectors=False,
         num_removed_components_for_word_vectors=0,
         count_words_out_of_vocabulary=True,
         use_idf_weights=True,
         shuffle_corpus=True):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    game_names, _ = load_game_names(include_genres=False, include_categories=False)

    steam_tokens = load_tokens()

    documents = list(steam_tokens.values())

    if shuffle_corpus:
        # Useful for Doc2Vec in 'doc2vec_model.py'. It might be useful for other methods.
        random.shuffle(documents)

    if compute_from_scratch:

        dct = Dictionary(documents)
        print('Dictionary size (before trimming): {}'.format(len(dct)))

        dct.filter_extremes(no_below=5, no_above=0.5)  # TODO choose parameters
        print('Dictionary size (after trimming): {}'.format(len(dct)))

        model = Word2Vec(documents, workers=multiprocessing.cpu_count())

        wv = model.wv

        if pre_process_word_vectors:
            # Jiaqi Mu, Pramod Viswanath, All-but-the-Top: Simple and Effective Postprocessing for Word Representations,
            # in: ICLR 2018 conference.
            # Reference: https://openreview.net/forum?id=HkuGJ3kCb

            wv.vectors -= np.array(wv.vectors).mean(axis=0)

            if num_removed_components_for_word_vectors > 0:
                wv.vectors = remove_pc(wv.vectors, npc=num_removed_components_for_word_vectors)

            wv.init_sims()

        if use_unit_vectors:
            wv.init_sims(replace=True)  # TODO IMPORTANT choose whether to normalize vectors

        index2word_set = set(wv.index2word)

        num_games = len(steam_tokens)

        word_counter = {}
        document_per_word_counter = {}

        counter = 0
        for app_id in steam_tokens:
            counter += 1

            if (counter % 1000) == 0:
                print('[{}/{}] appID = {} ({})'.format(counter, num_games, app_id, game_names[app_id]))

            reference_sentence = steam_tokens[app_id]
            if not count_words_out_of_vocabulary:
                # This has an impact on the value of 'total_counter'.
                reference_sentence = filter_out_words_not_in_vocabulary(reference_sentence, index2word_set)

            for word in reference_sentence:
                try:
                    word_counter[word] += 1
                except KeyError:
                    word_counter[word] = 1

            for word in set(reference_sentence):
                try:
                    document_per_word_counter[word] += 1
                except KeyError:
                    document_per_word_counter[word] = 1

        total_counter = sum(word_counter.values())

        # Inverse Document Frequency (IDF)
        idf = {}
        for word in document_per_word_counter:
            idf[word] = math.log((1 + num_games) / (1 + document_per_word_counter[word]))

        # Word frequency. Caveat: over the whole corpus!
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
            num_words_in_reference_sentence = len(reference_sentence)

            reference_sentence = filter_out_words_not_in_vocabulary(reference_sentence, index2word_set)
            if not count_words_out_of_vocabulary:
                num_words_in_reference_sentence = len(reference_sentence)

            weighted_vector = np.zeros(wv.vector_size)

            for word in reference_sentence:
                if use_idf_weights:
                    weight = idf[word]
                else:
                    weight = (alpha / (alpha + word_frequency[word]))

                # TODO IMPORTANT Why use the normalized word vectors instead of the raw word vectors?
                if use_unit_vectors:
                    word_vector = wv.vectors_norm[wv.vocab[word].index]
                else:
                    word_vector = wv.vectors[wv.vocab[word].index]

                weighted_vector += weight * word_vector

            if len(reference_sentence) > 0:
                sentence_vector[app_id] = weighted_vector / num_words_in_reference_sentence
            else:
                sentence_vector[app_id] = weighted_vector

            X[i, :] = sentence_vector[app_id]

        # Reference: https://stackoverflow.com/a/11620982
        X = np.where(np.isfinite(X), X, 0)

        print('Saving the sentence embedding.')
        np.save('data/X.npy', X)

    else:
        print('Loading the sentence embedding.')
        X = np.load('data/X.npy', mmap_mode='r')

    if num_removed_components_for_sentence_vectors > 0:
        X = remove_pc(X, npc=num_removed_components_for_sentence_vectors)

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

        for (rank, j) in enumerate(sorted_ind):
            app_id = app_ids[j]
            store_url = get_store_url_as_bb_code(app_id)
            print('{:2}) similarity: {:.1%} ; appID: {} ({})'.format(rank + 1, v[j], store_url, game_names[app_id]))

    return


if __name__ == '__main__':
    main()
