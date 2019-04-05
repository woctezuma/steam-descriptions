# Objective: learn a Word2Vec model, then build a sentence embedding based on a weighted average of word embeddings.
# References:
# [1] Sanjeev Arora, Yingyu Liang, Tengyu Ma, "A Simple but Tough-to-Beat Baseline for Sentence Embeddings", 2016.
# [2] Jiaqi Mu, Pramod Viswanath, All-but-the-Top: Simple and Effective Postprocessing for Word Representations, 2018.

import logging
import math
import multiprocessing
import random

import numpy as np
import spacy
from gensim.corpora import Dictionary
from gensim.models import Word2Vec

from SIF_embedding import remove_pc
from benchmark_utils import load_benchmarked_app_ids, print_ranking
from hard_coded_ground_truth import compute_retrieval_score, plot_retrieval_scores
from sentence_models import filter_out_words_not_in_vocabulary
from universal_sentence_encoder import perform_knn_search_with_app_ids_as_input
from utils import load_tokens, load_game_names


def retrieve_similar_store_descriptions(compute_from_scratch=True,
                                        use_unit_vectors=False,
                                        alpha=1e-3,  # in SIF weighting scheme, parameter in the range [3e-5, 3e-3]
                                        num_removed_components_for_sentence_vectors=0,  # in SIF weighting scheme
                                        pre_process_word_vectors=False,
                                        num_removed_components_for_word_vectors=0,
                                        count_words_out_of_vocabulary=True,
                                        use_idf_weights=True,
                                        shuffle_corpus=True,
                                        use_glove_with_spacy=True,
                                        use_cosine_similarity=True,
                                        num_neighbors=10,
                                        no_below=5,  # only relevant with Word2Vec
                                        no_above=0.5,  # only relevant with Word2Vec
                                        only_print_banners=True):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    game_names, _ = load_game_names(include_genres=False, include_categories=False)

    steam_tokens = load_tokens()

    documents = list(steam_tokens.values())

    if shuffle_corpus:
        # Useful for Doc2Vec in 'doc2vec_model.py'. It might be useful for other methods.
        random.shuffle(documents)

    if compute_from_scratch:

        if not use_glove_with_spacy:
            # Use self-trained Word2Vec vectors

            dct = Dictionary(documents)
            print('Dictionary size (before trimming): {}'.format(len(dct)))

            dct.filter_extremes(no_below=no_below, no_above=no_above)
            print('Dictionary size (after trimming): {}'.format(len(dct)))

            model = Word2Vec(documents, workers=multiprocessing.cpu_count())

            wv = model.wv

        else:
            # Use pre-trained GloVe vectors loaded from spaCy
            # Reference: https://spacy.io/models/en#en_vectors_web_lg

            spacy_model_name = 'en_vectors_web_lg'  # either 'en_core_web_lg' or 'en_vectors_web_lg'
            nlp = spacy.load(spacy_model_name)

            wv = nlp.vocab

        if pre_process_word_vectors:
            # Jiaqi Mu, Pramod Viswanath, All-but-the-Top: Simple and Effective Postprocessing for Word Representations,
            # in: ICLR 2018 conference.
            # Reference: https://openreview.net/forum?id=HkuGJ3kCb

            if use_glove_with_spacy:
                wv.vectors.data -= np.array(wv.vectors.data).mean(axis=0)

                if num_removed_components_for_word_vectors > 0:
                    wv.vectors.data = remove_pc(wv.vectors.data, npc=num_removed_components_for_word_vectors)

            else:
                wv.vectors -= np.array(wv.vectors).mean(axis=0)

                if num_removed_components_for_word_vectors > 0:
                    wv.vectors = remove_pc(wv.vectors, npc=num_removed_components_for_word_vectors)

                wv.init_sims()

        if use_unit_vectors and not use_glove_with_spacy:
            # Pre-computations of unit word vectors, which replace the unnormalized word vectors. A priori not required
            # here, because another part of the code takes care of it. A fortiori not required when using spaCy.
            wv.init_sims(replace=True)  # TODO IMPORTANT choose whether to normalize vectors

        if not use_glove_with_spacy:
            index2word_set = set(wv.index2word)
        else:
            index2word_set = None

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
                reference_sentence = filter_out_words_not_in_vocabulary(reference_sentence, index2word_set, wv)

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
        if not use_glove_with_spacy:
            word_vector_length = wv.vector_size
        else:
            word_vector_length = wv.vectors_length
        X = np.zeros([num_games, word_vector_length])

        counter = 0
        for (i, app_id) in enumerate(steam_tokens.keys()):
            counter += 1

            if (counter % 1000) == 0:
                print('[{}/{}] appID = {} ({})'.format(counter, num_games, app_id, game_names[app_id]))

            reference_sentence = steam_tokens[app_id]
            num_words_in_reference_sentence = len(reference_sentence)

            reference_sentence = filter_out_words_not_in_vocabulary(reference_sentence, index2word_set, wv)
            if not count_words_out_of_vocabulary:
                # NB: Out-of-vocabulary words are not counted in https://stackoverflow.com/a/35092200
                num_words_in_reference_sentence = len(reference_sentence)

            weighted_vector = np.zeros(word_vector_length)

            for word in reference_sentence:
                if use_idf_weights:
                    weight = idf[word]
                else:
                    weight = (alpha / (alpha + word_frequency[word]))

                # TODO IMPORTANT Why use the normalized word vectors instead of the raw word vectors?
                if not use_glove_with_spacy:
                    if use_unit_vectors:
                        # Reference: https://github.com/RaRe-Technologies/movie-plots-by-genre
                        word_vector = wv.vectors_norm[wv.vocab[word].index]
                    else:
                        word_vector = wv.vectors[wv.vocab[word].index]
                else:
                    word_vector = wv.get_vector(word)
                    if use_unit_vectors:
                        word_vector_norm = wv[word].vector_norm
                        if word_vector_norm > 0:
                            word_vector = word_vector / word_vector_norm

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

    app_ids = list(int(app_id) for app_id in steam_tokens.keys())

    query_app_ids = load_benchmarked_app_ids(append_hard_coded_app_ids=True)

    matches_as_app_ids = perform_knn_search_with_app_ids_as_input(query_app_ids,
                                                                  label_database=X,
                                                                  app_ids=app_ids,
                                                                  use_cosine_similarity=use_cosine_similarity,
                                                                  num_neighbors=num_neighbors)

    print_ranking(query_app_ids,
                  matches_as_app_ids,
                  num_elements_displayed=num_neighbors,
                  only_print_banners=only_print_banners)

    retrieval_score = compute_retrieval_score(query_app_ids,
                                              matches_as_app_ids,
                                              num_elements_displayed=num_neighbors,
                                              verbose=False)

    return retrieval_score


if __name__ == '__main__':
    # Initialize 'data/X.npy'
    retrieve_similar_store_descriptions(compute_from_scratch=True)

    # Try different values for the number of sentence components to remove.
    # NB: 'data/X.npy' will be read from the disk, which avoids redundant computations.
    retrieval_scores = dict()
    for i in range(0, 20, 5):
        retrieval_scores[i] = retrieve_similar_store_descriptions(compute_from_scratch=False,
                                                                  num_removed_components_for_sentence_vectors=i)

    print(retrieval_scores)

    plot_retrieval_scores(retrieval_scores)
