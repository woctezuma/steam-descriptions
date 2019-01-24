import operator

import numpy as np
from gensim.models import Word2Vec

from utils import get_word_model_file_name, load_game_names, load_tokens
from word_model import get_word_model_vocabulary


def avg_feature_vector(tokenized_sentence, model):
    # Reference: https://stackoverflow.com/a/35092200

    index2word_set = get_word_model_vocabulary(model)

    num_features = model.wv.vector_size

    feature_vec = np.zeros((num_features,), dtype='float32')
    n_words = 0
    for word in tokenized_sentence:
        if word in index2word_set:
            n_words += 1
            feature_vec = np.add(feature_vec, model.wv[word])
    if n_words > 0:
        feature_vec = np.divide(feature_vec, n_words)
    return feature_vec


def filter_out_words_not_in_vocabulary(tokenized_sentence, index2word_set):
    filtered_tokenized_sentence = [word for word in tokenized_sentence if word in index2word_set]
    return filtered_tokenized_sentence


def compute_similarity_with_all_other_steam_sentences(query_app_id, steam_tokens=None, model=None, game_names=None,
                                                      filter_out_words_out_of_vocabulary=True):
    if steam_tokens is None:
        steam_tokens = load_tokens()

    if model is None:
        model = Word2Vec.load(get_word_model_file_name())

    if game_names is None:
        game_names = load_game_names()

    index2word_set = get_word_model_vocabulary(model)

    query_sentence = steam_tokens[query_app_id]
    if filter_out_words_out_of_vocabulary:
        query_sentence = filter_out_words_not_in_vocabulary(query_sentence, index2word_set)

    similarity_scores = {}

    counter = 0
    num_games = len(steam_tokens)

    for app_id in steam_tokens:
        counter += 1

        if (counter % 1000) == 0:
            print('[{}/{}] appID = {} ({})'.format(counter, num_games, app_id, game_names[app_id]))

        reference_sentence = steam_tokens[app_id]
        if filter_out_words_out_of_vocabulary:
            reference_sentence = filter_out_words_not_in_vocabulary(reference_sentence, index2word_set)

        similarity_scores[app_id] = model.wv.n_similarity(query_sentence, reference_sentence)

    return similarity_scores


def get_store_url_as_bb_code(app_id):
    store_url = '[URL=https://store.steampowered.com/app/' + app_id + ' ]' + app_id + '[/URL]'
    return store_url


def print_most_similar_sentences(similarity_scores, num_items_displayed=10, game_names=None):
    if game_names is None:
        game_names = load_game_names()

    counter = 0

    sorted_similarity_scores = sorted(similarity_scores.items(), key=operator.itemgetter(1), reverse=True)

    similar_app_ids = []

    for app_id, sim_value in sorted_similarity_scores:

        store_url = get_store_url_as_bb_code(app_id)

        if counter == 0:
            print('Query appID: {} ({})'.format(store_url, game_names[app_id]))
            print('\n\nTop similar games:')
        else:
            print('{:2}) appID: {} ({})'.format(counter, store_url, game_names[app_id]))
            similar_app_ids.append(app_id)

        counter += 1

        if counter > num_items_displayed:
            break

    return similar_app_ids


if __name__ == '__main__':
    query_app_id = '583950'  # Artifact
    similarity_scores = compute_similarity_with_all_other_steam_sentences(query_app_id)
    similar_app_ids = print_most_similar_sentences(similarity_scores)
    print(similar_app_ids)
