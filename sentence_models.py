# Objective: learn a Word2Vec model, then build a sentence embedding based on an average of word embeddings.
# Reference: n_similarity() in Gensim Word2Vec (https://radimrehurek.com/gensim/models/word2vec.html)

import operator

from gensim.models import Word2Vec

from utils import get_word_model_file_name, load_game_names, load_tokens
from word_model import get_word_model_vocabulary


def filter_out_words_not_in_vocabulary(tokenized_sentence, index2word_set=None, wv_spacy=None):
    if index2word_set is not None:
        # Using Gensim for the Word2Vec word embedding
        filtered_tokenized_sentence = [word for word in tokenized_sentence if word in index2word_set]
    else:
        # Using spaCy for the GloVe word embedding
        filtered_tokenized_sentence = [word for word in tokenized_sentence if wv_spacy[word].has_vector]
    return filtered_tokenized_sentence


def compute_similarity_with_all_other_steam_sentences(query_app_id, steam_tokens=None, model=None, game_names=None,
                                                      filter_out_words_out_of_vocabulary=True):
    if steam_tokens is None:
        steam_tokens = load_tokens()

    if model is None:
        model = Word2Vec.load(get_word_model_file_name())

    if game_names is None:
        game_names, _ = load_game_names()

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

        try:
            similarity_scores[app_id] = model.wv.n_similarity(query_sentence, reference_sentence)
        except ZeroDivisionError:
            similarity_scores[app_id] = 0

    return similarity_scores


def get_store_url_as_bb_code(app_id):
    store_url = '[URL=https://store.steampowered.com/app/' + app_id + ' ]' + app_id + '[/URL]'
    return store_url


def print_most_similar_sentences(similarity_scores, num_items_displayed=10, game_names=None, verbose=True):
    if game_names is None:
        game_names, _ = load_game_names()

    counter = 0

    sorted_similarity_scores = sorted(similarity_scores.items(), key=operator.itemgetter(1), reverse=True)

    similar_app_ids = []

    if verbose:
        print('Top similar games:')

    for app_id, sim_value in sorted_similarity_scores:
        counter += 1

        store_url = get_store_url_as_bb_code(app_id)

        if verbose:
            try:
                print('{:2}) similarity: {:.1%} ; appID: {} ({})'.format(counter, sim_value, store_url,
                                                                         game_names[app_id]))
            except KeyError:
                print('{:2}) similarity: {:.1%} ; tag: {}'.format(counter, sim_value, app_id))

        similar_app_ids.append(app_id)

        if counter >= num_items_displayed:
            print()
            break

    return similar_app_ids


if __name__ == '__main__':
    query_app_id = '583950'  # Artifact
    similarity_scores = compute_similarity_with_all_other_steam_sentences(query_app_id)
    similar_app_ids = print_most_similar_sentences(similarity_scores)
    print(similar_app_ids)
