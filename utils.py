# Objective: load and tokenize data with spaCy

import json

import spacy
from gensim.parsing.preprocessing import strip_tags, remove_stopwords
from gensim.utils import simple_preprocess


def get_data_folder():
    folder_name = 'data/'
    return folder_name


def get_raw_data_file_name():
    file_name = get_data_folder() + 'aggregate_prettyprint.json'
    return file_name


def get_token_file_name():
    file_name = get_data_folder() + 'tokens.json'
    return file_name


def get_word_model_file_name():
    file_name = get_data_folder() + 'word2vec.model'
    return file_name


def get_doc_model_file_name():
    file_name = get_data_folder() + 'doc2vec.model'
    return file_name


def load_raw_data(verbose=False):
    if verbose:
        print('Loading raw data')

    with open(get_raw_data_file_name(), 'r') as f:
        steam_sentences = json.load(f)

    return steam_sentences


def load_game_names(include_genres=True, include_categories=True):
    steam_sentences = load_raw_data()

    game_names = dict()
    game_tags = dict()

    for app_id in steam_sentences:
        game_names[app_id] = steam_sentences[app_id]['name']

        game_tags[app_id] = []

        if include_genres:
            try:
                game_tags[app_id] += steam_sentences[app_id]['genres']
            except KeyError:
                pass

        if include_categories:
            try:
                game_tags[app_id] += steam_sentences[app_id]['categories']
            except KeyError:
                pass

    return game_names, game_tags


def load_tokens():
    print('Loading tokens')

    with open(get_token_file_name(), 'r') as f:
        steam_tokens = json.load(f)

    return steam_tokens


def compute_tokens(steam_sentences=None, save_to_disk=False, use_spacy=False):
    print('Computing tokens')

    if steam_sentences is None:
        steam_sentences = load_raw_data()

    counter = 0
    num_games = len(steam_sentences)

    steam_tokens = {}

    # You need to have downloaded the model first. Reference: https://spacy.io/models/en#section-en_core_web_lg
    nlp = spacy.load('en_core_web_lg')

    for app_id in steam_sentences:
        game_data = steam_sentences[app_id]
        counter += 1

        if (counter % 1000) == 0:
            print(
                '[{}/{}] appID = {} ({})'.format(
                    counter,
                    num_games,
                    app_id,
                    game_data['name'],
                ),
            )

        if use_spacy:
            original_str = str(strip_tags(game_data['text']))

            original_str = original_str.replace('\t', ' ')

            # Reference: https://nicschrading.com/project/Intro-to-NLP-with-spaCy/
            original_str = original_str.strip().replace('\n', ' ').replace('\r', ' ')
            original_str = (
                original_str.replace('&amp;', 'and')
                .replace('&gt;', '>')
                .replace('&lt;', '<')
            )

            doc = nlp(original_str)

            ents = [str(entity).strip() for entity in doc.ents]  # Named entities.

            # Keep only words (no numbers, no punctuation).
            # Lemmatize tokens, remove punctuation and remove stopwords.
            doc = [
                token.lemma_ for token in doc if token.is_alpha and not token.is_stop
            ]

            # Add named entities, but only if they are a compound of more than word.
            relevant_entities = [str(entity) for entity in ents if len(entity) > 1]
            doc.extend(relevant_entities)

            game_tokens = doc
        else:
            game_tokens = simple_preprocess(
                remove_stopwords(strip_tags(game_data['text'])),
                deacc=True,
                min_len=3,
            )

        steam_tokens[app_id] = list(game_tokens)

    if save_to_disk:
        with open(get_token_file_name(), 'w') as f:
            json.dump(steam_tokens, f)

    return steam_tokens


if __name__ == '__main__':
    compute_tokens(save_to_disk=True, use_spacy=True)
