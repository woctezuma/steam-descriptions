import json

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


def load_raw_data():
    print('Loading raw data')

    with open(get_raw_data_file_name(), 'r') as f:
        steam_sentences = json.load(f)

    return steam_sentences


def load_game_names():
    steam_sentences = load_raw_data()

    game_names = dict()
    game_tags = dict()

    for app_id in steam_sentences:
        game_names[app_id] = steam_sentences[app_id]['name']

        try:
            game_tags[app_id] = steam_sentences[app_id]['genres'] + steam_sentences[app_id]['categories']
        except KeyError:
            game_tags[app_id] = []

    return game_names, game_tags


def load_tokens(filter_a_few_special_characters=True):
    print('Loading tokens')

    with open(get_token_file_name(), 'r') as f:
        steam_tokens = json.load(f)

    if filter_a_few_special_characters:
        steam_tokens = filter_tokens(steam_tokens)

    return steam_tokens


def compute_tokens(steam_sentences=None, save_to_disk=False):
    print('Computing tokens')

    if steam_sentences is None:
        steam_sentences = load_raw_data()

    counter = 0
    num_games = len(steam_sentences)

    steam_tokens = {}

    for app_id in steam_sentences:
        game_data = steam_sentences[app_id]
        counter += 1

        if (counter % 1000) == 0:
            print('[{}/{}] appID = {} ({})'.format(counter, num_games, app_id, game_data['name']))

        game_tokens = simple_preprocess(remove_stopwords(strip_tags(game_data['text'])), deacc=True, min_len=3)
        steam_tokens[app_id] = list(game_tokens)

    if save_to_disk:
        with open(get_token_file_name(), 'w') as f:
            json.dump(steam_tokens, f)

    return steam_tokens


def filter_tokens(steam_tokens):
    print('Filtering tokens')

    characters_to_filter = ['_']

    filtered_steam_tokens = {}

    for app_id in steam_tokens:
        game_tokens = steam_tokens[app_id]

        filtered_tokens = [word for word in game_tokens
                           if all(character not in word for character in characters_to_filter)]

        filtered_steam_tokens[app_id] = filtered_tokens

    return filtered_steam_tokens


if __name__ == '__main__':
    word_model_file_name = get_word_model_file_name()
    print(word_model_file_name)
