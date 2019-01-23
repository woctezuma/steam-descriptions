import json
from pprint import pprint

from gensim.models import Word2Vec
from gensim.utils import simple_preprocess


def get_data_folder():
    folder_name = 'data/'
    return folder_name


def get_raw_data_file_name():
    file_name = 'aggregate_prettyprint.json'
    return file_name


def get_token_file_name():
    file_name = 'tokens.json'
    return file_name


def get_model_file_name():
    file_name = 'word2vec.model'
    return file_name


def load_data():
    with open(get_data_folder() + get_raw_data_file_name(), 'r') as f:
        data = json.load(f)

    return data


def load_tokens(data=None):
    try:
        print('Loading')

        with open(get_data_folder() + get_token_file_name(), 'r') as f:
            steam_texts = json.load(f)

    except FileNotFoundError:
        print('Computing')

        if data is None:
            data = load_data()

        counter = 0
        num_games = len(data)

        steam_texts = {}
        for app_id in data:
            game = data[app_id]
            counter += 1

            print('[{}/{}] appID = {} ({})'.format(counter, num_games, app_id, game['name']))

            tokens = list(simple_preprocess(game['text'], deacc=True))
            steam_texts[app_id] = tokens

        if len(steam_texts) > 0:
            with open(get_data_folder() + get_token_file_name(), 'w') as f:
                json.dump(steam_texts, f)

    return steam_texts


def filter_tokens(steam_texts):
    filtered_steam_texts = {}

    print('Filtering tokens')
    characters_to_filter = ['_']

    for app_id in steam_texts:
        tokens = steam_texts[app_id]

        filtered_tokens = [word for word in tokens if all(character not in word for character in characters_to_filter)]

        filtered_steam_texts[app_id] = filtered_tokens

    return filtered_steam_texts


def load_corpus():
    steam_texts = load_tokens()

    steam_texts = filter_tokens(steam_texts)

    documents = list(steam_texts.values())

    return documents


def initialize_model(steam_texts):
    model = Word2Vec(steam_texts)

    return model


def save_model(model, model_name=None):
    if model_name is None:
        model_name = get_data_folder() + get_model_file_name()

    model.save(model_name)

    return


def load_model(model_name=None):
    if model_name is None:
        model_name = get_data_folder() + get_model_file_name()

    model = Word2Vec.load(model_name)

    return model


def train_model_on_steam_data():
    documents = load_corpus()

    model = initialize_model(documents)

    model.train(documents, total_examples=len(documents), epochs=10)

    save_model(model)

    return model


def test_word(model, query_word='anime'):
    similar_words = model.wv.most_similar(positive=query_word)
    print('\nThe most similar words to the word "{}":'.format(query_word))
    pprint(similar_words)

    return


def main():
    model = load_model()

    # Vocabulary
    index2word_set = set(model.wv.index2word)

    for query_word in ['anime', 'fun', 'violent']:
        test_word(model, query_word)


if __name__ == '__main__':
    main()
