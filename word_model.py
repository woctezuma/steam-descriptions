from pprint import pprint

from gensim.models import Word2Vec

from utils import get_word_model_file_name, load_tokens


def train_word_model_on_steam_tokens(steam_tokens=None, num_epochs=10):
    if steam_tokens is None:
        steam_tokens = load_tokens()

    documents = list(steam_tokens.values())

    model = Word2Vec(documents)

    model.train(documents, total_examples=len(documents), epochs=num_epochs)

    model.save(get_word_model_file_name())

    return model


def test_word(model, query_word='anime'):
    similar_words = model.wv.most_similar(positive=query_word)

    print('\nThe most similar words to the word "{}" are:'.format(query_word))
    pprint(similar_words)

    return


def get_word_model_vocabulary(model):
    index2word_set = set(model.wv.index2word)
    return index2word_set


if __name__ == '__main__':
    model = Word2Vec.load(get_word_model_file_name())

    for query_word in ['anime', 'fun', 'violent']:
        if query_word in get_word_model_vocabulary(model):
            test_word(model, query_word)
        else:
            print('The word {} is not part of the word model vocabulary.'.format(query_word))
