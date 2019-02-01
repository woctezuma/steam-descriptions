# Objective: learn a Word2Vec model

from pprint import pprint

from gensim.models import Word2Vec

from utils import get_word_model_file_name, load_tokens


def train_word_model_on_steam_tokens(model=None, steam_tokens=None, num_epochs=10):
    # Warning: training will happen several times, which might be detrimental to your model!

    if steam_tokens is None:
        steam_tokens = load_tokens()

    documents = list(steam_tokens.values())

    if model is None:
        model = Word2Vec(documents)  # training already happens here, due to the 'documents' argument!

    model.train(documents, total_examples=len(documents), epochs=num_epochs)

    model.save(get_word_model_file_name())

    return model


def test_word(model, query_word='anime'):
    similar_words = model.wv.most_similar(positive=query_word)

    print('\nThe most similar words to the word "{}" are:'.format(query_word))
    pprint(similar_words)

    return similar_words


def get_word_model_vocabulary(model):
    index2word_set = set(model.wv.index2word)
    return index2word_set


def compute_similarity_using_word2vec_model(query_word, steam_tokens=None, model=None,
                                            enforce_training=False):
    if steam_tokens is None:
        steam_tokens = load_tokens()

    if model is None:
        try:
            print('Loading Word2Vec model.')
            model = Word2Vec.load(get_word_model_file_name())

            if enforce_training:
                model = train_word_model_on_steam_tokens(model=model, steam_tokens=steam_tokens)

        except FileNotFoundError:
            print('Training Word2Vec model from scratch.')
            model = train_word_model_on_steam_tokens(model=None, steam_tokens=steam_tokens)

    if query_word in get_word_model_vocabulary(model):
        similar_words = test_word(model, query_word)
    else:
        print('The word {} is not part of the word model vocabulary.'.format(query_word))
        similar_words = None

    return similar_words


if __name__ == '__main__':
    steam_tokens = load_tokens()

    model = Word2Vec.load(get_word_model_file_name())

    for query_word in ['anime', 'fun', 'violent']:
        compute_similarity_using_word2vec_model(query_word, steam_tokens, model)
