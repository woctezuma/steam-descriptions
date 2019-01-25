from gensim.models import doc2vec

from doc2vec_model import compute_similarity_using_doc2vec_model, train_doc_model_on_steam_tokens, read_corpus
from utils import load_tokens, get_doc_model_file_name
from word_model import compute_similarity_using_word2vec_model


def main(train_from_scratch=False):
    steam_tokens = load_tokens()

    if train_from_scratch:
        print('Creating a new Doc2Vec model from scratch.')
        documents = list(read_corpus(steam_tokens))
        model = doc2vec.Doc2Vec(documents)
    else:
        print('Loading previous Doc2Vec model.')
        model = doc2vec.Doc2Vec.load(get_doc_model_file_name())

    model = train_doc_model_on_steam_tokens(model=model, steam_tokens=steam_tokens, num_epochs=30)

    # Test doc2vec
    for query_app_id in ['583950', '531640', '364470', '292030']:
        compute_similarity_using_doc2vec_model(query_app_id, steam_tokens, model)

    # Check the relevance of the corresponding word2vec
    for query_word in ['anime', 'fun', 'violent']:
        compute_similarity_using_word2vec_model(query_word, steam_tokens, model)

    return True


if __name__ == '__main__':
    main()
