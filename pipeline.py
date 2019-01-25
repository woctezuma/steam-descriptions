from gensim.models import doc2vec

from doc2vec_model import compute_similarity_using_doc2vec_model, train_doc_model_on_steam_tokens, read_corpus
from utils import load_tokens, get_doc_model_file_name
from word_model import compute_similarity_using_word2vec_model


def main(train_from_scratch=False, enforce_training=False):
    steam_tokens = load_tokens()

    if train_from_scratch:
        print('Creating a new Doc2Vec model from scratch.')
        documents = list(read_corpus(steam_tokens))
        model = doc2vec.Doc2Vec(documents)
    else:
        print('Loading previous Doc2Vec model.')
        model = doc2vec.Doc2Vec.load(get_doc_model_file_name())

    if enforce_training:
        model = train_doc_model_on_steam_tokens(model=model, steam_tokens=steam_tokens, num_epochs=20)

    # Test doc2vec
    for query_app_id in ['10', '620', '105600', '264710', '292030', '294100', '364470', '504230', '519860', '531640',
                         '560130', '582010', '583950', '588650', '590380', '620980', '638970', '644560', '646570',
                         '653530', '683320', '698780', '731490', '742120', '812140', '863550', '973760']:
        compute_similarity_using_doc2vec_model(query_app_id, steam_tokens, model, avoid_inference=True,
                                               num_items_displayed=3)

    # Check the relevance of the corresponding word2vec
    for query_word in ['anime', 'fun', 'violent']:
        compute_similarity_using_word2vec_model(query_word, steam_tokens, model)

    return True


if __name__ == '__main__':
    main()
