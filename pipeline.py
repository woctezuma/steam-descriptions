import logging
import multiprocessing

from gensim.models import doc2vec

from doc2vec_model import check_analogy
from doc2vec_model import compute_similarity_using_doc2vec_model, read_corpus
from utils import load_tokens, load_game_names
from word_model import compute_similarity_using_word2vec_model


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    game_names, game_tags = load_game_names()

    steam_tokens = load_tokens()

    documents = list(read_corpus(steam_tokens, game_tags))

    model = doc2vec.Doc2Vec(documents,
                            num_epochs=20,
                            workers=multiprocessing.cpu_count())

    # Test doc2vec
    check_analogy(model, pos=[239350, 646570], neg=[557410])  # Spelunky + (Slay the Spire) - (Dream Quest)
    check_analogy(model, pos=[70, 20920], neg=[20900])  # Half-Life + (Witcher 2) - (Witcher)

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
