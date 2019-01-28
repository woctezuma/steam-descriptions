from gensim.corpora import Dictionary
from gensim.models import LsiModel
from gensim.similarities import MatrixSimilarity

from doc2vec_model import reformat_similarity_scores_for_doc2vec
from sentence_models import print_most_similar_sentences
from utils import load_tokens, load_game_names


def main(num_items_displayed=10):
    game_names, _ = load_game_names(include_genres=False, include_categories=False)

    steam_tokens = load_tokens()

    documents = list(steam_tokens.values())

    dct = Dictionary(documents)
    print(len(dct))
    dct.filter_extremes(no_below=7, no_above=0.2)
    print(len(dct))

    corpus = [dct.doc2bow(doc) for doc in documents]

    # model = TfidfModel(corpus, id2word=dct)  # TF-IDF

    model = LsiModel(corpus, id2word=dct, num_topics=20)  # LSI/LSA #TODO num_topics

    index = MatrixSimilarity(model[corpus], num_best=10, num_features=len(dct))

    query_app_ids = ['620', '364470', '504230', '583950', '646570', '863550', '794600']

    app_ids = list(steam_tokens.keys())

    for query_app_id in query_app_ids:
        print('Query appID: {} ({})'.format(query_app_id, game_names[query_app_id]))

        query = steam_tokens[query_app_id]
        vec_bow = dct.doc2bow(query)
        vec_lsi = model[vec_bow]
        sims = index[vec_lsi]

        similarity_scores_as_tuples = [(app_ids[i], sim) for (i, sim) in sims]

        similarity_scores = reformat_similarity_scores_for_doc2vec(similarity_scores_as_tuples)
        print_most_similar_sentences(similarity_scores, num_items_displayed=num_items_displayed)

    return


if __name__ == '__main__':
    main()
