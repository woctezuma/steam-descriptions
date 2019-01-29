from gensim.corpora import Dictionary
from gensim.models import TfidfModel, LsiModel, RpModel, LdaModel, HdpModel
from gensim.similarities import MatrixSimilarity

from doc2vec_model import reformat_similarity_scores_for_doc2vec
from sentence_models import print_most_similar_sentences
from utils import load_tokens, load_game_names


def main(chosen_model_no=6, num_items_displayed=10):
    possible_model_names = [
        'tf_idf',
        'lsi_bow', 'lsi_tf_idf',
        'rp_bow', 'rp_tf_idf',
        'lda_bow', 'lda_tf_idf',
        'hdp_bow', 'hdp_tf_idf',
    ]
    chosen_model_name = possible_model_names[chosen_model_no]
    print(chosen_model_name)

    game_names, _ = load_game_names(include_genres=False, include_categories=False)

    steam_tokens = load_tokens()

    documents = list(steam_tokens.values())

    dct = Dictionary(documents)
    print(len(dct))
    dct.filter_extremes(no_below=5, no_above=0.5)  # TODO choose parameters
    print(len(dct))

    corpus = [dct.doc2bow(doc) for doc in documents]

    # Pre-processing

    pre_process_corpus_with_tf_idf = chosen_model_name.endswith('_tf_idf')

    tfidf_model = TfidfModel(corpus, id2word=dct, normalize=False)  # TODO choose whether to normalize

    if pre_process_corpus_with_tf_idf:
        # Caveat: the leading underscore is important. Do not use this pre-processing if the chosen model is Tf-Idf!
        print('Corpus as Tf-Idf')
        pre_processed_corpus = tfidf_model[corpus]
    else:
        print('Corpus as Bag-of-Words')
        pre_processed_corpus = corpus

    # Model

    if chosen_model_name == 'tf_idf':
        print('Term Frequency * Inverse Document Frequency (Tf-Idf)')
        model = tfidf_model

    elif chosen_model_name.startswith('lsi'):
        print('Latent Semantic Indexing (LSI/LSA)')
        model = LsiModel(pre_processed_corpus, id2word=dct, num_topics=200)  # TODO choose num_topics

    elif chosen_model_name.startswith('rp'):
        print('Random Projections (RP)')
        model = RpModel(pre_processed_corpus, id2word=dct, num_topics=300)  # TODO choose num_topics

    elif chosen_model_name.startswith('lda'):
        print('Latent Dirichlet Allocation (LDA)')
        model = LdaModel(pre_processed_corpus, id2word=dct, num_topics=100)  # TODO choose num_topics
        pass

    elif chosen_model_name.startswith('hdp'):
        print('Hierarchical Dirichlet Process (HDP)')
        model = HdpModel(pre_processed_corpus, id2word=dct)
        pass

    else:
        print('No model specified.')
        model = None
        pass

    index = MatrixSimilarity(model[pre_processed_corpus], num_best=10, num_features=len(dct))

    query_app_ids = ['620', '364470', '504230', '583950', '646570', '863550', '794600']

    app_ids = list(steam_tokens.keys())

    for query_app_id in query_app_ids:
        print('Query appID: {} ({})'.format(query_app_id, game_names[query_app_id]))

        query = steam_tokens[query_app_id]
        vec_bow = dct.doc2bow(query)
        if pre_process_corpus_with_tf_idf:
            pre_preoccessed_vec = tfidf_model[vec_bow]
        else:
            pre_preoccessed_vec = vec_bow
        vec_lsi = model[pre_preoccessed_vec]
        sims = index[vec_lsi]

        similarity_scores_as_tuples = [(app_ids[i], sim) for (i, sim) in sims]

        similarity_scores = reformat_similarity_scores_for_doc2vec(similarity_scores_as_tuples)
        print_most_similar_sentences(similarity_scores, num_items_displayed=num_items_displayed)

    return


if __name__ == '__main__':
    main()
