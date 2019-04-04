# Objective: try different methods to retrieve semantically similar sentences, including a few based on Word2Vec models.

import logging

import spacy
from gensim.corpora import Dictionary
from gensim.models import TfidfModel, LsiModel, RpModel, LdaModel, HdpModel, KeyedVectors, Word2Vec
from gensim.models import WordEmbeddingSimilarityIndex
from gensim.similarities import MatrixSimilarity, SparseTermSimilarityMatrix, SoftCosineSimilarity
from spacy.tokens import Doc

from benchmark_utils import load_benchmarked_app_ids, print_ranking, get_app_name
from doc2vec_model import reformat_similarity_scores_for_doc2vec
from sentence_models import print_most_similar_sentences, filter_out_words_not_in_vocabulary
from utils import load_tokens, load_game_names


def main(chosen_model_no=0, num_items_displayed=10, use_spacy=False, use_soft_cosine_similarity=False,
         num_topics=None, no_below=5, no_above=0.5, normalize_vectors=False):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    if num_topics is None:
        num_topics = 100

    possible_model_names = [
        'tf_idf',  # 0
        'lsi_bow', 'lsi_tf_idf',  # 1, 2
        'rp_bow', 'rp_tf_idf',  # 3, 4
        'lda_bow', 'lda_tf_idf',  # 5, 6
        'hdp_bow', 'hdp_tf_idf',  # 7, 8
        'word2vec',  # 9
    ]
    chosen_model_name = possible_model_names[chosen_model_no]
    print(chosen_model_name)

    game_names, _ = load_game_names(include_genres=False, include_categories=False)

    steam_tokens = load_tokens()

    nlp = spacy.load('en_core_web_lg')

    documents = list(steam_tokens.values())

    dct = Dictionary(documents)
    print(len(dct))
    dct.filter_extremes(no_below=no_below, no_above=no_above)
    print(len(dct))

    corpus = [dct.doc2bow(doc) for doc in documents]

    # Pre-processing

    pre_process_corpus_with_tf_idf = chosen_model_name.endswith('_tf_idf')

    tfidf_model = TfidfModel(corpus, id2word=dct, normalize=normalize_vectors)

    if pre_process_corpus_with_tf_idf:
        # Caveat: the leading underscore is important. Do not use this pre-processing if the chosen model is Tf-Idf!
        print('Corpus as Tf-Idf')
        pre_processed_corpus = tfidf_model[corpus]
    else:
        print('Corpus as Bag-of-Words')
        pre_processed_corpus = corpus

    # Model

    model = None
    wv = None
    index2word_set = None

    if chosen_model_name == 'tf_idf':
        print('Term Frequency * Inverse Document Frequency (Tf-Idf)')
        model = tfidf_model

    elif chosen_model_name.startswith('lsi'):
        print('Latent Semantic Indexing (LSI/LSA)')
        model = LsiModel(pre_processed_corpus, id2word=dct, num_topics=num_topics)

    elif chosen_model_name.startswith('rp'):
        print('Random Projections (RP)')
        model = RpModel(pre_processed_corpus, id2word=dct, num_topics=num_topics)

    elif chosen_model_name.startswith('lda'):
        print('Latent Dirichlet Allocation (LDA)')
        model = LdaModel(pre_processed_corpus, id2word=dct, num_topics=num_topics)

    elif chosen_model_name.startswith('hdp'):
        print('Hierarchical Dirichlet Process (HDP)')
        model = HdpModel(pre_processed_corpus, id2word=dct)

    elif chosen_model_name == 'word2vec':
        use_a_lot_of_ram = False

        if use_a_lot_of_ram:
            model = None

            print('Loading Word2Vec based on Google News')
            # Warning: this takes a lot of time and uses a ton of RAM!
            wv = KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin.gz', binary=True)
        else:
            if use_spacy:
                print('Using Word2Vec with spaCy')
            else:
                print('Training Word2Vec')

                model = Word2Vec(documents)

                wv = model.wv

        if not use_spacy:
            wv.init_sims(replace=normalize_vectors)

            index2word_set = set(wv.index2word)

    else:
        print('No model specified.')
        model = None

    if chosen_model_name != 'word2vec':
        if not use_soft_cosine_similarity:
            index = MatrixSimilarity(model[pre_processed_corpus], num_best=10, num_features=len(dct))
        else:
            w2v_model = Word2Vec(documents)
            similarity_index = WordEmbeddingSimilarityIndex(w2v_model.wv)
            similarity_matrix = SparseTermSimilarityMatrix(similarity_index, dct, tfidf_model, nonzero_limit=100)
            index = SoftCosineSimilarity(model[pre_processed_corpus], similarity_matrix)
    else:
        index = None

    query_app_ids = load_benchmarked_app_ids(append_hard_coded_app_ids=True)

    app_ids = list(int(app_id) for app_id in steam_tokens.keys())

    matches_as_app_ids = []

    for query_count, query_app_id in enumerate(query_app_ids):
        print('[{}/{}] Query appID: {} ({})'.format(query_count + 1, len(query_app_ids),
                                                    query_app_id, get_app_name(query_app_id, game_names)))

        query = steam_tokens[str(query_app_id)]

        if use_spacy:
            spacy_query = Doc(nlp.vocab, query)
        else:
            spacy_query = None

        if chosen_model_name != 'word2vec':
            vec_bow = dct.doc2bow(query)
            if pre_process_corpus_with_tf_idf:
                pre_preoccessed_vec = tfidf_model[vec_bow]
            else:
                pre_preoccessed_vec = vec_bow
            vec_lsi = model[pre_preoccessed_vec]
            sims = index[vec_lsi]

            if use_soft_cosine_similarity:
                sims = enumerate(sims)

            similarity_scores_as_tuples = [(str(app_ids[i]), sim) for (i, sim) in sims]
            similarity_scores = reformat_similarity_scores_for_doc2vec(similarity_scores_as_tuples)
        else:
            if use_spacy:
                similarity_scores = {}
                for app_id in steam_tokens:
                    reference_sentence = steam_tokens[app_id]
                    spacy_reference = Doc(nlp.vocab, reference_sentence)
                    similarity_scores[app_id] = spacy_query.similarity(spacy_reference)
            else:
                query_sentence = filter_out_words_not_in_vocabulary(query, index2word_set)

                similarity_scores = {}

                counter = 0
                num_games = len(steam_tokens)

                for app_id in steam_tokens:
                    counter += 1

                    if (counter % 1000) == 0:
                        print('[{}/{}] appID = {} ({})'.format(counter, num_games, app_id, game_names[app_id]))

                    reference_sentence = steam_tokens[app_id]
                    reference_sentence = filter_out_words_not_in_vocabulary(reference_sentence, index2word_set)

                    try:
                        similarity_scores[app_id] = wv.n_similarity(query_sentence, reference_sentence)
                    except ZeroDivisionError:
                        similarity_scores[app_id] = 0

        similar_app_ids = print_most_similar_sentences(similarity_scores, num_items_displayed=num_items_displayed,
                                                       verbose=False)
        matches_as_app_ids.append(similar_app_ids)

    print_ranking(query_app_ids,
                  matches_as_app_ids,
                  only_print_banners=True)

    return


if __name__ == '__main__':
    main(chosen_model_no=0,
         use_spacy=False,
         use_soft_cosine_similarity=False,
         num_topics=None,
         no_below=5,
         no_above=0.5,
         normalize_vectors=False)
