from time import time

from gensim.models import doc2vec

from sentence_models import print_most_similar_sentences
from utils import load_tokens, load_game_names, get_doc_model_file_name
from word_model import compute_similarity_using_word2vec_model


def read_corpus(steam_tokens):
    for app_id, tokens in steam_tokens.items():
        yield doc2vec.TaggedDocument(tokens, [int(app_id)])


def reformat_similarity_scores_for_doc2vec(similarity_scores_as_tuples, game_names=None):
    if game_names is None:
        game_names = load_game_names()

    dummy_app_ids = []

    similarity_scores = dict()
    for app_id, similarity_value in similarity_scores_as_tuples:
        # Remove dummy appIDs
        if str(app_id) in game_names:
            similarity_scores[str(app_id)] = similarity_value
        else:
            dummy_app_ids.append(app_id)

    print('Dummy appIDs: {}'.format(dummy_app_ids))

    return similarity_scores


def train_doc_model_on_steam_tokens(model=None, steam_tokens=None, num_epochs=10):
    if steam_tokens is None:
        steam_tokens = load_tokens()

    documents = list(read_corpus(steam_tokens))

    if model is None:
        model = doc2vec.Doc2Vec(documents)

    start = time()
    model.train(documents, total_examples=len(documents), epochs=num_epochs)
    print('Elapsed time: {%.2f}' % (time() - start))

    model.save(get_doc_model_file_name())

    return model


def compute_similarity_using_doc2vec_model(query_app_id, steam_tokens=None, model=None,
                                           enforce_training=False, avoid_inference=False):
    if steam_tokens is None:
        steam_tokens = load_tokens()

    if model is None:
        try:
            print('Loading Doc2Vec model.')
            model = doc2vec.Doc2Vec.load(get_doc_model_file_name())

            if enforce_training:
                model = train_doc_model_on_steam_tokens(model=model, steam_tokens=steam_tokens)

        except FileNotFoundError:
            print('Training Doc2Vec model from scratch.')
            model = train_doc_model_on_steam_tokens(model=None, steam_tokens=steam_tokens)

    if avoid_inference:
        print('Finding most similar documents based on the query appID.')
        # For games which are part of the training corpus, we do not need to call model.infer_vector()
        similarity_scores_as_tuples = model.docvecs.most_similar(positive=int(query_app_id))

        # Hack for display with print_most_similar_sentences():
        # if model.docvecs.most_similar() is called with an integer doctag found in the training set,
        # then the doctag is not returned! So, we add it to the list of tuples for later display!
        perfect_similarity_score = 1.0
        if all(query_app_id != app_id for (app_id, similarity_value) in similarity_scores_as_tuples):
            similarity_scores_as_tuples.append((query_app_id, perfect_similarity_score))
    else:
        print('Finding most similar documents based on an inferred vector, which represents the query document.')
        query = steam_tokens[query_app_id]
        # Caveat: « Subsequent calls to this function may infer different representations for the same document. »
        # Reference: https://radimrehurek.com/gensim/models/doc2vec.html#gensim.models.doc2vec.Doc2Vec.infer_vector
        inferred_vector = model.infer_vector(query)
        similarity_scores_as_tuples = model.docvecs.most_similar([inferred_vector])

    similarity_scores = reformat_similarity_scores_for_doc2vec(similarity_scores_as_tuples)
    print_most_similar_sentences(similarity_scores)

    return similarity_scores


if __name__ == '__main__':
    steam_tokens = load_tokens()

    model = doc2vec.Doc2Vec.load(get_doc_model_file_name())

    # Test doc2vec
    for query_app_id in ['583950', '531640', '364470', '292030']:
        compute_similarity_using_doc2vec_model(query_app_id, steam_tokens, model, avoid_inference=False)

    # Check the relevance of the corresponding word2vec
    for query_word in ['anime', 'fun', 'violent']:
        compute_similarity_using_word2vec_model(query_word, steam_tokens, model)
