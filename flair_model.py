# Objective: load data, tokenize & embed sentences with the 'flair' library @ https://github.com/zalandoresearch/flair/

import json

from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentPoolEmbeddings, Sentence
from gensim.parsing.preprocessing import strip_tags

from utils import load_raw_data, get_embedding_file_name


def compute_steam_embeddings(steam_sentences=None, save_to_disk=False):
    if steam_sentences is None:
        steam_sentences = load_raw_data()

    counter = 0
    num_games = len(steam_sentences)

    steam_embeddings = {}

    # initialize the word embeddings
    glove_embedding = WordEmbeddings('glove')
    flair_embedding_forward = FlairEmbeddings('news-forward')
    flair_embedding_backward = FlairEmbeddings('news-backward')

    # initialize the document embeddings, mode = mean
    document_embeddings = DocumentPoolEmbeddings([glove_embedding,
                                                  flair_embedding_backward,
                                                  flair_embedding_forward])

    for app_id in steam_sentences:
        game_data = steam_sentences[app_id]
        counter += 1

        if (counter % 1000) == 0:
            print('[{}/{}] appID = {} ({})'.format(counter, num_games, app_id, game_data['name']))

        original_str = str(strip_tags(game_data['text']))

        original_str = original_str.replace('\t', ' ')

        # Reference: https://nicschrading.com/project/Intro-to-NLP-with-spaCy/
        original_str = original_str.strip().replace('\n', ' ').replace('\r', ' ')
        original_str = original_str.replace('&amp;', 'and').replace('&gt;', '>').replace('&lt;', '<')

        # Tokenize the text with the 'flair' library
        sentence = Sentence(original_str, use_tokenizer=True)

        # embed the sentence with our document embedding
        document_embeddings.embed(sentence)

        steam_embeddings[app_id] = sentence.get_embedding()

    if save_to_disk:
        with open(get_embedding_file_name(), 'w') as f:
            json.dump(steam_embeddings, f)

    return steam_embeddings


if __name__ == '__main__':
    steam_embeddings = compute_steam_embeddings()
