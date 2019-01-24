from gensim.models import Word2Vec

from utils import get_word_model_file_name, load_game_names, load_tokens


def main():
    steam_tokens = load_tokens()

    model = Word2Vec.load(get_word_model_file_name())

    game_names = load_game_names()

    return True


if __name__ == '__main__':
    main()
