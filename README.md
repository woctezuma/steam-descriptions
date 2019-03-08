# Steam Descriptions

[![Build status][build-image]][build]
[![Updates][dependency-image]][pyup]
[![Python 3][python3-image]][pyup]
[![Code coverage][codecov-image]][codecov]
[![Code Quality][codacy-image]][codacy]

This repository contains Python code to retrieve semantically similar Steam games.

## Requirements

-   Install the latest version of [Python 3.X](https://www.python.org/downloads/).
-   Install the required packages:

```bash
pip install -r requirements.txt
```

## Method

Each game is described by the concatenation of:
 
-   a short text below its banner on the Steam store:

![short game description](https://i.imgur.com/qSiN3Hh.png)

-   a long text in the section called "About the game":

![long game description](https://i.imgur.com/zpLKiqh.png)

The text is tokenized with [spaCy](https://spacy.io/) by running [`utils.py`](utils.py).
The tokens are then fed as input to different methods to retrieve semantically similar game descriptions.

For instance, a word embedding can be learnt with [Word2Vec](https://en.wikipedia.org/wiki/Word2vec) and then used for a
sentence embedding based on a weighted average of word embeddings (cf. [`sif_embedding_perso.py`](sif_embedding_perso.py)).
A [pre-trained GloVe](https://spacy.io/models/en#section-en_vectors_web_lg) embedding can also be used instead of the self-trained Word2Vec embedding.

Or a document embedding can be learnt with [Doc2Vec](https://radimrehurek.com/gensim/models/doc2vec.html) (cf. [`doc2vec_model.py`](doc2vec_model.py)), although, in
our experience, this is more useful to learn document tags, e.g. game genres, rather than to retrieve similar documents.

Different baseline algorithms are suggested in [`sentence_baseline.py`](sentence_baseline.py).

Embeddings can also be computed with [Universal Sentence Encoder](https://tfhub.dev/google/universal-sentence-encoder/2) on [Google Colab](https://colab.research.google.com/) with [this notebook](universal_sentence_encoder.ipynb).

## References

-   [Tutorial on the official website of 'gensim' module](https://radimrehurek.com/gensim/models/word2vec.html)
-   [Tutorial on a blog](http://kavita-ganesan.com/gensim-word2vec-tutorial-starter-code/)
-   [Tool: spaCy](https://spacy.io/)
-   [Tool: Gensim](https://radimrehurek.com/gensim/)
-   [Word2Vec](https://en.wikipedia.org/wiki/Word2vec)
-   Sanjeev Arora, Yingyu Liang, Tengyu Ma, [A Simple but Tough-to-Beat Baseline for Sentence Embeddings](https://openreview.net/forum?id=SyK00v5xx), in: ICLR 2017 conference.

<!-- Definitions -->

[build]: <https://travis-ci.org/woctezuma/steam-descriptions>
[build-image]: <https://travis-ci.org/woctezuma/steam-descriptions.svg?branch=master>

[pyup]: <https://pyup.io/repos/github/woctezuma/steam-descriptions/>
[dependency-image]: <https://pyup.io/repos/github/woctezuma/steam-descriptions/shield.svg>
[python3-image]: <https://pyup.io/repos/github/woctezuma/steam-descriptions/python-3-shield.svg>

[codecov]: <https://codecov.io/gh/woctezuma/steam-descriptions>
[codecov-image]: <https://codecov.io/gh/woctezuma/steam-descriptions/branch/master/graph/badge.svg>

[codacy]: <https://www.codacy.com/app/woctezuma/steam-descriptions>
[codacy-image]: <https://api.codacy.com/project/badge/Grade/710292a19eff45e08a53e8b0028e02d4>
