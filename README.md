# Steam Descriptions

[![Build status][build-image]][build]
[![Code coverage][codecov-image]][codecov]
[![Code Quality][codacy-image]][codacy]

This repository contains Python code to retrieve semantically similar Steam games.

![Sekiro: similar store descriptions with GloVe](https://github.com/woctezuma/steam-descriptions/wiki/img/fuUtQ5Z.jpg)

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
For Tf-Idf, the code is duplicated in [`export_tfidf_for_javascript_visualization.py`](export_tfidf_for_javascript_visualization.py), with the addition of an export for visualization of the matches as a graph [in the web browser](https://woctezuma.github.io/).

Embeddings can also be computed with [Universal Sentence Encoder](https://tfhub.dev/google/universal-sentence-encoder/2) on [Google Colab](https://colab.research.google.com/) with [this notebook][universal_sentence_encoder].
[![Open In Colab][colab-badge]][universal_sentence_encoder]
Results are shown with [`universal_sentence_encoder.py`](universal_sentence_encoder.py). 

## Results

### Similar games

An in-depth commentary is provided on the [Wiki](https://github.com/woctezuma/steam-descriptions/wiki/Commentary).
Matches obtained with Tf-Idf are shown as a graph [in the web browser](https://woctezuma.github.io/).
Overall, I would suggest to match store descriptions with:
-   either [Term Frequency * Inverse Document Frequency (Tf-Idf)](https://github.com/woctezuma/steam-descriptions/wiki/baseline_0),

![Witcher: similar store descriptions with Tf-Idf](https://github.com/woctezuma/steam-descriptions/wiki/img/fH7gjaS.png)

-   or a weighted average of GloVe word embeddings, with Tf-Idf reweighting, after removing some components:
    - either [only sentence components](https://github.com/woctezuma/steam-descriptions/wiki/sif_embedding_glove_cosine_sent_comp_10),
    - or [both sentence and word components](https://github.com/woctezuma/steam-descriptions/wiki/sif_embedding_glove_toggle_preprocess_word_vectors_num_comp_10_sent_comp_10) (for slighly better results, by a tiny margin).

![Neverwinter: similar store descriptions with GloVe](https://github.com/woctezuma/steam-descriptions/wiki/img/PYzT6ol.png)

A retrieval score can be computed, thanks to a ground truth of games set in the same fictional universe.
Alternative scores can be computed as the proportions of genres or tags shared between the query and the retrieved games.

When using average of word embeddings as sentence embeddings:
-   removing only [sentence components](https://openreview.net/forum?id=SyK00v5xx) provided a very large increase of the score (+105%),
-   removing only [word components](https://openreview.net/forum?id=HkuGJ3kCb) provided a large increase of the score (+51%),
-   removing both components provided a very large increase of the score (+108%),
-   relying on a weighted average instead of a simple average lead to better results,
-   Tf-Idf reweighting lead to better results than [Smooth Inverse Frequency](https://openreview.net/forum?id=SyK00v5xx) reweighting,
-   GloVe word embeddings lead to better results than Word2Vec.

![Influence of the removal of sentence components](https://github.com/woctezuma/steam-descriptions/wiki/img/plot.png)

A table with scores for each major experiment is [available](https://github.com/woctezuma/steam-descriptions/wiki/steam_descriptions).
For each game series, the score is the number of games from this series which are found among the top 10 most similar games (excluding the query). The higher the score, the better the retrieval.

Results can be accessed from the [Wiki homepage](https://github.com/woctezuma/steam-descriptions/wiki/).

### Unique games

It is possible to highlight games with *unique* store descriptions, by applying a threshold to similarity values output by the algorithm.
This is done in [`find_unique_games.py`](find_unique_games.py):
-   the Tf-Idf model is used to compute similarity scores between store descriptions,
-   a game is *unique* if the similarity score between a query game and its most similar game (other than itself) is lower than or equal to an arbitrary threshold.

Results are shown [here](https://github.com/woctezuma/steam-descriptions/wiki/Unique_Games).

## References

-   [Microsoft's tutorial](https://github.com/microsoft/nlp/tree/master/examples/sentence_similarity) about sentence similarity,
-   [My answer on StackOverlow](https://stackoverflow.com/a/54330582/), about sentence embeddings
-   [Tutorial on the official website of 'gensim' module](https://radimrehurek.com/gensim/models/word2vec.html)
-   [Tutorial on a blog](http://kavita-ganesan.com/gensim-word2vec-tutorial-starter-code/)
-   [Tool: spaCy](https://spacy.io/)
-   [Tool: Gensim](https://radimrehurek.com/gensim/)
-   [Word2Vec](https://en.wikipedia.org/wiki/Word2vec)
-   [GloVe](https://github.com/stanfordnlp/GloVe)
-   [Universal Sentence Encoder](https://tfhub.dev/google/universal-sentence-encoder/2)
-   Sanjeev Arora, Yingyu Liang, Tengyu Ma, [A Simple but Tough-to-Beat Baseline for Sentence Embeddings](https://openreview.net/forum?id=SyK00v5xx), in: ICLR 2017 conference.
-   Jiaqi Mu, Pramod Viswanath, [All-but-the-Top: Simple and Effective Postprocessing for Word Representations](https://openreview.net/forum?id=HkuGJ3kCb), in: ICLR 2018 conference.

<!-- Definitions -->

[build]: <https://github.com/woctezuma/steam-descriptions/actions>
[build-image]: <https://github.com/woctezuma/steam-descriptions/workflows/Python package/badge.svg?branch=master>

[pyup]: <https://pyup.io/repos/github/woctezuma/steam-descriptions/>
[dependency-image]: <https://pyup.io/repos/github/woctezuma/steam-descriptions/shield.svg>
[python3-image]: <https://pyup.io/repos/github/woctezuma/steam-descriptions/python-3-shield.svg>

[codecov]: <https://codecov.io/gh/woctezuma/steam-descriptions>
[codecov-image]: <https://codecov.io/gh/woctezuma/steam-descriptions/branch/master/graph/badge.svg>

[codacy]: <https://www.codacy.com/app/woctezuma/steam-descriptions>
[codacy-image]: <https://api.codacy.com/project/badge/Grade/710292a19eff45e08a53e8b0028e02d4>

[universal_sentence_encoder]: <https://colab.research.google.com/github/woctezuma/steam-descriptions/blob/master/universal_sentence_encoder.ipynb>

[colab-badge]: <https://colab.research.google.com/assets/colab-badge.svg>
