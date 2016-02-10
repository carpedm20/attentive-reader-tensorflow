Attentive Reader
================

Tensorflow implementation of Google DeepMind's [Teaching Machines to Read and Comprehend](http://arxiv.org/pdf/1506.03340v3.pdf). This implementation contains:

![models](./assets/models.png)

1. Deep LSTM Reader
    - with [skip connections](http://arxiv.org/pdf/1308.0850.pdf) from the inputs to all hidden layers
    - with [peephole weights](http://www.jmlr.org/papers/volume3/gers02a/gers02a.pdf) that provide precise timing
2. Attentive Reader (in progress)
    - with [Bidirectional LSTMs](http://www.cs.toronto.edu/~graves/nn_2005.pdf) with peephole weights
3. Impatient Reader (in progress)
    - with [Bidirectional LSTMs](http://www.cs.toronto.edu/~graves/nn_2005.pdf) with peephole weights
    - with recurrent accumulation of information from the document while reading the query


Prerequisites
-------------

- Python 2.7 or Python 3.3+
- [Tensorflow](https://www.tensorflow.org/)
- [NLTK](http://www.nltk.org/)
- [Gensim](https://radimrehurek.com/gensim/index.html)


Usage
-----

First, you need to download [DeepMind Q&A Dataset](https://github.com/deepmind/rc-data) from [here](http://cs.nyu.edu/~kcho/DMQA/), save `cnn.tgz` and `dailymail.tgz` into the repo, and run:

    $ ./unzip.sh cnn.tgz dailymail.tgz

Then run the pre-processing code with:

    $ python data_utils.py data cnn

To train a model with `cnn` dataset:

    $ python main.py --dataset cnn

To test an existing model:

    $ python main.py --dataset cnn --forward_only True

(in progress)


Results
-------

in progres, should be like:

![result_from_paper](./assets/result_from_paper.png)


Author
------

Taehoon Kim / [@carpedm20](http://carpedm20.github.io/)
