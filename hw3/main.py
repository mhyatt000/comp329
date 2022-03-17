from functools import reduce
import itertools
import math
import os
import random
import re

from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm


def get_documents():

    # extraction
    features, targets = [], []
    for label in ["neg", "pos"]:
        with open(f"rt-polarity.{label}") as file:
            f = [
                re.sub("[^\w ]", " ", re.sub("'", "", line))
                for line in file.read().split("\n")
            ]
            f = [item for item in f if item]

            features += f

            t = 0 if label == "neg" else 1
            targets += [t for item in f]

    # shuffle
    data = [[f, t] for f, t in zip(features, targets)]
    random.shuffle(data)

    x, y = [col for col in zip(*data)]
    x = [[word for word in row.split(" ") if word] for row in x]

    n = len(y)
    a = int(n*0.85)

    train_x, train_y, test_x, test_y = x[:a], y[:a], x[a:], y[a:]
    return train_x, train_y, test_x, test_y


class Vectorizer():
    '''tfidf vectorizer from scratch'''

    def __init__(self):
        pass

    def fit_transform(self, *, documents):
        '''fits vectorizer and returns tfidf vectors for doc in documents'''

        self.vocab = set(reduce(lambda x, y: x + y, documents, []))

        docs_frequency = self._compute_freq(documents=documents, vocab=self.vocab)
        docs_tf = self._compute_TF(docs_frequency=docs_frequency, documents=documents)
        self.idf = self._compute_IDF(docs_frequency)

        tfidf = self._compute_tfidf(docs_tf=docs_tf, idf=self.idf)
        return tfidf

    def _compute_freq(self, *, documents, vocab):
        '''computes frequency of words in a doc ... NOT TF
        inspired by https://github.com/corymaklin/tfidf/blob/master/tfidf.ipynb'''

        docs_frequency = []
        for doc in tqdm(documents):
            freq = dict.fromkeys(vocab, 0)

            for word in doc:
                freq[word] += 1
            docs_frequency.append(freq)
        return docs_frequency

    def _compute_TF(self, *, docs_frequency, documents):
        """computes term frequency
        inspired by https://github.com/corymaklin/tfidf/blob/master/tfidf.ipynb"""

        docs_tf = []
        for freq, doc in tqdm(zip(docs_frequency, documents)):

            tf = {word: count / (1+len(doc)) for word, count in freq.items()}
            docs_tf.append(tf)

        return docs_tf

    def _compute_IDF(self, docs_frequency):
        """https://github.com/corymaklin/tfidf/blob/master/tfidf.ipynb"""

        N = len(docs_frequency)
        idf = dict.fromkeys(docs_frequency[0].keys(), 0)

        for freq in tqdm(docs_frequency):
            for word, val in freq.items():
                if val > 0:
                    idf[word] += 1

        idf = {word: math.log(N / float(val)) for word, val in idf.items()}
        return idf

    def _compute_tfidf(self, *, docs_tf, idf):
        '''computes tfidf'''

        tfidf = [[tf[word]*idf[word] for word in tf.keys()] for tf in tqdm(docs_tf)]
        return tfidf

    def transform(self, *, documents):
        '''returns tfidf vectors given previous word idf values'''

        documents = [[word for word in doc if word in self.vocab] for doc in documents]
        docs_frequency = self._compute_freq(documents=documents, vocab=self.vocab)
        docs_tf = self._compute_TF(docs_frequency=docs_frequency, documents=documents)

        tfidf = self._compute_tfidf(docs_tf=docs_tf, idf=self.idf)
        return tfidf


def main():

    train_x, train_y, test_x, test_y = get_documents()

    print('custom tfidf vectorizer...')

    vec = Vectorizer()
    vec_train_x = vec.fit_transform(documents=train_x)

    model = MLPClassifier(random_state=1, verbose=True, early_stopping=True)
    model.fit(vec_train_x, train_y)

    vec_test_x = vec.transform(documents=test_x)
    acc = model.score(vec_test_x, test_y)

    print(f'model test accuracy: {acc}')

    print('sklearn tfidf vectorizer...')

    vectorizer = TfidfVectorizer()
    raw_train_x = [' '.join(doc) for doc in train_x]
    vec_train_x = vectorizer.fit_transform(raw_train_x)

    model = MLPClassifier(random_state=1, max_iter=300, verbose=True, early_stopping=True)
    model.fit(vec_train_x, train_y)

    raw_test_x = [' '.join(doc) for doc in test_x]
    vec_test_x = vectorizer.transform(raw_test_x)
    acc = model.score(vec_test_x, test_y)

    print(f'model test accuracy: {acc}')


if __name__ == "__main__":
    main()
