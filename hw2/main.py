import itertools
import os
import random
import re

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from tqdm import tqdm


def get_data():

    # extraction
    features, targets = [], []
    for label in ["neg", "pos"]:
        with open(f"rt-polarity.{label}") as file:
            f = [
                re.sub("[^\w ]", " ", line)
                for line in file.read().split("\n")
            ]
            features += f

            t = 0 if label == "neg" else 1
            targets += [t for item in f]

    # shuffle
    data = [[f, t] for f, t in zip(features, targets)]
    random.shuffle(data)

    x, y = [col for col in zip(*data)]
    x = [[word for word in row.split(" ") if word] for row in x]

    return x, y


def fit(*, x, y):

    stop, p = {}, {}

    for xi, yi in zip(x, y):
        for word in xi:
            if word not in p.keys():
                p[word] = [0, 0, 0]
                stop[word] = 0
            p[word][yi] += 1
            p[word][2] += 1
            stop[word] += 1

    stop = [[word, val] for word, val in stop.items()]
    def freq(x): return x[1]
    stop.sort(reverse=True, key=freq)
    stop = stop[:150] + [item for item in stop if item[1] <= 1]
    stop = set([item[0] for item in stop])

    for word in stop:
        del p[word]

    vocab = set([word for word in p.keys()])
    for k, v in p.items():
        div = v[2]+len(vocab)
        p[k] = [(v[0]+1)/div, (v[1]+1)/div]

    # sum of 1s and 0s is num_pos
    prior = ((len(y)-sum(y))/len(y), sum(y)/len(y))

    from pprint import pprint
    temp = [(k, abs(v[0]-v[1])) for k, v in p.items()]
    temp.sort(key=(lambda x: x[1]))
    pprint(temp)

    return vocab, p, prior
    # p = {word: [sum(word in xi)] for word in vocab}
    # p = {word: probability for word in vocab}


def clf(*, x, y, vocab, p, prior):
    '''classifies all documents given the likelihood and prior'''

    x = [[word for word in xi if word in vocab] for xi in x]

    scores = [[np.log(prior[i]) + np.sum([np.log(p[word][i]) for word in xi])
               for i in [0, 1]] for xi in x]
    scores = [score[0] <= score[1] for score in scores]

    acc = sum([si == yi for si, yi in zip(scores, y)]) / len(scores)
    return acc, scores, y


def split(*, data):
    '''splits data into train,dev,test denoted a,b,c'''

    slices = [int(len(data)*i) for i in [0.70, 0.85]]
    train, dev, test = data[:slices[0]], data[slices[0]:slices[1]], data[slices[1]:]

    out = []
    for item in [train, dev, test]:
        x, y = [col for col in zip(*item)]
        [out.append(vec) for vec in (x, y)]
    return out


def confusion(*, scores, y):
    cmatx = [[0, 0], [0, 0]]

    for i, j in zip(y, scores):
        cmatx[i][j] += 1

    return cmatx


def main():

    x, y = get_data()
    data = [item for item in zip(x, y)]
    a_x, a_y, b_x, b_y, c_x, c_y = split(data=data)

    vocab, p, prior = fit(x=a_x+b_x, y=a_y+b_y)

    acc, scores, y = clf(x=c_x, y=c_y, vocab=vocab, p=p, prior=prior)
    print(acc)

    # plot(confusion(scores=scores, y=y))
    return acc


def plot(cmatx):

    fig = sns.heatmap(cmatx, cmap="YlGnBu", annot=True, fmt="d")
    fig.set_xlabel("Predicted Class")
    fig.set_ylabel("Class Label")
    fig.set_title("Sentiment Analysis Confusion Matrix")
    plt.show()


def hist(x):
    plt.hist(x, color='green')
    plt.title('Naive Bayes')
    plt.ylabel('Frequency')
    plt.xlabel('Accuracy (%)')
    plt.show()


def hist2d(x, y):

    fig, ax = plt.subplots()
    hh = ax.hist2d(x, y, cmap='Blues')
    fig.colorbar(hh[3], ax=ax)

    plt.title('Accuracy of Naive Bayes Algorithm')
    plt.ylabel('Minimum required word occurences')
    plt.xlabel('Accuracy (%)')
    plt.show()


if __name__ == "__main__":
    main()
    # hist([main() for i in range(50)])
