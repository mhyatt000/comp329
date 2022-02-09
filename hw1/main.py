import itertools
import os
import random
import re

import matplotlib.pyplot as plt
import seaborn as sns
import yaml


def get_data():

    # extraction
    features = []
    targets = []
    for label in ["neg", "pos"]:
        with open(f"rt-polarity.{label}") as file:
            f = [
                re.sub("\-", " ", re.sub("[\.,\(\)\[\]]", "", line))
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


def clf(x, y):
    total, c_pos, c_neg = 0, 0, 0
    num_pos, num_neg = 0, 0

    with open("config.yml", "r") as file:
        data = yaml.safe_load(file)
    pos = data["pos"]
    neg = data["neg"]

    for xi, yi in zip(x, y):
        num_neg = num_neg + 1 if yi == 0 else num_neg
        num_pos = num_pos + 1 if yi == 1 else num_pos

        if any([(item in xi) for item in neg]) or re.findall("n't", " ".join(xi)):
            if yi == 0:
                c_neg += 1
        elif any([(item in xi) for item in pos]):
            if yi == 1:
                c_pos += 1
        else:
            if yi == 1:
                c_pos += 1
        total += 1

    acc = (c_neg + c_pos) / total
    print(f"accuracy: {100*acc}%")
    print(f"c_pos={c_pos}, c_neg={c_neg}")
    return acc, [[c_neg, num_neg - c_neg], [num_pos - c_pos, c_pos]]


def train(x, y):
    tokens = []
    for item in x:
        tokens += item
    tokens = list(set(tokens))

    try:
        with open("config.yml", "r") as file:
            data = yaml.safe_load(file)
    except:
        data = []

    if not data:
        data = {"pos": [], "neg": [], "neither": [], "other": tokens}

    print("3: pos")
    print("2: neither")
    print("1: neg")
    print("0: quit...\n")

    i = 0
    while data["other"]:
        item = data["other"][0]
        label = input(f"{i}... {item}: ")

        data["other"].remove(item)
        if label == "3":
            data["pos"].append(item)
        if label == "2":
            data["neither"].append(item)
        if label == "1":
            data["neg"].append(item)
        if label == "":
            data["other"].append(item)

        if label == "0":
            break

        if i % 25 == 0:
            print("saving file")
            with open("config.yml", "w") as file:
                yaml.dump(data, file)
        i += 1

    with open("config.yml", "w") as file:
        yaml.dump(data, file)


def main():

    x, y = get_data()
    # train(x, y) if input('train? (y/n) ') == 'y' else None

    acc, cmatx = clf(x, y)
    # plot(cmatx)
    return acc


def plot(cmatx):

    fig = sns.heatmap(cmatx, cmap="YlGnBu", annot=True, fmt="d")
    fig.set_xlabel("Predicted Class")
    fig.set_ylabel("Class Label")
    fig.set_title("Sentiment Analysis Confusion Matrix")
    plt.show()


if __name__ == "__main__":
    main()
