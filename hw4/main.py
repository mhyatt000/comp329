import tempfile

from gensim import utils
import gensim.downloader as api
import gensim.models
from gensim.test.utils import datapath
from tqdm import tqdm
from scipy import spatial


class Corpus:
    """An iterator that yields sentences (lists of str)."""

    def __iter__(self):
        # corpus_path = datapath("lee_background.cor")
        corpus_path = 'all-the-text.txt'

        for line in open(corpus_path):
            # assume there's one document per line, tokens separated by whitespace
            yield utils.simple_preprocess(line)

def save(model, filename):
    model.save(filename)
    print(f'saved {filename}')

def load(filename):
    return gensim.models.Word2Vec.load(filename)

def analogy(wv, words):

    a = wv[words[0]] - wv[words[1]] + wv[words[2]]
    b = wv[words[3]]

    result = 1 - spatial.distance.cosine(a,b)

    print(f'{words[0]} - {words[1]} + {words[2]} = {words[3]}')
    print(result)


def main():

    # 1

    filename = 'gensim-model'

    try:
        model = load(filename)
    except:
        sentences = Corpus()
        model = gensim.models.Word2Vec(sentences=sentences)
        save(model, filename)

    wv = model.wv
    keys = wv.index_to_key

    pairs = [
        ("apple", "orange"),
        ("apple", "sauce"),
        ("apple", "pie"),
        ("apple", "tree"),
        ('apple','banana'),
        ('apple','orchard'),
        ('apple','running'),
        ('cake','pie'),
        ('teacher','student')
    ]

    [print(f"{w1}\t{w2}\t{wv.similarity(w1, w2):.2f}") for w1,w2 in pairs]
    print()

    filename = 'google-news-model'

    try:
        model = load(filename)
        wv = model.wv
    except:
        wv = api.load('word2vec-google-news-300')
        save(model, filename)

    keys = wv.index_to_key

    groups = ['car','fruit','exercise','president','earthquake']
    for word in groups:
        print(f'{word}:\t',[item[0] for item in wv.most_similar(positive=[word], topn=5)])
    print()

    # ['vehicle', 'cars', 'SUV', 'minivan', 'truck']
    # ['fruits', 'cherries', 'berries', 'pears', 'citrus_fruit']
    # ['excercise', 'exercises', 'Exercise', 'exercising', 'Fibromyalgia_aquatic']
    # ['President', 'chairman', 'vice_president', 'chief_executive', 'CEO']
    # ['quake', '#.#_magnitude_earthquake', '#.#_magnitude_quake', 'temblor', 'devastating_earthquake']

    corr = model.wv.evaluate_word_pairs('wordsim_similarity_goldstandard.txt')[1]
    print(corr)
    print()

    analogy(wv, ['swimmer','swim','run','runner'])
    analogy(wv, ['baker','bread','money','banker'])
    analogy(wv, ['fruit','sweet','sour','lemon'])
    analogy(wv, ['moon','night','day','sun'])


if __name__ == "__main__":
    main()


"""todo

3. evaluate pretrained model on wordsim-353
    - http://alfonseca.org/eng/research/wordsim353.html
    - wordsim_similarity_goldstandard.txt
    - measure with spearmans rank correlation coefficient
    - post on teams for comparrison
4. propose several word analogies
    - ex: king - man + woman = queen
    - test using pretrained model

"""
