import os
import math
import string
import random
import operator
from functools import reduce
from collections import Counter


def load_data_to_mini(path, per_class_max_docs=10, words_num=250):
    corpus = { }
    words_list = ""
    # with open(to_path, 'w') as f:
    for files in os.listdir(path):
        label = files[:-6]
        if label not in corpus.keys():
            corpus[label] = []

        file_path = os.path.join(path, files)
        with open(file_path) as f:
            text = f.read().lower()
            for c in string.punctuation:  # 去标点
                text = text.replace(c, ' ')
            for c in string.digits:  # 去数字
                text = text.replace(c, '')

            text = text.split()
            doc = []
            for i in range(1, per_class_max_docs * words_num + 1):
                words_list += text[i-1] + ' '
                doc.append(text[i-1])
                if i % words_num == 0:
                    corpus[label].append(doc)
                    doc = []

    words_list = list(set(words_list.split()))
    return corpus, words_list


def token_label(label):
    return {
        'Conan Doyle': 0,
        'Herman Melville': 1,
        'Jane Austen': 2,
    }.get(label, 'error')


def label_token(token):
    return {
        0: 'Conan Doyle',
        1: 'Herman Melville',
        2: 'Jane Austen',
    }.get(token, 'error')


def split_data_with_label(corpus):
    input_x = []
    input_y = []
    data = []
    for label in corpus:
        for x in corpus[label]:
            data.append((x, label))
    random.shuffle(data)
    for _ in data:
        input_x.append(_[0])
        input_y.append(token_label(_[1].split('.txt')[0]))
    return [input_x, input_y]


# calculate the term frequency
def tf(word, count):
    return count[word] / sum(count.values())


def n_containing(word, count_list):
    return sum(1 for count in count_list if word in count)

# calculate the idf value
def idf(word, count_list):
    return math.log(len(count_list)) / (1 + n_containing(word, count_list))

# calculate the tf-idf value
def tfidf(word, count, count_list):
    return tf(word, count) * idf(word, count_list)


def count_term(text):
    with open('../stopwords.txt') as f:
        stopwords = f.read().split()
    filtered = [w for w in text if not w in stopwords]
    count = Counter(filtered)

    return count


def feature_extractor(input_x, words_list):
    count_list = []
    for text in input_x:
        count_list.append(count_term(text))

    scores = []
    for count in count_list:
        tmp = {word: tfidf(word, count, count_list) for word in count}
        scores.append(tmp)

    features = []
    for i in range(len(input_x)):
        word_features = []
        for word in words_list:
            if word in scores[i].keys():
                word_features.append(scores[i][word])
            else:
                word_features.append(0)
        features.append(word_features)
    return features

# split the data to train and test
def split_data_to_train_and_test(x, y, rate=0.5):
    x_train, x_test, y_train, y_test =[], [], [], []
    index = list(range(len(x)))
    i = math.ceil(rate * len(x))
    train_index = index[0:i]
    test_index = index[i:]
    for i in train_index:
        x_train.append(x[i])
        y_train.append(y[i])

    for i in test_index:
        x_test.append(x[i])
        y_test.append(y[i])
    return x_train, x_test, y_train, y_test
