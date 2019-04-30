import os
import math
import random
import operator
from functools import reduce
from collections import Counter


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


def get_max_words(input_x, words_list):
    count_list = {}
    for label in input_x:
        count_list[label] = count_term(input_x[label])

    scores = {}
    for label in count_list:
        count = count_list[label]
        tmp = {word: tfidf(word, count, count_list) for word in count}
        if label not in scores.keys():
            scores[label] = tmp
        else:
            scores[label].append(tmp)

    words = []
    for label in scores:
        d = scores[label]
        sort_d = sorted(d.items(), key=lambda d: d[1], reverse=True)
        for i in range(5):
            words.append(sort_d[i][0])

    words = list(set(words))
    return words


def load_data_to_mini(path, per_class_max_docs=10, words_num=250):
    corpus = {}
    words_list = ""
    for files in os.listdir(path):
        label = files[:-6]
        if label not in corpus.keys():
            corpus[label] = []

        file_path = os.path.join(path, files)
        with open(file_path) as f:
            text = f.read().lower()
            text = text.split()
            doc = []
            for i in range(1, per_class_max_docs * words_num + 1):
                words_list += text[i - 1] + ' '
                doc.append(text[i-1])
                if i % words_num == 0:
                    corpus[label].append(doc)
                    doc = []

    texts = {}
    for label in corpus.keys():
        texts[label] = reduce(operator.add, corpus[label])
    words_list = list(set(words_list.split()))
    return corpus, words_list, texts

def load_single_text_to_test(path, words_num=250):
    corpus = []

    with open(path) as f:
        text = f.read().lower()
        text = text.split()
        doc = []
        for i in range(1, words_num + 1):
            doc.append(text[i-1])
            if i % words_num == 0:
                corpus.append(doc)
                doc = []
    return corpus

def norm(x):
    min_x = min(x)
    max_x = max(x)
    for i in range(len(x)):
        x[i] = float(x[i] - min_x)/(max_x - min_x)
    return x


def data_processing(input_x, words, texts, decision_tree=False):
    mean = {}
    if decision_tree:
        for key_word in words.split():
            sum = 0
            for text in texts:
                for word in text:
                    if word==key_word:
                        sum += 1
            mean[key_word] = sum/len(texts)
    features = []
    for text in input_x:
        row = [] # stores the feature for each doc
        sum_length = 0
        count = 0

        for word in text:
            sum_length += len(word)
            count += 1
        mean_length = math.ceil(sum_length / count)
        row.append(1 if mean_length>5 else 0)


        # count the key word frequency
        for key_word in words.split():
            sum = 0
            for word in text:
                if word == key_word:
                    sum += 1
            if decision_tree:
                if sum > mean[key_word]:
                    sum = 1
                else:
                    sum = 0
            row.append(sum)
        if not decision_tree:
            row = norm(row)
        features.append(row)

    return features
