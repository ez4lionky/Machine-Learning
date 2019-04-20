# import os
# import math
# import string
# import random
# import operator
# from functools import reduce
# from collections import Counter
# from sklearn.feature_extraction.text import TfidfVectorizer
#
# # 每个作者提取的文档数量（每个文档250 words）
# # per_class_max_docs = 100
#
#
#
# def load_data_to_mini(path, per_class_max_docs=10, words_num=250):
#     corpus = {}
#     # with open(to_path, 'w') as f:
#     for files in os.listdir(path):
#         if files=='stopwords.txt':
#             continue
#         docs = []
#         file_path = os.path.join(path, files)
#         with open(file_path) as f:
#             text = f.read().lower()
#             for c in string.punctuation:  # 去标点
#                 text = text.replace(c, ' ')
#             for c in string.digits:  # 去数字
#                 text = text.replace(c, '')
#
#             text = text.split()
#             doc = []
#             for i in range(1, per_class_max_docs * words_num + 1):
#                 doc.append(text[i-1])
#                 if i % words_num == 0:
#                     docs.append(doc)
#                     doc = []
#
#         corpus[files] = docs
#
#     return corpus
#
# # def load_data_to_mini(path, per_class_max_docs=10):
#     # corpus = {} #训练数据
#     # words_list = [] #存放所有文档的单词
#     # for files in os.listdir(path):
#     #     if not os.path.isdir(os.path.join(path, files)):
#     #         continue
#     #     docs = []
#     #     file_path = os.path.join(path, files)
#     #     i = 0
#     #
#     #     for _ in range(per_class_max_docs):
#     #         with open(file_path + '/text{}.txt'.format(i)) as f:
#     #             text = f.read().split()
#     #             docs.append(text)
#     #             words_list.append(text)
#     #         i += 1
#     #     corpus[files] = docs
#     #
#     # words_list = reduce(operator.add, words_list)
#     # return corpus, words_list
#
#
# def token_label(label):
#     return {
#         'Conan Doyle': 0,
#         'Herman Melville': 1,
#         'Jane Austen': 2,
#     }.get(label, 'error')
#
#
# def label_token(token):
#     return {
#         0: 'Conan Doyle',
#         1: 'Herman Melville',
#         2: 'Jane Austen',
#     }.get(token, 'error')
#
#
# def split_data_with_label(corpus):
#     input_x = []
#     input_y = []
#     data = []
#     for label in corpus:
#         for x in corpus[label]:
#             data.append((x, label))
#     # random.shuffle(data)
#     for _ in data:
#         input_x.append(_[0])
#         input_y.append(token_label(_[1]))
#     return [input_x, input_y]
#
#
# def tf(word, count):
#     return count[word] / sum(count.values())
#
#
# def n_containing(word, count_list):
#     return sum(1 for count in count_list if word in count)
#
#
# def idf(word, count_list):
#     return math.log(len(count_list)) / (1 + n_containing(word, count_list))
#
#
# def tfidf(word, count, count_list):
#     return tf(word, count) * idf(word, count_list)
#
#
# def count_term(text):
#     with open('data/stopwords.txt') as f:
#         stopwords = f.read().split()
#     filtered = [w for w in text if not w in stopwords]
#     count = Counter(filtered)
#     return count
#
#
# def feature_extractor(input_x, max_df=1.0, min_df=0.0):
#     count_list = []
#     for text in input_x:
#         count_list.append(count_term(text))
#
#     scores = []
#     i = 0
#     for count in count_list:
#         tmp = {word: tfidf(word, count, count_list) for word in count}
#         scores.append(tmp)
#         i += 1
#
#     # reduced_features = []
#     # for _ in features:
#     #     reduced_features.append(reduce(operator.add, _))
#     return reduced_features
#
#
# def split_data_to_train_and_test(x, y, indices=0.2, random_state=10, shuffle=True):
#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=indices, random_state=10)
#     return x_train, x_test, y_train, y_test


import os
import nltk
import math
import string
import operator
from collections import Counter
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from functools import reduce

# 每个作者提取的文档数量（每个文档250 words）
# per_class_max_docs = 100

def load_data_to_mini(path, per_class_max_docs=10, words_num=10):
    corpus = {}
    words_list = []
    # with open(to_path, 'w') as f:
    for files in os.listdir(path):
        if files == 'stopwords.txt':
            continue
        docs = []
        file_path = os.path.join(path, files)
        with open(file_path) as f:
            text = f.read().lower()
            for c in string.punctuation:  # 去标点
                text = text.replace(c, ' ')
            for c in string.digits:  # 去数字
                text = text.replace(c, '')

            text = nltk.word_tokenize(text)  # 分割成单词
            for _ in range(per_class_max_docs):
                doc = []
                count = 0
                for word in text:
                    doc.append(word)
                    count += 1
                    if(count>=words_num):
                        break
                docs.append(doc)
        corpus[files] = docs
        words_list.append(docs)
    words_list = reduce(operator.add, words_list)
    words_list = reduce(operator.add, words_list)
    return corpus, words_list


def token_label(label):
    return {
        'Conan Doyle.txt': 0,
        'Herman Melville.txt': 1,
        'Jane Austen.txt': 2,
    }.get(label, 'error')


def split_data_with_label(corpus):
    input_x = []
    input_y = []
    data = []
    for label in corpus.keys():
        for x in corpus[label]:
            data.append((x, label))
    data = shuffle(data)
    for _ in data:
        input_x.append(_[0])
        input_y.append(token_label(_[1]))

    return [input_x, input_y]


def tf(word, count):
    return count[word] / sum(count.values())


def n_containing(word, count_list):
    return sum(1 for count in count_list if word in count)


def idf(word, count_list):
    return math.log(len(count_list)) / (1 + n_containing(word, count_list))


def tfidf(word, count, count_list):
    return tf(word, count) * idf(word, count_list)


def feature_extractor(input_x, words_list, max_df=1.0, min_df=0.0):
    tf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_df=max_df, min_df=min_df)
    tf.fit(words_list)
    features = []
    print(len(input_x))
    for _ in input_x:
        tmp = tf.transform(_).toarray()
        features.append(tmp)
    reduced_features = []
    for _ in features:
        reduced_features.append(reduce(operator.add, _))
    return reduced_features


def count_term(text):

    filtered = [w for w in tokens if not w in stopwords.words('english')]
    stemmer = PorterStemmer()
    stemmed = stem_tokens(filtered, stemmer)
    count = Counter(stemmed)
    return count

def split_data_to_train_and_test(x, y, indices=0.2, random_state=10, shuffle=True):

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=indices, random_state=10)
    return x_train, x_test, y_train, y_test
