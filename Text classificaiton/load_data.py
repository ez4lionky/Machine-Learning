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

def load_data_to_mini(path, per_class_max_docs=10, words_num=250):
    corpus = {} #训练数据
    words_list = [] #存放所有文档的单词
    # with open(to_path, 'w') as f:
    for files in os.listdir(path):
        if os.path.isdir(os.path.join(path, files)):
            continue
        docs = []
        file_path = os.path.join(path, files)
        direc = os.path.join(path, str(files).split('.txt')[0])
        if not os.path.exists(direc):
            os.makedirs(direc)
        with open(file_path) as f:
            text = f.read().lower()
            for c in string.punctuation:  # 去标点
                text = text.replace(c, ' ')
            for c in string.digits:  # 去数字
                text = text.replace(c, '')

            text = nltk.word_tokenize(text)  # 分割成单词
            i = 0
            for _ in range(per_class_max_docs):
                doc = []
                count = 0
                for word in text:
                    doc.append(word)
                    count += 1
                    if(count>=words_num):
                        break
                with open(direc + '/text{}.txt'.format(i), 'w') as f:
                    f.write(' '.join(doc))
                docs.append(doc)
                i += 1
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
    for label in corpus:
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
