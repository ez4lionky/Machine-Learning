import os
import codecs
import jieba
import re

from sklearn.utils import shuffle


# 每篇文档保留的文档数量
# per_class_max_docs = 1000

def load_data_to_mini(path, to_path, per_class_max_docs=100):
    """
    处理清华大学语料库，将类别和文档处理成fasttext 所需要的格式
    :param path: 
    :param to_path: 
    :return: 
    """
    # 抽取后的语料库
    # 列举当前目录下的所有子列别目录

    corpus = []

    with codecs.open(to_path, 'w') as f:
        for files in os.listdir(path):
            curr_path = os.path.join(path, files)
            print('curr_path', curr_path)
            if os.path.isdir(curr_path):
                count = 0
                docs = []
                for file in os.listdir(curr_path):
                    count += 1
                    if count > per_class_max_docs:
                        break
                    file_path = os.path.join(curr_path, file)
                    # 读取文件中的内容
                    with codecs.open(file_path, 'r', encoding='utf-8') as fd:
                        docs.append(
                            '__label__' + files + ' ' + ' '.join(jieba.cut(re.sub('[  \n\r\t]+', '', fd.read()))))
                        f.write('__label__' + files + ' ' + ' '.join(jieba.cut(re.sub('[  \n\r\t]+', '', fd.read()))))
            corpus.append(docs)

    # 将数据写到一个新的文件中
    with codecs.open(to_path, 'a') as f:
        for docs in corpus:
            for doc in docs:
                f.write(doc + '\n')

    return corpus

def split_data_with_label(corpus):
    """
    将数据划分为训练数据和样本标签
    :param corpus: 
    :return: 
    """
    input_x = []
    input_y = []

    tag = []
    # if os.path.isfile(corpus):
    #     with codecs.open(corpus, 'r') as f:
    #         for line in f:
    #             print(line)
    #             tag.append(line)
    #
    # else:
    for docs in corpus:
        for doc in docs:
            tag.append(doc)
    tag = shuffle(tag)
    for doc in tag:
        index = doc.find(' ')
        input_y.append(doc[:index])
        input_x.append(doc[index + 1:])

    return [input_x, input_y]


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics.scorer import make_scorer
from sklearn import linear_model
from sklearn import metrics

from time import time

def feature_extractor(input_x, max_df=1.0, min_df=0.0):
    """
    特征抽取
    :param corpus: 
    :param case: 不同的特征抽取方法
    :return: 
    """
    return TfidfVectorizer(token_pattern='\w', ngram_range=(1,2), max_df=max_df, min_df=min_df).fit_transform(input_x)


def split_data_to_train_and_test(corpus, indices=0.2, random_state=10, shuffle=True):
    """
    将数据划分为训练数据和测试数据
    :param corpus: [input_x]
    :param indices: 划分比例
    :random_state: 随机种子
    :param shuffle: 是否打乱数据
    :return: 
    """
    input_x, y = corpus

    # 切分数据集
    x_train, x_test, y_train, y_test = train_test_split(input_x, y, test_size=indices, random_state=10)
    print("Vocabulary Size: {:d}".format(input_x.shape[1]))
    print("Train/test split: {:d}/{:d}".format(len(y_train), len(y_test)))
    return x_train, x_test, y_train, y_test

def fit_and_predicted(train_x, train_y, test_x, test_y, penalty='l2', C=1.0, solver='lbfgs'):
    """
    训练与预测
    :param train_x: 
    :param train_y: 
    :param test_x: 
    :param test_y: 
    :return: 
    """
    clf = linear_model.LogisticRegression(penalty=penalty, C=C, solver=solver, n_jobs=-1).fit(train_x, train_y)
    predicted = clf.predict(test_x)
    print(metrics.classification_report(test_y, predicted))
    print('accuracy_score: %0.5f' %(metrics.accuracy_score(test_y, predicted)))


corpus = load_data_to_mini('../data/THUCNews', 'thu_data_all', 100)
print('corpus size(%d,%d)' %(len(corpus), len(corpus[0])))

# 1. 加载语料
corpus = split_data_with_label(corpus)
input_x, y = corpus
# 2. 特征选择
input_x = feature_extractor(input_x)
# 3.切分训练数据和测试数据
train_x, test_x, train_y, test_y = split_data_to_train_and_test([input_x, y])

# 4. 训练以及测试
t0 = time()
print('\t\t使用 max_df,min_df=(1.0,0.0) 进行特征选择的逻辑回归文本分类\t\t')
fit_and_predicted(train_x, train_y, test_x, test_y)
print('time uesed: %0.4fs' %(time() - t0))