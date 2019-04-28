import sys
import pickle
sys.path.append("..")

import argparse
from math import *
from load_data import *


class DecisonTree:
    trainData = []
    trainLabel = []
    featureValues = {}
    max_depth = 0

    def __init__(self, trainData, trainLabel, threshold, max_depth):
        self.loadData(trainData, trainLabel)
        self.threshold = threshold
        self.max_depth = max_depth
        self.tree = self.createTree(range(0, len(trainLabel)), range(0, len(trainData[0])), 0)

    # load data
    def loadData(self, trainData, trainLabel):
        if len(trainData) != len(trainLabel):
            raise ValueError('input error')
        self.trainData = trainData
        self.trainLabel = trainLabel

        # calculate all the feature values, to split the data
        for data in trainData:
            for index, value in enumerate(data):
                if not index in self.featureValues.keys():
                    self.featureValues[index] = [value]
                if not value in self.featureValues[index]:
                    self.featureValues[index].append(value)


    # calculate the information entropy
    def caculateEntropy(self, dataset):
        labelCount = self.labelCount(dataset, self.trainLabel)
        size = len(dataset)
        result = 0
        for i in labelCount.values():
            pi = i / float(size)
            result -= pi * (log(pi) / log(2))
        return result

    # calculate the information gain
    def caculateGain(self, dataset_index, feature_index):
        values = self.featureValues[feature_index]
        result = 0
        for v in values:
            subDataset = self.splitDataset(dataset=dataset_index, feature=feature_index, value=v)
            result += len(subDataset) / float(len(dataset_index)) * self.caculateEntropy(subDataset)
        return self.caculateEntropy(dataset=dataset_index) - result

    # count the label occurrence
    def labelCount(self, dataset, trainLabel):
        labelCount = {}
        for i in dataset:
            if trainLabel[i] in labelCount.keys():
                labelCount[trainLabel[i]] += 1
            else:
                labelCount[trainLabel[i]] = 1

        return labelCount

    def createTree(self, dataset_index, features_index, depth):
        labelCount = self.labelCount(dataset_index, self.trainLabel)
        # If the feature set is empty,
        # the tree is a single-node tree,
        # the label with the largest number of samples is selected
        if not features_index:
            return max(list(labelCount.items()), key = lambda x:x[1])[0]

        if depth>=self.max_depth:
            return max(list(labelCount.items()), key=lambda x: x[1])[0]

        # If the data set contains only one tag, the tree is a single node tree
        if len(labelCount) == 1:
            return list(labelCount.keys())[0]

        # Calculate the information gain of each feature in the feature set
        l = map(lambda x : [x, self.caculateGain(dataset_index=dataset_index, feature_index=x)], features_index)
        # select the feature index has max info_gain
        max_feature_index, gain = max(list(l), key = lambda x: x[1])

        # If the maximum info_gain is less than the threshold, the label with the largest number of samples is selected
        if self.threshold > gain:
            return max(list(labelCount.items()), key=lambda x:x[1])[0]

        tree = {}
        # Select feature subset
        subFeatures = list(filter(lambda x : x != max_feature_index, features_index))
        tree['feature'] = max_feature_index
        # Build subtree
        for value in self.featureValues[max_feature_index]:
            subDataset = self.splitDataset(dataset=dataset_index, feature=max_feature_index, value=value)

            # Ensure that the sub-dataset is not empty
            if not subDataset:
                tree[value] = max(list(labelCount.items()), key = lambda x:x[1])[0]
                continue
            tree[value] = self.createTree(dataset_index=subDataset, features_index=subFeatures, depth=depth+1)
        return tree

    def splitDataset(self, dataset, feature, value):
        reslut = []
        for index in dataset:
            if self.trainData[index][feature] == value:
                reslut.append(index)
        return reslut


    def classify(self, data):
        def f(tree, data):
            if type(tree) != dict:
                return tree
            else:
                feature_index = tree['feature']
                value = data[feature_index]
                return f(tree[value], data)
        return f(self.tree, data)


parser = argparse.ArgumentParser(description='Decision tree for text classification')
parser.add_argument('--train', help='Training model', type=int, default=0)
parser.add_argument('--per_class_max_docs', help='Per-class select corresponding number of doc', type=int, default=100)
parser.add_argument('--predict', help='The file need to predict', type=str, default='')
parser.add_argument('--max_depth', help='The max depth of decision tree', type=int, default=10)
parser.add_argument('--threshold', help='The threshold of entropy cutoff', type=float, default=0.01)

args = parser.parse_args()
if args.train!=0:
    path = '../data'
    print('Loading data...')
    # Per_class_max_docs: short text extracted for each file
    corpus, _, texts = load_data_to_mini(path, per_class_max_docs=args.per_class_max_docs, words_num=250)
    x_train, y_train = split_data_with_label(corpus)
    # words = get_max_words(texts, words_list)
    with open('words.txt', 'r') as f:
        words = f.read()
    x_train = data_processing(x_train, words, texts, True)
    # x_train, x_test, y_train, y_test = split_data_to_train_and_test(x, y)
    tree = DecisonTree(trainData=x_train, trainLabel=y_train, threshold=args.threshold, max_depth=args.max_depth)
    fw = open('dtfile', 'wb')
    pickle.dump(tree, fw)
    fw.close()
    fw = open('texts', 'wb')
    pickle.dump(texts, fw)
    fw.close()

    path = '../test-data'
    corpus, _, __ = load_data_to_mini(path, per_class_max_docs=100, words_num=250)
    x_test, y_test = split_data_with_label(corpus)
    x_test = data_processing(x_test, words, texts, True)
    y_predict = []
    for data in x_test:
        y_predict.append(tree.classify(data))

    count = 0
    for i in range(len(y_test)):
        if y_predict[i] == y_test[i]:
            count += 1
    print('accuracy', count / len(y_test))


if args.predict!='':
    fr = open('dtfile', 'rb')
    tree = pickle.load(fr)
    fr.close()
    fr = open('texts', 'rb')
    texts = pickle.load(fr)
    fr.close()

    path = args.predict
    print('Loading data...')
    # Per_class_max_docs: short text extracted for each file
    corpus, words_list, _ = load_data_to_mini(path, per_class_max_docs=args.per_class_max_docs, words_num=250)
    x_test, y_test = split_data_with_label(corpus)
    with open('words.txt', 'r') as f:
        words = f.read()
    x_test = data_processing(x_test, words, texts, True)

    y_predict = []
    for data in x_test:
        y_predict.append(tree.classify(data))

    count = 0
    for i in range(len(y_test)):
        if y_predict[i] == y_test[i]:
            count += 1
    print('accuracy', count / len(y_test))
