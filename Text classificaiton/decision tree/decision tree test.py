
from math import *
from load_data import *


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
        for i in range(2):
            words.append(sort_d[i][0])
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
            for c in string.punctuation:  # Punctuation
                text = text.replace(c, ' ')
            for c in string.digits: # Digits
                text = text.replace(c, '')

            text = text.split()
            doc = []
            for i in range(1, per_class_max_docs * words_num + 1):
                words_list += text[i-1] + ' '
                doc.append(text[i-1])
                if i % words_num == 0:
                    corpus[label].append(doc)
                    doc = []
    data = {}
    for label in corpus.keys():
        data[label] = reduce(operator.add, corpus[label])

    words_list = list(set(words_list.split()))
    return corpus, words_list, data

def data_processing(input_x, words):
    features = []

    for text in input_x:
        row = [] # stores the feature for each doc

        # calculate the word average length
        sum = 0
        count = 0
        for word in text:
            sum += len(word)
            count += 1
        row.append(1 if sum/count>5 else 0)

        # count the number of 'the'
        sum = 0
        for word in text:
            if word=='the':
                sum += 1
        if sum<10:
            row.append(0)
        elif sum<=13:
            row.append(1)
        else:
            row.append(2)

        # count the key word frequency
        for key_word in words:
            sum = 0
            for word in text:
                if word==key_word:
                    sum += 1
            if sum>3:
                row.append(1)
            else:
                row.append(0)

        features.append(row)

    # print(np.shape(features))
    return features



# The input of decision tree should be [n, feature_dim] each feature is a discrete attribute, which can be calculated as the prob
class DecisonTree:
    trainData = []
    trainLabel = []
    featureValues = {} #每个特征所有可能的取值
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

        #计算 featureValues的可能值，用来根据特征的取值划分数据集
        for data in trainData:
            for index, value in enumerate(data):
                if not index in self.featureValues.keys():
                    self.featureValues[index] = [value]
                if not value in self.featureValues[index]:
                    self.featureValues[index].append(value)


    # 计算信息熵
    def caculateEntropy(self, dataset):
        labelCount = self.labelCount(dataset, self.trainLabel)
        size = len(dataset)
        result = 0
        for i in labelCount.values():
            pi = i / float(size)
            result -= pi * (log(pi) / log(2))
        return result

    # 计算信息增益
    def caculateGain(self, dataset_index, feature_index):
        values = self.featureValues[feature_index] #特征所有可能的取值
        result = 0
        for v in values:
            subDataset = self.splitDataset(dataset=dataset_index, feature=feature_index, value=v)
            result += len(subDataset) / float(len(dataset_index)) * self.caculateEntropy(subDataset)
        return self.caculateEntropy(dataset=dataset_index) - result

    # 计算数据集中，每个标签出现的次数
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
        feature, gain = max(list(l), key = lambda x: x[1])

        # If the maximum info_gain is less than the threshold, the label with the largest number of samples is selected
        if self.threshold > gain:
            return max(list(labelCount.items()), key=lambda x:x[1])[0]

        tree = {}
        # Select feature subset
        subFeatures = list(filter(lambda x : x != feature, features_index))
        tree['feature'] = feature
        # Build subtree
        for value in self.featureValues[feature]:
            subDataset = self.splitDataset(dataset=dataset_index, feature=feature, value=value)

            # Ensure that the sub-dataset is not empty
            if not subDataset:
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

if __name__ == '__main__':
    path = '../data'
    print('Loading data...')
    # Per_class_max_docs: short text extracted for each file
    corpus, words_list, data = load_data_to_mini(path, per_class_max_docs=50, words_num=250)
    x, y = split_data_with_label(corpus)
    words = get_max_words(data, words_list)
    x = data_processing(x, words)

    x_train, x_test, y_train, y_test = split_data_to_train_and_test(x, y)

    tree = DecisonTree(trainData=x_train, trainLabel=y_train, threshold=0.001, max_depth=5)

    y_predict = []
    for data in x_test:
        y_predict.append(tree.classify(data))

    count = 0
    for i in range(len(y_test)):
        if y_predict[i]==y_test[i]:
            count += 1
    print('accuracy', count/len(y_test))
