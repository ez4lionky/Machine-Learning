import numpy as np
import time
from load_data import *
from util import *
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier


def trainByGD(input, label, input_num, epoch=100, lr=0.01):
    Weight = []
    Bias = []
    for i in range(input_dim):
        row = []
        for j in range(classes):
            row.append(0)
        Weight.append(row)

    for _ in range(classes):
        Bias.append(0)

    x = input
    y = label
    for rounds in range(epoch):
        logits = mat_sub_add(mat_mul(x, Weight), Bias, operation='add')
        prob = softmax(logits)

        for i in range(input_num):
            prob[i][int(y[i])] -= 1

        w_grad = mat_mul(T(prob), x)
        b_grad = mat_mul( 1 / input_dim, mat_row_sum(T(prob)))
        Weight = mat_sub_add(Weight, mat_mul(lr, T(w_grad)), operation='sub')
        Bias = mat_sub_add(Bias, mat_mul(lr, b_grad), operation='sub')

    with open("Weight.txt", 'w') as f:
        for row in Weight:
            for _ in row:
                f.write(str(_) + ' ')
            f.write('\n')

    with open("Bias.txt", 'w') as f:
        for _ in Bias:
            f.write(str(_) + ' ')

    return Weight, Bias


def compute_accuracy(x_test, y_test, test_size):
    Weight = []
    Bias = []
    with open("Weight.txt") as f:
        line = f.readline()
        while line:
            row = []
            for _ in line.split():
                row.append(float(_))
            Weight.append(row)
            line = f.readline()

    with open("Bias.txt") as f:
        B = f.read()
        for _ in B.split():
            Bias.append(float(_))

    logits = mat_sub_add(mat_mul(x_test, Weight), Bias, operation='add')
    prob = softmax(logits)
    prediction = mat_argmax(prob)
    count = 0
    for i in range(len(prediction)):
        if prediction[i] == y_test[i]:
            count = count + 1
    print('Right prediction:', count)
    return (count + 0.0)/test_size


train_flag = False
test_flag = True

path = '../data'
print('Loading data...')
# Per_class_max_docs: short text extracted for each file
corpus, _, texts = load_data_to_mini(path, per_class_max_docs=100, words_num=250)
x_train, y_train = split_data_with_label(corpus)
with open('words.txt', 'r') as f:
    words = f.read()
x_train = data_processing(x_train, words)

input_dim = len(x_train[0])
classes = len(set(y_train))
print(classes)
train_size = len(x_train)

path = '../test-data'
corpus, _, texts = load_data_to_mini(path, per_class_max_docs=100, words_num=250)
x_test, y_test = split_data_with_label(corpus)
x_test = data_processing(x_test, words)
test_size = len(x_test)
print('Data loaded')
# clf = Perceptron()
clf = DecisionTreeClassifier(criterion='entropy', max_depth=4, min_impurity_split=0)
clf.fit(x_train, y_train)
print(clf.score(x_test, y_test))
# Weight = np.transpose(clf.coef_)
# Bias = clf.intercept_

# with open("Weight.txt", 'w') as f:
#     for row in Weight:
#         for _ in row:
#             f.write(str(_) + ' ')
#         f.write('\n')
#
# with open("Bias.txt", 'w') as f:
#     for _ in Bias:
#         f.write(str(_) + ' ')
#
# if train_flag:
#     print('training...')
#     s = time.time()
#     trainByGD(x_train, y_train, train_size)
#     e = time.time()
#     print('time: ', e - s)

# if test_flag:
#     print("accuracy:", compute_accuracy(x_test, y_test, test_size))