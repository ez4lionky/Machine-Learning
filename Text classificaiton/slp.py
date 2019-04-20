import math
from load_data import *
from util import *


def trainByGD(input, label, input_num, epoch=300, lr=0.01, C=1e-4):
    global Weight
    x = input
    y = label
    for rounds in range(epoch):
        logits = mat_mul(x, Weight)
        prob = softmax(logits)

        for i in range(input_num):
            prob[i][int(y[i])] += 1

        penalty = mat_mul(C, Weight)
        l = mat_mul(-1.0 / input_num, mat_mul(T(prob), x))
        grad = mat_sub_add(T(l), penalty, operation='add')
        Weight = mat_sub_add(Weight, mat_mul(lr, grad), operation='sub')


def compute_accuracy(x_test, y_test, test_size):
    global Weight
    logits = mat_mul(x_test, Weight)
    prob = softmax(logits)

    prediction = mat_argmax(prob)
    count = 0
    for i in range(len(prediction)):
        if prediction[i] == y_test[i]:
            count = count + 1
    return (count + 0.0)/test_size


path = 'data'
corpus, _ = load_data_to_mini(path, per_class_max_docs=2)
x, y = split_data_with_label(corpus)
x = feature_extractor(x, _)
x_train, x_test, y_train, y_test = split_data_to_train_and_test(x, y)
input_dim = len(x_train[0])
classes = len(set(y_train))
train_size = len(x_train)
test_size = len(x_test)

Weight = []
for i in range(input_dim):
    row = []
    for j in range(classes):
        row.append(0.01)
    Weight.append(row)

trainByGD(x_train, y_train, train_size)
print("accuracy:", compute_accuracy(x_test, y_test, test_size))
print("Weight:", len(Weight))
