import math
from load_data import *


def mat_row_sum(x):
    if isinstance(x, (tuple, list)):
        row_sum = []
        for _ in x:
            row_sum.append(sum(_))
        return row_sum


def mat_exp(x):
    if isinstance(x, (tuple, list)):
        exp = []
        for _ in x:
            row = []
            for __ in _:
                row.append(math.exp(__))
            exp.append(row)
        return exp


def mat_argmax(x):
    row_argmax = []
    for row in x:
        for i, val in enumerate(row):
            if (val == max(row)):
                row_argmax.append(i)
    return row_argmax


def mat_sub_add(M1, M2, operation='sub'):
    if isinstance(M1, (tuple, list)) and isinstance(M2, (tuple, list)):
        assert len(M1)==len(M2)
        assert len(M1[0])==len(M2[0])
        result = []
        for i in range(len(M1)):
            row = []
            for j in range(len(M1[0])):
                if operation == 'sub':
                    row.append(M1[i][j] - M2[i][j])
                if operation == 'add':
                    row.append(M1[i][j] + M2[i][j])
            result.append(row)
        return result


def mat_mul(M1, M2):
    if isinstance(M1, (float, int)) and isinstance(M2, (tuple, list)):
        return [[M1 * i for i in j] for j in M2]
    if isinstance(M1, (tuple, list)) and isinstance(M2, (tuple, list)):
        return [[sum(map(lambda x: x[0] * x[1], zip(i, j))) for j in zip(*M2)] for i in M1]


def T(x):
    if isinstance(x, (tuple, list)):
        return list(zip(*x))


def softmax(x):
    prob = mat_exp(x)
    row_sum = mat_row_sum(prob)
    for i in range(len(prob)):
        for j in range(len(prob[i])):
            prob[i][j] /= row_sum[i]
    return prob


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
corpus, words_list = load_data_to_mini(path, per_class_max_docs=100)
x, y = split_data_with_label(corpus)
x = feature_extractor(x, words_list)
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

