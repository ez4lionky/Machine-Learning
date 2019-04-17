import math
import csv
import numpy as np
import matplotlib.pyplot as plt


def load_data(path):
    csvfile = csv.reader(open(path, 'r'))

    index = 0
    indices = []
    notes_index = 0

    header = next(csvfile)
    for _ in header:
        if _ == 'Temperature' or _ == 'MPG':
            indices.append(index)
        elif  _ =='Notes':
            notes_index = index
        index += 1

    x = []
    y = []
    for row in csvfile:
        line = []
        flag = True
        for i in indices:
            if row[notes_index] == 'sch' :
                flag = False
            else:
                line.append(float(row[i]))
        if flag == True:
            x.append(line[0])
            y.append(line[1])
    return x, y


train_x, train_y = load_data('data/mpg-2018.csv')
train_x = np.reshape(train_x, newshape=(-1, 1))
train_y = np.reshape(train_y, newshape=(-1, 1))

test_x, test_y = load_data('data/mpg-2017.csv')
test_x = np.reshape(test_x, newshape=(-1, 1))
test_y = np.reshape(test_y, newshape=(-1, 1))
# a, b = load_data('data/mpg-2017.csv')
# print(len(a))
# print(len(b))


def model(w1, b, w2, w3, x):
    return w1 * x + b + w2 * np.log(x + w3)


def optimize(w1, b, w2, w3, x, y):
    n = len(x)
    alpha = 1e-4
    y_hat = model(w1, b, w2, w3, x)
    dw1 = (1.0 / n) * ((y_hat - y) * x).sum()
    db = (1.0 / n) * ((y_hat - y).sum())
    dw2 = (1.0 / n) * ((y_hat - y) * np.log(x + w3)).sum()
    dw3 = (1.0 / n) * ((y_hat - y) * w2 * 1.0 / np.log(x + w3)).sum()
    w1 = w1 - alpha * dw1
    b = b - alpha * db
    w2 = w2 - alpha * dw2
    w3 = w3 - alpha * dw3
    return w1, b, w2, w3


def iterate(w1, b, w2, w3, x, y, epochs):
    for i in range(epochs):
        w1, b, w2, w3 = optimize(w1, b, w2, w3, x, y)

    y_hat = model(w1, b, w2, w3, x)
    print(w1, b, w2, w3)
    plt.scatter(x, y)
    plt.plot(x, y_hat)
    return w1, b, w2, w3


# MAE metric
def evaluate(w1, b, w2, w3, x, y):
    sum = 0
    y_hat = model(w1, b, w2, w3, x)
    for i in range(len(y)):
        sum += math.fabs(y[i] - y_hat[i])
    return sum / len(y)

w1, w2, w3 = 0.01, 0.01, 0.01
b = 0
epochs = 1000000
w1, b, w2, w3 = iterate(w1, b, w2, w3, train_x, train_y, epochs)
error = evaluate(w1, b, w2, w3, test_x, test_y)
print(error)
plt.title('{} iteration / mean absolute error: {:.2f}'.format(epochs, error))
plt.savefig('graphs/{}epochs-mae{:.2f}.jpg'.format(epochs, error))
