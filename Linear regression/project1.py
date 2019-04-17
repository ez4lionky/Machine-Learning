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


def model(w, b, x):
    return w * x + b


def cost_function(w, b, x, y):
    n = len(x)
    return 0.5 / n * (np.square(y - w * x - b)).sum()


def optimize(w, b , x, y):
    n = len(x)
    alpha = 1e-4
    y_hat = model(w, b, x)
    dw = (1.0 / n) * ((y_hat - y) * x).sum()
    db = (1.0 / n) * ((y_hat - y).sum())
    w = w - alpha * dw
    b = b - alpha * db
    return w, b


def iterate(w, b, x, y, epochs):
    for i in range(epochs):
        w, b = optimize(w, b, x, y)

    y_hat = model(w, b, x)
    cost = cost_function(w, b, x, y)
    print(w, b, cost)
    plt.scatter(x,y)
    plt.plot(x,y_hat)
    return w, b


# MAE metric
def evaluate(w, b, x, y):
    sum = 0
    y_hat = model(w, b, x)
    for i in range(len(y)):
        sum += math.fabs(y[i] - y_hat[i])
    return sum / len(y)

w = 0.01
b = 0
epochs = 1000000
w, b = iterate(w, b, train_x, train_y, epochs)
error = evaluate(w, b, test_x, test_y)
print(error)
plt.title('{} iteration / mean absolute error: {:.2f}'.format(epochs, error))
plt.savefig('graphs/{}epochs-mae{:.2f}.jpg'.format(epochs, error))