import csv
from util import *
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
test_x, test_y = load_data('data/mpg-2017.csv')

def model(w1, b, w2, w3, x):
    term1 = mat_sub_add(mat_mul(w1, x), b, 'add')
    mat_log(mat_sub_add(x, w3, 'add'))
    term2 = mat_mul(w2, mat_log(mat_sub_add(x, w3, 'add')))
    return mat_sub_add(term1, term2, 'add')


def optimize(w1, b, w2, w3, x, y):
    n = len(x)
    alpha = 1e-4
    y_hat = model(w1, b, w2, w3, x)
    delta_y = mat_sub_add(y_hat, y)
    dw1 = sum(mat_mul(1.0 / n, mat_mul(delta_y, x)))
    db = sum(mat_mul(1.0 / n, delta_y))
    dw2 = sum(mat_mul(1.0 / n, mat_mul(delta_y, mat_log(mat_sub_add(x, w3, 'add')))))
    t = mat_div(1.0, mat_sub_add(x, w3, 'add'))
    t = mat_mul(w2, t)
    dw3 = sum(mat_mul(1.0 / n, mat_mul(delta_y, t)))
    w1 = w1 - alpha * dw1
    b = w2 - alpha * db
    w2 = w2 - alpha * dw2
    w3 = w3 - alpha * dw3
    return w1, b, w2, w3


def iterate(w1, b, w2, w3, x, y, epochs):
    for i in range(epochs):
        w1, b, w2, w3 = optimize(w1, b, w2, w3, x, y)

    y_hat = model(w1, b, w2, w3, x)
    plt.scatter(x, y)
    plt.scatter(x, y_hat, c='r')
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
epochs = 1000
w1, b, w2, w3 = iterate(w1, b, w2, w3, train_x, train_y, epochs)
error = evaluate(w1, b, w2, w3, test_x, test_y)
print('MAE error: ', error)
plt.title('{} iteration / mean absolute error: {:.2f}'.format(epochs, error))
plt.savefig('graphs/{}epochs-mae{:.2f}.jpg'.format(epochs, error))
