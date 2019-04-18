import csv
import math
# import matplotlib.pyplot as plt


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


train_file = 'mpg-2018.csv'
test_file = 'mpg-2017.csv'
train_x, train_y = load_data('data/' + train_file)
test_x, test_y = load_data('data/' + test_file)

def model(w1, b, w2, w3, x):
    term1 = []
    for x_i in x:
        term1.append(w1 * x_i + b)

    term2 = []
    for x_i in x:
        term2.append(w2 * math.log(x_i + w3))

    result = []
    for i in range(len(x)):
        result.append(term1[i] + term2[i])
    return result


def optimize(w1, b, w2, w3, x, y, lr):
    n = len(x)
    y_hat = model(w1, b, w2, w3, x)
    dw1, db, dw2, dw3 = 0, 0, 0, 0
    for i in range(n):
        dw1 += 2 * (y_hat[i] - y[i]) * x[i]
        db += 2 * (y_hat[i] - y[i])
        dw2 += 2 * (y_hat[i] - y[i]) * math.log(x[i] + w3)
        dw3 += 2 * (y_hat[i] - y[i]) * w2 / (x[i] + w3)
    w1 -= lr * dw1 / n
    b -= lr * db / n
    w2 -= lr * dw2 / n
    w3 -= lr * dw3 / n
    return w1, b, w2, w3


def iterate(w1, b, w2, w3, x, y, epochs, lr=1e-4):
    for i in range(epochs):
        w1, b, w2, w3 = optimize(w1, b, w2, w3, x, y, lr)

    y_hat = model(w1, b, w2, w3, x)
    # plt.scatter(x, y)
    # plt.scatter(x, y_hat, c='r')
    return w1, b, w2, w3


def cost_function(w1, b, w2, w3, x, y):
    n = len(x)
    y_hat = model(w1, b, w2, w3, x)
    sum = 0
    for i in range(n):
        sum += (y_hat[i] - y[i]) ** 2
    return sum

w1, b, w2, w3 = 0.01, 0, 0.01, 0.01
lr = 1e-4
epochs = 500000
w1, b, w2, w3 = iterate(w1, b, w2, w3, train_x, train_y, epochs, lr)
error = cost_function(w1, b, w2, w3, test_x, test_y)
print('learning rate:', lr)
print('epochs:', epochs)
print('w1:', w1)
print('b:', b)
print('w2:', w2)
print('w3:', w3)
print('Test data', test_file, 'train with', train_file, 'error:', error)
# plt.title('{} iteration / error: {:.2f}'.format(epochs, error))
# plt.savefig('graphs/{}epochs-error{:.2f}.jpg'.format(epochs, error))
