import csv
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

train_file = 'mpg-2017.csv'
test_file = 'mpg-2017.csv'
train_x, train_y = load_data('data/' + train_file)
test_x, test_y = load_data('data/' + test_file)

def model(w, b, x):
    result = []
    for x_i in x:
        result.append(w * x_i + b)
    return result


def optimize(w, b, x, y, lr):
    n = len(x)
    y_hat = model(w, b, x)
    dw, db = 0, 0
    for i in range(n):
        dw += 2 * (y_hat[i] - y[i]) * x[i]
        db += 2 * (y_hat[i] - y[i])
    w -= lr * dw / n
    b -= lr * db / n
    return w, b


def iterate(w, b, x, y, epochs, lr=1e-4):
    for i in range(epochs):
        w, b = optimize(w, b, x, y, lr)
    y_hat = model(w, b, x)
    plt.scatter(x, y)
    plt.plot(x, y_hat)
    return w, b


def cost_function(w, b, x, y):
    n = len(x)
    y_hat = model(w, b, x)
    sum = 0
    for i in range(n):
        sum += (y_hat[i] - y[i]) ** 2
    return sum


w, b = 0.01, 0
lr = 1e-4
epochs = 500000
w, b = iterate(w, b, train_x, train_y, epochs)
print('learning rate:', lr)
print('epochs:', epochs)
print('w:', w)
print('b:', b)
error = cost_function(w, b, test_x, test_y)
print('Test data', test_file, 'train with', train_file, 'error:', error)
plt.title('{} iteration / error: {:.2f}'.format(epochs, error))
plt.savefig('graphs/{}epochs-error{:.2f}.jpg'.format(epochs, error))
