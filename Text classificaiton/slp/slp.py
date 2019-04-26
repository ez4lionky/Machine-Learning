import sys
sys.path.append("..")


import argparse
import time
from load_data import *
from util import *

parser = argparse.ArgumentParser(description='Perceptron for text classification')
parser.add_argument('--train', help='Training model', type=int, default=0)
parser.add_argument('--per_class_max_docs', help='Per-class select corresponding number of doc', type=int,  default=100)
parser.add_argument('--predict', help='The file need to predict', type=str, default='')
parser.add_argument('--lr', help='Learning rate', type=float, default=0.001)
parser.add_argument('--epoch', help='Training iterations', type=int, default=1000)
args = parser.parse_args()

def trainByGD(input, label, input_num, epoch=args.epoch, lr=args.lr):
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
    accs = []
    for rounds in range(epoch):
        logits = mat_sub_add(mat_mul(x, Weight), Bias, operation='add')
        prob = softmax(logits)

        for i in range(input_num):
            prob[i][int(y[i])] -= 1

        w_grad = mat_mul(T(prob), x)
        b_grad = mat_mul( 1 / input_dim, mat_row_sum(T(prob)))
        Weight = mat_sub_add(Weight, mat_mul(lr, T(w_grad)), operation='sub')
        Bias = mat_sub_add(Bias, mat_mul(lr, b_grad), operation='sub')

        logits = mat_sub_add(mat_mul(x_test, Weight), Bias, operation='add')
        prob = softmax(logits)
        prediction = mat_argmax(prob)
        count = 0
        for i in range(len(prediction)):
            if prediction[i] == y_test[i]:
                count = count + 1
        acc = (count + 0.0)/test_size
        accs.append(acc)
    # plt.figure()
    # ax = plt.gca()
    # plt.title('Accuracy - epochs')
    # ax.set_xlabel('Epochs')
    # ax.set_ylabel('Accuracy')
    # plt.plot(range(epoch), accs)
    # plt.savefig('figure1')
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

if args.train!=0:
    path = '../data'
    print('Loading data...')
    # Per_class_max_docs: short text extracted for each file
    corpus, _, texts = load_data_to_mini(path, per_class_max_docs=args.per_class_max_docs, words_num=250)
    x_train, y_train = split_data_with_label(corpus)
    with open('words.txt', 'r') as f:
        words = f.read()
    x_train = data_processing(x_train, words, texts)

    input_dim = len(x_train[0])
    classes = len(set(y_train))
    train_size = len(x_train)

    path = '../test-data'
    corpus, _, texts = load_data_to_mini(path, per_class_max_docs=args.per_class_max_docs, words_num=250)
    x_test, y_test = split_data_with_label(corpus)
    x_test = data_processing(x_test, words, texts)
    test_size = len(x_test)
    print('Data loaded')

    print('training...')
    s = time.time()
    trainByGD(x_train, y_train, train_size)
    e = time.time()
    print('time: %.3fs' % (e - s))
    print("accuracy:", compute_accuracy(x_test, y_test, test_size))

if args.predict!='':
    path = args.predict
    corpus, _, texts = load_data_to_mini(path, per_class_max_docs=args.per_class_max_docs, words_num=250)
    x_test, y_test = split_data_with_label(corpus)
    with open('words.txt', 'r') as f:
        words = f.read()
    x_test = data_processing(x_test, words, texts)

    test_size = len(x_test)
    print("accuracy:", compute_accuracy(x_test, y_test, test_size))