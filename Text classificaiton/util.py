import math


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
        if isinstance(M1[0], (tuple, list)) and isinstance(M2[0], (tuple, list)):
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
        elif isinstance(M1[0], (tuple, list)):
            assert len(M1[0]) == len(M2)
            result = []
            for i in range(len(M1)):
                row = []
                for j in range(len(M1[0])):
                    if operation == 'sub':
                        row.append(M1[i][j] - M2[j])
                    if operation == 'add':
                        row.append(M1[i][j] + M2[j])
                result.append(row)
            return result
        else:
            assert len(M1) == len(M2)
            result = []
            for i in range(len(M1)):
                if operation == 'sub':
                    result.append(M1[i] - M2[i])
                if operation == 'add':
                    result.append(M1[i] + M2[i])
            return result

    if isinstance(M1, (tuple, list)) and isinstance(M2, (float, int)):
        if isinstance(M1[0], (tuple, list)):
            if operation=='sub':
                return [[i - M2 for i in j] for j in M1]
            if operation=='add':
                return [[i + M2 for i in j] for j in M1]
        else:
            if operation=='sub':
                return [i - M2 for i in M1]
            if operation=='add':
                return [i + M2 for i in M1]


def mat_mul(M1, M2):
    if isinstance(M1, (float, int)) and isinstance(M2, (tuple, list)):
        if isinstance(M2[0], (tuple, list)):
            return [[M1 * i for i in j] for j in M2]
        else:
            return [M1 * i for i in M2]
    if isinstance(M1, (tuple, list)) and isinstance(M2, (tuple, list)):
        if isinstance(M1[0], (tuple, list)) and isinstance(M2[0], (tuple, list)):
            return [[sum(map(lambda x: x[0] * x[1], zip(i, j))) for j in zip(*M2)] for i in M1]
        else:
            return [(i * j) for (i, j) in zip(M1, M2)]


def T(x):
    if isinstance(x, (tuple, list)):
        return list(zip(*x))


def softmax(x):
    for i in range(len(x)):
        max_x = max(x[i])
        for j in range(len(x[i])):
            x[i][j] = x[i][j] - max_x
    prob = mat_exp(x)
    row_sum = mat_row_sum(prob)
    for i in range(len(prob)):
        for j in range(len(prob[i])):
            prob[i][j] /= row_sum[i]
    return prob

