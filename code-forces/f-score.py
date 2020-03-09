def recall(t, p):
    return float(t) / float(p)


def precision(t, c):
    return float(t) / float(c)


def f_measure(t, c, p):
    if t == 0:
        return 0
    prec = precision(t, p)
    rec = recall(t, c)
    return 2 * prec * rec / (prec + rec)


def cal_precision_w(t, c, p):
    return float(t) * float(c) / float(p)


def calculate_f_measure(confusion_matrix):
    classes_count = len(confusion_matrix)
    ts = []
    cs = []
    ps = []

    all = 0
    for i in range(classes_count):
        ci = 0
        pi = 0
        for j in range(classes_count):
            all += confusion_matrix[i][j]
            ci += confusion_matrix[i][j]
            pi += confusion_matrix[j][i]
        ts.append(confusion_matrix[i][i])
        cs.append(ci)
        ps.append(pi)

    precision_w = 0
    recall_w = 0
    micro_f = 0
    for i in range(classes_count):
        if cs[i] != 0 and ps[i] != 0:
            precision_w += cal_precision_w(ts[i], cs[i], ps[i])
            recall_w += ts[i]
            micro_f += cs[i] * f_measure(ts[i], cs[i], ps[i])

    precision_w /= all
    recall_w /= all
    micro_f /= all

    macro_f = 2 * precision_w * recall_w / (precision_w + recall_w)

    return macro_f, micro_f


def main():
    k = int(input())
    confusion_matrix = []
    for i in range(k):
        c = [int(item) for item in input().split(' ')]
        confusion_matrix.append(c)

    macro_f, micro_f = calculate_f_measure(confusion_matrix)

    print(macro_f)
    print(micro_f)


if __name__ == "__main__":
    main()
