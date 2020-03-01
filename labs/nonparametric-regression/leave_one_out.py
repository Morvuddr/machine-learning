from enums import *
from calculate_fmeasure import *
from nonparametric_regression import nonparametric_regression


def leave_one_out_naive(entities, distances, kernel_type, window_type, window_size=0,
                        neighbours_count=0):
    confusion_matrix = [[0] * len(entities[0].class_names)] * len(entities[0].class_names)
    for i in range(len(entities)):
        result = -1
        result = int(round(nonparametric_regression(entities[i],
                                                    entities[:i] + entities[i + 1:],
                                                    ReductionType.naive,
                                                    kernel_type,
                                                    window_type,
                                                    window_size,
                                                    neighbours_count,
                                                    distances[i])))
        if result != -1:
            confusion_matrix[entities[i].class_number][result] += 1
    return calculate_fmeasure(confusion_matrix)


def leave_one_out_one_hot(entities, distances, kernel_type, window_type, window_size=0,
                          neighbours_count=0):
    confusion_matrix = [[0] * len(entities[0].class_names)] * len(entities[0].class_names)
    return
