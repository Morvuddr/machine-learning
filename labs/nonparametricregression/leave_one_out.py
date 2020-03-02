from labs.nonparametricregression.enums import *
from labs.nonparametricregression.calculate_fmeasure import *
from labs.nonparametricregression.nonparametric_regression import nonparametric_regression


def leave_one_out_naive(entities, distances, kernel_type, window_type, window_size=0,
                        neighbours_count=0):
    n = len(entities[0].class_list)
    m = len(entities[0].class_list)
    confusion_matrix = [[0] * m for i in range(n)]

    for i in range(len(entities)):
        fresult = nonparametric_regression(entities[i],
                                                    entities,
                                                    ReductionType.naive,
                                                    kernel_type,
                                                    window_type,
                                                    window_size,
                                                    neighbours_count,
                                                    distances[i])
        result = int(round(fresult))
        confusion_matrix[entities[i].class_number][result] += 1
    return calculate_fmeasure(confusion_matrix)


def leave_one_out_one_hot(entities, distances, kernel_type, window_type, window_size=0,
                          neighbours_count=0):
    confusion_matrix = [[0] * len(entities[0].class_names)] * len(entities[0].class_names)
    return
