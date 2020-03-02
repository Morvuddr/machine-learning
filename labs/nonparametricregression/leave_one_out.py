from labs.nonparametricregression.enums import *
from labs.nonparametricregression.calculate_fmeasure import *
from labs.nonparametricregression.nonparametric_regression import nonparametric_regression


def leave_one_out_naive(entities, distance_type, distances, kernel_type, window_type, window_sizes=None,
                        neighbours_count=0):
    n = len(entities[0].class_list)
    m = len(entities[0].class_list)
    confusion_matrix = [[0] * m for i in range(n)]

    window_size = 0.0
    if distance_type == DistanceFuncType.MANHATTAN:
        window_size = window_sizes[0]
    if distance_type == DistanceFuncType.EUCLIDEAN:
        window_size = window_sizes[1]
    if distance_type == DistanceFuncType.CHEBYSHEV:
        window_size = window_sizes[2]

    for i in range(len(entities)):
        f_result = nonparametric_regression(ReductionType.naive,
                                            entities[i],
                                            entities,
                                            kernel_type,
                                            window_type,
                                            distance_type,
                                            window_size,
                                            neighbours_count,
                                            distances[i])
        result = int(round(f_result))
        confusion_matrix[entities[i].class_number][result] += 1
    return calculate_fmeasure(confusion_matrix)


def leave_one_out_one_hot(entities, distance_type, distances, kernel_type, window_type, window_sizes=None,
                          neighbours_count=0):
    n = len(entities[0].class_list)
    m = len(entities[0].class_list)
    confusion_matrix = [[0] * m for i in range(n)]

    window_size = 0.0
    if distance_type == DistanceFuncType.MANHATTAN:
        window_size = window_sizes[0]
    if distance_type == DistanceFuncType.EUCLIDEAN:
        window_size = window_sizes[1]
    if distance_type == DistanceFuncType.CHEBYSHEV:
        window_size = window_sizes[2]

    for i in range(len(entities)):
        query_classes = []
        for class_index in range(len(entities[0].class_list)):
            f_result = nonparametric_regression(ReductionType.one_hot,
                                                entities[i],
                                                entities,
                                                kernel_type,
                                                window_type,
                                                distance_type,
                                                window_size,
                                                neighbours_count,
                                                distances[i],
                                                class_index)
            query_classes.append(f_result)
        result = query_classes.index(max(query_classes))
        confusion_matrix[entities[i].class_number][result] += 1

    return calculate_fmeasure(confusion_matrix)
