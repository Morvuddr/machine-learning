import math
from labs.nonparametricregression.enums import DistanceFuncType, KernelFuncType, WindowType, ReductionType


class Distance:
    def __init__(self, value, index):
        self.value = value
        self.index = index


def cal_average(entities, reduction_type):
    if reduction_type == ReductionType.naive:
        sum_num = 0
        for entity in entities:
            sum_num += entity.class_number
        avg = sum_num / len(entities)
        return avg
    elif reduction_type == ReductionType.one_hot:
        avg_list = []
        for i in range(len(entities[0].class_list)):
            avg = 0
            for j in range(len(entities)):
                avg += entities[j].class_list[i]
            avg /= len(entities)
            avg_list.append(avg)
        return avg_list.index(max(avg_list))
    return 0


def cal_similar_entities(query_entity, entities):
    similar = []
    for entity in entities:
        if entity.normalized_attributes == query_entity.normalized_attributes:
            similar.append(entity)
    return similar


def calculate_distance(d_type, m, entity1, entity2):
    d = 0
    if d_type == DistanceFuncType.MANHATTAN:
        for i in range(m):
            d += abs(entity1.normalized_attributes[i] - entity2.normalized_attributes[i])
    if d_type == DistanceFuncType.EUCLIDEAN:
        for i in range(m):
            d += pow(entity1.normalized_attributes[i] - entity2.normalized_attributes[i], 2)
        d = math.sqrt(d)
    if d_type == DistanceFuncType.CHEBYSHEV:
        for i in range(m):
            i_distance = abs(entity1.normalized_attributes[i] - entity2.normalized_attributes[i])
            if i_distance > d:
                d = i_distance
    return d


def calculate_kernel(k_type, distance):
    kernel = 0
    if k_type == KernelFuncType.UNIFORM:
        if abs(distance) < 1:
            kernel = 0.5
    if k_type == KernelFuncType.TRIANGULAR:
        if abs(distance) < 1:
            kernel = 1 - abs(distance)
    if k_type == KernelFuncType.EPANECHNIKOV:
        if abs(distance) < 1:
            kernel = 0.75 * (1 - pow(distance, 2))
    if k_type == KernelFuncType.QUARTIC:
        if abs(distance) < 1:
            kernel = 15 / 16 * pow(1 - pow(distance, 2), 2)
    if k_type == KernelFuncType.TRIWEIGHT:
        if abs(distance) < 1:
            kernel = 35 / 32 * pow(1 - pow(distance, 2), 3)
    if k_type == KernelFuncType.TRICUBE:
        if abs(distance) < 1:
            kernel = 70 / 81 * pow(1 - pow(abs(distance), 3), 3)
    if k_type == KernelFuncType.GAUSSIAN:
        kernel = pow(math.e, -pow(distance, 2) / 2) / math.sqrt(2 * math.pi)
    if k_type == KernelFuncType.COSINE:
        if abs(distance) < 1:
            kernel = math.pi * math.cos(math.pi * distance / 2) / 4
    if k_type == KernelFuncType.LOGISTIC:
        kernel = 1 / (pow(math.e, distance) + 2 + pow(math.e, -distance))
    if k_type == KernelFuncType.SIGMOID:
        kernel = 2 / (math.pi * (pow(math.e, distance) + pow(math.e, -distance)))
    return kernel


def nonparametric_regression(query_entity, entities, reduction_type, kernel_type, window_type, window_size=0,
                             neighbours_count=0, distances=None, distance_type=DistanceFuncType.MANHATTAN):
    n = len(entities) - 1
    m = len(entities[0].normalized_attributes)

    if distances is None:
        distances = []
        for i in range(n):
            value = calculate_distance(distance_type, m, query_entity, entities[i])
            distances.append(Distance(value, i))
        distances = sorted(distances, key=lambda d: d.value)

    if window_type == WindowType.FIXED:
        for i in range(n):
            if distances[i].value >= window_size:
                neighbours_count = i
                break
            if i == n - 1:
                neighbours_count = n
                break
    elif window_type == WindowType.VARIABLE:
        window_size = distances[neighbours_count].value
        for i in range(n):
            if distances[i].value > window_size:
                neighbours_count = i
                break
            if i == n - 1:
                neighbours_count = n
                break

    if window_size == 0:
        similar_entities = cal_similar_entities(query_entity, entities)
        if len(similar_entities) != 0:
            return cal_average(similar_entities, reduction_type)
        else:
            return cal_average(entities, reduction_type)

    if neighbours_count == 0:
        return cal_average(entities, reduction_type)

    numerator = 0
    denominator = 0
    for i in range(n):

        kernel = calculate_kernel(kernel_type, (distances[i].value / window_size))
        target_value = entities[distances[i].index].class_number
        numerator += target_value * kernel
        denominator += kernel

    if denominator == 0:
        return cal_average(entities, reduction_type)
    else:
        return numerator / denominator
