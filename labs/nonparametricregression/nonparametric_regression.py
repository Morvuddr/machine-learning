import math
from labs.nonparametricregression.enums import DistanceFuncType, KernelFuncType, WindowType, ReductionType


class Distance:
    def __init__(self, value, index):
        self.value = value
        self.index = index


def cal_average_naive(query_entity, entities, z):
    sum_num = 0
    for entity in entities:
        sum_num += entity.class_number
    avg = (sum_num - query_entity.class_number) / len(entities)
    return avg


def cal_average_one_hot(query_entity, entities, class_index):
    avg = 0
    for i in range(len(entities)):
        avg += entities[i].class_list[class_index]
    avg = (avg - query_entity.class_list[class_index]) / len(entities)
    return avg


def cal_similar_entities(query_entity, entities):
    similar = []
    for entity in entities:
        if entity.normalized_attributes == query_entity.normalized_attributes and \
                entities.index(query_entity) != entities.index(entity):
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


def nonparametric_regression(reduction_type,
                             query_entity,
                             entities,
                             kernel_type,
                             window_type,
                             distance_type,
                             window_size=0,
                             neighbours_count=0,
                             distances=None,
                             class_index=-1):
    n = len(entities) - 1
    m = len(entities[0].normalized_attributes)

    if reduction_type == ReductionType.naive:
        cal_average = cal_average_naive
    else:
        cal_average = cal_average_one_hot

    if distances is None:
        distances = []
        for i in range(n):
            value = calculate_distance(distance_type, m, query_entity, entities[i])
            distances.append(Distance(value, i))
        distances = sorted(distances, key=lambda d: d.value)

    if window_type == WindowType.VARIABLE:
        window_size = distances[neighbours_count].value

    if window_size == 0:
        similar_entities = cal_similar_entities(query_entity, entities)
        if len(similar_entities) != 0:
            return cal_average(query_entity, similar_entities, class_index)
        else:
            return cal_average(query_entity, entities, class_index)

    numerator = 0
    denominator = 0
    for i in range(n):
        kernel = calculate_kernel(kernel_type, (distances[i].value / window_size))
        if reduction_type == ReductionType.naive:
            target_value = entities[distances[i].index].class_number
        else:
            target_value = entities[distances[i].index].class_list[class_index]
        numerator += target_value * kernel
        denominator += kernel

    if denominator == 0:
        return cal_average(query_entity, entities, class_index)
    else:
        return numerator / denominator
