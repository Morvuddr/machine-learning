import math
from enum import Enum


class DistanceFuncType(Enum):
    MANHATTAN = "manhattan"
    EUCLIDEAN = "euclidean"
    CHEBYSHEV = "chebyshev"


class KernelFuncType(Enum):
    UNIFORM = "uniform"
    TRIANGULAR = "triangular"
    EPANECHNIKOV = "epanechnikov"
    QUARTIC = "quartic"
    TRIWEIGHT = "triweight"
    TRICUBE = "tricube"
    GAUSSIAN = "gaussian"
    COSINE = "cosine"
    LOGISTIC = "logistic"
    SIGMOID = "sigmoid"


class WindowType(Enum):
    FIXED = "fixed"
    VARIABLE = "variable"


class Entity:
    def __init__(self, target_value, attributes):
        self.target_value = target_value
        self.attributes = attributes


class Distance:
    def __init__(self, value, index):
        self.value = value
        self.index = index


def calculate_distance(d_type, m, entity1, entity2):
    d = 0
    if d_type == DistanceFuncType.MANHATTAN.value:
        for i in range(m):
            d += abs(entity1.attributes[i] - entity2.attributes[i])
    if d_type == DistanceFuncType.EUCLIDEAN.value:
        for i in range(m):
            d += pow(entity1.attributes[i] - entity2.attributes[i], 2)
        d = math.sqrt(d)
    if d_type == DistanceFuncType.CHEBYSHEV.value:
        for i in range(m):
            i_distance = abs(entity1.attributes[i] - entity2.attributes[i])
            if i_distance > d:
                d = i_distance
    return d


def calculate_kernel(k_type, distance):
    kernel = 0
    if k_type == KernelFuncType.UNIFORM.value:
        if abs(distance) < 1:
            kernel = 0.5
    if k_type == KernelFuncType.TRIANGULAR.value:
        if abs(distance) < 1:
            kernel = 1 - abs(distance)
    if k_type == KernelFuncType.EPANECHNIKOV.value:
        if abs(distance) < 1:
            kernel = 0.75 * (1 - pow(distance, 2))
    if k_type == KernelFuncType.QUARTIC.value:
        if abs(distance) < 1:
            kernel = 15 / 16 * pow(1 - pow(distance, 2), 2)
    if k_type == KernelFuncType.TRIWEIGHT.value:
        if abs(distance) < 1:
            kernel = 35 / 32 * pow(1 - pow(distance, 2), 3)
    if k_type == KernelFuncType.TRICUBE.value:
        if abs(distance) < 1:
            kernel = 70 / 81 * pow(1 - pow(abs(distance), 3), 3)
    if k_type == KernelFuncType.GAUSSIAN.value:
        kernel = pow(math.e, -pow(distance, 2) / 2) / math.sqrt(2 * math.pi)
    if k_type == KernelFuncType.COSINE.value:
        if abs(distance) < 1:
            kernel = math.pi * math.cos(math.pi * distance / 2) / 4
    if k_type == KernelFuncType.LOGISTIC.value:
        kernel = 1 / (pow(math.e, distance) + 2 + pow(math.e, -distance))
    if k_type == KernelFuncType.SIGMOID.value:
        kernel = 2 / (math.pi * (pow(math.e, distance) + pow(math.e, -distance)))
    return kernel


def cal_average(entities):
    sum_num = 0
    for entity in entities:
        sum_num += entity.target_value
    avg = sum_num / len(entities)
    return avg


def cal_similar_entities(query_entity, entities):
    similar = []
    for entity in entities:
        if entity.attributes == query_entity.attributes:
            similar.append(entity)
    return similar


def main():
    entities = []

    s = [int(item) for item in input().split(' ')]
    n = s[0]
    m = s[1]
    for i in range(n):
        attr = [int(item) for item in input().split(' ')]
        target_value = attr.pop()
        entities.append(Entity(target_value, attr))
    attr = [int(item) for item in input().split(' ')]
    query_entity = Entity(None, attr)
    distance_function_type = input()
    kernel_function_type = input()
    window_type = input()
    neighbours_count = 0
    window_size = 0
    if window_type == WindowType.FIXED.value:
        window_size = int(input())
    elif window_type == WindowType.VARIABLE.value:
        neighbours_count = int(input())

    distances = []
    for i in range(n):
        value = calculate_distance(distance_function_type, m, query_entity, entities[i])
        distances.append(Distance(value, i))
    distances = sorted(distances, key=lambda d: d.value)

    if window_type == WindowType.FIXED.value:
        for i in range(n):
            if distances[i].value >= window_size:
                neighbours_count = i
                break
            if i == n - 1:
                neighbours_count = n
                break
    elif window_type == WindowType.VARIABLE.value:
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
            query_entity.target_value = cal_average(similar_entities)
            print(query_entity.target_value)
            return
        else:
            query_entity.target_value = cal_average(entities)
            print(query_entity.target_value)
            return

    if neighbours_count == 0:
        query_entity.target_value = cal_average(entities)
        print(query_entity.target_value)
        return

    numerator = 0
    denominator = 0
    for i in range(len(entities)):
        kernel = calculate_kernel(kernel_function_type, (distances[i].value / window_size))
        target_value = entities[distances[i].index].target_value
        numerator += target_value * kernel
        denominator += kernel

    if denominator == 0:
        query_entity.target_value = cal_average(entities)
        print(query_entity.target_value)
        return
    else:
        query_entity.target_value = numerator / denominator
        print(query_entity.target_value)
        return


if __name__ == "__main__":
    main()
