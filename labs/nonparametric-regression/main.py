import csv
from entity import Entity
from normalize import normalize_attributes
from leave_one_out import leave_one_out_naive, leave_one_out_one_hot
from enums import DistanceFuncType, KernelFuncType, WindowType, ReductionType
from nonparametric_regression import calculate_distance

class Result:
    def __init__(self, reduction_type, distance_type, kernel_type, window_type, fmeasure, window_size=0,
                  neighbours_count=0):
        self.reduction_type = reduction_type
        self.distance_type = distance_type
        self.kernel_type = kernel_type
        self.window_type = window_type
        self.fmeasure = fmeasure
        self.window_size = window_size
        self.neighbours_count = neighbours_count


def main():
    entities = []
    class_names = []
    naive_results = []
    one_hot_results = []
    manhattan_distances = [[]]
    euclidean_distances = [[]]
    chebyshev_distances = [[]]

    with open('solar-flare.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        line_number = 0
        for row in csv_reader:
            if line_number != 0:
                attributes = [int(i) for i in row]
                target_value = attributes.pop()
                entity = Entity(target_value, attributes)

                if target_value not in class_names:
                    class_names.append(target_value)
                entity.class_number = class_names.index(target_value)
                entities.append(entity)
            line_number += 1

        for entity in entities:
            entity.class_list = [0] * len(class_names)
            entity.class_list[entity.class_number] = 1

    normalize_attributes(entities)

    for distance_type in DistanceFuncType:
        for i in range(len(entities)):
            distances_for_entity = []
            for j in range(len(entities) - 1):
                if i != j:
                    distance = calculate_distance(distance_type, len(entities[0].attributes), entities[i], entities[j])
                    distances_for_entity.append(distance)
            if distance_type == DistanceFuncType.MANHATTAN.value:
                manhattan_distances.append(distances_for_entity)
            elif distance_type == DistanceFuncType.EUCLIDEAN.value:
                euclidean_distances.append(distances_for_entity)
            elif distance_type == DistanceFuncType.CHEBYSHEV.value:
                chebyshev_distances.append(distances_for_entity)



    for window_type in WindowType:
        for distance_type in DistanceFuncType:
            for kernel_type in KernelFuncType:
                distances = [[]]
                if distance_type == DistanceFuncType.MANHATTAN.value:
                    distances = manhattan_distances
                elif distance_type == DistanceFuncType.EUCLIDEAN.value:
                    distances = euclidean_distances
                elif distance_type == DistanceFuncType.CHEBYSHEV.value:
                    distances = chebyshev_distances

                # calculate all fmeasures for naive reduction
                naive_fmeasure = leave_one_out_naive(entities, distance_type, kernel_type, window_type)
                naive_results.append(Result(ReductionType.naive, distance_type, kernel_type, window_type, naive_fmeasure))
                # calculate all fmeasures for OneHot reduction
                one_hot_fmeasure = leave_one_out_one_hot(entities, distance_type, kernel_type, window_type)
                one_hot_results.append(Result(ReductionType.one_hot, distance_type, kernel_type, window_type, one_hot_fmeasure))

    print(max(naive_results, key=lambda result: result.fmeasure))
    print(max(one_hot_results, key=lambda result: result.fmeasure))


if __name__ == "__main__":
    main()
