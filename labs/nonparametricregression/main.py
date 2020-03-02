import csv
from labs.nonparametricregression.entity import Entity
from labs.nonparametricregression.normalize import normalize_attributes
from labs.nonparametricregression.leave_one_out import leave_one_out_naive, leave_one_out_one_hot
from labs.nonparametricregression.enums import DistanceFuncType, KernelFuncType, WindowType, ReductionType
from labs.nonparametricregression.nonparametric_regression import calculate_distance, Distance


class Result:
    def __init__(self, reduction_type, distance_type, kernel_type, window_type, fmeasure, window_size=None,
                 neighbours_count=0):
        if window_size is None:
            window_size = [0.0]
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
    window_sizes = [5.0, 2.0, 0.5]
    neighbours_count = 10
    manhattan_distances = []
    euclidean_distances = []
    chebyshev_distances = []

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
            for j in range(len(entities)):
                if i != j:
                    distance = calculate_distance(distance_type, len(entities[0].attributes), entities[i], entities[j])
                    distances_for_entity.append(Distance(distance, j))
            if distance_type == DistanceFuncType.MANHATTAN:
                manhattan_distances.append(distances_for_entity)
            elif distance_type == DistanceFuncType.EUCLIDEAN:
                euclidean_distances.append(distances_for_entity)
            elif distance_type == DistanceFuncType.CHEBYSHEV:
                chebyshev_distances.append(distances_for_entity)

    for i in range(len(manhattan_distances)):
        manhattan_distances[i] = sorted(manhattan_distances[i], key=lambda d: d.value)
    for i in range(len(euclidean_distances)):
        euclidean_distances[i] = sorted(euclidean_distances[i], key=lambda d: d.value)
    for i in range(len(manhattan_distances)):
        chebyshev_distances[i] = sorted(chebyshev_distances[i], key=lambda d: d.value)

    for window_type in WindowType:
        for distance_type in DistanceFuncType:
            for kernel_type in KernelFuncType:
                distances = []
                if distance_type == DistanceFuncType.MANHATTAN:
                    distances = manhattan_distances
                elif distance_type == DistanceFuncType.EUCLIDEAN:
                    distances = euclidean_distances
                elif distance_type == DistanceFuncType.CHEBYSHEV:
                    distances = chebyshev_distances

                # calculate all fmeasures for naive reduction
                naive_fmeasure = leave_one_out_naive(entities, distance_type, distances, kernel_type, window_type,
                                                     window_sizes,
                                                     neighbours_count)
                naive_result = Result(ReductionType.naive, distance_type, kernel_type, window_type, naive_fmeasure,
                                      window_sizes, neighbours_count)

                naive_results.append(naive_result)

                # calculate all fmeasures for OneHot reduction
                one_hot_fmeasure = leave_one_out_one_hot(entities,
                                                         distance_type,
                                                         distances,
                                                         kernel_type,
                                                         window_type,
                                                         window_sizes,
                                                         neighbours_count)
                one_hot_results.append(
                    Result(ReductionType.one_hot,
                           distance_type,
                           kernel_type,
                           window_type,
                           one_hot_fmeasure,
                           window_sizes,
                           neighbours_count))

    naive_max = max(naive_results, key=lambda x: x.fmeasure)
    one_hot_max = max(one_hot_results, key=lambda x: x.fmeasure)








    # print all results
    for result in naive_results:
        print(result.window_type.value, result.distance_type.value, result.kernel_type.value,
              " = ",
              result.fmeasure)

    print("---- Best NAIVE result: ",
          naive_max.window_type.value, naive_max.distance_type.value, naive_max.kernel_type.value,
          " = ",
          naive_max.fmeasure,
          " ----")

    for result in one_hot_results:
        print(result.window_type.value, result.distance_type.value, result.kernel_type.value,
              " = ",
              result.fmeasure)

    print("---- Best OneHot result: ",
          one_hot_max.window_type.value, one_hot_max.distance_type.value, one_hot_max.kernel_type.value,
          " = ",
          one_hot_max.fmeasure,
          " ----")


if __name__ == "__main__":
    main()
