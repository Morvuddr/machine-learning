import math


def normalize_attributes(entities):
    for i in range(len(entities[0].attributes)):
        avg = 0
        rmse = 0

        # calculate average for attribute
        for j in range(len(entities)):
            avg += entities[j].attributes[i]
        avg /= len(entities)

        # calculate root mean square error for attribute
        for j in range(len(entities)):
            rmse += pow(entities[j].attributes[i], 2)
        rmse = math.sqrt(rmse / len(entities))

        # calculate normalized attribute
        for j in range(len(entities)):
            normalized = (entities[j].attributes[i] - avg) / rmse
            entities[j].normalized_attributes.append(normalized)
