import numpy as np
import pandas as pd
import random
import sklearn
from scipy.spatial.distance import euclidean
from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt


def fmeasure(contingency_matrix, count):
    objects_count = count
    horizontal_p = []
    verticals_p = []
    localmax = 0.0
    for i in range(len(contingency_matrix)):
        localsum = 0
        for j in range(len(contingency_matrix[0])):
            localsum += contingency_matrix[i][j]
        horizontal_p.append(localsum/objects_count)
    for i in range(len(contingency_matrix[0])):
        localsum = 0
        for j in range(len(contingency_matrix)):
            localsum += contingency_matrix[j][i]
        verticals_p.append(localsum / objects_count)

    for i in range(len(contingency_matrix)):
        for j in range(len(contingency_matrix[0])):
            a = (contingency_matrix[i][j]/objects_count/horizontal_p[i])
            b = (contingency_matrix[i][j]/objects_count/verticals_p[j])
            if a + b == 0:
                number = 0
            else:
                number = (2 * a * b) / (a + b)
            if number > localmax:
                localmax = number

    fmeasurenumber = 0
    for i in range(len(contingency_matrix[0])):
        fmeasurenumber += verticals_p[i] * localmax

    return fmeasurenumber


class kMeans:
    def __init__(self, n_clusters=5, distanceFunction="euclidean"):
        if distanceFunction == "euclidean":
            self.distance = euclidean
            self.n_clusters = n_clusters
            self.clusterCenters = []

    def fit(self, X):
        clusterCentersIndices = random.sample([i for i in range(X.shape[0])], k=self.n_clusters)
        self.clusterCenters = [X[i] for i in clusterCentersIndices]
        self.dataClusters = [-1 for i in range(X.shape[0])]
        self.X = X
        while True:
            prevClusterCenters = self.clusterCenters
            #assign cluster for points
            for i in range(X.shape[0]):
                point = X[i]
                distances = []
                for j in self.clusterCenters:
                    distances.append(self.distance(point, j))
                self.dataClusters[i] = np.argmin(distances)
            #find new centers
            accumulative = [np.zeros_like(self.clusterCenters[0]) for i in range(self.n_clusters)]
            countOfPointsInCluster = [0 for i in range(self.n_clusters)]
            for i in range(X.shape[0]):
                accumulative[self.dataClusters[i]] += X[i]
                countOfPointsInCluster[self.dataClusters[i]] +=1
            for i in range(self.n_clusters):
                accumulative[i] /= countOfPointsInCluster[i]
            self.clusterCenters = accumulative
            if np.array_equal(prevClusterCenters, self.clusterCenters):
                break
        self.labels_ = self.dataClusters


    def getClusterCohesion(self):
        result = 0.0
        for i in range(self.X.shape[0]):
            result += self.distance(self.clusterCenters[self.dataClusters[i]], self.X[i])
        return result


def main():
    data = pd.read_csv('../nonparametricregression/solar-flare.csv')
    #print(data)
    data = pd.get_dummies(data=data, columns=['largest_spot_size',
                                              'spot_distribution',
                                              'Activity',
                                              'Evolution',
                                              'Previous_24_hour_flare_activity_code',
                                              'Historically-complex',
                                              'Did_region_become_historically_complex',
                                              'Area',
                                              'Area_of_the_largest_spot',
                                              'C-class_flares_production_by_this_region',
                                              'M-class_flares_production_by_this_region',
                                              'X-class_flares_production_by_this_region'])
    target = data['class']
    data = data.drop('class', 1)

    target = target.array

    data = minmax_scale(data)

    clusterizator = kMeans(n_clusters=5)
    clusterizator.fit(data)

    pca = PCA(n_components=2)
    dataTwoDimensional = pca.fit_transform(data)
    x = []
    y = []
    for i in dataTwoDimensional:
        x.append(i[0])
        y.append(i[1])

    df1 = pd.DataFrame()
    ax = sns.scatterplot(x=x, y=y, hue=target, palette='bright')
    plt.show()
    ax = sns.scatterplot(x=x, y=y, hue=clusterizator.labels_, palette='dark')
    plt.show()

    cohesionList = []
    fmeasureList = []
    for i in range(1, 6):
        clusterizator = kMeans(n_clusters=i)
        clusterizator.fit(data)
        cohesionList.append(clusterizator.getClusterCohesion())
        # Create a DataFrame with labels and varieties as columns: df
        df = pd.DataFrame({'Labels': target, 'Clusters': clusterizator.labels_})
        # Create crosstab: ct
        ct = pd.crosstab(df['Labels'], df['Clusters'])
        contingency_matrix = sklearn.metrics.cluster.contingency_matrix(target, clusterizator.labels_)
        fmeasureList.append(fmeasure(contingency_matrix, len(clusterizator.labels_)))

    sns.lineplot(x=[i for i in range(1, 6)], y=cohesionList)
    plt.show()
    sns.lineplot(x=[i for i in range(1, 6)], y=fmeasureList)
    plt.show()


if __name__ == "__main__":
    main()