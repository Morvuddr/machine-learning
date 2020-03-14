import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import SGDRegressor, Ridge, LinearRegression, RANSACRegressor
from sklearn import preprocessing
import matplotlib.pyplot as plt


def nrmse(actual: np.ndarray, predicted: np.ndarray):
    return rmse(actual, predicted) / (actual.max() - actual.min())


def rmse(actual: np.ndarray, predicted: np.ndarray):
    return np.sqrt(mse(actual, predicted))


def mse(actual: np.ndarray, predicted: np.ndarray):
    return np.mean(np.square(_error(actual, predicted)))


def _error(actual: np.ndarray, predicted: np.ndarray):
    return actual - predicted

def main():

    int_data = []
    with open("datasets/1.txt", 'r') as data:
        for row in data:
            content_row = row.rstrip('\n')
            int_data.append([int(i) for i in content_row.split(' ')])

    # Считывание тренировочных данных
    attributes_count = int_data.pop(0)[0]
    training_objects_count = int_data.pop(0)[0]
    x_train = []
    y_train = []
    for i in range(training_objects_count):
        line = int_data.pop(0)
        y_train.append(line.pop())
        x_train.append(line)

    # Считывание тестовых данных
    test_objects_count = int_data.pop(0)[0]
    test_objects = []
    x_test = []
    y_test = []
    for i in range(test_objects_count):
        line = int_data.pop(0)
        y_test.append(line.pop())
        x_test.append(line)

    # Нормализация
    x_train = preprocessing.normalize(x_train)
    x_test = preprocessing.normalize(x_test)

    # Преобразование массивов с классами в np.array
    y_train = np.asarray(y_train, dtype=np.float)
    y_test = np.array(y_test, dtype=np.float)

    # добавляем к атрибутам единицу
    x_train = np.hstack((x_train, np.ones((x_train.shape[0], 1))))
    x_test = np.hstack((x_test, np.ones((x_test.shape[0], 1))))

    # метод наименьших квадратов через svd
    modelLS = Ridge(alpha=0.5, solver='svd')
    modelLS.fit(x_train, y_train)
    y = modelLS.predict(x_train)
    error1 = nrmse(y, y_train)
    print(error1)
    y_pred = modelLS.predict(x_test)
    error2 = nrmse(y_pred, y_test)
    print(error2)

    # стохастический градиентный спуск
    graph_x = []
    graph_y = []
    for i in range(50, 1001, 50):
        modelGD = SGDRegressor(shuffle=True,
                               max_iter=i,
                               penalty="elasticnet",
                               alpha=0.01,
                               learning_rate="invscaling",
                               eta0=0.001,
                               l1_ratio=0.6,
                               power_t=0.3)
        modelGD.fit(x_train, y_train)
        y_pred = modelGD.predict(x_test)
        graph_x.append(i)
        graph_y.append(nrmse(y_pred, y_test))
        print(nrmse(y_pred, y_test))

    plt.plot(graph_x, graph_y, label="nrmse - iter number")
    plt.xlabel("iterations")
    plt.ylabel("NRMSE")
    plt.legend()
    plt.show()

    # оптимизация черного ящика
    graphXBB = []
    graphYBB = []
    for i in range(100, 5001, 100):
        modelBB = RANSACRegressor(max_trials=i, max_skips=100, stop_score=0.95)
        modelBB.fit(x_train, y_train)
        y_pred = modelBB.predict(x_test)
        graphXBB.append(i)
        graphYBB.append(nrmse(y_pred, y_test))
        print(nrmse(y_pred, y_test))

    plt.plot(graphXBB, graphYBB, label="nrmse - sample number")
    plt.xlabel("samples")
    plt.ylabel("NRMSE")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
