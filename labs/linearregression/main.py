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
    # для тестовых данных
    graph_x = []
    graph_y = []
    # сохраняем все ошибки для градиентного спуска
    all_descent_nrmse = []
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
        error = nrmse(y_pred, y_test)
        graph_y.append(error)
        all_descent_nrmse.append((error, i))

    # вывод минимальной ошибки
    min_nrmse = min(all_descent_nrmse, key= lambda e: e[0])
    print('Ошибка для градиентного спуска и тестовых данных')
    print('минимальная ошибка = ', min_nrmse[0], 'количество итераций =', min_nrmse[1])

    # построение графика
    plt.plot(graph_x, graph_y, label="Test data. NRMSE - iter number")
    plt.xlabel("iterations")
    plt.ylabel("NRMSE")
    plt.legend()
    plt.show()

    # для тренировочных данных
    graph_x = []
    graph_y = []
    # сохраняем все ошибки для градиентного спуска
    all_descent_nrmse = []
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
        y_pred = modelGD.predict(x_train)
        graph_x.append(i)
        error = nrmse(y_pred, y_train)
        graph_y.append(error)
        all_descent_nrmse.append((error, i))

    # вывод минимальной ошибки
    min_nrmse = min(all_descent_nrmse, key=lambda e: e[0])
    print('Ошибка для градиентного спуска и тренировочных данных')
    print('минимальная ошибка = ', min_nrmse[0], 'количество итераций =', min_nrmse[1])

    # построение графика
    plt.plot(graph_x, graph_y, label="Test data. NRMSE - iter number")
    plt.xlabel("iterations")
    plt.ylabel("NRMSE")
    plt.legend()
    plt.show()

    # оптимизация черного ящика
    # для тестовых данных
    graphXBB = []
    graphYBB = []
    # сохраняем все ошибки для черного ящика
    all_black_box_nrmse = []
    for i in range(100, 5001, 100):
        modelBB = RANSACRegressor(max_trials=i, max_skips=100, stop_score=0.95)
        modelBB.fit(x_train, y_train)
        y_pred = modelBB.predict(x_test)
        graphXBB.append(i)
        error = nrmse(y_pred, y_test)
        graphYBB.append(error)
        all_black_box_nrmse.append((error, i))

    # вывод минимальной ошибки
    min_nrmse = min(all_black_box_nrmse, key=lambda e: e[0])
    print('минимальная ошибка = ', min_nrmse[0], 'количество итераций =', min_nrmse[1])

    # построение графика
    plt.plot(graphXBB, graphYBB, label="Test data. NRMSE - sample number")
    plt.xlabel("samples")
    plt.ylabel("NRMSE")
    plt.legend()
    plt.show()

    # для тренировочных данных
    graphXBB = []
    graphYBB = []
    # сохраняем все ошибки для черного ящика
    all_black_box_nrmse = []
    for i in range(100, 5001, 100):
        modelBB = RANSACRegressor(max_trials=i, max_skips=100, stop_score=0.95)
        modelBB.fit(x_train, y_train)
        y_pred = modelBB.predict(x_train)
        graphXBB.append(i)
        error = nrmse(y_pred, y_train)
        graphYBB.append(error)
        all_black_box_nrmse.append((error, i))

    # вывод минимальной ошибки
    min_nrmse = min(all_black_box_nrmse, key=lambda e: e[0])
    print('минимальная ошибка = ', min_nrmse[0], 'количество итераций =', min_nrmse[1])

    # построение графика
    plt.plot(graphXBB, graphYBB, label="Train data. NRMSE - sample number")
    plt.xlabel("samples")
    plt.ylabel("NRMSE")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
