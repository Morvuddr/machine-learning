from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
import numpy as np


def colors(labels):
    return ['green' if label == 'P' else 'purple' for label in labels]


def solution(filename):
    # Считывание данных
    data = pd.read_csv(filename + '.csv')
    # data['class'] = pd.factorize(data['class'])[0]
    y = data['class'].values
    x = data.drop(['class'], axis=1).values

    # Разбиение данных на тестовую и тренировочную выборки
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15)

    # Алгоритм бустинга
    if filename == 'chips':
        learning_rate = 0.15
    else:
        learning_rate = 1.15
    model = AdaBoostClassifier(learning_rate=learning_rate, n_estimators=150)
    model.fit(x, y)

    # Построение графика для каждого шага алгоритма
    plt.figure()
    plt.scatter(x[:, 0], x[:, 1], c=colors(y), cmap=plt.cm.Paired)
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T

    y_pred = model.staged_predict(x)
    decision_funcs = model.staged_decision_function(xy)

    i = 1
    for _, func in zip(y_pred, decision_funcs):
        plt.clf()
        plt.scatter(x[:, 0], x[:, 1], c=colors(y), cmap=plt.cm.Paired)
        f = func.reshape(XX.shape)
        ax = plt.gca()
        ax.contour(XX, YY, f, colors='k', levels=[-1, 0, 1], alpha=0.5,
                   linestyles=['--', '-', '--'])
        plt.legend([f'Step number {i}'])
        plt.savefig(filename + '/' + str(i) + '.png')

        i += 1

    # Построение графика зависимости качества от номера шага.
    # model.learning_rate = 1
    # model.fit(x_train, y_train)
    # quality_functions = model.staged_score(x_test, y_test)
    # graph_x = []
    # graph_y = []
    # for i, predicted in enumerate(quality_functions, start=0):
    #     graph_x.append(i)
    #     graph_y.append(predicted)
    # plt.clf()
    # plt.plot(graph_x, graph_y, label="accuracy depending on number of steps")
    # plt.xlabel("steps")
    # plt.ylabel("accuracy")
    # plt.legend()
    # plt.savefig(filename + '/accuracy_depending_on_number_of_steps')


def main():
    solution("chips")
    solution("geyser")


if __name__ == "__main__":
    main()