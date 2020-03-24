from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def colors(labels):
    return ['green' if label == 'P' else 'purple' for label in labels]


def solution(filename):
    # Считывание данных
    data = pd.read_csv(filename + '.csv')
    y = data['class'].values
    x = data.drop(['class'], axis=1).values

    # Разбиение данных на тестовую и тренировочную выборки
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15)

    # Алгоритм бустинга
    model = AdaBoostClassifier(learning_rate=0.23, n_estimators=150)
    model.fit(x_train, y_train)

    # Построение графика для каждого шага алгоритма
    y_pred = model.staged_predict(x)
    for i, item in enumerate(y_pred, start=1):
        plt.clf()
        plt.xlabel('Step number ' + str(i))
        plt.scatter(x[:, 0], x[:, 1], c=colors(y), cmap=plt.cm.Paired)
        ax = plt.gca()
        ax.scatter(x[:, 0], x[:, 1], facecolors='none', edgecolors=colors(item), s=100, linewidths=1)
        plt.savefig(filename + '/' + str(i) + '.png')

    # Построение графика зависимости качества от номера шага.
    model.learning_rate = 1
    model.fit(x_train, y_train)
    quality_functions = model.staged_score(x_test, y_test)
    graph_x = []
    graph_y = []
    for i, predicted in enumerate(quality_functions, start=0):
        graph_x.append(i)
        graph_y.append(predicted)
    plt.clf()
    plt.plot(graph_x, graph_y, label="accuracy depending on number of steps")
    plt.xlabel("steps")
    plt.ylabel("accuracy")
    plt.legend()
    plt.savefig(filename + '/accuracy_depending_on_number_of_steps')


def main():
    solution("chips")
    solution("geyser")


if __name__ == "__main__":
    main()