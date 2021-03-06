import sys
from sklearn.metrics import confusion_matrix
import keras
from keras.datasets import fashion_mnist, mnist
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.utils import to_categorical
import numpy
import matplotlib.pyplot as pyplot
import pandas
import tensorflow
import seaborn


def main():
    # загружаем данные
    (trainX, trainY), (testX, testY) = mnist.load_data()
    testXOriginal = testX
    trainX = trainX.reshape(-1, 28, 28, 1)
    testX = testX.reshape(-1, 28, 28, 1)
    trainX = trainX.astype('float32')
    testX = testX.astype('float32')
    trainX = trainX / 255
    testX = testX / 255
    trainYOneHot = to_categorical(trainY)
    testYOneHot = to_categorical(testY)


    # методы адаптивного градиентного спуска
    optimizers = ['adam', 'adagrad', 'adadelta', 'adamax', 'nadam']
    activations = ['relu', 'tanh', 'elu', 'selu', 'sigmoid', 'exponential', 'linear']
    bestLoss = sys.float_info.max
    bestOptimizer = 'sgd'
    bestActivation = ''
    optimizer_results = []

    for optimizer in optimizers:
        for activation in activations:
                model = Sequential()

                model.add(Conv2D(64, (3, 3), input_shape=(28, 28, 1)))
                model.add(Activation(activation))
                model.add(MaxPooling2D(pool_size=(2, 2)))

                model.add(Conv2D(64, (3, 3)))
                model.add(Activation(activation))
                model.add(MaxPooling2D(pool_size=(2, 2)))

                model.add(Flatten())
                model.add(Dense(64))
                model.add(Dense(10))

                model.add(Activation('softmax'))
                model.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])
                history = model.fit(trainX, trainYOneHot, batch_size=64, epochs=2)
                testLoss, testAccuracy = model.evaluate(testX, testYOneHot)
                optimizer_results.append((optimizer, testLoss, testAccuracy, activation))
                if testLoss < bestLoss:
                    bestLoss = testLoss
                    bestOptimizer = optimizer
                    bestActivation = activation

    for result in optimizer_results:
        print('Optimizer ', result[0])
        print('Test loss', result[1])
        print('Test accuracy', result[2])
        print('Activation', result[3])
    print('Best optimizer is {}, best activation is {}, with loss {}'.format(bestOptimizer, bestActivation, bestLoss))

    model = Sequential()

    model.add(Conv2D(64, (3, 3), input_shape=(28, 28, 1)))
    model.add(Activation(bestActivation))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation(bestActivation))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Dense(10))
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=bestOptimizer, metrics=['accuracy'])
    model.fit(trainX, trainYOneHot, batch_size=64, epochs=10)
    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    predictions = model.predict(testX)
    testYPredicted = []
    for i in range(len(testY)):
        testYPredicted.append(numpy.argmax(predictions[i]))
    print(confusion_matrix(testY, testYPredicted))

    # теперь обучаем и тестируем лучшие параметры на fashion_mnist

    (trainX, trainY), (testX, testY) = fashion_mnist.load_data()
    testXOriginal = testX
    trainX = trainX.reshape(-1, 28, 28, 1)
    testX = testX.reshape(-1, 28, 28, 1)
    trainX = trainX.astype('float32')
    testX = testX.astype('float32')
    trainX = trainX / 255
    testX = testX / 255
    trainYOneHot = to_categorical(trainY)
    testYOneHot = to_categorical(testY)

    model = Sequential()

    model.add(Conv2D(64, (3, 3), input_shape=(28, 28, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))

    model.add(Dense(10))

    model.add(Activation('softmax'))

    print('Best optimizer is {} with loss {}'.format(bestOptimizer, bestLoss))
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=bestOptimizer, metrics=['accuracy'])
    model.fit(trainX, trainYOneHot, batch_size=64, epochs=10)
    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    predictions = model.predict(testX)
    testYPredicted = []
    for i in range(len(testY)):
        testYPredicted.append(numpy.argmax(predictions[i]))
    print(confusion_matrix(testY, testYPredicted))

    for i in range(len(testY)):
        if testYPredicted[i] != testY[i]:
            pyplot.imshow(testXOriginal[i])
            pyplot.text(1, 1, s=str(testY[i]) + " | " + str(testYPredicted[i]))
            pyplot.show()
            pyplot.clf()


if __name__ == "__main__":
    main()