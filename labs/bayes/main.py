from sklearn import preprocessing
from sklearn.metrics import roc_curve
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import matplotlib.pyplot as plt
import os
import re


def main():
    folders = []
    x = []
    y = []
    text = []
    for i in os.walk('messages'):
        if len(i[2]) != 0:
            folders.append((i[0], i[2]))

    # Считывание сообщений из директорий
    for item in folders:
        for file in item[1]:
            if file.find('spmsg') != -1:
                y.append(1)
            if file.find('legit') != -1:
                y.append(0)
            file_handler = open('{0}/{1}'.format(item[0], file), 'r')
            for line in file_handler:
                text.extend(re.findall(r'\d+', line))
            x.append(' '.join(text))
            text.clear()

    counts = CountVectorizer(ngram_range=(1, 30)).fit_transform(x)
    clf = MultinomialNB(fit_prior=False)
    clf.fit(counts, y)
    false_positive_rate, true_positive_rate, threshold = roc_curve(y, clf.predict_proba(counts)[:, 0], pos_label=0)

    plt.title('ROC')
    plt.plot(false_positive_rate, true_positive_rate)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

    graphX, graphY = [], []
    vectorizer = CountVectorizer(ngram_range=(2, 5))
    counts = vectorizer.fit_transform(x)
    for j in range(1, 100, 10):
        clf = MultinomialNB(class_prior=[j/100, 1-j/100])
        scores = cross_val_score(clf, counts, y, cv=10)
        graphX.append(j/100)
        graphY.append(scores.mean())

    plt.plot(graphX, graphY, label="Accuracy depending on prior probability")
    plt.xlabel("prior probability")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()



if __name__ == "__main__":
    main()