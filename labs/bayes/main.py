from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
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
    print(folders)
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

    vectorizer = CountVectorizer()
    counts = vectorizer.fit_transform(x)
    model = MultinomialNB()
    print(cross_val_score(model, counts, y, cv=10))


if __name__ == "__main__":
    main()