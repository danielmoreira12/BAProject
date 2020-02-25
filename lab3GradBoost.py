from sklearn.naive_bayes import GaussianNB

from functions import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.ensemble import GradientBoostingClassifier

data = pd.read_csv('covertype1.csv')
#----------------------------------------------Grandient Boosting--------------------------------------------
y: np.ndarray = data.pop('Cover_Type').values
X: np.ndarray = data.values
labels = pd.unique(y)

trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)

estimators = [5, 50, 100, 150, 200, 300]
l_rate = [0.01, 0.1]
max_depth = [1, 2, 3, 4, 5]

yvalues = []
"""for l in range(len(l_rate)):
    values = {}
    for d in range(len(max_depth)):
        yvalues = []
        for e in range(len(estimators)):
            gbc = GradientBoostingClassifier(n_estimators=estimators[e], learning_rate=l_rate[l],
                                             max_depth=max_depth[d])
            gbc.fit(trnX, trnY)
            gbc.score(tstX, tstY)
            prdY = gbc.predict(tstX)

            yvalues.append(metrics.accuracy_score(tstY, prdY))
            print('l_rate: ', l, ' max_depth ', d, ' estimators ', e, ' accuracy: ',
                  str(metrics.accuracy_score(tstY, prdY)))

plt.show()
"""
plt.figure()
fig, axs = plt.subplots(1, 2, figsize=(10, 4), squeeze=False)
for k in range(len(l_rate)):
    values = {}
    for d in max_depth:
        yvalues = []
        for n in estimators:
            rf = GradientBoostingClassifier(n_estimators=n, learning_rate= l_rate[k], max_depth=d)
            rf.fit(trnX, trnY)
            prdY = rf.predict(tstX)
            yvalues.append(metrics.accuracy_score(tstY, prdY))
        values[d] = yvalues
    multiple_line_chart(axs[0, k], estimators, values, 'Random Forests with %s features' % k, 'nr estimators',
                             'accuracy', percentage=False)

plt.show()

plt.figure()
fig, axs = plt.subplots(1, 2, figsize=(10, 4), squeeze=False)
for k in range(len(l_rate)):
    values = {}
    for d in max_depth:
        yvalues = []
        for n in estimators:
            rf = GradientBoostingClassifier(n_estimators=n, learning_rate= l_rate[k], max_depth=d)
            rf.fit(trnX, trnY)
            prdY = rf.predict(tstX)
            yvalues.append(metrics.recall_score(tstY, prdY, average='macro'))
            values[d] = yvalues
        multiple_line_chart(axs[0, k], estimators, values, 'Random Forests with %s features' % k, 'nr estimators',
                            'sensibility', percentage=False)
plt.show()
"""clf = GaussianNB()
clf.fit(trnX, trnY)
prdY = clf.predict(tstX)
plt.figure()
fig, axs = plt.subplots(1, 2, figsize=(8, 4), squeeze=False)
plot_confusion_matrix(axs[0, 0], metrics.confusion_matrix(tstY, prdY, labels), labels)
plot_confusion_matrix(axs[0, 1], metrics.confusion_matrix(tstY, prdY, labels), labels, normalize=True)
plt.tight_layout()
plt.show()"""