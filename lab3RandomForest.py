from sklearn.naive_bayes import GaussianNB

from functions import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('covertype1.csv')
#----------------------------------------------Random Forest------------------------------------------------
y: np.ndarray = data.pop('Cover_Type').values
X: np.ndarray = data.values
labels = pd.unique(y)

trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)

n_estimators = [5, 10, 25, 50, 75, 100, 150, 200, 250, 300]
max_depths = [5, 10, 25, 50]
max_features = ['sqrt', 'log2']

plt.figure()
fig, axs = plt.subplots(1, 2, figsize=(10, 4), squeeze=False)
for k in range(len(max_features)):
    f = max_features[k]
    values = {}
    for d in max_depths:
        yvalues = []
        for n in n_estimators:
            rf = RandomForestClassifier(n_estimators=n, max_depth=d, max_features=f)
            rf.fit(trnX, trnY)
            prdY = rf.predict(tstX)
            yvalues.append(metrics.accuracy_score(tstY, prdY))
        values[d] = yvalues
    multiple_line_chart(axs[0, k], n_estimators, values, 'Random Forests with %s features' % f, 'nr estimators',
                             'accuracy', percentage=False)

plt.show()

plt.figure()
fig, axs = plt.subplots(1, 2, figsize=(10, 4), squeeze=False)
for k in range(len(max_features)):
    f = max_features[k]
    values = {}
    for d in max_depths:
        yvalues = []
        for n in n_estimators:
            rf = RandomForestClassifier(n_estimators=n, max_depth=d, max_features=f)
            rf.fit(trnX, trnY)
            prdY = rf.predict(tstX)
            yvalues.append(metrics.recall_score(tstY, prdY, average='macro'))
        values[d] = yvalues
    multiple_line_chart(axs[0, k], n_estimators, values, 'Random Forests with %s features' % f, 'nr estimators',
                             'sensibility', percentage=False)

plt.show()

"""clf = GaussianNB()
clf.fit(trnX, trnY)
prdY = clf.predict(tstX)

plt.figure()
fig, axs = plt.subplots(1, 2, figsize=(8, 4), squeeze=False)
plot_confusion_matrix(axs[0,0], metrics.confusion_matrix(tstY, prdY, labels), labels)
plot_confusion_matrix(axs[0,1], metrics.confusion_matrix(tstY, prdY, labels), labels, normalize=True)
plt.tight_layout()
plt.show()"""
