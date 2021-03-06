from functions import *
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

data = pd.read_csv('Balanced_Data_SMOTE.csv')

#----------------------------------------------Training strategy---------------------------------------------------
y: np.ndarray = data.pop('Cover_Type').values
X: np.ndarray = data.values
labels: np.ndarray = pd.unique(y)

trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)

#----------------------------------------------Naive Bayes---------------------------------------------------------
"""clf = GaussianNB()
clf.fit(trnX, trnY)
prdY = clf.predict(tstX)
cnf_mtx = metrics.confusion_matrix(tstY, prdY, labels)
plot_confusion_matrix(plt.gca(), cnf_mtx, labels)"""

estimators = {'GaussianNB': GaussianNB(),
              'BernoulyNB': BernoulliNB()}

#'MultinomialNB': MultinomialNB(),

xvalues = []
yvalues = []
for clf in estimators:
    xvalues.append(clf)
    estimators[clf].fit(trnX, trnY)
    prdY = estimators[clf].predict(tstX)
    yvalues.append(metrics.accuracy_score(tstY, prdY))

xvaluesS = []
yvaluesS = []
for clfS in estimators:
    xvaluesS.append(clfS)
    estimators[clfS].fit(trnX, trnY)
    prdYS = estimators[clfS].predict(tstX)
    yvaluesS.append(metrics.recall_score(prdYS, tstY, average='macro'))

print('Sensibility: ', xvaluesS, yvaluesS)
print('Accuracy: ', xvalues, yvalues,)

plt.figure()
bar_chart(plt.gca(), xvaluesS, yvaluesS, 'Comparison of Naive Bayes Models', '', 'sensibility', percentage=True)
plt.show()

plt.figure()
bar_chart(plt.gca(), xvalues, yvalues, 'Comparison of Naive Bayes Models', '', 'accuracy', percentage=True)
plt.show()