from sklearn.naive_bayes import GaussianNB

from functions import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

data = pd.read_csv('Balanced_Data_SMOTE.csv')

#----------------------------------------------Decision Trees---------------------------------------------------
y: np.ndarray = data.pop('Cover_Type').values
X: np.ndarray = data.values
labels = pd.unique(y)

trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)

min_samples_leaf = [.01, .0075, .005, .0025, .001]
max_depths = [5, 10, 25, 50]
criteria = ['entropy', 'gini']

plt.figure()
fig, axs = plt.subplots(1, 2, figsize=(16, 4), squeeze=False)
for k in range(len(criteria)):
    f = criteria[k]
    values = {}
    for d in max_depths:
        yvalues = []
        for n in min_samples_leaf:
            tree = DecisionTreeClassifier(min_samples_leaf=n, max_depth=d, criterion=f)
            tree.fit(trnX, trnY)
            prdY = tree.predict(tstX)
            yvalues.append(metrics.accuracy_score(tstY, prdY))
        values[d] = yvalues
        print("Max Depth: ", d, "Accuracy: ", yvalues)
    multiple_line_chart(axs[0, k], min_samples_leaf, values, 'Decision Trees with %s criteria' % f,
                             'nr estimators',
                             'accuracy', percentage=True)

plt.show()

plt.figure()
fig, axs = plt.subplots(1, 2, figsize=(16, 4), squeeze=False)
for k in range(len(criteria)):
    f = criteria[k]
    values = {}
    for d in max_depths:
        yvalues = []
        for n in min_samples_leaf:
            tree = DecisionTreeClassifier(min_samples_leaf=n, max_depth=d, criterion=f)
            tree.fit(trnX, trnY)
            prdY = tree.predict(tstX)
            yvalues.append(metrics.recall_score(tstY, prdY, average='macro'))
        values[d] = yvalues
        print("Max Depth: ", d, "Sensibility: ", yvalues)
    multiple_line_chart(axs[0, k], min_samples_leaf, values, 'Decision Trees with %s criteria' % f,
                             'nr estimators',
                             'sensibility', percentage=False)

plt.show()
"""
tree = DecisionTreeClassifier(max_depth=3)
tree.fit(trnX, trnY)

dot_data = export_graphviz(tree, out_file='dtree.dot', filled=True, rounded=True, special_characters=True)
print('step 2')

# Convert to png
print('step 3')
plt.figure(figsize=(14, 18))
plt.axis('off')

import pydot

(graph,) = pydot.graph_from_dot_file('dtree.dot')
graph.write_png('decision_tree_schema_raw_data.png')

plt.show()

clf = GaussianNB()
clf.fit(trnX, trnY)
prdY = clf.predict(tstX)

plt.figure()
fig, axs = plt.subplots(1, 2, figsize=(8, 4), squeeze=False)
plot_confusion_matrix(axs[0,0], metrics.confusion_matrix(tstY, prdY, labels), labels)
plot_confusion_matrix(axs[0,1], metrics.confusion_matrix(tstY, prdY, labels), labels, normalize=True)
plt.tight_layout()
plt.show()"""