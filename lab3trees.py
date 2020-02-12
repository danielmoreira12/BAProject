from functions import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.tree import DecisionTreeClassifier

colnames = ['Elevation', 'Aspect', 'Slope',
                'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
                'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon',
                'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points',
                'Wilderness_Area 1', 'Wilderness_Area 2', 'Wilderness_Area 3', 'Wilderness_Area 4',
                'Soil_Type 1', 'Soil_Type 2', 'Soil_Type 3', 'Soil_Type 4', 'Soil_Type 5', 'Soil_Type 6', 'Soil_Type 7',
                'Soil_Type 8', 'Soil_Type 9', 'Soil_Type 10',
                'Soil_Type 11', 'Soil_Type 12', 'Soil_Type 13', 'Soil_Type 14', 'Soil_Type 15', 'Soil_Type 16',
                'Soil_Type 17', 'Soil_Type 18', 'Soil_Type 19', 'Soil_Type 20',
                'Soil_Type 21', 'Soil_Type 22', 'Soil_Type 23', 'Soil_Type 24', 'Soil_Type 25', 'Soil_Type 26',
                'Soil_Type 27', 'Soil_Type 28', 'Soil_Type 29', 'Soil_Type 30',
                'Soil_Type 31', 'Soil_Type 32', 'Soil_Type 33', 'Soil_Type 34', 'Soil_Type 35', 'Soil_Type 36',
                'Soil_Type 37', 'Soil_Type 38', 'Soil_Type 39', 'Soil_Type 40',
                'Cover_Type']
data = pd.read_csv('covtype.csv', names=colnames)
data = data.astype({"Wilderness_Area 1": 'category', 'Wilderness_Area 2': 'category',
                    'Wilderness_Area 3': 'category', 'Wilderness_Area 4': 'category',
                    'Soil_Type 1': 'category', 'Soil_Type 2': 'category', 'Soil_Type 3': 'category',
                    'Soil_Type 4': 'category', 'Soil_Type 5': 'category', 'Soil_Type 6': 'category',
                    'Soil_Type 7': 'category', 'Soil_Type 8': 'category', 'Soil_Type 9': 'category',
                    'Soil_Type 10': 'category',
                    'Soil_Type 11': 'category', 'Soil_Type 12': 'category', 'Soil_Type 13': 'category',
                    'Soil_Type 14': 'category', 'Soil_Type 15': 'category', 'Soil_Type 16': 'category',
                    'Soil_Type 17': 'category',
                    'Soil_Type 18': 'category', 'Soil_Type 19': 'category', 'Soil_Type 20': 'category',
                    'Soil_Type 21': 'category', 'Soil_Type 22': 'category', 'Soil_Type 23': 'category',
                    'Soil_Type 24': 'category', 'Soil_Type 25': 'category', 'Soil_Type 26': 'category',
                    'Soil_Type 27': 'category', 'Soil_Type 28': 'category', 'Soil_Type 29': 'category',
                    'Soil_Type 30': 'category', 'Soil_Type 31': 'category', 'Soil_Type 32': 'category',
                    'Soil_Type 33': 'category', 'Soil_Type 34': 'category', 'Soil_Type 35': 'category',
                    'Soil_Type 36': 'category', 'Soil_Type 37': 'category', 'Soil_Type 38': 'category',
                    'Soil_Type 39': 'category', 'Soil_Type 40': 'category'})

#----------------------------------------------Decision Trees---------------------------------------------------
y: np.ndarray = data.pop('Cover_Type').values
X: np.ndarray = data.values
labels = pd.unique(y)

trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)

min_samples_leaf = [.05, .025, .01, .0075, .005, .0025, .001]
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
    multiple_line_chart(axs[0, k], min_samples_leaf, values, 'Decision Trees with %s criteria' % f,
                             'nr estimators',
                             'accuracy', percentage=True)

plt.show()