import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import SMOTE
from functions import *

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
#---------------------------------Data balancing - part1------------------------------------------------------
unbal = pd.read_csv('covtype.csv', sep=',', decimal='.', names=colnames)
target_count = unbal['Cover_Type'].value_counts()
plt.figure()
plt.title('Cover_Type balance')
plt.bar(target_count.index, target_count.values)
plt.show()

min_class = target_count.idxmin()
ind_min_class = target_count.index.get_loc(min_class)

print('Minority class:', target_count[ind_min_class])
print('Majority class:', target_count[7-ind_min_class])
print('Proportion:', round(target_count[ind_min_class] / target_count[7-ind_min_class], 7), ': 7')

#---------------------------------Data balancing - part2-------------------------------------------------------
RANDOM_STATE = 42
unbal = unbal.sample(frac=.25)
values = {'Original': [target_count.values[ind_min_class], target_count.values[7-ind_min_class]]}

df_class_min = unbal[unbal['Cover_Type'] == min_class]
df_class_max = unbal[unbal['Cover_Type'] != min_class]

df_under = df_class_max.sample(len(df_class_min))
values['UnderSample'] = [target_count.values[ind_min_class], len(df_under)]

df_over = df_class_min.sample(len(df_class_max), replace=True)
values['OverSample'] = [len(df_over), target_count.values[7-ind_min_class]]

smote = SMOTE(random_state=RANDOM_STATE)
y = unbal.pop('Cover_Type').values
X = unbal.values
_, smote_y = smote.fit_sample(X, y)
smote_target_count = pd.Series(smote_y).value_counts()
values['SMOTE'] = [smote_target_count.values[ind_min_class], smote_target_count.values[7-ind_min_class]]

plt.figure()
multiple_bar_chart(plt.gca(),
                        [target_count.index[ind_min_class], target_count.index[7-ind_min_class]],
                        values, 'Target', 'frequency', 'Class balance')
plt.show()