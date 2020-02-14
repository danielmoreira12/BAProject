import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import SMOTE
from functions import *

#---------------------------------Data balancing - part1------------------------------------------------------
unbal = pd.read_csv('covertypeTreated.csv', sep=',', decimal='.')
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