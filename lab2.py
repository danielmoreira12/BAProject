from functions import *
import pandas as pd
import numpy as np
from pandas.plotting import register_matplotlib_converters
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE, RandomOverSampler

register_matplotlib_converters()
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
original = pd.read_csv('covtype.csv', names=colnames)
original = original.astype({"Wilderness_Area 1": 'category', 'Wilderness_Area 2': 'category',
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

sb_vars = original.select_dtypes(include='object')
original[sb_vars.columns] = original.select_dtypes(['object']).apply(lambda x: x.astype('category'))

cols_nr = original.select_dtypes(include='number')
cols_sb = original.select_dtypes(include='category')

original.describe(include='all')

#---------------------------------Missing Value Imputation - Part 1---------------------------------------------

imp = SimpleImputer(strategy='constant', fill_value='NA', missing_values=np.nan, copy=True)
imp.fit(original.values)
mat = imp.transform(original.values)
data = pd.DataFrame(mat, columns=original.columns)
data.describe(include='all')

#---------------------------------Missing Value Imputation - Parte 2------------------------------------------
imp_nr = SimpleImputer(strategy='mean', missing_values=np.nan, copy=True)
imp_sb = SimpleImputer(strategy='most_frequent', missing_values='', copy=True)
df_nr = pd.DataFrame(imp_nr.fit_transform(cols_nr), columns=cols_nr.columns)
df_sb = pd.DataFrame(imp_sb.fit_transform(cols_sb), columns=cols_sb.columns)

data = df_nr.join(df_sb, how='right')
data.describe(include='all')

#---------------------------------Normalization---------------------------------------------------------------

transf = Normalizer().fit(df_nr)
df_nr = pd.DataFrame(transf.transform(df_nr, copy=True), columns= df_nr.columns)
norm_data = df_nr.join(df_sb, how='right')
norm_data.describe(include='all')

#---------------------------------Variable Dummification--------------------------------------------------------
"""from sklearn.preprocessing import OneHotEncoder


def dummify(df, cols_to_dummify):
    one_hot_encoder = OneHotEncoder(sparse=False)

    for var in cols_to_dummify:
        one_hot_encoder.fit(data[var].values.reshape(-1, 1))
        feature_names = one_hot_encoder.get_feature_names([var])
        transformed_data = one_hot_encoder.transform(data[var].values.reshape(-1, 1))
        df = pd.concat((df, pd.DataFrame(transformed_data, columns=feature_names)), 1)
        df.pop(var)
    return df


df = dummify(data, cols_sb.columns)
df.describe(include='all')
"""
#Deu erro! Não sei porque ainda
#---------------------------------Data balancing - part1------------------------------------------------------
#Deu erro! Falta de memoria. Temos de ver o que a professora falou de tirar uma amostra dos dados
unbal = pd.read_csv('covtype.csv', sep=',', decimal='.')
target_count = unbal['Outcome'].value_counts()
plt.figure()
plt.title('Class balance')
plt.bar(target_count.index, target_count.values)
plt.show()

min_class = target_count.idxmin()
ind_min_class = target_count.index.get_loc(min_class)

print('Minority class:', target_count[ind_min_class])
print('Majority class:', target_count[1-ind_min_class])
print('Proportion:', round(target_count[ind_min_class] / target_count[1-ind_min_class], 2), ': 1')

#---------------------------------Data balancing - part2-------------------------------------------------------
RANDOM_STATE = 42
values = {'Original': [target_count.values[ind_min_class], target_count.values[1-ind_min_class]]}

df_class_min = unbal[unbal['Outcome'] == min_class]
df_class_max = unbal[unbal['Outcome'] != min_class]

df_under = df_class_max.sample(len(df_class_min))
values['UnderSample'] = [target_count.values[ind_min_class], len(df_under)]

df_over = df_class_min.sample(len(df_class_max), replace=True)
values['OverSample'] = [len(df_over), target_count.values[1-ind_min_class]]

smote = SMOTE(ratio='minority', random_state=RANDOM_STATE)
y = unbal.pop('Outcome').values
X = unbal.values
_, smote_y = smote.fit_sample(X, y)
smote_target_count = pd.Series(smote_y).value_counts()
values['SMOTE'] = [smote_target_count.values[ind_min_class], smote_target_count.values[1-ind_min_class]]

plt.figure()
multiple_bar_chart(plt.gca(),
                        [target_count.index[ind_min_class], target_count.index[1-ind_min_class]],
                        values, 'Target', 'frequency', 'Class balance')