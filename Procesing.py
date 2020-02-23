import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from functions import *
original = pd.read_csv('covertypeBalanced.csv')
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
                    'Soil_Type 39': 'category', 'Soil_Type 40': 'category', 'Cover_Type':'category'})


sb_vars = original.select_dtypes(include='object')
original[sb_vars.columns] = original.select_dtypes(['object']).apply(lambda x: x.astype('category'))

cols_nr = original.select_dtypes(include='number')
cols_sb = original.select_dtypes(include='category')

df_nr = pd.DataFrame(cols_nr, columns=cols_nr.columns)
df_sb = pd.DataFrame(cols_sb, columns=cols_sb.columns)

data = df_nr.join(df_sb, how='right')
data.describe(include='all')

transf = StandardScaler(with_mean=True, with_std=True, copy=True).fit(df_nr)
df_nr = pd.DataFrame(transf.transform(df_nr), columns= df_nr.columns)
norm_data_zscore = df_nr.join(df_sb, how='right')
norm_data_zscore.describe(include='all')

transf = MinMaxScaler(feature_range=(0, 1), copy=True).fit(df_nr)
df_nr = pd.DataFrame(transf.transform(df_nr), columns= df_nr.columns)
norm_data_minmax = df_nr.join(df_sb, how='right')
norm_data_minmax.describe(include='all')

print(norm_data_minmax)

fig, axs = plt.subplots(1, 3, figsize=(20,10),squeeze=False)
axs[0, 0].set_title('Original data')
df_nr.boxplot(ax=axs[0, 0])
axs[0, 1].set_title('Z-score normalization')
norm_data_zscore.boxplot(ax=axs[0, 1])
axs[0, 2].set_title('MinMax normalization')
norm_data_minmax.boxplot(ax=axs[0, 2])
#fig.tight_layout()
plt.show()

norm_data_zscore.to_csv('covertypeZNormalized.csv',index=False,index_label=False)
norm_data_minmax.to_csv('covertypeMinNormalized.csv',index=False,index_label=False)