import ipynb.fs.defs.functions as func
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import register_matplotlib_converters

from functions import *

register_matplotlib_converters()
colnames=['Elevation', 'Aspect',
'Slope',
'Horizontal_Distance_To_Hydrology',
'Vertical_Distance_To_Hydrology',
'Horizontal_Distance_To_Roadways',
'Hillshade_9am',
'Hillshade_Noon',
'Hillshade_3pm',
'Horizontal_Distance_To_Fire_Points',
'Wilderness_Area 1',
'Wilderness_Area 2',
'Wilderness_Area 3',
'Wilderness_Area 4',
'Soil_Type 1', 'Soil_Type 2','Soil_Type 3','Soil_Type 4','Soil_Type 5','Soil_Type 6','Soil_Type 7','Soil_Type 8','Soil_Type 9','Soil_Type 10',
'Soil_Type 11', 'Soil_Type 12','Soil_Type 13','Soil_Type 14','Soil_Type 15','Soil_Type 16','Soil_Type 17','Soil_Type 18','Soil_Type 19','Soil_Type 20',
'Soil_Type 21', 'Soil_Type 22','Soil_Type 23','Soil_Type 24','Soil_Type 25','Soil_Type 26','Soil_Type 27','Soil_Type 28','Soil_Type 29','Soil_Type 30',
'Soil_Type 31', 'Soil_Type 32','Soil_Type 33','Soil_Type 34','Soil_Type 35','Soil_Type 36','Soil_Type 37','Soil_Type 38','Soil_Type 39','Soil_Type 40',
'Cover_Type']

data = pd.read_csv('covtype.csv', names = colnames)

print(data.shape)

print(data.dtypes)

cat_vars = data.select_dtypes(include='object')
for att in cat_vars:
    print(att, data[att].unique())

#data[cat_vars.columns] = data.select_dtypes(['object']).apply(lambda x: x.astype('category'))
#data.dtypesdata[cat_vars.columns] = data.select_dtypes(['object']).apply(lambda x: x.astype('category'))

fig = plt.figure(figsize=(10,7))
mv = {}
for var in data:
    mv[var] = data[var].isna().sum()
    func.bar_chart(plt.gca(), mv.keys(), mv.values(), 'Number of missing values per variable', var, 'nr. missing values')
fig.tight_layout()
plt.show()

data.describe(include = 'all')

data.boxplot()
plt.show()

