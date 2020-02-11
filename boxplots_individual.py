from functions import *

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

columns = data.select_dtypes(include='number').columns
rows, cols = choose_grid(12)
plt.figure()
fig, axs = plt.subplots(rows, cols, figsize=(cols*4, rows*4), squeeze=False)
i, j = 0, 0

for n in range(12):
    axs[i, j].set_title('Boxplot for %s'%columns[n])
    axs[i, j].boxplot(data[columns[n]].dropna().values)
    i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
fig.tight_layout()
plt.show()
"""
for n in range(12, 24):
    axs[i, j].set_title('Boxplot for %s'%columns[n])
    axs[i, j].boxplot(data[columns[n]].dropna().values)
    i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
fig.tight_layout()
plt.show()"""

"""for n in range(24, 36):
    axs[i, j].set_title('Boxplot for %s'%columns[n])
    axs[i, j].boxplot(data[columns[n]].dropna().values)
    i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
fig.tight_layout()
plt.show()"""

"""for n in range(36, 48):
    axs[i, j].set_title('Boxplot for %s'%columns[n])
    axs[i, j].boxplot(data[columns[n]].dropna().values)
    i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
fig.tight_layout()
plt.show()
"""
"""columns = data.select_dtypes(include='number').columns
rows, cols = choose_grid(1)
plt.figure()
fig, axs = plt.subplots(figsize=(cols*4, rows*4), squeeze=False)
i, j = 0, 0

axs[i, j].set_title('Boxplot for %s'%columns[54])
axs[i, j].boxplot(data[columns[54]].dropna().values)
i, j = (i + 1, 0) if (54+1) % cols == 0 else (i, j + 1)
fig.tight_layout()
plt.show()"""