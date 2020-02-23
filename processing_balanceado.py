from functions import *

colnames = ['Elevation', 'Aspect', 'Slope',
                'HorDistToHydrology', 'VerDistToHydrology',
                'HorDistToRoadways', 'Hillshade_9am', 'Hillshade_Noon',
                'Hillshade_3pm', 'HorDistToFirePoints',
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

category = 'Cover_Type'
target_count = data[category].value_counts()
print(target_count)
min_class = target_count.idxmin()
print('Minority class:', target_count[min_class])
df_class_min = data[data[category] == min_class]

c1 = data.loc[data['Cover_Type'] == 1]
c2 = data.loc[data['Cover_Type'] == 2]
c3 = data.loc[data['Cover_Type'] == 3]
c4 = data.loc[data['Cover_Type'] == 4]
c5 = data.loc[data['Cover_Type'] == 5]
c6 = data.loc[data['Cover_Type'] == 6]
c7 = data.loc[data['Cover_Type'] == 7]

c1 = c1.sample(len(df_class_min))
c2 = c2.sample(len(df_class_min))
c3 = c3.sample(len(df_class_min))
c5 = c5.sample(len(df_class_min))
c6 = c6.sample(len(df_class_min))
c7 = c7.sample(len(df_class_min))

frames = [c2, c3, c4, c5, c6, c7]
data_balanced = c1.append(frames)

print(data_balanced.shape)

print(data_balanced)

data_balanced.to_csv('covertypeBalanced.csv', index=False, index_label=False)