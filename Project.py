from functions import *
import seaborn as sns

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
                    'Soil_Type 39': 'category', 'Soil_Type 40': 'category', 'Cover_Type': 'category'})


fig = plt.figure(figsize=(10,7))
mv = {}
for var in data:
    mv[var] = data[var].isna().sum()
    bar_chart(plt.gca(), mv.keys(), mv.values(), 'Number of missing values per variable', var, 'nr. missing values')
fig.tight_layout()
plt.savefig("MissingValues.png")

#---------------------HistogramForCategory---------------------------------------------------------------------------
columns = data.select_dtypes(include='category').columns
rows, cols = choose_grid(len(columns))
plt.figure()
fig, axs = plt.subplots(rows, cols, figsize=(cols*4, rows*4), squeeze=False)
i, j = 0, 0
for n in range(len(columns)):
    counts = data[columns[n]].dropna().value_counts(normalize=True)
    bar_chart(axs[i, j], counts.index, counts.values, 'Histogram for %s'%columns[n], columns[n], 'presence')
    i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
fig.tight_layout()
plt.savefig("HistogramCategory.png")

#---------------------BoxplotIndividual----------------------------------------------------------------------------
columns = data.select_dtypes(include='number').columns
rows, cols = choose_grid(len(columns))
plt.figure()
fig, axs = plt.subplots(rows, cols, figsize=(cols*4, rows*4), squeeze=False)
i, j = 0, 0

for n in range(len(columns)):
    axs[i, j].set_title('Boxplot for %s'%columns[n])
    axs[i, j].boxplot(data[columns[n]].dropna().values)
    i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
fig.tight_layout()
plt.savefig("BoxplotIndividual.png")

#---------------------Histogram----------------------------------------------------------------------------------
columns = data.select_dtypes(include='number').columns
rows, cols = choose_grid(len(columns))
plt.figure()
fig, axs = plt.subplots(rows, cols, figsize=(cols*4, rows*4), squeeze=False)
i, j = 0, 0

for n in range(len(columns)):
    axs[i, j].set_title('Histogram for %s'%columns[n])
    axs[i, j].set_xlabel(columns[n])
    #axs[i, j].set_ylabel("0 to 255 index")
    axs[i, j].hist(data[columns[n]].dropna().values, 'auto')
    i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
fig.tight_layout()
plt.savefig("HistogramNumber.png")

#---------------------HistogramTrends-----------------------------------------------------------------------------
columns = data.select_dtypes(include='number').columns
rows, cols = choose_grid(len(columns))
plt.figure()
fig, axs = plt.subplots(rows, cols, figsize=(cols*4, rows*4), squeeze=False)
i, j = 0, 0
for n in range(len(columns)):
    axs[i, j].set_title('Histogram with trend for %s'%columns[n])
    #axs[i, j].set_ylabel("probability")
    sns.distplot(data[columns[n]].dropna().values, norm_hist=True, ax=axs[i, j], axlabel=columns[n])
    i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
fig.tight_layout()
plt.savefig("HistogramTrends.png")

#-------------------------------Sparsity-----------------------------------------------------------------------
columns = data.select_dtypes(include='number').columns
rows, cols = len(columns)-1, len(columns)-1
plt.figure()
fig, axs = plt.subplots(rows, cols, figsize=(cols*4, rows*4), squeeze=False)
for i in range(len(columns)):
    var1 = columns[i]
    for j in range(i+1, len(columns)):
        var2 = columns[j]
        axs[i, j-1].set_title("%s x %s"%(var1,var2))
        axs[i, j-1].set_xlabel(var1)
        axs[i, j-1].set_ylabel(var2)
        axs[i, j-1].scatter(data[var1], data[var2])
fig.tight_layout()
plt.savefig('Sparsity.png')

#-------------------------------Correlation analysis---------------------------------------------------------------
fig = plt.figure(figsize=[14, 14])
corr_mtx = data.corr()
sns.heatmap(corr_mtx, xticklabels=corr_mtx.columns, yticklabels=corr_mtx.columns, annot=True, cmap='Blues')
plt.title('Correlation analysis')
plt.savefig('CorrelationAnalysis.png')

