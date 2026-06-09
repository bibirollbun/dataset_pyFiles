import warnings
warnings.filterwarnings('ignore')


import pandas #provides data structures to quickly analyze data
#Since this code runs on Kaggle server, train data can be accessed directly in the 'input' folder
dataset = pandas.read_csv("../input/forest-cover-type-prediction/train.csv") 

dataset = dataset.iloc[:,1:]


print(dataset.shape)
dataset.head()


dataset.dtypes


dataset.describe()





print(dataset.skew())
#Skewness is the degree of asymmetry observed in a probability distribution. When data points on a bell curve are not distributed symmetrically to the left and right sides of the median, the bell curve is skewed.


dataset.plot.hist(column=["Elevation"], figsize=(10, 8), bins=200)
dataset.plot.hist(column=["Slope"], figsize=(10, 8), bins=20)


print(dataset.groupby('Soil_Type25').size(), '\n')
print(dataset.groupby('Soil_Type29').size(), '\n')
print(dataset.groupby('Cover_Type').size())


print(dataset.groupby('Wilderness_Area1').size(), '\n')
print(dataset.groupby('Wilderness_Area2').size(), '\n')
print(dataset.groupby('Wilderness_Area3').size(), '\n')
print(dataset.groupby('Wilderness_Area4').size(), '\n')


dataset.cov()


dataset.corr()


import numpy

size = 10 
data=dataset.iloc[:,:size] 
cols=data.columns 
data_corr = data.corr()
threshold = 0.3

corr_list = []

#Search for the highly correlated pairs
for i in range(0,size):
    for j in range(i+1,size):
        if (data_corr.iloc[i,j] >= threshold and data_corr.iloc[i,j] < 1) or (data_corr.iloc[i,j] < 0 and data_corr.iloc[i,j] <= -threshold):
            corr_list.append([data_corr.iloc[i,j],i,j])

s_corr_list = sorted(corr_list,key=lambda x: -abs(x[0]))

for v,i,j in s_corr_list:
    print ("%s and %s = %.2f" % (cols[i],cols[j],v))


import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(dataset.corr(method='pearson', min_periods=1))
plt.show()


unique_vals = set()

for v, i, j in corr_list:
    unique_vals.add(cols[i])
    unique_vals.add(cols[j])

unique_vals_list = list(unique_vals)

print(unique_vals_list)


dataset_with_corelating_cols = dataset[unique_vals_list].copy()

plt.figure(figsize = (16,5))

ax = sns.heatmap(dataset_with_corelating_cols.corr(method='pearson', min_periods=1), annot=True, linewidths=.5)


for v,i,j in s_corr_list:
    sns.pairplot(dataset, hue="Cover_Type", size=6, x_vars=cols[i],y_vars=cols[j] )
    plt.show()


sns.pairplot(dataset, hue="Cover_Type", vars=['Elevation', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology'])


sns.boxplot(
    dataset[unique_vals_list[:4]],
    whis=[0, 100], width=.6, palette="vlag"
)


sns.boxplot(
    dataset[unique_vals_list[4:]],
    whis=[0, 100], width=10, palette="vlag"
)


for col in unique_vals_list:
    sns.violinplot(dataset[col], y=data[col])
    plt.show()




