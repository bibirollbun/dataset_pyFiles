#import libraries

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns ##vizualization
import matplotlib.pyplot as plt #understanding statistics


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



df = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
df_train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")


df.dtypes


df.info() ##does not contain nulls


df.describe()


### Understand outlier in num_reported_accidents

Q1 = df['num_reported_accidents'].quantile(0.25)
Q3 = df['num_reported_accidents'].quantile(0.75)
IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR



outliers = df[(df['num_reported_accidents'] < lower_limit) | 
              (df['num_reported_accidents'] > upper_limit)]


print(f"Number outliers: {len(outliers)}")
print("Unique outlier values:")
print(outliers['num_reported_accidents'].unique())


#Visualization outliers in num_reported_accidents

for col in outliers.columns:
    sns.histplot(data=outliers, x=col, kde=True).set_title(f"Outliers em {col}")
    plt.show()



###adding the outliers in dataframe

df['is_outlier'] = ((df['num_reported_accidents'] < lower_limit) |
                    (df['num_reported_accidents'] > upper_limit))


#separating categorical and numerical variables

# Numeric
num_cols = df.select_dtypes(include="number").columns.tolist()



# Categoric
cat_cols = df.select_dtypes(exclude="number").columns.tolist()


correlation_matrix = df[num_cols].corr()


## numerical correlation
sns.heatmap(
    correlation_matrix,
    annot=True,         
    cmap='viridis',    
    fmt=".2f"           
)

plt.title('Correlation numeric')
plt.show()



##showing the proportion of accidents in risk zones by type of weather.

bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
df['risk_bin'] = pd.cut(df['accident_risk'], bins)

cause_accidents = pd.crosstab(df['weather'], df['risk_bin'], normalize='index')





sns.histplot(data=df, x='accident_risk', kde=True)

print("Most accidents fall into the moderate risk range (0.2–0.5)")
print("Few accidents reach high risk (0.8–1.0)")
print("This may indicate that most accidents occur under common conditions, and extremes are rare")

