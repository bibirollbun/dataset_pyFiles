# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_data = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
sub_data = pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")



print(f"train data shape {train_data.shape}")
print(f"test data shape {test_data.shape}")


train_data.columns


train_data.isna().sum().sum()


train_data = train_data.drop_duplicates()
train_data.duplicated().sum()


num_cols = train_data.select_dtypes(include="number").columns.tolist()
cat_cols = train_data.select_dtypes(exclude="number").columns.tolist()
num_cols.remove("accident_risk")
print(f"Numerical colums {num_cols}")
print(f"Categorical colums {cat_cols}")


correlation_matrix = train_data[num_cols + ['accident_risk']].corr()
correlation_matrix


plt.figure(figsize=(8, 6))
correlation_matrix = train_data[num_cols + ['accident_risk']].corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            linewidths=1)
plt.show()


for c in cat_cols:
    print(f"{c} (Uniques):{train_data[c].unique()}")


fig, axes = plt.subplots(2,4, figsize=(16,8))
axes = axes.flatten()
cmap = plt.get_cmap('magma')
colors = cmap([0.9,0.66,0.33])
target = 'accident_risk'
for i, col in enumerate(cat_cols):
    grouped = train_data.groupby(col)[target].mean()
    axes[i].bar(grouped.index.astype(str), grouped.values, color=colors)
    axes[i].set_ylabel(f'Mean{target}')
    axes[i].set_title(f'{col} vs {target}')
    axes[i].tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.show()


plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.hist(train_data['accident_risk'], bins=50, edgecolor='black')
plt.xlabel('Accident Risk')
plt.title('Accident Risk Distribution')


plt.subplot(1,2,2)
plt.boxplot(train_data['accident_risk'])
plt.xlabel('Accident Risk')
plt.title('Accident Risk Distribution')


bool_cols = ["road_signs_present", "public_road", "holiday", "school_season"]
for col in bool_cols:
    train_data[col]=train_data[col].astype(int)
    test_data[col]=test_data[col].astype(int)


le = LabelEncoder()
cate_cols = train_data.select_dtypes(exclude="number").columns.tolist()
for col in cate_cols:
    train_data[col]=le.fit_transform(train_data[col])
    test_data[col]=le.fit_transform(test_data[col])


train_data.head()


road_type = train_data.loc[train_data.road_type == 'highway']["accident_risk"]
rate_acc = sum(road_type/len(road_type))
print ("% of accidents on highway:", rate_acc)


road_type = train_data.loc[train_data.road_type == 'urban']["accident_risk"]
rate_acc = sum(road_type/len(road_type))
print ("% of accidents on urban:", rate_acc)


road_type = train_data.loc[train_data.road_type == 'rural']["accident_risk"]
rate_acc = sum(road_type/len(road_type))
print ("% of accidents on rural:", rate_acc)


from sklearn.ensemble import RandomForestRegressor
import pandas as pd

# Target variable
y = train_data["accident_risk"]

# Feature selection
features = ['road_type', 'weather', 'speed_limit', 'lighting', 'road_signs_present']
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])
X_test = X_test.reindex(columns=X.columns, fill_value=0)

# Model setup
model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)

# Predictions
predictions = model.predict(X_test)

# Save submission
output = pd.DataFrame({'id': test_data.id, 'accident_risk': predictions})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")


print(output.head())



corr = test_data.corr(numeric_only=True)
corr


corr = test_data.corr(numeric_only=True)['num_reported_accidents'].sort_values(ascending=False)
corr



import matplotlib.pyplot as plt
import seaborn as sns

# Remove the target itself from correlation
corr = corr.drop('num_reported_accidents', errors='ignore')

# Set the plot size
plt.figure(figsize=(10,6))

# Create a barplot
sns.barplot(
    x=corr.values,
    y=corr.index,
    palette='coolwarm'
)

# Add labels and title
plt.title("Feature Correlation with Number of Reported Accidents", fontsize=14)
plt.xlabel("Correlation Coefficient")
plt.ylabel("Feature Name")

# Show a vertical line at 0 to separate positive and negative correlations
plt.axvline(0, color='black', linestyle='--', linewidth=1)

plt.tight_layout()
plt.show()


