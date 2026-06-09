# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_data = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
train_extra_data = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv")


missing_values_train = pd.DataFrame({'Feature': train_data.columns,
                              '[TRAIN] No. of Missing Values': train_data.isnull().sum().values,
                              '[TRAIN] % of Missing Values': ((train_data.isnull().sum().values)/len(train_data)*100)})

missing_values_test = pd.DataFrame({'Feature': test_data.columns,
                             '[TEST] No.of Missing Values': test_data.isnull().sum().values,
                             '[TEST] % of Missing Values': ((test_data.isnull().sum().values)/len(test_data)*100)})

missing_values_train_extra = pd.DataFrame({'Feature': train_extra_data.columns,
                             '[EXTRA] No.of Missing Values': train_extra_data.isnull().sum().values,
                             '[EXTRA] % of Missing Values': ((train_extra_data.isnull().sum().values)/len(train_extra_data)*100)})

unique_values = pd.DataFrame({'Feature': train_data.columns,
                              'No. of Unique Values[FROM TRAIN]': train_data.nunique().values})

feature_types = pd.DataFrame({'Feature': train_data.columns,
                              'DataType': train_data.dtypes})

merged_df = pd.merge(missing_values_train, missing_values_test, on='Feature', how='left')
merged_df = pd.merge(merged_df, missing_values_train_extra, on='Feature', how='left')
merged_df = pd.merge(merged_df, unique_values, on='Feature', how='left')
merged_df = pd.merge(merged_df, feature_types, on='Feature', how='left')

merged_df


train_data.describe()


train_extra_data.describe()


full_train = pd.concat([train_data, train_extra_data], ignore_index=True)
full_train.shape


for col in full_train:
    if full_train[col].dtype == 'object':
        print(col,full_train[col].unique() )


import matplotlib.pyplot as plt



full_train['Brand'].value_counts(dropna=False).plot(kind='bar')
plt.title('Distribution of Brands')
plt.xlabel('Brand')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



full_train['Material'].value_counts(dropna=False).plot(kind='bar')
plt.title('Distribution of Materials')
plt.xlabel('Material')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



full_train['Size'].value_counts(dropna=False).plot(kind='bar')
plt.title('Distribution of Sizes')
plt.xlabel('Size')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



full_train['Laptop Compartment'].value_counts(dropna=False).plot(kind='bar')
plt.title('Distribution of Laptop Compartments')
plt.xlabel('Laptop Compartment')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



full_train['Waterproof'].value_counts(dropna=False).plot(kind='bar')
plt.title('Distribution of Waterproofness')
plt.xlabel('Waterproof')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



full_train['Style'].value_counts(dropna=False).plot(kind='bar')
plt.title('Distribution of Styles')
plt.xlabel('Style')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


full_train['Color'].value_counts(dropna=False).plot(kind='bar')
plt.title('Distribution of Color')
plt.xlabel('Color')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


import seaborn as sns
sns.displot(full_train['Price'], kde=True)


plt.hist(full_train['Compartments'].dropna(), bins=10, edgecolor='black')
plt.title('Distribution of Compartments')
plt.xlabel('Number of Compartments')
plt.ylabel('Count')
plt.show()



plt.hist(full_train['Weight Capacity (kg)'].dropna(), bins=30, edgecolor='black')
plt.title('Distribution of Weight Capacity (kg)')
plt.xlabel('Weight Capacity (kg)')
plt.ylabel('Count')
plt.show()



from sklearn.impute import SimpleImputer
num_cols = ['Compartments', 'Weight Capacity (kg)']
cat_cols = ['Brand', 'Material', 'Size', 'Style', 'Color', 'Laptop Compartment', 'Waterproof']

num_imputer = SimpleImputer(strategy='median')
full_train[num_cols] = num_imputer.fit_transform(full_train[num_cols])
test_data[num_cols] = num_imputer.transform(test_data[num_cols])

cat_imputer = SimpleImputer(strategy='most_frequent')
full_train[cat_cols] = cat_imputer.fit_transform(full_train[cat_cols])
test_data[cat_cols] = cat_imputer.transform(test_data[cat_cols])


from sklearn.preprocessing import LabelEncoder

def feature_engineering(data):

    # converting to binary for easier analysis
    data['Laptop_Comp'] = data['Laptop Compartment'].map({'Yes': 1, 'No': 0})
    data['Water_Resistant'] = data['Waterproof'].map({'Yes': 1, 'No': 0})

    # certain styles may be limited by size, for example a messenger bag is often S or M
    data['Style_Size'] = data['Style'] + '_' + data['Size']

    # certain materials may correspond to certain sizes, for example a rucksack is not commonly made of leather
    data['Material_Style'] = data['Material'].astype(str) + '_' + data['Style'].astype(str)

    # normalizing the weight capacity by dividing to the gratest value
    data['Capacity_Scaled'] = data['Weight Capacity (kg)'] / data['Weight Capacity (kg)'].max()

    # some bags might hold more weight with less compartments
    data['Capacity_per_Compartment'] = data['Weight Capacity (kg)'] / (data['Compartments'] + 1)

    return data
    


full_train = feature_engineering(full_train)
test_data = feature_engineering(test_data)


id_test = test_data['id']
full_train.drop('id', axis=1, inplace=True)
test_data.drop('id', axis=1, inplace=True)


def detect_outliers_iqr(series):
    Q1 = series.quantile(0.15)
    Q3 = series.quantile(0.85)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (series < lower_bound) | (series > upper_bound)

outliers_weight_mask = detect_outliers_iqr(full_train['Weight Capacity (kg)'])
outliers_capacity_mask = detect_outliers_iqr(full_train['Capacity_per_Compartment'])
outliers_capacity_scaled_mask = detect_outliers_iqr(full_train['Capacity_Scaled'])

combined_outliers_mask = outliers_weight_mask | outliers_capacity_mask | outliers_capacity_scaled_mask

print("Weight Capacity outliers:\n", full_train[outliers_weight_mask])
print('--------------------------------------------------------------')
print("Capacity per Compartment outliers:\n", full_train[outliers_capacity_mask])
print('--------------------------------------------------------------')
print("Capacity Scaled outliers:\n", full_train[outliers_capacity_scaled_mask])



full_train = full_train[~combined_outliers_mask].reset_index(drop=True)


import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew

def check_skewness(data, column):
    sns.histplot(data[column], kde=True)
    plt.title(f"Histogram of {column}")
    plt.show()
    
    skewness = skew(data[column].dropna())
    print(f"Skewness of {column}: {skewness}")

    if skewness > 0:
        print(f"{column} is right-skewed (positive skew)")
    elif skewness < 0:
        print(f"{column} is left-skewed (negative skew)")
    else:
        print(f"{column} is approximately symmetric")

check_skewness(full_train, 'Weight Capacity (kg)')
check_skewness(test_data, 'Weight Capacity (kg)')


print(full_train.dtypes)



columns_to_encode = ['Brand', 'Material', 'Size', 'Laptop Compartment','Waterproof', 
                     'Style', 'Color', 'Style_Size', 'Laptop_Comp',
                     'Water_Resistant', 'Material_Style']
train_data_to_encode = full_train[columns_to_encode]
test_data_to_encode = test_data[columns_to_encode]

train_data_to_scale = full_train.drop(columns_to_encode, axis=1)
test_data_to_scale = test_data.drop(columns_to_encode, axis=1)

train_data_encoded = pd.get_dummies(train_data_to_encode, columns=columns_to_encode, drop_first=True)
test_data_encoded = pd.get_dummies(test_data_to_encode, columns=columns_to_encode, drop_first=True)


train_data_encoded.head()


test_data_encoded.head()


from sklearn.preprocessing import MinMaxScaler

minmax_scaler = MinMaxScaler()

minmax_scaler.fit(train_data_to_scale.drop(['Price'], axis=1))

scaled_data_train = minmax_scaler.transform(train_data_to_scale.drop(['Price'], axis=1))
scaled_train_df = pd.DataFrame(scaled_data_train, columns=train_data_to_scale.drop(['Price'], axis=1).columns)

scaled_data_test = minmax_scaler.transform(test_data_to_scale)
scaled_test_df = pd.DataFrame(scaled_data_test, columns=test_data_to_scale.columns)


scaled_test_df.head()


train_data_combined = pd.concat([train_data_encoded.reset_index(drop=True), scaled_train_df.reset_index(drop=True)], axis=1)
test_data_combined = pd.concat([test_data_encoded.reset_index(drop=True), scaled_test_df.reset_index(drop=True)], axis=1)


train_data_combined.head()


test_data_combined.head()


print(train_data_combined.dtypes)


from sklearn.neural_network import MLPRegressor
ann = MLPRegressor(hidden_layer_sizes=(100,),
    activation='tanh',
    max_iter=10,
    learning_rate = 'constant',
    verbose=True)

ann.fit(train_data_combined, full_train["Price"])


from lightgbm import LGBMRegressor, log_evaluation
model = LGBMRegressor()
model.fit(
    train_data_combined,
    full_train["Price"],
    eval_set=[(train_data_combined, full_train["Price"])],  
    eval_metric="rmse", 
    callbacks=[log_evaluation(period=1)]
)


predictions = model.predict(test_data_combined)
print(predictions[:10])


submission = pd.DataFrame({
    "id": id_test,  
    "Price": predictions     
})
submission.to_csv("submission.csv", index=False)



submission.head()




