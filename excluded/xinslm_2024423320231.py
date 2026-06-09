import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import shap
from sklearn.preprocessing import LabelEncoder

from xgboost import XGBClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter


train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv', index_col='id')
train.sample(5)


print('Train shape =',train.shape)

print('Null values =',train.isnull().sum().sum())


# Plot pie chart
target_classes = train['Fertilizer Name'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(target_classes, labels=target_classes.index, autopct='%1.1f%%', startangle=90)
plt.title(f"Distribution of Fertilizer Name")
plt.axis("equal")
plt.show()

# Print unique and missing values
print(f"Number of Unique Fertilizer Name: {train['Fertilizer Name'].nunique()}")


train.dtypes


train.describe()


train.duplicated().sum()


# 1. Combining two categorical columns

x, y = train.drop(columns = ['Fertilizer Name']), train.loc[:, 'Fertilizer Name']

x['soil_crop_type'] = x['Soil Type'] + '_' + x['Crop Type']


# 2. Aggregating num columns

# Ratios
x['N_to_P'] = x['Nitrogen'] / (x['Phosphorous'] + 1e-5)
x['N_to_K'] = x['Nitrogen'] / (x['Potassium'] + 1e-5)
x['K_to_P'] = x['Potassium'] / (x['Phosphorous'] + 1e-5)
x['Moisture_to_Temp'] = x['Moisture'] / (x['Temparature'] + 1e-5)
x['Humidity_to_Temp'] = x['Humidity'] / (x['Temparature'] + 1e-5)

# Sums / interactions
x['NPK_sum'] = x['Nitrogen'] + x['Phosphorous'] + x['Potassium']
x['Moisture_Temp_Interaction'] = x['Moisture'] * x['Temparature']
x['Humidity_Temp_Interaction'] = x['Humidity'] * x['Temparature']

# Differences
x['N_minus_P'] = x['Nitrogen'] - x['Phosphorous']
x['N_minus_K'] = x['Nitrogen'] - x['Potassium']
x['P_minus_K'] = x['Phosphorous'] - x['Potassium']
x['Temp_minus_Humidity'] = x['Temparature'] - x['Humidity']
x['Moisture_minus_Temp'] = x['Moisture'] - x['Temparature']

# Quadratic / polynomial features
x['Temparature_sq'] = x['Temparature'] ** 2
x['Humidity_sq'] = x['Humidity'] ** 2
x['Moisture_sq'] = x['Moisture'] ** 2
x['Nitrogen_Phosphorous'] = x['Nitrogen'] * x['Phosphorous']
x['Nitrogen_Potassium'] = x['Nitrogen'] * x['Potassium']
x['Phosphorous_Potassium'] = x['Phosphorous'] * x['Potassium']


# 3. Binning / Discretization

x['Temp_bin'] = pd.cut(x['Temparature'], bins=[0, 20, 30, 50], labels=['low', 'medium', 'high'])


# 4. pH levels of soils
ph_range_map = {
    'Loamy': (6.2, 6.8),
    'Sandy': (4.5, 5.5),
    'Clayey': (7.5, 10.0),
    'Red' : (4.5, 7.5),
    'Black' : (7.2, 8.5),
}

x['Soil_pH_min'] = x['Soil Type'].map(lambda s: ph_range_map[s][0])
x['Soil_pH_max'] = x['Soil Type'].map(lambda s: ph_range_map[s][1])

ph_map = {
    'Loamy': (ph_range_map['Loamy'][0] + ph_range_map['Loamy'][1]) / 2,
    'Sandy': (ph_range_map['Sandy'][0] + ph_range_map['Sandy'][1]) / 2,
    'Clayey': (ph_range_map['Clayey'][0] + ph_range_map['Clayey'][1]) / 2,
    'Red' : (ph_range_map['Red'][0] + ph_range_map['Red'][1]) / 2,
    'Black' : (ph_range_map['Black'][0] + ph_range_map['Black'][1]) / 2,
}

x['Soil_pH'] = x['Soil Type'].map(ph_map)


objects = [col for col in x.columns if x[col].dtype == 'O']


x[objects] = x[objects].astype('category')


cats = [col for col in x.columns if x[col].dtype == 'category']
nums = [col for col in x.columns if col not in cats]


def reduce_numeric_memory(df, nums, verbose=True):
    df = df[nums].copy()
    start_mem = df.memory_usage(deep=True).sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtypes

        if pd.api.types.is_numeric_dtype(col_type):
            c_min = df[col].min()
            c_max = df[col].max()

            if pd.api.types.is_integer_dtype(col_type):
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)

            elif pd.api.types.is_float_dtype(col_type):
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage(deep=True).sum() / 1024**2

    if verbose:
        print(f'Memory usage reduced from {start_mem:.2f} MB to {end_mem:.2f} MB '
              f'({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)')

    return df


x[nums] = reduce_numeric_memory(x, nums)


correlation_matrix = x[nums].corr()
plt.figure(figsize=(20, 18))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix of Numerical Features")
plt.show()


# Adding random values to select features by importance
x['random'] = np.random.randint(10, 50 + 1, size=len(x))


le = LabelEncoder()
y_transformed = le.fit_transform(y)

y_transformed = y_transformed.astype("int32")


def mapk(actual, predicted, k=3):
    def apk(a, p, k):
        score = 0.0
        for i in range(min(k, len(p))):
            if p[i] == a:
                score += 1.0 / (i + 1)
                break
        return score
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])

