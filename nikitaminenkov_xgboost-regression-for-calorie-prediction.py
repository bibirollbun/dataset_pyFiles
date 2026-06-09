# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
%matplotlib inline



d_train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')


print(f'Data Shape: {d_train.shape}')

print(f'\nData Info:')
d_train.info()

print(f'\nNumerical Features Summary:')
display(d_train.describe().transpose())

print(f'\nFirst 10 rows of the Dataset:')
d_train.head(10)


d_train.nunique()


sns.set_style('whitegrid')


numerical_features = d_train.columns[2:]

for feature in numerical_features:  
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    if abs(d_train[feature].skew()) < 1: 
        sns.histplot(d_train[feature], kde=True, bins=50)
        plt.xlabel(feature)
        plt.title(f'Histogram of {feature}')
        plt.ylabel('Frequency')

    else:
        sns.histplot(np.log1p(d_train[feature]), kde=True, bins=50)
        plt.title(f'Log Transformed Histogram of {feature}')
        plt.xlabel(f'log({feature})')

    plt.subplot(1, 2, 2)
    sns.boxplot(x=d_train[feature])
    plt.title(f'Box Plot of {feature}')

    plt.tight_layout()
    plt.show()

    print(f'\nStatistics for {feature}:')
    print(f'\nSkewness: {d_train[feature].skew():.2f}')
    print(f'\nNumber of Missing Values: {d_train[feature].isnull().sum()}')
    


sex_counts = d_train['Sex'].value_counts()

plt.figure(figsize=(10, 5))
plt.pie(x=sex_counts, labels=sex_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Distribution of Sex')
plt.axis('equal')

plt.show()

print(f"Number of Unique {feature}: {d_train[feature].nunique()}")
print(f"Missing Values in {feature}: {d_train[feature].isnull().sum()}")


colors = sns.color_palette('husl', len(numerical_features))

rows = -(-len(numerical_features) // 4)
plt.figure(figsize=(20, 5 * rows))

for i, (col, color) in enumerate(zip(numerical_features, colors), 1):
    plt.subplot(rows, 4, i)
    sns.kdeplot(data=d_train, x=col, fill=True, color=color)
    plt.title(f'KDE Plot of {col}', fontsize=14, color=color)
    plt.xlabel(col)
    plt.ylabel('Density')

plt.tight_layout()
plt.show()


numeric_df = d_train.select_dtypes(include='number')

sns.pairplot(data=numeric_df, corner=True)
plt.suptitle('Pairwise Scatter Plots', y=1.02)
plt.show()


for feature in numerical_features[:-1]:
    plt.figure(figsize=(10, 6))
    
    sns.scatterplot(x=d_train[feature], y=d_train['Calories'], hue=d_train['Sex'], palette='deep', alpha=0.5)
    plt.title(f'{feature} vs Calories')
    plt.xlabel(feature)
    plt.ylabel('Calories')
    plt.show()

df_corr = d_train[numerical_features].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(df_corr, cmap='plasma', annot=True)
plt.title('Correlation Matrix of Numerical Features')
plt.show()


plt.figure(figsize=(12, 8))
sns.boxplot(x=d_train['Sex'], y=d_train['Calories'], )
plt.title('Sex vs Calories')
plt.xlabel('Sex')
plt.ylabel('Calories')
plt.tight_layout()
plt.show()


colors = sns.color_palette('husl', len(numerical_features))
rows = -(-len(numerical_features) // 4)
plt.figure(figsize=(20, 5 * rows))

for i, (col, color) in enumerate(zip(numerical_features, colors), 1):
    plt.subplot(rows, 4, i)
    sns.violinplot(data=d_train, y=col, color=color)
    plt.title(f'Violin Plot of {col}', fontsize=14, color=color)
    plt.xlabel('')
    plt.ylabel(col)

plt.tight_layout()
plt.show()


for col in numerical_features:
    Q1 = d_train[col].quantile(0.25)
    Q3 = d_train[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = d_train[(d_train[col] < Q1 - 1.5 * IQR) | (d_train[col] > Q3 + 1.5 * IQR)]
    print(f'{col}: {len(outliers)} outliers')


d_test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
d_test.head(5)


train = d_train.copy()
test = d_test.copy()


train = pd.get_dummies(train, columns=['Sex'], drop_first=True)
test = pd.get_dummies(test, columns=['Sex'], drop_first=True)


from sklearn.preprocessing import StandardScaler

data_to_scale = [feat for feat in numerical_features if feat != 'Calories']

scaler = StandardScaler()

train[data_to_scale] = scaler.fit_transform(train[data_to_scale])

test[data_to_scale] = scaler.transform(test[data_to_scale])


# BMI
train['BMI'] = train['Weight'] / (train['Height'] / 100)**2
test['BMI'] = test['Weight'] / (test['Height'] / 100)**2

# Intensity
train['Intensity'] = train['Heart_Rate'] / train['Duration']
test['Intensity'] = test['Heart_Rate'] / test['Duration']


man_temp_mean = train.loc[train['Sex_male'] == True, 'Body_Temp'].mean()
woman_temp_mean = train.loc[train['Sex_male'] == False, 'Body_Temp'].mean()

# Temp Deviation
train['Temp_Deviation'] = train.apply(
    lambda row: row['Body_Temp'] - man_temp_mean if row['Sex_male'] else row['Body_Temp'] - woman_temp_mean,
    axis=1
)
test['Temp_Deviation'] = test.apply(
    lambda row: row['Body_Temp'] - man_temp_mean if row['Sex_male'] else row['Body_Temp'] - woman_temp_mean,
    axis=1
)

# High Heart Rate
high_hr_threshold = train['Heart_Rate'].quantile(0.95)

train['High_Heart_Rate'] = (train['Heart_Rate'] > high_hr_threshold).astype(int)
test['High_Heart_Rate'] = (test['Heart_Rate'] > high_hr_threshold).astype(int)

# Age Group
bins = [0, 18, 30, 45, 60, 100]
labels = ['Teen', 'Young', 'Adult', 'Middle', 'Senior']

train['Age_Group'] = pd.cut(train['Age'], bins=bins, labels=labels)
test['Age_Group'] = pd.cut(test['Age'], bins=bins, labels=labels)

# One-hot encoding Age_Group
train = pd.get_dummies(train, columns=['Age_Group'], drop_first=True)
test = pd.get_dummies(test, columns=['Age_Group'], drop_first=True)


from sklearn.model_selection import train_test_split

X = train.drop(columns='Calories')
y = train['Calories']

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)


from xgboost import XGBRegressor

model = XGBRegressor(n_estimator=100, learning_rate=0.1, max_depth=6, random_state=42)

model.fit(X_train, y_train)

preds = model.predict(X_valid)


from sklearn.metrics import mean_squared_log_error


preds = np.clip(preds, 0, None)

RMSLE = mean_squared_log_error(preds, y_valid)

print(f'RMSLE: {RMSLE:.4f}')


errors = y_valid - preds

plt.figure(figsize=(10,6))
plt.hist(errors, bins=50)
plt.title('Distribusion of errors')
plt.xlabel('Error')
plt.ylabel('Friquancy')
plt.grid(True)
plt.show()


plt.figure(figsize=(8, 6))

sns.scatterplot(x=preds, y=y_valid, alpha=0.6)

max_val = max(max(y_valid), max(preds))
min_val = min(min(y_valid), min(preds))
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Perfect Prediction')

plt.xlabel('Predicted Calories')
plt.ylabel('Actual Calories')
plt.title('Predictions vs. True Values')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()


from xgboost import plot_importance

plt.figure(figsize=(10, 6))
plot_importance(model, max_num_features=10)
plt.title("Top 10 most important features")
plt.show()


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer


def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, np.maximum(0, y_pred)))

rmsle_scorer = make_scorer(rmsle, greater_is_better=False)


xgb = XGBRegressor(random_state=42)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.05, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring=rmsle_scorer,
    cv=3,
    verbose=2,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best RMSLE:", -grid_search.best_score_)


best_model = grid_search.best_estimator_


val_preds = best_model.predict(X_valid)
val_rmsle = rmsle(y_valid, val_preds)
print(f"Validation RMSLE: {val_rmsle:.5f}")


test_submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')


test_preds = best_model.predict(test)


submission = pd.DataFrame({
    'id': test_submission['id'],  
    'Calories': test_preds
})

submission.to_csv('submission.csv', index=False)

print("✅ Submission is done!")

