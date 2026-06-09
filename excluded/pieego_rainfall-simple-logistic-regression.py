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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='whitegrid')

import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score, precision_score

from sklearn.linear_model import LogisticRegression


train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')


train.info()


test.info()


train.describe().T


train.head()


miss_train = train.isnull().sum()
miss_percent_train = miss_train / len(train)
unique_train = train. nunique()

miss_test = test.isnull().sum()
miss_percent_test = miss_test / len(test)
unique_test = test.nunique()

summary_train = pd.DataFrame(
    data ={
    "Train_Missing_Values": miss_train.values,
    "Train_%_Missing_Values": miss_percent_train.values,
    "Train_Unique": unique_train.values,
    'Feature': train.columns})

summary_test = pd.DataFrame(
    data ={
    "Test_Missing_Values": miss_test.values,
    "Test_%_Missing_Values": miss_percent_test.values,
    "Test_Unique": unique_test.values,
    'Feature': test.columns})

combined_df = pd.merge(summary_train, summary_test, how='left', on='Feature')
combined_df.set_index('Feature', drop=True, inplace=True)
combined_df


test['winddirection'] = test.winddirection.fillna(test.winddirection.mean())


train[train.duplicated()]


test[test.duplicated()]


def get_numeric_features(df: pd.DataFrame, columns: list) -> list:
    num_features = [col for col in columns if pd.api.types.is_numeric_dtype(df[col])]
    return num_features

def get_category_features(df: pd.DataFrame, columns: list) -> list:
    cat_features = [col for col in columns if pd.api.types.is_categorical_dtype(df[col])]
    return cat_features
    
def get_object_features(df: pd.DataFrame, columns: list) -> list:
    obj_features = [col for col in columns if pd.api.types.is_object_dtype(df[col])]
    return obj_features


num_features = get_numeric_features(train, train.columns)
num_features.remove('id')
num_features.remove('day')


def create_numeric_plots(feature):
    
    if feature not in train.columns or feature not in test.columns:
        return None
    
    combined_df = pd.concat(objs=(train.assign(Dataset='Train'), test.assign(Dataset='Test')), axis=0).reset_index(drop=True)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,4))
    COLORS_PALLETE = ['#EFDFBB', '#722F37']
    
    ax = sns.histplot(data=combined_df,
                    x = feature,
                    kde=True,
                    bins=30,
                    hue='Dataset',
                    palette=COLORS_PALLETE,
                    ax=axes[0])
    ax.set_title(f'Distribution of {feature}')
    ax.set_ylabel('Frequency')
    ax.set_xlabel(f'{feature}'.capitalize())
    
    ax = sns.boxplot(data=combined_df,
                     palette=COLORS_PALLETE,
                     x = feature,
                     y = 'Dataset',
                     ax=axes[1])
    
    ax.set_title(f'BoxPlot of {feature}')
    ax.set_ylabel('Dataset')
    ax.set_xlabel(f'{feature}'.capitalize())
    
    plt.tight_layout()

for numf in num_features:
    create_numeric_plots(numf)


def iqr_method(df: pd.DataFrame, feature: str) -> int:
    q1 = np.quantile(df[feature], .25)
    me = df[feature].median()
    q3 = np.quantile(df[feature], .75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outliers_cond = (df[feature] < lower_bound) | (df[feature] > upper_bound)
    
    outliers = df.loc[outliers_cond, feature]    
    return outliers, lower_bound, upper_bound


outliers_data = []
outliers_idx = {}

for numf in num_features:
    if numf not in ('id', 'day', 'rainfall'):
        outliers, lower_bound, upper_bound = iqr_method(train, numf)
        num_outliers = len(outliers)
        percent_outliers = num_outliers / len(train) * 100
        
        outliers_idx[numf] = outliers.index
        
        # Append row data
        outliers_data.append({'Feature': numf, 'Outliers': num_outliers, 'Outlier_Percent': percent_outliers})

# Convert list of dicts to DataFrame
outliers_df = pd.DataFrame(outliers_data)

outliers_df


for feature, idx in outliers_idx.items():
    print(train.loc[idx, [feature, 'rainfall']].head(10))


corr_matrix = round(train.corr(), 2)
annot_matrix = corr_matrix.copy().astype(str)
np.fill_diagonal(annot_matrix.values, train.columns) 

plt.figure(figsize=(16, 12))
sns.heatmap(corr_matrix, annot=annot_matrix, fmt="", cmap="mako", linewidths=0.5)
plt.title("Correlation Heatmap with Feature Names on Diagonal")


import statsmodels.api as sm

X = train.drop(['id', 'day', 'rainfall'], axis=1)
y = train['rainfall']

X = sm.add_constant(X)

model = sm.Logit(y, X).fit()

influence = model.get_influence()

leverage = influence.hat_matrix_diag
standardized_residuals = influence.resid_studentized

dffits_values = standardized_residuals*np.sqrt(leverage / (1 - leverage))
k = X.shape[1]
n = X.shape[0]

threshold = 3*np.sqrt(( k + 1)/ (n-k-1))

influential_points = np.where(np.abs(dffits_values) > threshold)[0]
print("Influential observations (Standardized Residuals):", influential_points)

train = train.drop(influential_points, axis=0)


train['high_cloud_humidity'] = ((train['humidity'] > 80) & (train['cloud'] > 60)).astype(int)
train['cloud_sunshine_ratio'] = train['cloud'] / (train['sunshine'] + 1)
train['humidity_sunshine_ratio'] = train['humidity'] / (train['sunshine'] + 1)
train['low_sunshine'] = (train['sunshine'] < 6).astype(int)
train['temperature'] = train[['mintemp', 'temparature', 'maxtemp']].mean(axis=1)
train['cloud_windspeed'] = train['cloud'] * train['windspeed']


train['humidity_previous_day'] = train['humidity'].shift(1).fillna(0)
train['pressure_previous_day'] = train['pressure'].shift(1).fillna(0)

test['humidity_previous_day'] = test['humidity'].shift(1).fillna(0)
test['pressure_previous_day'] = test['pressure'].shift(1).fillna(0)


test['high_cloud_humidity'] = ((test['humidity'] > 80) & (test['cloud'] > 60)).astype(int)
test['cloud_sunshine_ratio'] = test['cloud'] / (test['sunshine'] + 1)
test['humidity_sunshine_ratio'] = test['humidity'] / (test['sunshine'] + 1)
test['low_sunshine'] = (test['sunshine'] < 6).astype(int)
test['temperature'] = test[['mintemp', 'temparature', 'maxtemp']].mean(axis=1)
test['cloud_windspeed'] = test['cloud'] * test['windspeed']


train = train.drop(['mintemp', 'temparature', 'maxtemp'], axis=1)
test = test.drop(['mintemp', 'temparature', 'maxtemp'], axis=1)


def assign_season(day: int):
    if not ( 1 <= day <= 366):
        return None
    
    if 80 <= day < 172: return 'spring'
    elif 172 <= day < 266: return 'summer'
    elif 266 <= day < 355: return 'autumn'
    else : return 'winter'

train['season'] = train['day'].map(assign_season).astype('category')
test['season'] = test['day'].map(assign_season).astype('category')

def get_month_from_day(day: int):
    if not (1 <= day <= 366):
        return "Invalid day. Must be between 1 and 366."
    
    if 1 <= day <= 31:
        return "January"
    if 32 <= day <= 60:
        return "February"
    if 61 <= day <= 91:
        return "March"
    if 92 <= day <= 121:
        return "April"
    if 122 <= day <= 152:
        return "May"
    if 153 <= day <= 182:
        return "June"
    if 183 <= day <= 213:
        return "July"
    if 214 <= day <= 244:
        return "August"
    if 245 <= day <= 274:
        return "September"
    if 275 <= day <= 305:
        return "October"
    if 306 <= day <= 335:
        return "November"
    if 336 <= day <= 366:
        return "December"
    
    return "Error: Day out of range"

train['month'] = train['day'].map(get_month_from_day).astype('category')
test['month'] = test['day'].map(get_month_from_day).astype('category')


from scipy.stats import skew, kurtosis
new_num_features = get_numeric_features(train, train.columns)
new_num_features.remove('rainfall')

skewness = {col:skew(train[col]) for col in new_num_features}
kurtosis_value = {col:kurtosis(train[col]) for col in new_num_features}

print(pd.DataFrame({'Skewness': skewness, 'Kurtosis': kurtosis_value}))
    
features_to_transform = [col for col in new_num_features if abs(skewness[col]) > 0.5 or kurtosis_value[col] > 3]
features_to_transform


for feature in features_to_transform:
    train[f"{feature}_log"] = np.log1p(train[feature])
    test[f"{feature}_log"] = np.log1p(test[feature])


train = train.drop(features_to_transform, axis=1)
test = test.drop(features_to_transform, axis=1)


train = pd.get_dummies(train, columns=['season','month'], drop_first=True, dtype=int)
test = pd.get_dummies(test, columns=['season', 'month'], drop_first=True, dtype=int)


scaler = StandardScaler()
columns_to_scale = [col for col in train.columns if not (col.startswith("season_") or  col.startswith("month_") or col in ('rainfall', 'id'))]
train[columns_to_scale] = scaler.fit_transform(train[columns_to_scale])
test[columns_to_scale] = scaler.transform(test[columns_to_scale])


strat_kfold = StratifiedKFold(shuffle=True, random_state=42)
models = {
    'Logistic Regression': (LogisticRegression(solver='liblinear'), {
        'penalty': ['l1', 'l2'],
        'C': np.linspace(1e-2, 10, 100),
        'max_iter': [100, 500, 1000]
    })
}   
 
best_models = {}

# Prepare dataset
X = train.drop(['rainfall', 'id'], axis=1)
y = train['rainfall']
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Iterate over models and perform hyperparameter tuning
for model_name, (model, params) in models.items():
    print(f'\nTraining {model_name}...')
    grid_search = GridSearchCV(estimator=model, param_grid=params, cv=strat_kfold, verbose=1, scoring='neg_log_loss')
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    best_models[model_name] = best_model
    y_pred = best_model.predict(X_valid)
    y_pred_proba = best_model.predict_proba(X_valid)[:, 1]
    
    # Evaluate model
    accuracy = accuracy_score(y_valid, y_pred)
    f1 = f1_score(y_valid, y_pred, average='weighted')
    recall = recall_score(y_valid, y_pred, average='weighted')
    precision = precision_score(y_valid, y_pred, average='weighted')
    auc = roc_auc_score(y_valid, y_pred_proba)
    
    print(f'Best parameters for {model_name}: {grid_search.best_params_}')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'F1-score: {f1:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'AUC: {auc:.4f}')



def plot_cost_function_vs_C(model, param_grid, X_train, y_train, strat_kfold, scoring='neg_log_loss'):

    if 'C' in param_grid:
        grid_search = GridSearchCV(estimator=model, param_grid={'C': param_grid['C']}, 
                                   cv=strat_kfold, scoring=scoring, verbose=1, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        C_values = param_grid['C']
        mean_scores = grid_search.cv_results_['mean_test_score']
        
        if scoring == 'neg_log_loss':
            mean_scores = -mean_scores

        sns.set_style("whitegrid")
        plt.figure(figsize=(10, 5))
        plt.plot(C_values, mean_scores, marker='o', linestyle='dashed', color='b')
        plt.xscale('log')  # Log scale for better visualization
        plt.xlabel("C (Regularization Parameter)")
        plt.ylabel("Cost Function Value (Log Loss)")
        plt.title(f"Cost Function vs. C for {model.__class__.__name__}")
        plt.show()

plot_cost_function_vs_C(LogisticRegression(), models['Logistic Regression'][1], X_train, y_train, strat_kfold)



best_model = best_models['Logistic Regression']
X = test.drop('id', axis=1)
y_pred = best_model.predict_proba(X)[:, 1]
submission = pd.DataFrame({"id": test.id,
                           "rainfall": y_pred})
submission.to_csv('submission.csv', index=False)




