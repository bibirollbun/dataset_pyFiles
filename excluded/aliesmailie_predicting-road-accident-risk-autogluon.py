!pip install sweetviz
!pip install vegafusion vegafusion-python-embed
!pip install vl-convert-python
!pip install ray==2.10.0
!pip install autogluon.tabular
!pip install -U ipywidgets


#system handling
import os
import time
import warnings
warnings.filterwarnings('ignore')

#data handling
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.preprocessing import StandardScaler, LabelEncoder
from autogluon.tabular import TabularDataset, TabularPredictor

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')


train.shape


test.shape


train.info()


train.dtypes


print("Target column statistics (accident_risk):")

train['accident_risk'].describe()


train.isnull().sum()


print("Duplicated Rows:",train.duplicated().sum())


train_num_cols = train.select_dtypes(include='number').columns.tolist()
train_num_cols.remove('id')
correlation_matrix = train[train_num_cols].corr()

plt.figure(figsize=(6,5))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numerical Features')
plt.show()


# from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

# nominalFeatures = ['road_type', 'weather']

# ohe = OneHotEncoder(drop='first', sparse_output=False)
# encoded_train = ohe.fit_transform(train[nominalFeatures])
# encoded_test = ohe.fit_transform(test[nominalFeatures])

# feature_names_dropped = ohe.get_feature_names_out(nominalFeatures)

# encoded_train_df = pd.DataFrame(encoded_train, columns=feature_names_dropped)
# encoded_test_df = pd.DataFrame(encoded_test, columns=feature_names_dropped)
# train = pd.concat([train.drop(columns=nominalFeatures), encoded_train_df], axis=1)
# test = pd.concat([test.drop(columns=nominalFeatures), encoded_test_df], axis=1)

# train.head()


# ordinal_features = ['lighting', 'time_of_day']

# oe = OrdinalEncoder()
# for feature in ordinal_features: 
#     train[feature] = oe.fit_transform(train[feature].values.reshape(-1,1))
#     test[feature] = oe.fit_transform(test[feature].values.reshape(-1,1))

# train.head()


# bool_features = ["road_signs_present", "public_road","holiday", "school_season"]
# for feature in bool_features :
#     train[feature] = train[feature].astype(int)
#     test[feature] = test[feature].astype(int)

# train.head()


# from sklearn.preprocessing import MinMaxScaler

# scaler = MinMaxScaler()
# num_cols = train.select_dtypes(include='number').columns.tolist()
# num_cols.remove('id')
# num_cols.remove('accident_risk')

# train[num_cols] = scaler.fit_transform(train[num_cols])
# test[num_cols] = scaler.fit_transform(test[num_cols])

# train.head()


CATEGORICAL_FEATURES = ['lighting', 'time_of_day']
BOOLEAN_FEATURES = ['road_signs_present', 'public_road', 'holiday', 'school_season']
NUMERICAL_FEATURES = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents']
TARGET = 'accident_risk'
ID_COL = 'id'


def engineer_features(df):
    df_eng = df.copy()
    
    # Core interactions
    df_eng['curv_speed'] = df_eng['curvature'] * df_eng['speed_limit']
    df_eng['lane_speed'] = df_eng['num_lanes'] * df_eng['speed_limit']
    df_eng['accidents_speed'] = df_eng['num_reported_accidents'] * df_eng['speed_limit']
    df_eng['accidents_curv'] = df_eng['num_reported_accidents'] * df_eng['curvature']
    
    # Polynomial features
    df_eng['curvature_sq'] = df_eng['curvature'] ** 2
    df_eng['curvature_cube'] = df_eng['curvature'] ** 3
    df_eng['speed_sq'] = df_eng['speed_limit'] ** 2
    
    # Risk scores
    df_eng['risk_intensity'] = (df_eng['curvature'] * df_eng['speed_limit']) / 50
    df_eng['lane_capacity_risk'] = (5 - df_eng['num_lanes']) * df_eng['speed_limit']
    df_eng['accidents_per_lane'] = df_eng['num_reported_accidents'] / (df_eng['num_lanes'] + 1)
    
    # Binary indicators
    df_eng['high_risk_combo'] = ((df_eng['curvature'] > 0.5) & 
                                  (df_eng['speed_limit'] >= 60)).astype(int)

    df_eng['meta_curvature'] = 0.3 * df_eng['curvature']
    df_eng['meta_night'] = 0.2 * (df_eng['lighting'] == 'night').astype(int)
    df_eng['meta_speed'] = 0.2 * (df_eng['speed_limit'] >= 60).astype(int)
    df_eng['meta_accidents'] = 0.1 * (df_eng['num_reported_accidents'] > 2).astype(int)
    df['meta_weather'] = 0.1 * (df['weather'] != 'clear').astype(int)

    df_eng['weather_lighting'] = df_eng['weather'].astype(str) + '_' + df_eng['lighting'].astype(str)

    
    return df_eng

# Preprocessing
train_processed = train.copy()
test_processed = test.copy()

# Convert booleans
for col in BOOLEAN_FEATURES:
    train_processed[col] = train_processed[col].astype(int)
    test_processed[col] = test_processed[col].astype(int)

# Label encode categoricals
label_encoders = {}
for col in CATEGORICAL_FEATURES:
    le = LabelEncoder()
    train_processed[f'{col}_enc'] = le.fit_transform(train_processed[col])
    test_processed[f'{col}_enc'] = le.transform(test_processed[col])
    label_encoders[col] = le

# Apply feature engineering
train = engineer_features(train_processed)
test = engineer_features(test_processed)

print(f"Feature engineering complete")
print(f"Original features: {len(CATEGORICAL_FEATURES + BOOLEAN_FEATURES + NUMERICAL_FEATURES)}")
print(f"Engineered features: {train.shape[1]}")
print(f"New features created: {test.shape[1] - train_processed.shape[1]}")


train['log_curvature'] = np.log1p(train['curvature'])
test['log_curvature'] = np.log1p(test['curvature'])


def f(X):
    return \
    0.3 * X["curvature"] + \
    0.2 * (X["lighting"] == "night").astype(int) + \
    0.1 * (X["weather"] != "clear").astype(int) + \
    0.2 * (X["speed_limit"] >= 60).astype(int) + \
    0.1 * (X["num_reported_accidents"] > 2).astype(int)


    
train['meta'] = f(train)
test['meta'] = f(test)


train.head()


y = train['accident_risk']
X = train.drop(['accident_risk', 'id'], axis=1, errors='ignopre')
X_test = test.drop('id', axis=1, errors='ignore')

# AutoGluon expects the target in the same dataframe
train = X.copy()
train['accident_risk'] = y

print('Shape of Train data is : ' , train.shape)
print('Shape of Test data is : ' , X_test.shape)


train.columns


X_test.columns



train_ag = TabularDataset(train)
X_test_ag = TabularDataset(X_test)
target = 'accident_risk'

predictor_main = TabularPredictor(label=target, eval_metric ='rmse', 
                            problem_type="regression").fit(train_ag, 
                                                           presets='best_quality',
                                                           # presets = 'extreme',
                                                           # auto_stack = True,
                                                           time_limit=3600*9.3,
                                                           verbosity=3,
                                                           excluded_model_types=['KNN',
                                                                                 'NN_FASTAI',
                                                                                 'NN_TORCH',
                                                                                 'LinearModel',
                                                                                 'XT'
                                                                                ],
                                                           ag_args_fit={'num_gpus': 2}
                                                          )




# Source directory
# import shutil

# source_dir = "/kaggle/input//accidentprediction13/AutogluonModels/ag-20251023_182907"

# # Destination directory
# destination_dir = "/kaggle/working/accidentprediction13/AutogluonModels/ag-20251023_182907"

# shutil.copytree(source_dir, destination_dir)



# predictor_main = TabularPredictor.load('/kaggle/working/accidentprediction13/AutogluonModels/ag-20251023_182907')


results = predictor_main.fit_summary()
print(results)


predictor_main.fit_summary()


predictor_main.leaderboard()



y_pred = predictor_main.predict(X_test)



# Feature importance
importances = predictor_main.feature_importance(train_ag)
print("Feature importances:")
print(importances.head(14))


# Plot feature importances

plt.figure(figsize=(12, 10))
sns.barplot(
    x=importances['importance'],
    y=importances.index,
    palette='viridis'
)
plt.title('Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()



submission = pd.DataFrame({
    "id": submission.id,          
    "accident_risk": y_pred    
})


submission.to_csv("submission.csv", index=False)
print("submission dataset saved!")

