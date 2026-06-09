!pip install -q xgboost==3.0.0


import numpy as np # linear algebra
import seaborn as sns
import pandas as pd
import time, os, gc, random, warnings, math
import seaborn as sb
import matplotlib.pyplot as plt
import xgboost as xgb
import numpy as np
import itertools
from itertools import combinations
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer 
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from catboost import CatBoostRegressor, Pool
from catboost import CatBoostClassifier, Pool

warnings.filterwarnings('ignore')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


file_path = '../input/playground-series-s5e5/train.csv'
test_path = '../input/playground-series-s5e5/test.csv'

data = pd.read_csv(file_path) 
test_data = pd.read_csv(test_path) 
sample_submission = pd.read_csv('../input/playground-series-s5e5/sample_submission.csv')

def preprocessing(df, train):
    df['Sex'] = df['Sex'].map({'male': 1, 'female': 2})
    df['BMI'] = df['Weight'] / (df['Height'] / 100) ** 2

    numerical_cols = ['Age', 'Sex', 'Weight', 'Height', 'Heart_Rate', 'Body_Temp', 'Duration'] 

    combination_orders=[1, 2, 3, 4]
    for order in combination_orders:
        if order == 1:
            for col in numerical_cols:
                df[f"{col}_log"] = np.log1p(df[col])
        else:
            for cols_tuple in combinations(numerical_cols, order):
                product_val = 1
                product_feature_name_parts = []
                for col in cols_tuple:
                    product_val *= df[col]
                    product_feature_name_parts.append(col)
                df[f"{'_m_'.join(product_feature_name_parts)}"] = np.log1p(product_val)
                if order >= 2: # Division makes sense for at least 2 columns
                    numerator_col = cols_tuple[0]
                    denominator_product = 1
                    denominator_feature_name_parts = []
                    for col_idx in range(1, order): # Start from the second column
                        denominator_product *= df[cols_tuple[col_idx]]
                        denominator_feature_name_parts.append(cols_tuple[col_idx])
                    denominator = denominator_product + 1e-5
                    df[f"{numerator_col}_d_{'_d_'.join(denominator_feature_name_parts)}"] = np.log1p(df[numerator_col] / denominator)

    for col in ['Height', 'Weight', 'BMI', 'Heart_Rate', 'Body_Temp', 'Duration']:
        for agg in ['min', 'max']:
            for group in ['Sex']:
                g = df.groupby(group)[col].agg(agg).rename(f'{group}_{col}_{agg}')
                df = df.merge(g, on=group, how='left')
    
    columns_to_drop = ['Weight_log', 'Age_log', 'Duration_log', 'Body_Temp_log', 'Heart_Rate_log', 'Height_log', 'Sex_log', 'Sex', 'Weight_d_Body_Temp_d_Duration', 'Weight_m_Height_m_Duration', 'Sex_d_Height', 'Sex_d_Weight_d_Height_d_Heart_Rate', 'Sex_d_Weight_d_Height_d_Duration', 'Age_m_Weight_m_Height_m_Body_Temp', 'Age_m_Weight_m_Body_Temp', 'Age_m_Height_m_Body_Temp_m_Duration', 'Weight_m_Height_m_Heart_Rate_m_Body_Temp', 'Age_m_Height_m_Duration', 'Age_m_Weight_m_Body_Temp_m_Duration', 'Sex_d_Weight_d_Height_d_Body_Temp', 'Age_d_Sex_d_Height_d_Heart_Rate', 'Sex_d_Height_d_Body_Temp', 'Age_m_Weight_m_Duration', 'Age_d_Weight_d_Duration', 'Weight_m_Heart_Rate_m_Body_Temp_m_Duration', 'Sex_m_Duration', 'Sex_d_Weight_d_Body_Temp_d_Duration', 'Age_d_Weight_d_Heart_Rate', 'Sex_d_Weight_d_Height', 'Sex_d_Weight_d_Duration', 'Age_d_Sex_d_Heart_Rate_d_Duration', 'Age_d_Height_d_Heart_Rate_d_Duration', 'Age_d_Weight_d_Heart_Rate_d_Duration', 'Age_m_Weight_m_Height', 'Age_d_Weight_d_Heart_Rate_d_Body_Temp', 'Age_d_Weight_d_Body_Temp', 'Age_d_Body_Temp_d_Duration', 'Age_m_Height', 'Age_m_Height_m_Body_Temp', 'Weight_d_Duration', 'Sex_d_Height_d_Duration', 'Age_d_Sex_d_Weight_d_Body_Temp', 'Weight_m_Height_m_Body_Temp_m_Duration', 'Sex_m_Height_m_Body_Temp_m_Duration', 'Sex_m_Weight_m_Height_m_Heart_Rate', 'Weight_m_Body_Temp_m_Duration', 'Height_m_Body_Temp_m_Duration', 'Height_m_Duration', 'Age_d_Weight_d_Height_d_Heart_Rate', 'Age_d_Weight_d_Height']
    df = df.drop(columns=columns_to_drop, errors='ignore')
                
    if train:
        df.drop_duplicates(subset=df.columns, keep='first').reset_index(drop=True)
        if 'id' in df.columns:
            df.drop(columns=['id'], inplace=True)
        if 'User_ID' in df.columns:
            df.drop(columns=['User_ID'], inplace=True)  
    return df

data = preprocessing(data, True)
test_data = preprocessing(test_data, False)

columns_to_remove = [col for col in data.columns if col.startswith("Sex") or col.startswith("Calories")]
df_cleaned = data.copy().drop(columns=columns_to_remove)
features = df_cleaned.columns

plt.subplots(figsize=(15, 3 * math.ceil(len(features))))
for i, col in enumerate(features):
    plt.subplot(math.ceil(len(features)), 3, i + 1)
    x = data.sample(1000)
    sb.scatterplot(x=col, y='Calories', data=x)
plt.tight_layout()
plt.show()

data.head()


y = data.Calories
X = data.copy().drop(columns='Calories')
test_X = test_data.copy()

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.005, random_state=4)

model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=500, 
    max_depth=10,
    learning_rate=0.03, 
    random_state=4,
    n_jobs=-1,
    early_stopping_rounds = 100
) 

model.fit(X_train, np.log1p(y_train),
          eval_set=[(X_val, np.log1p(y_val))],
          verbose=100) 

val_predictions_log = model.predict(X_val)
val_predictions_original_scale = np.expm1(val_predictions_log)
val_predictions_original_scale[val_predictions_original_scale < 0] = 0

rmsle = np.sqrt(mean_squared_error(np.log1p(y_val), np.log1p(val_predictions_original_scale)))

print("\nValidation RMSLE for XGBoost Model: {:,.5f}".format(rmsle)) #0.0544


importance_types = ['weight', 'gain', 'cover', 'total_gain', 'total_cover']
# all_feature_ranks = pd.DataFrame(index=X_train.columns) 
for importance_type in importance_types:
    feature_importances = model.get_booster().get_score(importance_type=importance_type)
    importance_series = pd.Series(feature_importances)
    sorted_importances = importance_series.sort_values(ascending=False)
    importance_df = pd.DataFrame({
        'Feature': sorted_importances.index,
        'Importance': sorted_importances.values
    })
    plt.figure(figsize=(10, 20))
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
    plt.title('XGBoost Feature Importance (Type: '+importance_type+')', fontsize=16)
    plt.xlabel('Importance (Number of times used in a tree)', fontsize=12)
    plt.ylabel('Feature', fontsize=6)
    plt.tight_layout()
    plt.show()


trained_features = [col for col in test_X.columns if col != 'id']
test_X_for_prediction = test_X[trained_features]
test_preds_log = model.predict(test_X_for_prediction)

test_preds_original_scale = np.expm1(test_preds_log)
test_preds_final = test_preds_original_scale.clip(min=0)

sample_submission['Calories'] = test_preds_final
submission_filename = 'XGBoost_submission.csv' # Example: Use a name reflecting the model
sample_submission.to_csv(submission_filename, index=False)
print(f"Submission file saved as '{submission_filename}'")

