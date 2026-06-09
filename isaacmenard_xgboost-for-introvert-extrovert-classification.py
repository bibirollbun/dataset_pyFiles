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
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier


warnings.filterwarnings('ignore')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


file_path = '../input/playground-series-s5e7/train.csv'
test_path = '../input/playground-series-s5e7/test.csv'

data = pd.read_csv(file_path) 
test_data = pd.read_csv(test_path) 
sample_submission = pd.read_csv('../input/playground-series-s5e7/sample_submission.csv')
data.head()


data = pd.read_csv(file_path) 
test_data = pd.read_csv(test_path) 

def preprocessing(df, train):
    df = df.replace([np.nan, -np.inf], 1)
    if(train):
        df['Personality'] = df['Personality'].replace({'Introvert': 0, 'Extrovert': 1}).astype(float)
    df['Stage_fear'] = df['Stage_fear'].replace({'Yes': 0.1, 'No': 1}).astype(float)
    df['Time_spent_Alone'] = df['Time_spent_Alone'].replace({0: 0.1}).astype(float)
    df['Post_frequency'] = df['Post_frequency'].replace({0: 0.1}).astype(float)
    df['Drained_after_socializing'] = df['Drained_after_socializing'].replace({'No': 1, 'Yes': 0.1}).astype(float)

    # numerical_cols = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency'] 

    # combination_orders=[1, 2, 3]
    # for order in combination_orders:
    #     if order == 1:
    #         for col in numerical_cols:
    #             df[f"{col}_log"] = np.log1p(df[col])
    #     else:
    #         for cols_tuple in combinations(numerical_cols, order):
    #             product_val = 1
    #             product_feature_name_parts = []
    #             for col in cols_tuple:
    #                 product_val *= df[col]
    #                 product_feature_name_parts.append(col)
    #             df[f"{'_m_'.join(product_feature_name_parts)}"] = np.log1p(product_val)
    #             if order >= 2: # Division makes sense for at least 2 columns
    #                 numerator_col = cols_tuple[0]
    #                 denominator_product = 1
    #                 denominator_feature_name_parts = []
    #                 for col_idx in range(1, order): # Start from the second column
    #                     denominator_product *= df[cols_tuple[col_idx]]
    #                     denominator_feature_name_parts.append(cols_tuple[col_idx])
    #                 denominator = denominator_product + 1e-5
    #                 df[f"{numerator_col}_d_{'_d_'.join(denominator_feature_name_parts)}"] = np.log1p(df[numerator_col] / denominator)
                
    if train:
        df.drop_duplicates(subset=df.columns, keep='first').reset_index(drop=True)
        if 'id' in df.columns:
            df.drop(columns=['id'], inplace=True)
        if 'User_ID' in df.columns:
            df.drop(columns=['User_ID'], inplace=True)  
    return df

data = preprocessing(data, True)
test_data = preprocessing(test_data, False)
features = data.copy().columns

# plt.subplots(figsize=(15, 3 * math.ceil(len(features))))
# for i, col in enumerate(features):
#     plt.subplot(math.ceil(len(features)), 3, i + 1)
#     x = data.sample(1000)
#     sb.scatterplot(x=col, y='Personality', data=x)
# plt.tight_layout()
# plt.show()

data.head()


y = data['Personality']
X = data.drop(columns='Personality')
test_X = test_data.copy()

# Split the data
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.0005, random_state=4, stratify=y
)

# 3. Train XGBClassifier
model = XGBClassifier(
    n_estimators=255,
    max_depth=10,
    learning_rate=0.03,
    random_state=4,
    n_jobs=-1,
    use_label_encoder=False,
    eval_metric='logloss'
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10000, verbose=100)

# Predict on validation set
val_predictions = model.predict(X_val)

# Calculate accuracy
accuracy = accuracy_score(y_val, val_predictions)
print("\nValidation Accuracy for XGBoost Classifier: {:.5f}".format(accuracy))


# Get only features used in training
trained_features = [col for col in test_X.columns if col != 'id']
test_X_for_prediction = test_X[trained_features]

# Predict class labels (0 or 1)
test_preds = model.predict(test_X_for_prediction)
test_preds_rounded = np.round(test_preds).astype(int)
test_preds_mapped = ['Introvert' if pred == 0 else 'Extrovert' for pred in test_preds_rounded]
preds_df = pd.DataFrame({'Personality': test_preds_mapped})

# Plot the distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=preds_df, x='Personality', palette='pastel')
plt.title('Distribution of Predicted Personality Types')
plt.ylabel('Count')
plt.xlabel('Personality')
plt.tight_layout()
plt.show()

# Create submission file
sample_submission['Personality'] = test_preds_mapped
submission_filename = 'XGBoost_submission.csv'
sample_submission.to_csv(submission_filename, index=False)

print(f"Submission file saved as '{submission_filename}'")

