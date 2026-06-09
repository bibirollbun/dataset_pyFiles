import pandas as pd
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

test_to_submit = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")
train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv", index_col='id')
sample = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')


target = 'diagnosed_diabetes'


# X_train contains your features, and y_train your labels (0s and 1s)
X_train = train.drop(columns=[target])
y_train = train[target]

# --- 1. Define numerical and categorical columns ---
# Adjust these lists according to the actual columns in your DataFrame 'train'
target = 'diagnosed_diabetes'
numeric_features = [col for col in train.select_dtypes(exclude=['object','category','bool']).columns.tolist() if col != target]
categorical_features = [col for col in train.select_dtypes(include=['object','category','bool']).columns.tolist() if col != target]

# Remove the target variable 'y' if it is still in X_train
# (Ensure that your feature lists do not include the column 'is_fraud' or similar)

# --- 2. Create preprocessing steps (Pipeline) ---

# Pipeline for numerical features: just scale
numeric_transformer = StandardScaler()

# Pipeline for categorical features: One-Hot Encoding
# handle_unknown='ignore' handles categories that may appear in future data (test/production)
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Use ColumnTransformer to apply transformations to the correct columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'  # Keeps other unspecified columns if there are any
)

# --- 3. Apply preprocessing to your data ---

# This creates a transformed and scaled numpy array
X_train_processed = preprocessor.fit_transform(X_train)

# --- 4. Implement Isolation Forest ---

# Define the contamination rate (expected proportion of outliers)
# Adjust this value (e.g. 0.01 = 1% outliers, 0.05 = 5% outliers)
contamination_rate = 0.05

# Initialize the Isolation Forest model
# n_estimators is the number of trees
iso_forest = IsolationForest(
    n_estimators=100,
    contamination=contamination_rate,
    random_state=42,
    n_jobs=-1  # Use all CPU cores for faster execution
)

# Train the model (it’s unsupervised, only uses X)
iso_forest.fit(X_train_processed)

# Predict the anomaly labels:
# y_pred_outliers will be -1 for outliers and 1 for inliers (normal points)
y_pred_outliers = iso_forest.predict(X_train_processed)

# --- 5. Clean the original DataFrame ---

# Create a boolean mask to select only the inliers (label 1)
mask_inliers = y_pred_outliers == 1

# Filter the original features X and labels Y
# The new DataFrame 'train_cleaned' is what you will use for XGBoost
train_cleaned = train[mask_inliers]

print(f"Original dimensions of the DataFrame: {train.shape}")
print(f"Dimensions of the cleaned DataFrame (only inliers): {train_cleaned.shape}")
print(f"Number of outliers detected and removed: {np.sum(y_pred_outliers == -1)}")



#In this way, I can create the correct encoding without missing labels
#for test_to_submit

test_to_submit[target] = 99 #99 ist a dummy value for me
combined_df = pd.concat([train_cleaned, test_to_submit], axis=0)
combined_df = combined_df.drop(columns=['id'])

combined_df_indices = combined_df[combined_df[target].isna()].index.tolist()


nunique = combined_df.nunique()
types = combined_df.dtypes

categorical_columns = []
categorical_dims =  {}
label_encoders = {}

for col in combined_df.columns:
    if types[col] == 'object' or nunique[col] < 200:
        print(col, combined_df[col].nunique())
        l_enc = LabelEncoder()
        combined_df[col] = combined_df[col].fillna("VV_likely")
        combined_df[col] = l_enc.fit_transform(combined_df[col].values)
        categorical_columns.append(col)
        categorical_dims[col] = len(l_enc.classes_)
        label_encoders[col] = l_enc
    else:
        combined_df.fillna(combined_df.loc[combined_df_indices, col].mean(), inplace=True)


#Label encoding is tranformed 99 in 2
train_cleaned = combined_df[combined_df[target]!=2] 
test_to_submit = combined_df[combined_df[target]==2]


target = 'diagnosed_diabetes'

X = train_cleaned.drop(target, axis=1)  
y = train_cleaned[target]  

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)

X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


from xgboost import XGBClassifier

clf_xgb = XGBClassifier(max_depth=8,
    learning_rate=0.1,
    n_estimators=1000,
    verbosity=0,
    silent=None,
    objective='binary:logistic',
    booster='gbtree',
    n_jobs=-1,
    nthread=None,
    gamma=0,
    min_child_weight=1,
    max_delta_step=0,
    subsample=0.7,
    colsample_bytree=1,
    colsample_bylevel=1,
    colsample_bynode=1,
    reg_alpha=0,
    reg_lambda=1,
    scale_pos_weight=1,
    base_score=0.5,
    random_state=0,
    seed=None,)

clf_xgb.fit(X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        early_stopping_rounds=40,
        verbose=10)


preds_xgb_valid = np.array(clf_xgb.predict_proba(X_valid))
valid_auc = roc_auc_score(y_score=preds_xgb_valid[:,1], y_true=y_valid)
print(f'Roc auc of valid data: {valid_auc}')

preds_xgb_test = np.array(clf_xgb.predict_proba(X_test))
test_auc = roc_auc_score(y_score=preds_xgb_test[:,1], y_true=y_test)
print(f'Roc auc of test data: {test_auc}')


# Retrieve feature importance from the trained model
importance = clf_xgb.get_booster().get_score(importance_type='weight')

# Create a DataFrame for feature importance
importance_df = pd.DataFrame(importance.items(), columns=['Feature', 'Importance'])

# Sort the DataFrame by importance
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot using Pandas
importance_df.set_index('Feature')['Importance'].plot(kind='barh', figsize=(12, 8), color='skyblue')
plt.xlabel('Importance')
plt.title('Feature Importance from XGBoost Model')
plt.show()


X_submit = test_to_submit.drop(columns=[target])
y_pred_submit = np.array(clf_xgb.predict_proba(X_submit))

sample[target] = y_pred_submit[:, 1]
sample.to_csv('test_xgb_e512.csv', index=False)
sample.head()

