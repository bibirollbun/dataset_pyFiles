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


df_train=pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
df_test=pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
sample_submission=pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")


df_train.head(5)


df_train.drop(columns=["id"], axis=1, inplace=True)
df_test.drop(columns=["id"], axis=1, inplace=True)


import seaborn as sns
import matplotlib.pyplot as plt
target_variable="Fertilizer Name"  
def eda_pipeline(df_train, df_test):
    
    # Display first few rows
    print("\n--- First few rows of train data ---")
    display(df_train.head())
    
    print("\n--- First few rows of test data ---")
    display(df_test.head())
    
    # Dataset info
    print("\n--- Train Data Info ---")
    print(df_train.info())
    
    print("\n--- Test Data Info ---")
    print(df_test.info())
    
    # Missing values
    print("\n--- Missing Values in Train Data ---")
    print(df_train.isnull().sum())
    
    print("\n--- Missing Values in Test Data ---")
    print(df_test.isnull().sum())
    
    print("\n--- Percentage of Missing Values in Train Data ---")
    print((df_train.isnull().sum() / len(df_train)) * 100)
    
    print("\n--- Percentage of Missing Values in Test Data ---")
    print((df_test.isnull().sum() / len(df_test)) * 100)
    
    # Summary statistics
    print("\n--- Train Data Summary Statistics ---")
    print(df_train.describe())
    
    print("\n--- Test Data Summary Statistics ---")
    print(df_test.describe())
    
    # Identify categorical columns
    train_cat_columns = [col for col in df_train.columns if df_train[col].dtype == 'O']
    test_cat_columns = [col for col in df_test.columns if df_test[col].dtype == 'O']
    
    print("\n--- Categorical Columns in Train Data ---")
    print(train_cat_columns)
    
    print("\n--- Unique Values in Categorical Columns (Train) ---")
    print(df_train[train_cat_columns].nunique())
    
    print("\n--- Categorical Columns in Test Data ---")
    print(test_cat_columns)
    
    print("\n--- Unique Values in Categorical Columns (Test) ---")
    print(df_test[test_cat_columns].nunique())
    
    # Identify numerical columns
    train_num_columns = [col for col in df_train.columns if df_train[col].dtype in ['int64', 'float64']]
    test_num_columns = [col for col in df_test.columns if df_test[col].dtype in ['int64', 'float64']]
    
    print("\n--- Numerical Columns in Train Data ---")
    print(train_num_columns)
    
    print("\n--- Numerical Columns in Test Data ---")
    print(test_num_columns)
    
    # Check for duplicate rows
    print("\n--- Duplicate Rows in Train Data ---")
    print(df_train.duplicated().sum())
    
    print("\n--- Duplicate Rows in Test Data ---")
    print(df_test.duplicated().sum())
    
    # Correlation matrix (excluding non-numeric columns)
    print("\n--- Correlation Matrix ---")
    plt.figure(figsize=(12, 6))
    sns.heatmap(df_train[train_num_columns].corr(), annot=True, cmap='coolwarm')
    plt.show()
       
    # Correlation with Target Variable
    # print("\n--- Correlation with Target Variable ---")
    # target_corr = df_train[train_num_columns].corr()[target_variable].sort_values(ascending=False)
    # print(target_corr)
    
    # plt.figure(figsize=(12, 6))
    # sns.barplot(x=target_corr.index, y=target_corr.values, palette='coolwarm')
    # plt.xticks(rotation=90)
    # plt.title(f'Feature Correlation with {target_variable}')
    # plt.show()   
    
    # Distribution plots for numerical features
    print("\n--- Distribution of Numerical Features ---")
    df_train[train_num_columns].hist(figsize=(12, 10), bins=30)
    plt.show()
    
    # Box plots for outlier detection
    print("\n--- Box Plots for Outlier Detection ---")
    for col in train_num_columns:
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=df_train[col])
        plt.title(f'Box plot of {col}')
        plt.show()
    
    # Value counts for categorical features
    print("\n--- Value Counts for Categorical Columns ---")
    for col in train_cat_columns:
        print(f"\nValue counts for {col}:")
        print(df_train[col].value_counts())
    
    
    


eda_pipeline(df_train, df_test)


# Target Distribution Check
print("\n--- Distribution of Target Variable for Class Balance Check ---\n")
df_train[target_variable].value_counts(normalize=True).plot(kind='barh')


from sklearn.preprocessing import LabelEncoder

def data_preprocessing_pipeline(df_train, df_test, target_column='Fertilizer Name'):
    """
    Preprocess the dataset by handling missing values and encoding categorical variables.
    Returns processed DataFrames and the label encoder for the target column.
    """
    # Fill missing values
    for column in df_train.columns:
        if df_train[column].dtype == 'object':
            mode_value = df_train[column].mode()[0]
            df_train[column].fillna(mode_value, inplace=True)
        elif df_train[column].dtype in ['int64', 'float64']:
            mean_value = df_train[column].mean()
            df_train[column].fillna(mean_value, inplace=True)
    
    for column in df_test.columns:
        if df_test[column].dtype == 'object':
            mode_value = df_test[column].mode()[0]
            df_test[column].fillna(mode_value, inplace=True)
        elif df_test[column].dtype in ['int64', 'float64']:
            mean_value = df_test[column].mean()
            df_test[column].fillna(mean_value, inplace=True)
    
    # Encode categorical features
    label_encoders = {}
    target_encoder = None  # separate encoder for target column

    for column in df_train.columns:
        if df_train[column].dtype == 'object':
            le = LabelEncoder()
            df_train[column] = le.fit_transform(df_train[column].astype(str))
            label_encoders[column] = le

            if column == target_column:
                target_encoder = le  # store encoder for target

    for column in df_test.columns:
        if df_test[column].dtype == 'object':
            if column in label_encoders:
                le = label_encoders[column]
                df_test[column] = df_test[column].apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
            else:
                df_test[column] = -1

    return df_train, df_test, target_encoder


df_train, df_test, target_le = data_preprocessing_pipeline(df_train, df_test, target_column='Fertilizer Name')


df_train.head(5)


from sklearn.preprocessing import StandardScaler

def standardize_data(df_train, df_test):
    """
    Standardize all numerical features using StandardScaler,
    ensuring both train and test have the same columns, while preserving the target variable.
    """
    # Separate target column from train data
    target_values = df_train[target_variable]
    df_train = df_train.drop(columns=[target_variable])
    
    # Ensure both datasets have the same feature columns
    common_columns = df_train.columns.intersection(df_test.columns)
    df_train = df_train[common_columns]
    df_test = df_test[common_columns]
    
    # Initialize StandardScaler
    scaler = StandardScaler()
    
    # Fit on train data and transform both train and test data
    df_train_scaled = pd.DataFrame(scaler.fit_transform(df_train), columns=common_columns)
    df_test_scaled = pd.DataFrame(scaler.transform(df_test), columns=common_columns)
    
    # Reattach the target column to the scaled train data
    df_train_scaled[target_variable] = target_values.reset_index(drop=True)
    
    return df_train_scaled, df_test_scaled


# df_train_scaled, df_test_scaled = standardize_data(df_train, df_test)


# **Label Encode Target**
# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
# df_train["Fertilizer Name"] = le.fit_transform(df_train["Fertilizer Name"])


X = df_train.drop(columns=[target_variable])
y = df_train[target_variable]


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier
# from catboost import CatBoostClassifier
# from lightgbm import LGBMClassifier
# from sklearn.metrics import accuracy_score

# # Initialize models
# rf_model = RandomForestClassifier(n_estimators=500, random_state=42)
# xgb_model = XGBClassifier(n_estimators=500, use_label_encoder=False, eval_metric='logloss', random_state=42)
# cat_model = CatBoostClassifier(n_estimators=500, verbose=0, random_state=42)
# lgbm_model = LGBMClassifier(n_estimators=500, random_state=42,verbose=-1)

# # Train all models
# rf_model.fit(X_train, y_train)
# xgb_model.fit(X_train, y_train)
# cat_model.fit(X_train, y_train)
# lgbm_model.fit(X_train, y_train)

# # Make predictions on X_test
# rf_pred = rf_model.predict(X_test)
# xgb_pred = xgb_model.predict(X_test)
# cat_pred = cat_model.predict(X_test)
# lgbm_pred = lgbm_model.predict(X_test)

# # Accuracy Score
# print("Accuracy Score rf is ",accuracy_score(y_test,rf_pred))
# print("Accuracy Score xgb is ",accuracy_score(y_test,xgb_pred))
# print("Accuracy Score catboost is ",accuracy_score(y_test,cat_pred))
# print("Accuracy Score lgbm is ",accuracy_score(y_test,lgbm_pred))


# cat_features = ['Soil Type', 'Crop Type']


from catboost import CatBoostClassifier
cat_model = CatBoostClassifier(n_estimators=500, verbose=100,random_state=42)
cat_model.fit(X_train, y_train)


from sklearn.metrics import accuracy_score
cat_pred = cat_model.predict(X_test)
print("Accuracy Score catboost is ",accuracy_score(y_test,cat_pred))


from xgboost import XGBClassifier
xgb_model = XGBClassifier(n_estimators=500, use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
print("Accuracy Score xgb is ",accuracy_score(y_test,xgb_pred))


from xgboost import plot_importance
import matplotlib.pyplot as plt

plot_importance(xgb_model, max_num_features=15)
plt.show()



from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score

# Initialize LGBM for multiclass classification
lgbm_model = LGBMClassifier(
    objective='multiclass',
    n_estimators=500,
    learning_rate=0.05,
    num_class=len(np.unique(y_train)),  # number of classes
    random_state=42, verbose=-1
)

# Train
lgbm_model.fit(X_train, y_train)

# Predict
lgbm_pred = lgbm_model.predict(X_test)

# Accuracy
print("Accuracy Score LGBM is", accuracy_score(y_test, lgbm_pred))



# Predict class probabilities for each test sample
# pred_probs = xgb_model.predict_proba(df_test)  # shape: (n_samples, n_classes)


# Predict probabilities from all models
probs_cat = cat_model.predict_proba(df_test)     
probs_xgb = xgb_model.predict_proba(df_test)
probs_lgb = lgbm_model.predict_proba(df_test)



pred_probs = (probs_cat + probs_xgb + probs_lgb) / 3


import numpy as np

def mapk(actual, predicted, k=3):
    """
    Computes the mean average precision at k.
    
    Parameters:
    - actual: list/array of true label indices (ints or labels)
    - predicted: list of lists of predicted indices/labels
    - k: maximum number of predictions to consider (default=3)
    
    Returns:
    - MAP@k score
    """
    def apk(a, p, k):
        """Average precision at k for a single prediction"""
        if a in p[:k]:
            return 1.0 / (p[:k].index(a) + 1)
        return 0.0
    
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])



# Get top 3 predicted class indices for each row
top3_idx = np.argsort(-pred_probs, axis=1)[:, :3]
flat = target_le.inverse_transform(top3_idx.ravel())
top3_names = flat.reshape(top3_idx.shape)


top3_names.shape,sample_submission.shape


# Ensure y_test is a NumPy array or Series
y_true = target_le.inverse_transform(y_test)


y_pred = top3_names.tolist()


score = mapk(y_true, y_pred, k=3)
print(f"MAP@3 Score: {score:.4f}")


sample_submission['Fertilizer Name'] = [' '.join(row) for row in top3_names]
sample_submission.to_csv('submission.csv', index=False)
print('Submission file saved.')




