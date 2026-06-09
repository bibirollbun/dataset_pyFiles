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


import warnings
warnings.filterwarnings('ignore')


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')


train_df = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")


import seaborn as sns
import matplotlib.pyplot as plt

# Select only numerical columns (skip categorical/string data)
numerical_cols = train_df.select_dtypes(include=['int64', 'float64']).columns

# Adjust figure size dynamically based on number of plots
n_cols = len(numerical_cols)
plt.figure(figsize=(15, 4 * n_cols))  # Wider to accommodate labels

# Loop through numerical columns
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(n_cols, 1, i)  # Arrange plots vertically
    sns.boxplot(x=train_df[col])
    plt.title(f'Boxplot of {col}', fontsize=12)
    plt.tight_layout()  # Prevent overlapping labels

plt.show()


import pandas as pd
import numpy as np

def detect_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers

# Identify outliers for all numerical columns
outliers_dict = {col: detect_outliers(train_df, col) for col in numerical_cols}

# Print outlier counts
for col, outliers in outliers_dict.items():
    print(f"{col}: {len(outliers)} outliers ({(len(outliers)/len(train_df))*100:.2f}%)")


def cap_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[column] = np.where(df[column] < lower, lower, 
                         np.where(df[column] > upper, upper, df[column]))
    return df

train_df_capped = train_df.copy()
for col in numerical_cols:
    train_df_capped = cap_outliers(train_df_capped, col)


train_df_capped.columns


def create_safe_features(df):
    """Calculate features without using Calories"""
    df = df.copy()

    # Body Mass Index (BMI)
    df['BMI'] = df['Weight'] / (df['Height']/100)**2

    # Body Surface Area (BSA)
    df['BSA'] = 0.007184 * (df['Height']**0.725) * (df['Weight']**0.425)

    # Weight-to-Height Ratio
    df['Weight_Height_Ratio'] = df['Weight'] / df['Height']
    
    # Basic transformations
    df['Max_HR_Estimate'] = 220 - df['Age']
    df['HR_Reserve'] = df['Heart_Rate'] / (220 - df['Age'])
    df['Temp_HR_Index'] = df['Body_Temp'] * df['Heart_Rate'] / 1000
    df['Age_Duration_Ratio'] = df['Age'] / df['Duration']
    df['Weighted_Duration'] = df['Duration'] * df['Weight'] / 100
    df['HR_squared'] = df['Heart_Rate'] ** 2
    df['Age_squared'] = df['Age']**2
    df['HR_Duration_Interaction'] = df['Heart_Rate'] * df['Duration']

    # Age groups
    df['Age_Group'] = pd.cut(df['Age'], 
                                  bins=[0,20,30,40,50,100], 
                                  labels=['Teen','20s','30s','40s','50+'])

    # Heart Rate Intensity
    df['HR_Zone'] = pd.cut(df['Heart_Rate'],
                                bins=[0,120,150,180,300],
                                labels=['Light','Moderate','Vigorous','Max'])
    
    # MET calculation without Calories
    df['MET_safe'] = (df['Heart_Rate'] * 0.0175) / df['Weight']
    
    # Estimated calories/min (hypothetical scaling)
    df['Calories_per_min_est'] = df['Heart_Rate'] * df['Weight'] * 0.0005
    
    # CV_Load 
    df['CV_Load'] = (df['Heart_Rate'] * df['Duration']) / (df['Age'] * df['Body_Temp'])

    df['HR_Duration_Temp'] = df['Heart_Rate'] * df['Duration'] * df['Body_Temp']
    df['HR_Weight_Ratio'] = df['Heart_Rate'] / df['Weight']
    df['HR_Change_Rate'] = df['Heart_Rate'] / df['Duration']
    
    
    return df


# Apply to your data
train_df_safe = create_safe_features(train_df_capped)


train_df_safe.columns


categorical_cols = train_df_safe.select_dtypes(include=['object', 'category']).columns.tolist()
print("Categorical columns to encode:", categorical_cols)


from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def encode_cat(df):
    # Copy the original DataFrame to preserve it
    df_encoded = df.copy()

    # A. Label Encoding for Ordinal Categories
    ordinal_cols = ['Age_Group', 'HR_Zone']  # If these exist
    le = LabelEncoder()
    for col in ordinal_cols:
        if col in df_encoded.columns:
            df_encoded[col+'_encoded'] = le.fit_transform(df_encoded[col])
            # Optional: Save the mapping for reference
            print(f"{col} mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

    # B. One-Hot Encoding for Nominal Categories
    nominal_cols = ['Sex']  # Add others if needed
    df_encoded = pd.get_dummies(
        df_encoded, 
        columns=nominal_cols, 
        prefix=nominal_cols,
        drop_first=True  # Avoid dummy variable trap
    )

    # C. Drop original categorical columns (optional)
    cols_to_drop = categorical_cols
    df_encoded = df_encoded.drop(columns=cols_to_drop, errors='ignore')
    return df_encoded


train_df_encoded = encode_cat(train_df_safe)


cols = [col for col in train_df_encoded.columns if col != 'Calories'] + ['Calories']
train_df_encoded = train_df_encoded[cols]
train_df_encoded.head()


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from skopt import BayesSearchCV
from sklearn.metrics import make_scorer
from xgboost import XGBRegressor  # Changed from LGBMRegressor
from scipy.stats import randint, uniform
from sklearn.metrics import mean_squared_log_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Selected features (unchanged)
selected_features = [
    'Age', 'Height', 'Weight',
    'Duration', 'Heart_Rate', 'Body_Temp', 
    'HR_Reserve', 'Temp_HR_Index', 'Age_Duration_Ratio',
    'Weighted_Duration', 'HR_squared', 'HR_Duration_Interaction',
    'CV_Load' ,'MET_safe', 'Calories_per_min_est',
    'BSA', 'BMI', 'Weight_Height_Ratio', 'Max_HR_Estimate', 'Age_Group_encoded' ,'Sex_male',
    'HR_Duration_Temp', 'HR_Weight_Ratio', 'HR_Change_Rate'
]
target = 'Calories'

# 1. Data Preparation (unchanged)
X = train_df_encoded[selected_features]
y = train_df_encoded[target]

# Train-test split (unchanged)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train-validation split (unchanged)
X_train_sub, X_val, y_train_sub, y_val = train_test_split(
    X_train, y_train, 
    test_size=0.2, 
    random_state=42
)

# 2. KFold (unchanged)
from sklearn.model_selection import KFold
cv = KFold(n_splits=3, shuffle=True, random_state=42)

# 3. Subsample (unchanged)
sample_size = min(100000, len(X_train_sub))
tune_idx = np.random.choice(X_train_sub.index, size=sample_size, replace=False)
X_tune = X_train_sub.loc[tune_idx]
y_tune = y_train_sub.loc[tune_idx]

# 4. RMSLE Scorer (unchanged)
def rmsle(y_true, y_pred):
    y_pred = np.clip(y_pred, 0, None)
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

rmsle_scorer = make_scorer(rmsle, greater_is_better=False)

# ----------------------------
# 5. XGBOOST TUNING (MODIFIED SECTION)
# ----------------------------
xgb_params = {
    'n_estimators': randint(100, 300),
    'max_depth': randint(3, 10),
    'learning_rate': uniform(0.01, 0.3),
    'subsample': uniform(0.6, 0.3),
    'colsample_bytree': uniform(0.6, 0.3),
    'gamma': uniform(0, 0.5),
    'reg_alpha': uniform(0, 1),
    'reg_lambda': uniform(0, 1)
}

# search = RandomizedSearchCV(
#     XGBRegressor(
#         random_state=42, 
#         tree_method='hist',  # Faster for large datasets
#         enable_categorical=True,  # If using categorical features
#     ),
#     param_distributions=xgb_params,
#     n_iter=25,
#     cv=cv,
#     scoring=rmsle_scorer,
#     random_state=42,
#     n_jobs=-1,
#     verbose=1
# )

search = BayesSearchCV(
    XGBRegressor(tree_method='hist', random_state=42),
    {
        'n_estimators': (100, 300),
        'learning_rate': (0.01, 0.3, 'log-uniform'),
        'max_depth': (3, 10)
    },
    n_iter=30,
    cv=cv,
    scoring=rmsle_scorer
)

# 6. Run the search (unchanged)
search.fit(X_tune, y_tune)

# 7. Retrain best model on full data with early stopping
best_xgb = XGBRegressor(
    **search.best_params_,
    random_state=42,
    tree_method='hist')

# You MUST provide validation data for early_stopping_rounds
best_xgb.fit(
    X_train, 
    y_train,
    eval_set=[(X_val, y_val)],  # Validation data required
    verbose=10  # Shows evaluation every 10 iterations
)

# 8. Evaluate
test_pred = best_xgb.predict(X_test)
print(f"Final Test RMSLE: {rmsle(y_test, test_pred):.4f}")

# ----------------------------
# 9. (NEW) Feature Importance Plot
# ----------------------------
plt.figure(figsize=(12, 8))
pd.Series(best_xgb.feature_importances_, index=selected_features)\
  .sort_values()\
  .plot(kind='barh', title='XGBoost Feature Importance')
plt.show()


# Compare train vs test performance
train_pred = best_xgb.predict(X_train)
test_pred = best_xgb.predict(X_test)

print(f"Train RMSLE: {rmsle(y_train, train_pred):.4f}") 
print(f"Test RMSLE: {rmsle(y_test, test_pred):.4f}")


test_df = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
print("Shape of Test Data", test_df.shape)
display(test_df.head(10))


# Apply to your data
test_df_safe = create_safe_features(test_df)


test_df_encoded = encode_cat(test_df_safe)


test_df_encoded.columns


test_pred = best_xgb.predict(test_df_encoded[selected_features])

# Clip negative predictions (if any)
test_pred = np.clip(test_pred, 0, None)

# Create submission DataFrame
submission = pd.DataFrame({
    'id': test_df['id'], 
    'Calories': test_pred
})

# Save predictions
submission.to_csv('submission.csv', index=False)
print("Predictions saved to submission.csv")


submission.head()

