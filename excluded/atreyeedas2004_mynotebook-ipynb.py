import pandas as pd

# ✅ Use the correct dataset folder path
train = pd.read_csv('/kaggle/input/burnout-datathon-ieeecsmuj/train.csv')
test = pd.read_csv('/kaggle/input/burnout-datathon-ieeecsmuj/test.csv')
sample = pd.read_csv('/kaggle/input/burnout-datathon-ieeecsmuj/sample_submission.csv')

# Preview data
print("Train shape:", train.shape)
print("Test shape:", test.shape)
train.head()


# Check missing values
missing = train.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
print("Missing values in train set:\n", missing)

# Optional: check % of missing
missing_percent = (train.isnull().sum() / len(train)) * 100
missing_percent[missing_percent > 0].sort_values(ascending=False)
print("Percentage of Missing values in train set:\n",missing_percent)



# Check type and unique values
print("Penalty dtype:", train['Penalty'].dtype)
print("Unique values in Penalty:\n", train['Penalty'].unique())



# Fill missing Penalty values
train['Penalty'] = train['Penalty'].fillna('No Penalty')
test['Penalty'] = test['Penalty'].fillna('No Penalty')
val['Penalty'] = val['Penalty'].fillna('No Penalty')


cat_cols = train.select_dtypes(include='object').columns.tolist()
print("Categorical columns:", cat_cols)


from sklearn.preprocessing import OneHotEncoder

# Define columns to encode
low_card_cols = [
    'category_x', 'Track_Condition', 'Tire_Compound_Front',
    'Tire_Compound_Rear', 'Penalty', 'Session', 'weather'
]

# One-hot encode with pandas (for simplicity)
train_encoded = pd.get_dummies(train, columns=low_card_cols, drop_first=True)
test_encoded = pd.get_dummies(test, columns=low_card_cols, drop_first=True)
val_encoded = pd.get_dummies(val, columns=low_card_cols, drop_first=True)

# Align the columns so train/test have the same features
train_encoded, test_encoded = train_encoded.align(test_encoded, join='left', axis=1, fill_value=0)
train_encoded, val_encoded = train_encoded.align(val_encoded, join='left', axis=1, fill_value=0)



from sklearn.preprocessing import LabelEncoder

# Encode high-cardinality categorical variables
label_cols = ['rider_name', 'team_name', 'bike_name', 'circuit_name']

# Use LabelEncoder for each
for col in label_cols:
    le = LabelEncoder()
    # Fit on combined train/test/val to avoid unseen labels
    combined = pd.concat([train_encoded[col], test_encoded[col], val_encoded[col]], axis=0).astype(str)
    le.fit(combined)
    train_encoded[col] = le.transform(train_encoded[col].astype(str))
    test_encoded[col] = le.transform(test_encoded[col].astype(str))
    val_encoded[col] = le.transform(val_encoded[col].astype(str))

# Drop low-importance columns
train_encoded = train_encoded.drop(columns=['shortname', 'track'], errors='ignore')
test_encoded = test_encoded.drop(columns=['shortname', 'track'], errors='ignore')
val_encoded = val_encoded.drop(columns=['shortname', 'track'], errors='ignore')



import numpy as np

# Define features and target
target = 'Lap_Time_Seconds'
features = [col for col in train_encoded.columns if col != target and col != 'Unique ID']

X_train = train_encoded[features]
y_train = train_encoded[target]

X_val = val_encoded[features]
y_val = val_encoded[target]


pip install --upgrade lightgbm


import lightgbm
print(lightgbm.__version__)



from lightgbm import LGBMRegressor, early_stopping, log_evaluation

model = LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=7,
    random_state=42,
    objective='regression'
)



model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[
        early_stopping(stopping_rounds=50),
        log_evaluation(period=100)
    ]
)


from sklearn.metrics import mean_squared_error
# Predict on validation set
y_pred = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print("Validation RMSE:", rmse)


import matplotlib.pyplot as plt
import pandas as pd

# Get feature importances
importances = model.feature_importances_
feature_names = X_train.columns
feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False)

# Plot top 20
plt.figure(figsize=(12, 6))
plt.barh(feat_imp_df['Feature'].head(20)[::-1], feat_imp_df['Importance'].head(20)[::-1])
plt.xlabel("Importance")
plt.title("Top 20 Feature Importances")
plt.tight_layout()
plt.show()



test.head()


#  Define features (excluding target and ID)
features = [col for col in test_encoded.columns if col != 'Unique ID' or col!='Lap_Time_Seconds']



# Ensure test has same features and order as train
X_test = test_encoded[features]  # 'features' comes from train_encoded

# Align test to training columns, fill any missing with 0
X_test_aligned = X_test.reindex(columns=X_train.columns, fill_value=0)



# Predict using aligned test data
test_preds = model.predict(X_test_aligned)

# Create solution file
solution_df = pd.DataFrame({
    'Unique ID': test_encoded['Unique ID'],
    'Lap_Time_Seconds': test_preds
})

solution_df.to_csv('solution.csv', index=False)

print("✅ created solution.csv")


