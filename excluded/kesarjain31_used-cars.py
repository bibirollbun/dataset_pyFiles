# ==============================
# Step 1: Import Libraries
# ==============================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# ==============================
# Step 2: Load Datasets
# ==============================
train = pd.read_csv('/kaggle/input/playground-series-s4e9/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s4e9/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s4e9/sample_submission.csv')

# ==============================
# Step 3: Handle Missing / Infinite Values
# ==============================
train = train.replace([np.inf, -np.inf], np.nan).fillna('missing')
test = test.replace([np.inf, -np.inf], np.nan).fillna('missing')

# ==============================
# Step 4: Identify Target Column
# ==============================
target = 'price'
X = train.drop(target, axis=1)
y = train[target]

# Drop ID column if exists
if 'id' in X.columns:
    X = X.drop('id', axis=1)
    test = test.drop('id', axis=1)

# ==============================
# Step 5: Encode Categorical Columns
# ==============================
label_encoders = {}
for col in X.select_dtypes(include='object').columns:
    le = LabelEncoder()
    combined = pd.concat([X[col], test[col]], axis=0).astype(str)
    le.fit(combined)
    X[col] = le.transform(X[col].astype(str))
    test[col] = le.transform(test[col].astype(str))
    label_encoders[col] = le

# ==============================
# Step 6: Train-Test Split
# ==============================
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# ==============================
# Step 7: Train RandomForest Regressor (Fast)
# ==============================
model = RandomForestRegressor(n_estimators=50, max_depth=12, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# ==============================
# Step 8: Evaluate Model
# ==============================
val_preds = model.predict(X_valid)
mae = mean_absolute_error(y_valid, val_preds)
print(f"ðŸ“ˆ Validation MAE: {mae:.2f}")

# ==============================
# Step 9: Manual Example Prediction (Fixed)
# ==============================
example = pd.DataFrame({
    'model_year': [2018],
    'milage': [35000],
    'engine': [1197],
    'fuel_type': ['Petrol'],
    'transmission': ['Manual'],
    'brand': ['Toyota'],
    'model': ['Corolla'],
    'ext_col': ['good'],
    'int_col': ['clean'],
    'accident': ['No'],
    'clean_title': ['Yes']
})

# Add missing columns
for col in X_train.columns:
    if col not in example.columns:
        example[col] = 0

# Encode categorical safely
for col in example.select_dtypes(include='object').columns:
    if col in label_encoders:
        le = label_encoders[col]
        example[col] = example[col].apply(lambda x: x if x in le.classes_ else 'missing')
        if 'missing' not in le.classes_:
            le.classes_ = np.append(le.classes_, 'missing')
        example[col] = le.transform(example[col].astype(str))

# Reorder columns to match training data
example = example[X_train.columns]

# Predict
example_pred = model.predict(example)
print(f"ðŸš— Example predicted {target}: {example_pred[0]:.2f}")

# ==============================
# Step 10: Feature Importance & Visualizations
# ==============================
# Target distribution
plt.figure(figsize=(8,4))
sns.histplot(train[target], bins=40, kde=True, color='teal')
plt.title("Distribution of Target Prices")
plt.xlabel("Price")
plt.ylabel("Count")
plt.show()

# Correlation heatmap
plt.figure(figsize=(10,6))
corr = train.corr(numeric_only=True)
sns.heatmap(corr, cmap='YlGnBu', annot=False)
plt.title("Feature Correlation Heatmap")
plt.show()

# Top 10 features correlated with target
top_corr = corr[target].sort_values(ascending=False)[1:11]
plt.figure(figsize=(8,4))
sns.barplot(x=top_corr.values, y=top_corr.index, palette='magma')
plt.title("Top 10 Features Correlated with Target")
plt.xlabel("Correlation Value")
plt.show()

# RandomForest feature importance
importances = pd.Series(model.feature_importances_, index=X_train.columns)
plt.figure(figsize=(10,5))
importances.sort_values(ascending=False)[:15].plot(kind='bar', color='orange')
plt.title("Top 15 Important Features")
plt.ylabel("Importance Score")
plt.show()

# ==============================
# Step 11: Predict Test Data & Create Submission
# ==============================
test_preds = model.predict(test)
submission = sample_submission.copy()
submission[submission.columns[-1]] = test_preds
submission.to_csv('/kaggle/working/submission.csv', index=False)
print("âœ… Submission file created!")
print(submission.head())


