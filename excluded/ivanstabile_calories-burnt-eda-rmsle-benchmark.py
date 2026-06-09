import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_log_error

import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv", index_col='id')
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv", index_col='id')


# inspect the basic structure of the datasets
print(f"Train shape is: {train.shape}")
print(f"Test shape is: {test.shape}")


train.head()


train.info()


# check for missing values
print("\nMissing values:\n", train.isnull().sum())


# summary statistics for numeric columns
print("\nDescriptive statistics:\n", train.describe())


# set aesthetic style for plots
sns.set(style="whitegrid")

numerical_cols = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Calories']
for col in numerical_cols:
    plt.figure(figsize=(6, 3))
    sns.histplot(train[col], kde=True, bins=30, color='skyblue')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


# plot categorical column
if 'Sex' in train.columns:
    sex_counts = train['Sex'].value_counts()
    labels = sex_counts.index
    sizes = sex_counts.values
    colors = ['#66b3ff', '#ff9999']  

    plt.figure(figsize=(5, 5))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    plt.title('Sex Distribution')
    plt.axis('equal')  
    plt.tight_layout()
    plt.show()


for col in numerical_cols[:-1]:
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=train[col], y=train['Calories'], alpha=0.5)
    plt.title(f'Calories vs {col}')
    plt.xlabel(col)
    plt.ylabel("Calories")
    plt.tight_layout()
    plt.show()


# correlation Heatmap
train_encoded = train.copy()

if 'Sex' in train.columns:
    train_encoded['Sex'] = train_encoded['Sex'].map({'female': 0, 'male': 1})

plt.figure(figsize=(8, 6))
sns.heatmap(train_encoded.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap (with Encoded Sex)")
plt.show()


for col in numerical_cols:
    plt.figure(figsize=(4, 3))
    sns.boxplot(y=train[col], color='orange')
    plt.title(f'Boxplot of {col}')
    plt.ylabel(col)
    plt.tight_layout()
    plt.show()


# encode categorical variable
train['Sex'] = train['Sex'].map({'female': 0, 'male': 1})
test['Sex'] = test['Sex'].map({'female': 0, 'male': 1})


# features & target
target = 'Calories'
features = [col for col in train.columns if col != target]

X = train[features]
y = train[target]
X_test = test.copy()


# train/validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# scale numeric columns
num_cols = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_val_scaled = X_val.copy()
X_test_scaled = X_test.copy()

X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
X_val_scaled[num_cols] = scaler.transform(X_val[num_cols])
X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])


# evaluation metric
def rmsle(y_true, y_pred):
    y_pred = np.maximum(y_pred, 0)
    return np.sqrt(mean_squared_log_error(y_true, y_pred))


models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

results = {}
predictions = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_val_scaled)
    y_pred = np.maximum(y_pred, 0)  # Ensure no negative predictions
    score = rmsle(y_val, y_pred)
    results[name] = score
    predictions[name] = y_pred
    print(f"{name} RMSLE: {score:.5f}")


plt.figure(figsize=(10, 6))
plt.bar(results.keys(), results.values(), color='cornflowerblue')
plt.xticks(rotation=30)
plt.ylabel("RMSLE (Lower is Better)")
plt.title("Model Comparison on Validation Set (RMSLE)")
plt.grid(axis='y')
plt.tight_layout()
plt.show()


best_model_name = min(results, key=results.get)
best_model = models[best_model_name]

X_scaled_full = pd.concat([X_train_scaled, X_val_scaled])
y_full = pd.concat([y_train, y_val])
best_model.fit(X_scaled_full, y_full)

test_preds = best_model.predict(X_test_scaled)
test_preds = np.maximum(test_preds, 0)

submission = pd.DataFrame({
    "id": test.index,
    "Calories": test_preds
})
submission_file = f"submission_{best_model_name.replace(' ', '_').lower()}.csv"
submission.to_csv(submission_file, index=False)
print(f"âœ… Best model: {best_model_name} â€” Submission saved as {submission_file}")

