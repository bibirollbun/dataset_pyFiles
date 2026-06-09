!pip install scikit-learn==1.5.0


!pip install -U imbalanced-learn



# Import required libraries
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Load the training data
train_df = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')

# Load the test data
test_df = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')

# Load the sample submission file
submission_df = pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')

# Preview the first few rows of train dataset
print(train_df.head())


 #Check basic info
train_df.info()

# Check for missing values
print("Missing values in each column:")
print(train_df.isnull().sum())
print(train_df.duplicated())


import seaborn as sns
import matplotlib.pyplot as plt
import importlib
importlib.reload(plt)
sns.countplot(x='y', data=train_df)
plt.title('Target Variable Distribution')
plt.xlabel('Subscribed (y)')
plt.ylabel('Count')
plt.show()



import matplotlib.pyplot as plt
import seaborn as sns
sns.histplot(train_df['age'],bins=30,kde=True)
plt.title("Age Distribution")
plt.show()


# Define age bins and labels
bins = [20, 30, 40, 50, 60, 70, 80, 90]
labels = ['20â€“30', '30â€“40', '40â€“50', '50â€“60', '60â€“70', '70â€“80', '80â€“90']

# Create a new column for age group
train_df['age_group'] = pd.cut(train_df['age'], bins=bins, labels=labels, right=False)

# Plot churn count by age group
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(data=train_df, x='age_group', hue='y')
plt.title('Churn Rate by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.legend(title='Churn')
plt.xticks(rotation=45)
plt.show()



sns.countplot(x='job',hue='y',data=train_df)
plt.title('Subscription by Job Type')
plt.xticks(rotation=45)
plt.show()



sns.countplot(x='housing',hue='y',data=train_df)
plt.title('Housing Loan vs Subscription')
plt.show()


sns.boxplot(data=train_df,x='y',y='balance')
plt.title=('Balance Distribution by Churn')
plt.xlabel('Churn')
plt.ylabel('Balance')
plt.show()


import pandas as pd
from sklearn.preprocessing import StandardScaler

# 1ï¸�âƒ£ Start with a fresh copy
df = train_df.copy()

# 2ï¸�âƒ£ Encode target variable ('y': yes/no â†’ 1/0)
df['y'] = df['y'].map({'yes': 1, 'no': 0})

# 3ï¸�âƒ£ Binary encoding for 'default', 'housing', 'loan'
binary_cols = ['default', 'housing', 'loan']
for col in binary_cols:
    df[col] = df[col].map({'yes': 1, 'no': 0})

# 4ï¸�âƒ£ One-hot encode multi-class categorical features
multi_cat_cols = ['job', 'marital', 'education', 'contact', 'month', 'poutcome']
df = pd.get_dummies(df, columns=multi_cat_cols, drop_first=True)

# 5ï¸�âƒ£ Create new interaction features
df['balance_to_duration'] = df['balance'] / (df['duration'] + 1)
df['previous_contacts'] = df['campaign'] + df['previous']
df['recent_contact'] = df['pdays'].apply(lambda x: 1 if x < 30 else 0)

# 6ï¸�âƒ£ Drop irrelevant column
df.drop('id', axis=1, inplace=True)

# 7ï¸�âƒ£ Save unscaled version for interpretation
df_unscaled = df.copy()

# 8ï¸�âƒ£ Scale numeric features for modeling
scaler = StandardScaler()
scaled_cols = ['balance', 'duration', 'campaign', 'pdays', 'previous']
df[scaled_cols] = scaler.fit_transform(df[scaled_cols])

# 9ï¸�âƒ£ Reverse scaling for human-readable columns
for i, col in enumerate(scaled_cols):
    df[f'{col}_original'] = df[col] * scaler.scale_[i] + scaler.mean_[i]

# âœ… Final check
# Compare column counts
print("ğŸ§¾ Original column count:", train_df.shape[1])
print("ğŸ§ª Engineered column count:", df.shape[1])

# Show first few rows before and after
print("\nğŸ“‚ Original data sample:")
display(train_df.head())

print("\nğŸ§ª Engineered data sample:")
# Show clean version for interpretation
df_unscaled.head()




# ğŸ“¦ Import libraries
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE

# ğŸ§¹ Sample and preprocess data
df_sample = train_df.sample(frac=0.3, random_state=42)

# ğŸ�¯ Select key features
selected_features = ['age', 'balance', 'duration', 'campaign', 'job', 'marital', 'education']
X = df_sample[selected_features]
y = df_sample['y']

# ğŸ”¢ Encode categorical features
X_encoded = pd.get_dummies(X, drop_first=True)

# âš–ï¸� Apply SMOTE to balance classes
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_encoded, y)

# ğŸ”€ Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# ğŸš€ Train XGBoost with tuned parameters
model = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    scale_pos_weight=1,  # SMOTE balances classes, so no need to adjust this
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)
model.fit(X_train, y_train)

# ğŸ“ˆ Make predictions
y_pred = model.predict(X_test)

# ğŸ“Š Evaluate model
print("âœ… Accuracy:", accuracy_score(y_test, y_pred))
print("\nğŸ“Š Classification Report:\n", classification_report(y_test, y_pred))
print("\nğŸ§® Confusion Matrix:\n", confusion_matrix(y_test, y_pred))



# ğŸ”� Traditional Feature Importance (XGBoost)
import importlib
importlib.reload(plt)
importances = model.feature_importances_
features = X_encoded.columns

# Plot
plt.figure(figsize=(8, 5))
sns.barplot(x=importances, y=features)
plt.title('XGBoost Feature Importance')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.tight_layout()
plt.show()



from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
# Get probabilities
y_scores = model.predict_proba(X_test)[:, 1]

# Precision-recall
precision, recall, _ = precision_recall_curve(y_test, y_scores)
avg_precision = average_precision_score(y_test, y_scores)

# Plot
plt.figure(figsize=(6, 4))
plt.plot(recall, precision, label=f'Avg Precision = {avg_precision:.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Resize the plot
plt.figure(figsize=(8, 6))  # Width=10, Height=6
shap.summary_plot(shap_values, X_test, plot_size=None, show=False)

plt.tight_layout()
plt.show()




from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Define the model
model = RandomForestClassifier(random_state=42, n_jobs=-1)

# Simplified hyperparameter space
param_dist = {
    'n_estimators': [50, 100, 150],  # Lower values for faster training
    'max_depth': [10, 20],           # Removed None and 30
    'min_samples_split': [2, 5],     # Fewer options
    'min_samples_leaf': [1, 2],      # Fewer options
    'bootstrap': [True]              # Removed False to reduce complexity
}

# Set up RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=model,
                                   param_distributions=param_dist,
                                   n_iter=10,  # Reduced from 20
                                   cv=3,
                                   scoring='accuracy',
                                   verbose=2,
                                   random_state=42,
                                   n_jobs=-1)

# Fit to training data
random_search.fit(X_train, y_train)

# Best result
print("Best Parameters:", random_search.best_params_)
print("Best Accuracy:", random_search.best_score_)



from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Select features (excluding 'id' and 'y')
selected_features = [col for col in train_df.columns if col not in ['id', 'y']]

# Encode training data
X_train = pd.get_dummies(train_df[selected_features], drop_first=True)
y_train = train_df['y']

# Train final model using best parameters
best_params = random_search.best_params_
final_model = RandomForestClassifier(**best_params, random_state=42)
final_model.fit(X_train, y_train)

# âœ… Filter selected_features to only those present in test_df
safe_features = [col for col in selected_features if col in test_df.columns]

# Encode test data
X_test_encoded = pd.get_dummies(test_df[safe_features], drop_first=True)

# âœ… Align test columns with training columns
X_test_encoded = X_test_encoded.reindex(columns=X_train.columns, fill_value=0)

# Predict churn probabilities (class 1 = churn)
y_pred_proba = final_model.predict_proba(X_test_encoded)[:, 1]

# Create submission file with correct format
submission = pd.DataFrame({
    'id': test_df['id'],
    'Exited': y_pred_proba
})
submission.to_csv('submission.csv', index=False)



submission.head()



import joblib

# Save the model
joblib.dump(final_model, 'random_forest_model.pkl')


