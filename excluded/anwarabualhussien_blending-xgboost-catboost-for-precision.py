from IPython.display import display, HTML

display(HTML("""
<div style="text-align: center;">
  <img src="https://raw.githubusercontent.com/ABUALHUSSEIN/kaggle-diabetes-prediction/refs/heads/main/DiabetesPrediction%20.png" width="1000">
</div>
"""))


import warnings
warnings.simplefilter('ignore')


import pandas as pd, numpy as np, gc
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
sns.set_palette("magma")
train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")
print('Train Shape:', train.shape)
print('Test Shape:', test.shape)



# 2. Basic Shape & Structure
print("--- Data Dimensions ---")
print(f"Number of Patients (Rows): {train.shape[0]}")
print(f"Number of Features (Columns): {train.shape[1]}")


train.head(10)


# 3. Check for Missing Values & Data Types
print("\n--- Data Health Check (Missing Values) ---")
missing_data = train.isnull().sum()
print(missing_data[missing_data > 0] if missing_data.any() else "No missing values found! The data is clean.")



# 4. Statistical Summary of Numeric Features
print("\n--- Summary Statistics ---")
display(train.describe().T)


# 5. Visualizing the Target Variable: diagnosed_diabetes
plt.figure(figsize=(8, 5))
ax = sns.countplot(x='diagnosed_diabetes', data=train)
plt.title('Target Distribution: How many patients have diabetes?', fontsize=15)
plt.xlabel('Diagnosed Diabetes (0 = No, 1 = Yes)')
plt.ylabel('Count')

# Add percentages on top of bars
total = len(train)
for p in ax.patches:
    percentage = f'{100 * p.get_height() / total:.1f}%'
    x = p.get_x() + p.get_width() / 2 - 0.05
    y = p.get_height()
    ax.annotate(percentage, (x, y), size=12, fontweight='bold')

plt.show()


# Step 2: The "Digital Diagnostic" - Identifying Numeric Red Flags

# 1. We officially drop 'id' from our analysis as it contains no medical value
train_numeric = train.drop(columns=['id'], errors='ignore')

# 2. Calculate correlations focusing only on numbers
correlations = train_numeric.corr(numeric_only=True)['diagnosed_diabetes'].sort_values(ascending=False)

# 3. Create a clean, professional visualization
plt.figure(figsize=(12, 8))

# We drop 'diagnosed_diabetes' so we don't compare the target to itself (which is always 1.0)
top_indicators = correlations.drop('diagnosed_diabetes')

# Color coding: Red for Risk Factors, Green for Protective Factors
colors = ['#e76f51' if x > 0 else '#2a9d8f' for x in top_indicators.values]

top_indicators.plot(kind='barh', color=colors)

plt.title('Clinical Risk Factors: What Drives Diabetes?', fontsize=18, fontweight='bold')
plt.xlabel('Correlation Strength (Pearson)', fontsize=12)
plt.ylabel('Health Metric', fontsize=12)
plt.axvline(x=0, color='black', linewidth=0.8) # Add a center line
plt.grid(axis='x', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()

# Print summary in a readable way
print("ğŸš¨ TOP RISK FACTORS (As these go UP, Diabetes risk increases):")
print(top_indicators[top_indicators > 0].head(5))

print("\nğŸ›¡ï¸� TOP PROTECTIVE FACTORS (As these go UP, Diabetes risk decreases):")
print(top_indicators[top_indicators < 0].sort_values().head(5))


#  Visualizing Categorical "Risk Profiles"

categorical_cols = ['gender', 'smoking_status', 'ethnicity', 'education_level']

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for i, col in enumerate(categorical_cols):
    # Calculate the percentage of diabetes for each category
    prop_data = train.groupby(col)['diagnosed_diabetes'].value_counts(normalize=True).unstack()
    
    prop_data.plot(kind='bar', stacked=True, ax=axes[i], color=['#264653', '#e76f51'], alpha=0.8)
    axes[i].set_title(f'Diabetes Rate by {col.replace("_", " ").title()}', fontsize=14)
    axes[i].set_ylabel('Proportion')
    axes[i].legend(['Healthy', 'Diabetes'], loc='upper right')
    axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45)

plt.tight_layout()
plt.show()




from sklearn.preprocessing import LabelEncoder, StandardScaler

# Make a copy so we don't mess up the original data
df_processed = train.drop(columns=['id'])

# 1. Encoding Categorical Variables (Turning words into numbers)
le = LabelEncoder()
cat_cols = ['gender', 'ethnicity', 'education_level', 'income_level', 
            'smoking_status', 'employment_status']

for col in cat_cols:
    df_processed[col] = le.fit_transform(df_processed[col].astype(str))

# 2. Feature Engineering: "The Healthy Lifestyle Score"
# Let's create a creative feature that combines exercise and diet
df_processed['lifestyle_balance'] = (df_processed['physical_activity_minutes_per_week'] / 100) + df_processed['diet_score']

# 3. Scaling Numeric Features
# This makes sure all numbers are on a similar scale (0 to 1 range roughly)
scaler = StandardScaler()
numeric_features = df_processed.drop(columns=['diagnosed_diabetes']).columns
df_processed[numeric_features] = scaler.fit_transform(df_processed[numeric_features])

print("âœ… Data Prepared!")
print(f"New Feature created: 'lifestyle_balance'")
display(df_processed.head())


from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# 1. Splitting the data
X = df_processed.drop(columns=['diagnosed_diabetes'])
y = df_processed['diagnosed_diabetes']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training on {len(X_train)} patients...")

# 2. Setting up our AI Doctor (XGBoost)
model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)

# 3. Training the model
model.fit(X_train, y_train)

# 4. Checking the results
val_probs = model.predict_proba(X_val)[:, 1]
final_score = roc_auc_score(y_val, val_probs)

print("--- Training Complete ---")
print(f"ROC-AUC Score: {final_score:.4f}")


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Get the importance scores from the model
importances = model.feature_importances_
feature_names = X.columns

# 2. Organize them into a DataFrame
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# 3. Plot the top 15 features
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(15), palette='viridis')
plt.title('The Decision Makers: Which Features does the AI trust most?', fontsize=16)
plt.xlabel('Importance Score (How much the model relies on this)')
plt.ylabel('Health Indicator')
plt.show()

# Print the top 3 most important features
print("--- The Model's Top 3 Evidence Pieces ---")
print(feature_importance_df.head(3))


!pip install catboost # Run this if you don't have it installed

from catboost import CatBoostClassifier

# 1. We go back to the original text data for CatBoost (it loves words!)
X_cb = train.drop(columns=['id', 'diagnosed_diabetes'])
y_cb = train['diagnosed_diabetes']

# Tell the model which columns are "words"
cat_features = ['gender', 'ethnicity', 'education_level', 'income_level', 
                'smoking_status', 'employment_status']

# 2. Split again
X_train_cb, X_val_cb, y_train_cb, y_val_cb = train_test_split(X_cb, y_cb, test_size=0.2, random_state=42)

# 3. Initialize CatBoost (The "Expert" Doctor)
cb_model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    eval_metric='AUC',
    random_state=42,
    verbose=100
)

# 4. Train
cb_model.fit(X_train_cb, y_train_cb, eval_set=(X_val_cb, y_val_cb), cat_features=cat_features)

# 5. Check the new score
cb_probs = cb_model.predict_proba(X_val_cb)[:, 1]
new_score = roc_auc_score(y_val_cb, cb_probs)

print(f"\nğŸš€ New CatBoost ROC-AUC Score: {new_score:.4f}")


# --- THE CORRECTED DIAGNOSTIC FACE-OFF ---

# 1. Get XGBoost Predictions correctly
# We use the X_val we created in Step 5 (which is already encoded and scaled)
xgb_val_probs = model.predict_proba(X_val)[:, 1]

# 2. Get CatBoost Predictions correctly
# We use the X_val_cb we created in Step 7 (the raw text version)
cb_val_probs = cb_model.predict_proba(X_val_cb)[:, 1]

# 3. Create the Ensemble (The "Averaged" Opinion)
# This is where the magic happens!
ensemble_val_probs = (xgb_val_probs + cb_val_probs) / 2

# 4. Calculate the Scores
xgb_score = roc_auc_score(y_val, xgb_val_probs)
cb_score = roc_auc_score(y_val, cb_val_probs)
ensemble_score = roc_auc_score(y_val, ensemble_val_probs)

# 5. Display the Results
comparison_df = pd.DataFrame({
    'Model Type': ['Single XGBoost (Manual Encoding)', 'Single CatBoost (Native Handling)', 'ğŸš€ ENSEMBLE (The Consensus)'],
    'ROC-AUC Score': [xgb_score, cb_score, ensemble_score]
}).sort_values(by='ROC-AUC Score', ascending=False)

print("--- ğŸ©º THE FINAL DIAGNOSTIC FACE-OFF ---")
display(comparison_df)

# Explain the "Alignment" trick
print("\nğŸ“� PRO-TIP: Feature Alignment")
print("We ensured that each model received the data in the exact format it was trained on.")
print("The Ensemble now shows the true combined strength of both AI Doctors!")


import glob
from sklearn.preprocessing import LabelEncoder

# 1. Locate and load the test data
test_file_path = glob.glob('/kaggle/input/**/test.csv', recursive=True)[0]
test_final = pd.read_csv(test_file_path)
test_ids = test_final['id']
X_test_raw = test_final.drop(columns=['id'])

# 2. Define exactly what the models are expecting
# These are the same steps we took during training
cat_cols = ['gender', 'ethnicity', 'education_level', 'income_level', 
            'smoking_status', 'employment_status']

# We define the column order based on our training data
# This is what 'correct_order' was missing
correct_order = df_processed.drop(columns=['diagnosed_diabetes']).columns

# 3. Prepare data for XGBoost (Engineering + Encoding + Scaling)
X_test_xgb = X_test_raw.copy()
X_test_xgb['lifestyle_balance'] = (X_test_xgb['physical_activity_minutes_per_week'] / 100) + X_test_xgb['diet_score']

for col in cat_cols:
    le_final = LabelEncoder()
    # We fit on the training data column to make sure categories match
    le_final.fit(train[col].astype(str))
    X_test_xgb[col] = le_final.transform(X_test_xgb[col].astype(str))

# Fix the column order and scale
X_test_xgb = X_test_xgb[correct_order]
X_test_scaled = scaler.transform(X_test_xgb)
xgb_final_probs = model.predict_proba(X_test_scaled)[:, 1]

# 4. Prepare data for CatBoost (It just needs the raw text version)
cb_final_probs = cb_model.predict_proba(X_test_raw)[:, 1]

# 5. The Ensemble: Blending the two Doctors (50/50)
final_consensus_probs = (xgb_final_probs + cb_final_probs) / 2

# 6. Save the Submission File
final_master_submission = pd.DataFrame({
    'id': test_ids,
    'diagnosed_diabetes': final_consensus_probs
})

final_master_submission.to_csv('final_master_submission.csv', index=False)

print("ğŸš€ BOOM! Mission Accomplished.")
print(f"Final submission created with {len(final_master_submission)} rows.")
display(final_master_submission.head())

