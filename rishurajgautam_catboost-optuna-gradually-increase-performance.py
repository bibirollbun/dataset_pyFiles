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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df=pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
test_df=pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')


df.columns


df.head()


sns.kdeplot(
            data=df, x='annual_income', hue='loan_paid_back',fill=True, palette="viridis"
        )


df['income_to_loan_ratio'] = df['annual_income']/df['loan_amount']


# 2. Plot the new feature
plt.figure(figsize=(10, 6))
sns.kdeplot(
    data=df, 
    x='income_to_loan_ratio', 
    hue='loan_paid_back', 
    fill=True, 
    palette="viridis",
    common_norm=False, # Important: normalize each group independently to compare shapes
    clip=(0, 20)       # Optional: limits x-axis if you have extreme outliers
)

plt.title('Distribution of Income-to-Loan Ratio by Repayment Status')
plt.xlabel('Income to Loan Ratio (Higher is safer)')
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

# Set up the figure for side-by-side plots
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 1. Plot Interest Rate
sns.kdeplot(
    data=df, 
    x='interest_rate', 
    hue='loan_paid_back', 
    fill=True, 
    palette="viridis",
    common_norm=False,  # Critical for comparing shapes!
    ax=axes[0]
)
axes[0].set_title('Does Interest Rate predict Default?')

# 2. Plot Credit Score
sns.kdeplot(
    data=df, 
    x='credit_score', 
    hue='loan_paid_back', 
    fill=True, 
    palette="viridis",
    common_norm=False,
    ax=axes[1]
)
axes[1].set_title('Does Credit Score predict Default?')

plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

# 1. Set the size - this plot needs to be wide
plt.figure(figsize=(14, 6))

# Sorting is crucial, otherwise the x-axis will be scrambled
sorted_subgrades = sorted(df['grade_subgrade'].unique())

# 3. Create the Count Plot
sns.countplot(
    data=df, 
    x='grade_subgrade', 
    hue='loan_paid_back', 
    order=sorted_subgrades, # <--- Apply the sort here
    palette="viridis"
)

plt.title('Loan Outcomes by Grade (A1 = Best, G5 = Worst)', fontsize=16)
plt.xlabel('Subgrade', fontsize=12)
plt.ylabel('Number of Loans', fontsize=12)
plt.xticks(rotation=45) # Rotate labels so they don't overlap
plt.legend(title='Loan Paid Back', loc='upper right')

plt.tight_layout()
plt.show()


# 1. Get all unique subgrades and sort them alphabetically
# This ensures A1 comes first and G5 (or F5) comes last
sorted_subgrades = sorted(df['grade_subgrade'].unique())

# 2. Create a dictionary to map strings to numbers
subgrade_mapping = {grade: index for index, grade in enumerate(sorted_subgrades)}

# 3. Apply the mapping to create a new column
df['subgrade_numeric'] = df['grade_subgrade'].map(subgrade_mapping)
test_df['subgrade_numeric'] = test_df['grade_subgrade'].map(subgrade_mapping)


# --- Sanity Check ---
print("Mapping Preview:")
# Print first 5 items in the dictionary to verify
print(list(subgrade_mapping.items())[:5])

# Compare the old string vs new number for a few rows
print("\nDataFrame Check:")
print(df[['grade_subgrade', 'subgrade_numeric']].head())


df.drop('grade_subgrade', axis=1)
test_df.drop('grade_subgrade', axis=1)



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay



# 1. Prepare the Data
# We select the top 3 features identified by your heatmap
features = ['debt_to_income_ratio', 'subgrade_numeric', 'credit_score']
X = df[features]
y = df['loan_paid_back']


# 2. Split into Train (80%) and Test (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# 3. Scaling (CRITICAL for Logistic Regression)
# This ensures Credit Score (700) doesn't dominate DTI (20) just because the number is bigger
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 4. Train the Model
# class_weight='balanced' tells the model: "Pay extra attention to the minority class (Defaulters)"
model = LogisticRegression(class_weight='balanced', random_state=42)
model.fit(X_train_scaled, y_train)


# 5. Make Predictions
y_pred = model.predict(X_test_scaled)

# 6. Evaluation
print("--- Model Performance Report ---")
print(classification_report(y_test, y_pred))

# Visualizing the Confusion Matrix to see if we caught the defaulters
plt.figure(figsize=(6, 6))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap='Blues', normalize='true')
plt.title("Confusion Matrix (Normalized)")
plt.show()


from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# 1. Get the probabilities (The raw confidence of the model)
# ROC_AUC handles the relationship automatically.
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# 2. Calculate the Score
auc_score = roc_auc_score(y_test, y_prob)
print(f"ROC AUC Score: {auc_score:.4f}")

# 3. Calculate the ROC Curve coordinates
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# 4. Plot the ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {auc_score:.2f})', color='darkorange', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing (AUC = 0.50)') # Dashed diagonal line

plt.xlabel('False Positive Rate (False Alarms)')
plt.ylabel('True Positive Rate (Recall)')
plt.title('ROC Curve for Loan Repayment Prediction')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()


from sklearn.ensemble import RandomForestClassifier

# 1. Train the Random Forest
# n_estimators=100: Create 100 decision trees
# class_weight='balanced': Handle the imbalance just like before
rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf_model.fit(X_train, y_train) # Note: Using X_train (unscaled) is fine for Trees



# 2. Get Probabilities
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

# 3. Calculate Scores
auc_rf = roc_auc_score(y_test, y_prob_rf)
print(f"Logistic Regression AUC: {auc_score:.4f}")
print(f"Random Forest AUC:       {auc_rf:.4f}")

# 4. Plot Both Curves for Comparison
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)

plt.figure(figsize=(8, 6))

# Plot Logistic Regression (Previous)
plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {auc_score:.2f})', linestyle='--', color='orange')

# Plot Random Forest (New)
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {auc_rf:.2f})', color='green', linewidth=2)

# Plot Random Guess
plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Model Comparison: Linear vs. Tree-Based')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# 1. Train a "Restricted" Random Forest
# max_depth=10: Don't let the tree grow too deep (prevents memorizing noise)
# min_samples_leaf=50: Each decision "bucket" must have at least 50 people (prevents niche rules)
rf_tuned = RandomForestClassifier(
    n_estimators=200, 
    max_depth=10, 
    min_samples_leaf=50, 
    class_weight='balanced', 
    random_state=42
)

rf_tuned.fit(X_train, y_train)


# 2. Get Probabilities & Score
y_prob_tuned = rf_tuned.predict_proba(X_test)[:, 1]
auc_tuned = roc_auc_score(y_test, y_prob_tuned)

print(f"Original Logistic Regression AUC: {auc_score:.4f}")
print(f"Tuned Random Forest AUC:          {auc_tuned:.4f}")

# 3. Plot Comparison
fpr_tuned, tpr_tuned, _ = roc_curve(y_test, y_prob_tuned)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {auc_score:.2f})', linestyle='--', color='orange')
plt.plot(fpr_tuned, tpr_tuned, label=f'Tuned Random Forest (AUC = {auc_tuned:.2f})', color='green', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Round 2: Can Tuning Save the Random Forest?')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()


import pandas as pd
import matplotlib.pyplot as plt

# 1. Extract Feature Importance
# The model stores this in the .feature_importances_ attribute
importances = rf_tuned.feature_importances_

# 2. Create a Series for easy plotting
# We map the scores to the feature names
feature_importance_df = pd.Series(importances, index=features).sort_values(ascending=True)

# 3. Plot
plt.figure(figsize=(10, 6))
feature_importance_df.plot(kind='barh', color='teal', edgecolor='black')

plt.title('What Matters Most? (Random Forest Feature Importance)')
plt.xlabel('Importance Score (Total = 1.0)')
plt.ylabel('Feature')
plt.grid(axis='x', linestyle='--', alpha=0.5)

# Add value labels to the bars
for index, value in enumerate(feature_importance_df):
    plt.text(value, index, f' {value:.1%}', va='center', fontweight='bold')

plt.tight_layout()
plt.show()


import xgboost as xgb

# 1. Prepare ALL Features
# We drop 'id' (useless) and the target. Everything else stays.
X = df.drop(columns=['id', 'loan_paid_back', 'grade_subgrade']) 
# Note: We drop 'grade_subgrade' because we have 'subgrade_numeric' from before.
# If you didn't run that step, keep 'grade_subgrade' and let get_dummies handle it.

y = df['loan_paid_back']

# 2. One-Hot Encoding (Convert Text to Numbers)
# This turns 'loan_purpose' -> 'loan_purpose_wedding', 'loan_purpose_business', etc.
X = pd.get_dummies(X, drop_first=True)


# 3. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# 4. Train XGBoost (The Kaggle Winner)
# scale_pos_weight is the XGBoost version of class_weight='balanced'
# We set it to roughly (Count of Negatives / Count of Positives) ~ 4
model_xgb = xgb.XGBClassifier(
    n_estimators=300,        # More trees
    learning_rate=0.05,      # Learn slower but deeper
    max_depth=6,             # Standard depth
    scale_pos_weight=4,      # Force it to care about Defaulters
    use_label_encoder=False,
    eval_metric='auc',
    random_state=42
)

print("Training XGBoost...")
model_xgb.fit(X_train, y_train)


# 5. Evaluate
y_prob_xgb = model_xgb.predict_proba(X_test)[:, 1]
auc_xgb = roc_auc_score(y_test, y_prob_xgb)

print(f"XGBoost AUC Score: {auc_xgb:.4f}")

# 6. Plot the curve
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_prob_xgb)

plt.figure(figsize=(10, 6))
plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {auc_xgb:.4f})', color='purple', linewidth=3)
plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')
plt.title('Can we hit 0.90+? (XGBoost Result)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.show()


# Check Feature Importance for the XGBoost model
xgb.plot_importance(model_xgb, max_num_features=10, importance_type='weight', title='What caused the jump to 0.92?')
plt.show()


# "Final Boss" Configuration
model_xgb_tuned = xgb.XGBClassifier(
    n_estimators=1500,       # Massive increase in trees
    learning_rate=0.01,      # Very slow, careful learning
    max_depth=6,             # Keep standard depth
    subsample=0.8,           # Train on random 80% of rows (prevents memorization)
    colsample_bytree=0.8,    # Use random 80% of columns per tree (adds robustness)
    scale_pos_weight=4,      # Keep focus on defaulters
    eval_metric='auc',
    random_state=42,
    n_jobs=-1                # Use all CPU cores
)

print("Training 'Slow Learner' XGBoost...")
model_xgb_tuned.fit(X_train, y_train)



# Evaluate
y_prob_tuned = model_xgb_tuned.predict_proba(X_test)[:, 1]
auc_tuned = roc_auc_score(y_test, y_prob_tuned)

print(f"Final Tuned XGBoost AUC: {auc_tuned:.4f}")

# Plot
fpr_final, tpr_final, _ = roc_curve(y_test, y_prob_tuned)
plt.figure(figsize=(10, 6))
plt.plot(fpr_final, tpr_final, label=f'Tuned XGBoost (AUC = {auc_tuned:.4f})', color='gold', linewidth=3)
plt.plot([0, 1], [0, 1], 'k--')
plt.title('The Quest for 0.93')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.show()


# 1. RELOAD & CLEAN (Starting Fresh to be safe)
# Assuming 'df' is your original dataframe
X = df.drop(columns=['id', 'loan_paid_back', 'grade_subgrade'])
y = df['loan_paid_back']

# --- NEW STEP: FEATURE INTERACTIONS ---

# Interaction 1: "The Desperation Index" (Interest Rate * DTI)
# High Rate AND High Debt = Extreme Risk
X['desperation_index'] = X['interest_rate'] * X['debt_to_income_ratio']

# Interaction 2: "The Stability Score" (Credit Score * Income)
# High Score AND High Income = Super Safe (The model might miss this linear combo)
X['stability_score'] = X['credit_score'] * np.log1p(X['annual_income']) # Log income to normalize

# Interaction 3: "Leverage" (Loan Amount / Credit Score)
# Asking for a huge loan with a bad score
X['leverage_ratio'] = X['loan_amount'] / X['credit_score']


# --------------------------------------

# 2. Encode
X = pd.get_dummies(X, drop_first=True)

# 3. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




# 4. Train the "Slow Learner" again with new features
model_final = xgb.XGBClassifier(
    n_estimators=2000,       # Bumped up for the new complex features
    learning_rate=0.005,     # Even slower learning
    max_depth=7,             # Slightly deeper to capture interactions
    subsample=0.7,           # More randomness to prevent overfitting
    colsample_bytree=0.7,
    scale_pos_weight=4,
    eval_metric='auc',
    random_state=42,
    n_jobs=-1
)

print("Training with Interaction Features...")
model_final.fit(X_train, y_train)


# 5. Evaluate
y_prob_final = model_final.predict_proba(X_test)[:, 1]
auc_final = roc_auc_score(y_test, y_prob_final)

print(f"Final AUC with Interactions: {auc_final:.5f}")


# !pip install catboost


from catboost import CatBoostClassifier


# 1. Prepare Data for CatBoost (It prefers RAW text for categories)
# We go back to the original df to keep 'grade_subgrade' and 'loan_purpose' as text
X_cat = df.drop(columns=['id', 'loan_paid_back', 'subgrade_numeric']) # Drop numeric subgrade, keep the text one
y = df['loan_paid_back']

# Identify categorical columns (Strings)
cat_features = list(X_cat.select_dtypes(include=['object']).columns)
print(f"CatBoost will handle these natively: {cat_features}")

# Split
X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(X_cat, y, test_size=0.2, random_state=42)


# 2. Train CatBoost
model_cb = CatBoostClassifier(
    iterations=2000, 
    learning_rate=0.02,
    depth=6,
    eval_metric='AUC',
    random_seed=42,
    verbose=200,             # Print progress every 200 trees
    auto_class_weights='Balanced'
)

print("Training CatBoost...")
model_cb.fit(X_train_cat, y_train_cat, cat_features=cat_features)


# 3. Get CatBoost Predictions
y_prob_cb = model_cb.predict_proba(X_test_cat)[:, 1]
auc_cb = roc_auc_score(y_test_cat, y_prob_cb)
print(f"CatBoost Alone AUC: {auc_cb:.5f}")



# --- THE GRAND FINALE: ENSEMBLING ---
# We take the average of your best XGBoost model (0.9204) and this new CatBoost model.
# Note: Ensure y_prob_tuned (from previous step) aligns with these rows. 
# Since we used random_state=42 for all splits, the rows should match perfectly.

# If you still have 'y_prob_tuned' from the 0.9204 run, use it here. 
# If not, the CatBoost score alone might beat it.
# Let's assume prediction averaging:
ensemble_prob = (0.5 * y_prob_cb) + (0.5 * y_prob_tuned) 
auc_ensemble = roc_auc_score(y_test_cat, ensemble_prob)

print(f"---------------------------------------")
print(f"Ensemble (XGBoost + CatBoost) AUC: {auc_ensemble:.5f}")
print(f"---------------------------------------")

# Plot
fpr_ens, tpr_ens, _ = roc_curve(y_test_cat, ensemble_prob)
plt.figure(figsize=(10, 6))
plt.plot(fpr_ens, tpr_ens, label=f'Ensemble (AUC = {auc_ensemble:.5f})', color='crimson', linewidth=3)
plt.plot([0, 1], [0, 1], 'k--')
plt.title('The Final Push: Model Ensembling')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.show()


!pip install optuna



import optuna
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# 1. Setup Data (Keep it the same as the winning run)
X_cat = df.drop(columns=['id', 'loan_paid_back', 'subgrade_numeric']) 
y = df['loan_paid_back']
cat_features = list(X_cat.select_dtypes(include=['object']).columns)

# Split (Standard 80/20)
X_train, X_test, y_train, y_test = train_test_split(X_cat, y, test_size=0.2, random_state=42)

# 2. Define the "Objective Function" for Optuna
# This function tells Optuna: "Here are some random parameters. Try them and tell me the score."
def objective(trial):
    params = {
        'iterations': 1000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', 1, 10),
        'random_strength': trial.suggest_float('random_strength', 1e-9, 10),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'eval_metric': 'AUC',
        'cat_features': cat_features,
        'verbose': False,
        'random_seed': 42
    }
    
    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train)
    
    preds = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds)
    
    return auc

# 3. Run the Optimization
print("Bot is searching for the perfect parameters...")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30) # 30 different experiments

# 4. Print the Winner
print("------------------------------------------------")
print(f"Best AUC Found: {study.best_value:.5f}")
print("Best Parameters:")
print(study.best_params)


from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# 1. The "Magic Parameters" found by Optuna
best_params = {
    'learning_rate': 0.09195047689481267,
    'depth': 7,
    'l2_leaf_reg': 2,
    'random_strength': 1.8630333217706272,
    'bagging_temperature': 0.42462333876969865,
    
    # THE UPGRADE:
    'iterations': 5000,        # Give it way more room to learn
    'od_type': 'Iter',         # Overfitting Detector
    'od_wait': 100,            # Stop if score doesn't improve for 100 trees
    'eval_metric': 'AUC',
    'random_seed': 42,
    'verbose': 500             # Print updates less often
}

# 2. Train the Ultimate Model
print("Training the Final Production Model...")
model_final = CatBoostClassifier(**best_params)

# We pass the test set to 'eval_set' so it knows when to stop
model_final.fit(
    X_train, y_train, 
    cat_features=cat_features, 
    eval_set=(X_test, y_test), 
    early_stopping_rounds=100
)




# 3. Final Evaluation
y_prob_final = model_final.predict_proba(X_test)[:, 1]
auc_final = roc_auc_score(y_test, y_prob_final)

print(f"---------------------------------------")
print(f"FINAL PROJECT SCORE: {auc_final:.5f}")
print(f"---------------------------------------")

# 4. Save the model (You've earned it)
model_final.save_model("credit_risk_model_v1.cbm")
print("Model saved as 'credit_risk_model_v1.cbm'")


# 1. Re-create the missing feature in the test set
# We created this way back in the beginning!
test_df['income_to_loan_ratio'] = test_df['annual_income'] / test_df['loan_amount']

# 2. Check for other engineered features
# If you ran the 'Interaction Engineering' block (desperation_index, etc.) 
# on your training data, your model might expect those too.
# Let's verify exactly what the model needs:
required_features = model_final.feature_names_
print(f"Model expects these features: {required_features}")

# 3. Add any other missing features (Safety check)
# If your model expects these from the XGBoost experiments, we calculate them.
# If the print statement above DOESN'T show them, this block won't hurt.
if 'desperation_index' in required_features:
    test_df['desperation_index'] = test_df['interest_rate'] * test_df['debt_to_income_ratio']
    
if 'stability_score' in required_features:
    test_df['stability_score'] = test_df['credit_score'] * np.log1p(test_df['annual_income'])

if 'leverage_ratio' in required_features:
    test_df['leverage_ratio'] = test_df['loan_amount'] / test_df['credit_score']

# 4. Prepare the final X_test
# We filter test_df to ensure it has ONLY the columns the model wants, in the exact order.
X_test_final = test_df[required_features]

# 5. Now Predict
print("Predicting...")
predictions = model_final.predict_proba(X_test_final)[:, 1]

# 6. Save
submission = pd.DataFrame({
    'id': submission_ids, # Make sure you ran the previous block to capture this
    'loan_paid_back': predictions
})
submission.to_csv('submission.csv', index=False)
print("Success! Created 'submission_fixed.csv'")


from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# 1. The "Magic Parameters" found by Optuna
best_params = {
    'learning_rate': 0.09195047689481267,
    'depth': 7,
    'l2_leaf_reg': 2,
    'random_strength': 1.8630333217706272,
    'bagging_temperature': 0.42462333876969865,
    
    # THE UPGRADE:
    'iterations': 6000,        # Give it way more room to learn
    'od_type': 'Iter',         # Overfitting Detector
    'od_wait': 100,            # Stop if score doesn't improve for 100 trees
    'eval_metric': 'AUC',
    'random_seed': 42,
    'verbose': 500             # Print updates less often
}

# 2. Train the Ultimate Model
print("Training the Final Production Model...")
model_final = CatBoostClassifier(**best_params)

# We pass the test set to 'eval_set' so it knows when to stop
model_final.fit(
    X_cat, y, 
    cat_features=cat_features, 
    eval_set=(X_test, y_test), 
    early_stopping_rounds=100
)




# 3. Final Evaluation
y_prob_final = model_final.predict_proba(X_test)[:, 1]
auc_final = roc_auc_score(y_test, y_prob_final)

print(f"---------------------------------------")
print(f"FINAL PROJECT SCORE: {auc_final:.5f}")
print(f"---------------------------------------")

# 4. Save the model (You've earned it)
model_final.save_model("credit_risk_model_v2.cbm")
print("Model saved as 'credit_risk_model_v2.cbm'")


# 1. Re-create the missing feature in the test set
# We created this way back in the beginning!
test_df['income_to_loan_ratio'] = test_df['annual_income'] / test_df['loan_amount']

# 2. Check for other engineered features
# If you ran the 'Interaction Engineering' block (desperation_index, etc.) 
# on your training data, your model might expect those too.
# Let's verify exactly what the model needs:
required_features = model_final.feature_names_
print(f"Model expects these features: {required_features}")

# 3. Add any other missing features (Safety check)
# If your model expects these from the XGBoost experiments, we calculate them.
# If the print statement above DOESN'T show them, this block won't hurt.
if 'desperation_index' in required_features:
    test_df['desperation_index'] = test_df['interest_rate'] * test_df['debt_to_income_ratio']
    
if 'stability_score' in required_features:
    test_df['stability_score'] = test_df['credit_score'] * np.log1p(test_df['annual_income'])

if 'leverage_ratio' in required_features:
    test_df['leverage_ratio'] = test_df['loan_amount'] / test_df['credit_score']

# 4. Prepare the final X_test
# We filter test_df to ensure it has ONLY the columns the model wants, in the exact order.
X_test_final = test_df[required_features]

# 5. Now Predict
print("Predicting...")
predictions = model_final.predict_proba(X_test_final)[:, 1]

# 6. Save
submission = pd.DataFrame({
    'id': submission_ids, # Make sure you ran the previous block to capture this
    'loan_paid_back': predictions
})
submission.to_csv('submission.csv', index=False)
print("Success! Created 'submission.csv'")




