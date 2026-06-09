import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt

# For displaying all of the columns in dataframes
pd.set_option('display.max_columns', None)

# For data modeling
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from xgboost import plot_importance

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# For metrics and helpful functions
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score,\
f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report, \
roc_auc_score
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.tree import plot_tree

import lightgbm as lgb

# For saving models
import pickle 


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df_train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')

# Check if it loaded correctly
df_train.head()


df_test.head()


#checking for imbalance in diagnosed diabestes

df_train['diagnosed_diabetes'].value_counts(normalize = True)


df_train.columns


def create_medical_features(df):
    df_eng = df.copy()
    
    # --- 1. Lipid Ratios (Crucial for Insulin Resistance) ---
    # The Triglyceride/HDL ratio is a major marker for insulin resistance
    # (even better than individual cholesterol numbers).
    df_eng['TG_to_HDL_Ratio'] = df_eng['triglycerides'] / df_eng['hdl_cholesterol']
    df_eng['LDL_to_HDL_Ratio'] = df_eng['ldl_cholesterol'] / df_eng['hdl_cholesterol']
    df_eng['Total_to_HDL_Ratio'] = df_eng['cholesterol_total'] / df_eng['hdl_cholesterol']
    df_eng['Non_HDL_Cholesterol'] = df_eng['cholesterol_total'] - df_eng['hdl_cholesterol']

    # --- 2. Blood Pressure Metrics ---
    # Pulse Pressure indicates arterial stiffness (hardening of arteries)
    df_eng['Pulse_Pressure'] = df_eng['systolic_bp'] - df_eng['diastolic_bp']
    # Mean Arterial Pressure (MAP)
    df_eng['MAP'] = (df_eng['systolic_bp'] + (2 * df_eng['diastolic_bp'])) / 3

    # --- 3. Interaction Terms (Risk Multipliers) ---
    # Age acts as a multiplier for BMI risk. 
    # A high BMI at 60 is riskier than a high BMI at 20.
    df_eng['BMI_x_Age'] = df_eng['bmi'] * df_eng['age']
    
    # Interaction between physical activity and weight
    # (High BMI + Low Activity = Maximum Risk)
    df_eng['Activity_Risk_Factor'] = df_eng['bmi'] / (df_eng['physical_activity_minutes_per_week'] + 1)

    # --- 4. Metabolic Syndrome Flag ---
    # This is a heuristic count of how many "bad" signs a patient has.
    # We create a simple score (0-4) based on standard medical thresholds.
    df_eng['Metabolic_Risk_Score'] = (
        (df_eng['bmi'] > 30).astype(int) +
        (df_eng['systolic_bp'] > 130).astype(int) +
        (df_eng['triglycerides'] > 150).astype(int) +
        (df_eng['hdl_cholesterol'] < 40).astype(int) # <40 for men, <50 for women usually, keeping simple
    )
    
    return df_eng

# Apply to BOTH train and test data
# Assuming 'df_train' is your full training set and 'df_test' is your test set
X_train_eng = create_medical_features(df_train) # Replace df_train with your variable name



X_train_eng.columns


y = df_train['diagnosed_diabetes']

X = df_train.drop(['diagnosed_diabetes','id'],axis = 1)


X.shape


x_enc = pd.get_dummies(X,drop_first = True, dtype=int)

x_enc.shape


X_train,X_val, y_train,y_val = train_test_split(x_enc,y,stratify = y, test_size = 0.2,random_state = 42)

print('training complete')


#defining models: 

clf = LogisticRegression(random_state=42, max_iter=500, class_weight='balanced')
rf = RandomForestClassifier(random_state=42, class_weight='balanced')
xgb = XGBClassifier(objective='binary:logistic', random_state=42)
lgbm = lgb.LGBMClassifier(
    objective='binary',
    metric='auc',
    random_state=42,
    n_jobs=-1,
    verbose=-1 # Silences warnings
)

print('model instantiated')


models_dict = {
    "Logistic Regression": clf,
    "Random Forest": rf,
    "XGBoost": xgb,
    "LGBM" : lgbm
}

def evaluate_model(models_dict, x_data, x_test, y_data, y_test):
    results = []
    
    # Loop through the dictionary items (name, model)
    for name, model in models_dict.items():
        # Fit the model
        model.fit(x_data, y_data)
        # Predict
        y_pred = model.predict(x_test)
        #Predict PROBABILITIES - GET PROBABILITIES (0.0 to 1.0)
        y_prob = model.predict_proba(x_test)[:, 1]

        #Use Probabilities (0.0 to 1.0) for: ROC AUC
        #Use Labels (0/1) for: Accuracy, Precision, Recall, F1.
        
        # Append results using the dictionary key 'name'
        results.append({
            'Model Name': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1 Score': f1_score(y_test, y_pred),
            'ROC AUC Probab': roc_auc_score(y_test, y_prob),
            
        })
        
    # Convert list to a Pandas DataFrame and round decimals
    df_model = pd.DataFrame(results).round(4)
    return df_model

# Pass the DICTIONARY, not the list
df_model = evaluate_model(models_dict, X_train, X_val, y_train, y_val)
df_model


# In above table, it is clear that the probability for getting a diabetic patient is 72% for XGBoost model,
# as compared to the second best model (logistic regression) with probability of identifying diabetic
# patient is 69.39%.


# Set up the plot size
plt.figure(figsize=(10, 8))

# Loop through your models to plot each curve
for name, model in models_dict.items():
    
    # 1. Get probabilities for the Positive Class (Diabetes)
    y_prob = model.predict_proba(X_val)[:, 1]
    
    # 2. Calculate ROC Curve metrics (False Positive Rate, True Positive Rate)
    fpr, tpr, _ = roc_curve(y_val, y_prob)
    
    # 3. Calculate AUC Score to display in the legend
    auc_score = roc_auc_score(y_val, y_prob)
    
    # 4. Plot the line
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.4f})', linewidth=2)

# Add a diagonal dashed line (The "Guessing" Line)
plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing (AUC = 0.5)')

# Formatting the chart
plt.xlabel('False Positive Rate (Incorrectly Scared Healthy People)')
plt.ylabel('True Positive Rate (Correctly Found Diabetics)')
plt.title('ROC Curve Comparison')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)

plt.show()


from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from scipy.stats import uniform, randint, loguniform

# 1. Initialize Model (Single threaded so Search can run parallel)
xgb = XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    random_state=42,
    n_jobs=1 
)

# 2. Define "Lite" Parameter Space (Optimized for speed)
cv_params_lite = {
    'learning_rate': uniform(0.01, 0.15),  # Slightly higher min rate = faster convergence
    'n_estimators': randint(300, 800),     # Capped at 800 to save time
    'max_depth': randint(3, 8),            # Capped at 8 (Depth 10 is very slow)
    'min_child_weight': randint(1, 7),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4),
    'reg_alpha': uniform(0, 2),            # Reduced regularization range
    'reg_lambda': uniform(0, 2)
}

# 3. Setup Stratified CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 4. Run "Lite" Randomized Search
# n_iter=50 is the sweet spot. It creates 250 fits (50 * 5 folds).
# This should take 15-20 minutes on standard Kaggle hardware.
print("Starting Fast-Track training...")

xgb_cv = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=cv_params_lite,
    n_iter=50,           
    scoring='roc_auc',
    cv=skf,
    refit='roc_auc',
    n_jobs=-1,               # Use all cores
    verbose=1,
    random_state=42
)

xgb_cv.fit(X_train, y_train)

print(f"Best ROC-AUC: {xgb_cv.best_score_:.5f}")
print(f"Best Params: {xgb_cv.best_params_}")




# Predict on Validation Set using the best model found
best_model = xgb_cv.best_estimator_
y_prob = best_model.predict_proba(X_val)[:, 1]
y_pred = best_model.predict(X_val)

print("\n--- Validation Set Results ---")
print('Accuracy: ', accuracy_score(y_val, y_pred))
print('Precision:', precision_score(y_val, y_pred))
print('Recall:   ', recall_score(y_val, y_pred))
print('F1 Score: ', f1_score(y_val, y_pred))
print('ROC AUC:  ', roc_auc_score(y_val, y_prob))


#now we will create the confusion matrix

def conf_matrix_plot(model,x_data,y_data):

    model_pred = model.predict(x_data)
    cm = confusion_matrix(y_data,model_pred, labels = model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = model.classes_)

    disp.plot()
    plt.show()

conf_matrix_plot(xgb_cv,X_val,y_val)


import pandas as pd
from catboost import CatBoostClassifier

# 1. Initialize CatBoost (The "Robust" Model)
# CatBoost is famous for working well with default parameters and fighting overfitting
cat_model = CatBoostClassifier(
    iterations=1000, 
    learning_rate=0.05, 
    depth=6, 
    eval_metric='AUC',
    random_seed=42,
    verbose=100,  # Print progress every 100 trees
    allow_writing_files=False
)

# 2. Retrain ALL models on FULL Data (Train + Val)
# (Assuming 'best_xgb' and 'best_lgbm' are still in memory from your last run)
# If not, we can reload them, but usually they are still there.

print("Training CatBoost on Full Data (Train + Val)...")
cat_model.fit(X_full, y_full)

# 3. Prepare Test Data (Safety Check)
df_test_eng = create_medical_features(df_test) 
X_test_submit = df_test_eng.drop('id', axis=1)
X_test_submit = pd.get_dummies(X_test_submit, drop_first=True, dtype=int)
_, X_test_submit_aligned = X_full.align(X_test_submit, join='left', axis=1, fill_value=0)

# 4. Get Predictions from the "Big Three"
print("Generating predictions...")
pred_xgb = best_xgb.predict_proba(X_test_submit_aligned)[:, 1]
pred_lgbm = best_lgbm.predict_proba(X_test_submit_aligned)[:, 1]
pred_cat = cat_model.predict_proba(X_test_submit_aligned)[:, 1]

# 5. THE TRI-BLEND (40% XGB, 30% LGBM, 30% CAT)
# We reduce XGBoost's influence slightly to let the others correct its errors
final_ensemble_preds = (0.4 * pred_xgb) + (0.3 * pred_lgbm) + (0.3 * pred_cat)

# 6. Save Submission
submission = pd.DataFrame({
    'id': df_test['id'],
    'diagnosed_diabetes': final_ensemble_preds
})

filename = 'submission_3_model_ensemble.csv'
submission.to_csv(filename, index=False)

print(f"DONE! Saved {filename}")


from IPython.display import FileLink

# Click the blue link that appears below to download your file
FileLink(r'submission_3_model_ensemble.csv')


'''
import lightgbm as lgb

# --- 1. Combine Data (Train + Val) ---
# We retrain on 100% of data to squeeze out every bit of information
X_full = pd.concat([X_train, X_val])
y_full = pd.concat([y_train, y_val])

print(f"Retraining on full dataset: {X_full.shape} samples")

# --- 2. Retrieve the BEST Models ---

# Model A: Your Tuned XGBoost (Found by the code you are running right now)
best_xgb = xgb_cv.best_estimator_ 

# Model B: LightGBM (The Second Opinion)
# We add this to stabilize predictions
best_lgbm = lgb.LGBMClassifier(
    objective='binary',
    metric='auc',
    n_estimators=500,       
    learning_rate=0.05,     
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

# --- 3. Retrain on FULL Data ---
print("Retraining Tuned XGBoost on full data...")
best_xgb.fit(X_full, y_full)

print("Retraining LightGBM on full data...")
best_lgbm.fit(X_full, y_full)

# --- 4. Prepare Test Data (Final Safety Check) ---
df_test_eng = create_medical_features(df_test) 
X_test_submit = df_test_eng.drop('id', axis=1)
X_test_submit = pd.get_dummies(X_test_submit, drop_first=True, dtype=int)

# Align with X_full to ensure columns match perfectly
_, X_test_submit_aligned = X_full.align(X_test_submit, join='left', axis=1, fill_value=0)

# --- 5. Predict & Blend ---
print("Generating predictions...")
pred_xgb = best_xgb.predict_proba(X_test_submit_aligned)[:, 1]
pred_lgbm = best_lgbm.predict_proba(X_test_submit_aligned)[:, 1]

# Weighted Blend: Tuned XGB gets 60%, LightGBM gets 40%
final_ensemble_preds = (0.6 * pred_xgb) + (0.4 * pred_lgbm)

# --- 6. Save ---
submission = pd.DataFrame({
    'id': df_test['id'],
    'diagnosed_diabetes': final_ensemble_preds
})

filename = 'submission_FINAL_ensemble.csv'
submission.to_csv(filename, index=False)

print(f"DONE! Saved {filename}")

'''


#model is very good at catching the disease. It identifies the vast majority of sick people.

#Also, Model is predicting "Diabetes" so often that it told approx 31,000 healthy people they were sick, 
#                                           while only correctly identifying approx 21,000 healthy people.

#"Diabetes" is the majority class (62%), the model has learned that if
#                                            it just guesses "Diabetes" it will be right most of the time


'''

# Apply the exact same feature engineering as training
df_test_eng = create_medical_features(df_test)
X_test = df_test_eng.drop('id', axis=1)

# Encode with dtype=int (MATCHING YOUR TRAINING DATA)
X_test = pd.get_dummies(X_test, drop_first=True, dtype=int)


'''

X_test = df_test.drop('id', axis = 1)

X_test = pd.get_dummies(X_test,drop_first = True, dtype = int)


'''

# We use [:, 1] to get the probability of '1' (Has Diabetes)
test_predictions = xgb_cv.predict_proba(X_test)[:, 1]

test_predictions


'''

#submission = pd.DataFrame({
#    'id': df_test['id'], 
#    'diagnosed_diabetes': test_predictions 
#})


#submission.to_csv('submission_diabetes_prediction.csv', index=False)

#submission.to_csv('tuned_diabetes_prediction.csv', index=False)

submission = pd.DataFrame({
    'id': df_test['id'], 
    'diagnosed_diabetes': test_predictions 
})

submission.to_csv('tuned_diabetes_prediction_corrected.csv', index=False)
print("Submission saved successfully.")


submission.isna().any()


#testing the percentage of patients getting identified as diabetic

plt.figure(figsize=(10, 5))
sns.histplot(submission['diagnosed_diabetes'], bins=30, kde=True)
plt.title("Distribution of Predictions")
plt.xlabel("Predicted Probability of Diabetes")
plt.show()

