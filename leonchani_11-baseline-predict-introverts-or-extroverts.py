# Table manipulation, calculating
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', 100) # increase the maximum number of columns

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler

# Learning
from sklearn.metrics import mean_squared_error, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import KFold
from collections import defaultdict
import lightgbm as lgb
import time

# Saving model
import joblib

# Ignore all warnings
import warnings
warnings.simplefilter("ignore")


df_train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


df_train


df_test


# value for mapping 
mapping_yes_no = {'Yes': 1, 'No': 0}
mapping_persolity = {'Extrovert': 1, 'Introvert': 0}

# df_train
df_train['Stage_fear'] = df_train['Stage_fear'].replace(mapping_yes_no)
df_train['Drained_after_socializing'] = df_train['Drained_after_socializing'].replace(mapping_yes_no)
df_train['Personality'] = df_train['Personality'].replace(mapping_persolity)

# df_test
df_test['Stage_fear'] = df_test['Stage_fear'].replace(mapping_yes_no)
df_test['Drained_after_socializing'] = df_test['Drained_after_socializing'].replace(mapping_yes_no)


scale_pos_weight = df_train[df_train['Personality'] == 1]['Personality'].count() / df_train[df_train['Personality'] == 0]['Personality'].count()
scale_pos_weight








X = df_train.drop(columns=["id", "Personality"])
y = df_train["Personality"]


# --- Hyperparameters (initial values, room for tuning) ---
params = {
    # Parameters that have a relatively small or indirect effect on the prediction
    'objective': 'binary', # Set as a binary classification problem
    'metric': 'accuracy',  # Evaluation metric for binary classification (used for final output/summary, but not directly by eval_metric for early stopping)
    'boosting_type': 'gbdt',
    'verbosity': -1,
    'random_state': 42,
    'n_jobs': -1,

    # Parameters that are likely to have a large effect on the prediction
    'learning_rate': 0.1,
    'n_estimators': 2000,
    'max_depth': 7,
    'num_leaves': 32,
    'subsample': 0.9,
    'colsample_bytree': 0.9,

    # Parameters that may have a moderate impact on the forecast
    'min_child_samples': 20,
    'lambda_l1': 0.05,
    'lambda_l2': 0.05,
    
    # --- Parameters for Imbalanced Data---
    'scale_pos_weight': scale_pos_weight, # Weight given to the minority class (positive class)
    # 'is_unbalance': True,    # Whether to automatically handle imbalanced data
}

# --- K-Fold Cross-Validation ---
NUM_SPLITS = 10
SEED = 42
kf = KFold(n_splits=NUM_SPLITS, shuffle=True, random_state=SEED)
accuracy_scores = [] # A list to store accuracy scores for each fold
models = []          # A list to store trained models

# Initialize variables outside the K-Fold loop to hold the values of the last fold.
last_X_val = None
last_y_val = None
last_y_pred_val = None
last_y_pred_proba = None
last_model = None

# --- Total processing time measurement start ---
total_start_time = time.time()

for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
    print(f"Fold {fold+1}")
    
    # --- Fold processing time measurement start ---
    fold_start_time = time.time()

    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              eval_metric='binary_logloss', # Use 'binary_logloss' for early stopping
              callbacks=[lgb.early_stopping(100, verbose=False)])

    # Predict the classes directly for validation set
    y_pred_val = model.predict(X_val)
    # Predict probabilities for validation set (needed for probability distribution plot)
    y_pred_proba = model.predict_proba(X_val)

    # Calculate Accuracy Score
    acc_val = accuracy_score(y_val, y_pred_val)
    accuracy_scores.append(acc_val)
    models.append(model) # Save the trained model

    # æœ€å¾Œã�®ãƒ•ã‚©ãƒ¼ãƒ«ãƒ‰ã�®æƒ…å ±ã‚’ä¿�å­˜
    if fold == NUM_SPLITS - 1:
        last_X_val = X_val
        last_y_val = y_val
        last_y_pred_val = y_pred_val
        last_y_pred_proba = y_pred_proba
        last_model = model

    # --- Fold processing time measurement end and display ---
    fold_end_time = time.time()
    print(f"Fold {fold+1} completed in {fold_end_time - fold_start_time:.2f} seconds. Accuracy: {acc_val:.4f}")

# --- Total processing time measurement end and display ---
total_end_time = time.time()
print(f"\nTotal cross-validation process completed in {total_end_time - total_start_time:.2f} seconds.")

# --- Show cross-validation results ---
print("\nCross-validation Accuracy scores:", [f"{score:.4f}" for score in accuracy_scores])
print(f'Optimized Cross-validated Accuracy score: {np.mean(accuracy_scores):.4f} +/- {np.std(accuracy_scores):.4f}')
print(f'Max Accuracy score: {np.max(accuracy_scores):.4f}')
print(f'Min Accuracy score: {np.min(accuracy_scores):.4f}')


# --- Visualization for Binary Classification ---

# Define class labels for Introvert/Extrovert
# Ensure that these match the actual internal representation (0 and 1) if y is encoded.
# Assuming 0 is 'Introvert' and 1 is 'Extrovert' based on typical binary encoding.
class_labels = ['Introvert', 'Extrovert']
num_class_actual = len(class_labels) # Should be 2 for binary classification

fig = plt.figure(figsize=(18, 12)) # Adjusted figure size for binary classification plots
gs = fig.add_gridspec(2, 2) # Adjusted grid spec

# 1. Feature Importance
ax0 = fig.add_subplot(gs[0, 0])
# Ensure 'model' is the trained LightGBM model instance
feature_importance = model.booster_.feature_importance(importance_type='gain')
feature_names = X.columns # Use X.columns from your original dataset
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
importance_df = importance_df.sort_values(by="Importance", ascending=False)
sns.barplot(x="Importance", y="Feature", data=importance_df.head(15), palette="viridis", ax=ax0)
ax0.set_title("Feature Importance (LightGBM)", fontsize=14)
ax0.set_xlabel("Importance Score", fontsize=12)
ax0.set_ylabel("Features", fontsize=12)

# 2. Confusion Matrix
# y_pred_val is already the predicted class (0 or 1)
ax1 = fig.add_subplot(gs[0, 1])
cm = confusion_matrix(y_val, y_pred_val) # Use y_val and y_pred_val directly
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
disp.plot(cmap=plt.cm.Blues, ax=ax1, values_format='d')
ax1.set_title("Confusion Matrix", fontsize=14)
plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# 3. Predicted Probability Distribution (for Class 1, e.g., 'Extrovert')
ax2 = fig.add_subplot(gs[1, 0])
# For binary classification, y_pred_proba[:, 1] gives the probability of the positive class (class 1)
sns.histplot(y_pred_proba[:, 1], bins=30, kde=True, ax=ax2, color='skyblue')
ax2.set_title(f"Distribution of Predicted Probabilities for '{class_labels[1]}'", fontsize=14)
ax2.set_xlabel(f"Predicted Probability of '{class_labels[1]}'", fontsize=12)
ax2.set_ylabel("Frequency", fontsize=12)

# 4. Overall Accuracy Score (since Top-K for binary is just Accuracy)
ax3 = fig.add_subplot(gs[1, 1])
overall_accuracy = accuracy_score(y_val, y_pred_val)
ax3.bar(['Overall Accuracy'], [overall_accuracy], color='lightcoral')
ax3.set_title("Overall Accuracy Score", fontsize=14)
ax3.set_ylabel("Accuracy", fontsize=12)
ax3.set_ylim(0, 1)
ax3.text(0, overall_accuracy + 0.02, f'{overall_accuracy:.4f}', ha='center', va='bottom', fontsize=12, color='black') # Display score on bar
ax3.axis('off') # Hide axes for a cleaner look if just displaying the score

# You can add a text display for the overall accuracy
fig.text(0.5, 0.02, f"Overall Validation Accuracy: {overall_accuracy:.4f}", ha='center', va='center', fontsize=16, color='darkblue')

plt.tight_layout(rect=[0, 0.05, 1, 1]) # Adjust layout to make space for overall accuracy text
plt.show()





# import shap

# # LGBM SHAP values
# explainer_lgb = shap.TreeExplainer(model)
# shap_values_lgb = explainer_lgb.shap_values(X)


# shap.summary_plot(shap_values_lgb, X)


# # If shap_values_lgb is a list, convert it to a NumPy array
# if isinstance(shap_values_lgb, list):
#     shap_values_lgb = np.array(shap_values_lgb)

# # Handling the multiclass classification case
# if len(shap_values_lgb.shape) == 3:
#     shap_importance = np.abs(shap_values_lgb).mean(axis=1).mean(axis=0)
# # Handling binary classification cases
# else:
#     shap_importance = np.abs(shap_values_lgb).mean(axis=0)

# # Store in DataFrame
# df_importance = pd.DataFrame({
#     'feature': X.columns,
#     'shap_importance': shap_importance
# })

# # Sort by importance
# df_importance = df_importance.sort_values('shap_importance', ascending=False)

# # Show results
# display(df_importance)





test_id = df_test["id"]
test = df_test.drop(columns=['id'])
all_test_preds_proba = []

# --- Start measuring total test prediction time ---
test_prediction_start_time = time.time()

for fold_, model in enumerate(models):
    print(f"Predicting fold {fold_+1}...")
    
    # --- Start measuring prediction time for each fold ---
    fold_prediction_start_time = time.time()

    # Get prediction probabilities from each model
    # LGBMClassifier.predict_proba() returns an array of shape (n_samples, n_classes)
    pred_proba_ = model.predict_proba(test)
    all_test_preds_proba.append(pred_proba_)

    # --- End measuring and display prediction time for each fold ---
    fold_prediction_end_time = time.time()
    print(f"  Prediction for fold {fold_+1} completed in {fold_prediction_end_time - fold_prediction_start_time:.2f} seconds.")

# Average prediction probabilities for each fold (ensemble)
# Since all_test_preds_proba will be a list of lists (or a 3D array), average along axis=0
# The resulting shape will be (n_samples, n_classes)
ensembled_pred_proba = np.mean(all_test_preds_proba, axis=0)

# --- End measuring and display total test prediction time ---
test_prediction_end_time = time.time()
print(f"\nTotal test prediction process (including ensembling) completed in {test_prediction_end_time - test_prediction_start_time:.2f} seconds.")


ensembled_pred_classes_id = np.argmax(ensembled_pred_proba, axis=1)
inverse_mapping_persolity = {v: k for k, v in mapping_persolity.items()}
final_predictions = [inverse_mapping_persolity[pred_id] for pred_id in ensembled_pred_classes_id]


# Define the inverse mapping for Personality
# Make sure this matches your actual encoding: 0 for Introvert, 1 for Extrovert
inverse_mapping_persolity = {0: 'Introvert', 1: 'Extrovert'}

# --- Visualization ---
fig, axes = plt.subplots(1, 3, figsize=(20, 6)) # Arrange 3 graphs side by side

# 1. Training Data Target Variable Distribution
# Convert numerical labels to actual personality names for plotting
y_personality_names = y.map(inverse_mapping_persolity)
sns.countplot(y=y_personality_names, order=y_personality_names.value_counts().index, palette='viridis', ax=axes[0])
axes[0].set_title("Train Target Variable Distribution")
axes[0].set_xlabel("Count")
axes[0].set_ylabel("Personality Type")

# 2. Validation Prediction Distribution (using last fold's data)
# Convert numerical predicted labels to actual personality names
val_pred_personality_names = pd.Series(last_y_pred_val).map(inverse_mapping_persolity)
sns.countplot(y=val_pred_personality_names, order=val_pred_personality_names.value_counts().index, palette='magma', ax=axes[1])
axes[1].set_title("Validation Predictions Distribution (Last Fold)")
axes[1].set_xlabel("Count")
axes[1].set_ylabel("Personality Type")

# 3. Test Prediction Distribution (using ensembled predictions)
# Convert numerical predicted labels to actual personality names
test_pred_personality_names = pd.Series(ensembled_pred_classes_id).map(inverse_mapping_persolity)
sns.countplot(y=test_pred_personality_names, order=test_pred_personality_names.value_counts().index, palette='cividis', ax=axes[2])
axes[2].set_title("Test Predictions Distribution (Ensembled)")
axes[2].set_xlabel("Count")
axes[2].set_ylabel("Personality Type")

plt.tight_layout() # Adjust layout to prevent overlapping elements
plt.show() # Display the plots


# Define the inverse mapping for Personality
# Make sure this matches your actual encoding: 0 for Introvert, 1 for Extrovert
inverse_mapping_persolity = {0: 'Introvert', 1: 'Extrovert'}

# --- Visualization ---
fig, axes = plt.subplots(1, 3, figsize=(24, 8)) # Arrange 3 graphs side by side, increased size for pies

# Helper function to plot pie chart
def plot_pie_chart(data_series, title, ax, colors):
    counts = data_series.value_counts()
    labels = counts.index
    sizes = counts.values
    
    # Calculate percentages
    percentages = 100 * sizes / sizes.sum()
    
    # Autopct format string for percentage with one decimal place and count
    def autopct_format(pct):
        total = sum(sizes)
        val = int(round(pct*total/100.0))
        return f'{pct:.1f}%\n({val})'

    wedges, texts, autotexts = ax.pie(
        sizes, 
        labels=labels, 
        autopct=autopct_format, 
        startangle=90, 
        colors=colors, 
        wedgeprops=dict(width=0.4, edgecolor='w') # Donut-like appearance
    )
    ax.set_title(title, fontsize=16)
    ax.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
    
    # Improve readability of percentage labels: SET COLOR TO BLACK AND INCREASE SIZE
    plt.setp(autotexts, size=12, weight="bold", color="black") # Changed color to black and size to 12
    plt.setp(texts, size=12) # For slice labels (Introvert/Extrovert)

# 1. Training Data Target Variable Distribution
y_personality_names = y.map(inverse_mapping_persolity)
plot_pie_chart(y_personality_names, "Train Target Variable Distribution", axes[0], ['#4CAF50', '#FFC107']) # Green for Introvert, Yellow for Extrovert

# 2. Validation Prediction Distribution (using last fold's data)
val_pred_personality_names = pd.Series(last_y_pred_val).map(inverse_mapping_persolity)
plot_pie_chart(val_pred_personality_names, "Validation Predictions Distribution (Last Fold)", axes[1], ['#2196F3', '#FF9800']) # Blue for Introvert, Orange for Extrovert

# 3. Test Prediction Distribution (using ensembled predictions)
test_pred_personality_names = pd.Series(ensembled_pred_classes_id).map(inverse_mapping_persolity)
plot_pie_chart(test_pred_personality_names, "Test Predictions Distribution (Ensembled)", axes[2], ['#9C27B0', '#FF5722']) # Purple for Introvert, Red-Orange for Extrovert

plt.tight_layout() # Adjust layout to prevent overlapping elements
plt.show() # Display the plots





submission = pd.DataFrame({
    'id': test_id,
    'Personality': final_predictions
})

# Save
submission.to_csv('submission.csv', index=False)
submission





display(len(models))
display(models)


# Saving Models
joblib.dump(models,'LightGBM.joblib')

# Loading Models
light_gbm =joblib.load('LightGBM.joblib')
light_gbm





!pip install watermark


%load_ext watermark
%watermark -n -u -v -iv -w -p pytensor,aeppl,xarray




