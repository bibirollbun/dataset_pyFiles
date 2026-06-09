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


df_train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')


df_train


df_test





le_soil_type = LabelEncoder()
le_crop_type = LabelEncoder()
le_fertilizer_name = LabelEncoder()


df_train['Soil_Type_encoded'] = le_soil_type.fit_transform(df_train['Soil Type'])
df_train['Crop_Type_encoded'] = le_crop_type.fit_transform(df_train['Crop Type'])
df_train['Fertilizer_Name_encoded'] = le_fertilizer_name.fit_transform(df_train['Fertilizer Name'])

df_test['Soil_Type_encoded'] = le_soil_type.fit_transform(df_test['Soil Type'])
df_test['Crop_Type_encoded'] = le_crop_type.fit_transform(df_test['Crop Type'])


df_Fertilizer_Name = df_train[['Fertilizer_Name_encoded', 'Fertilizer Name']]


del df_train['Soil Type']
del df_train['Crop Type']
del df_train['Fertilizer Name']

del df_test['Soil Type']
del df_test['Crop Type']


columns_to_drop = ['id', 'Fertilizer_Name_encoded']
columns_to_standardize = df_train.copy().drop(columns=columns_to_drop).columns


scaler = StandardScaler()
scaler.fit(df_train[columns_to_standardize])

scaled_values_train = scaler.transform(df_train[columns_to_standardize])
df_train[columns_to_standardize] = scaled_values_train

scaled_values_test = scaler.transform(df_test[columns_to_standardize])
df_test[columns_to_standardize] = scaled_values_test





X = df_train.drop(columns=["id", "Fertilizer_Name_encoded"])
y = df_train["Fertilizer_Name_encoded"]


def apk(actual, predicted, k=3):
    """
    Calculates the Average Precision at k for a single user.
    """
    if not actual:
        return 0.0

    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    return score / min(len(actual), k)


def mapk(actual, predicted, k=3):
    """
    Calculates the Mean Average Precision at k.
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])

def lgbm_mapk(y_true, y_pred, k=3):
    """
    Function for evaluating MAP@k in LightGBM.
    This is a placeholder and needs careful implementation based on
    how y_true and y_pred are structured for a ranking problem.
    For a typical LightGBM regression/classification setup, you'd need
    to process y_pred into ranked lists for each user.
    """

    return 'MAP@3', 0.0, True


# Hyperparameters (initial values, room for tuning)
params = {
    # Parameters that have a relatively small or indirect effect on the prediction
    'objective': 'multiclass', # Set as a multiclass classification problem
    'num_class': 7,            # Adjust to the number of categories to predict
    'metric': 'multi_logloss', # Evaluation metric for multiclass classification
    'boosting_type': 'gbdt',
    'verbosity': -1,
    'random_state': 42,
    'n_jobs': -1,

    # Parameters that are likely to have a large effect on the prediction
    'learning_rate': 0.05,
    'n_estimators': 2000,
    'max_depth': 7,
    'num_leaves': 32,
    'subsample': 0.8,
    'colsample_bytree': 0.8,

    # Parameters that may have a moderate impact on the forecast
    'min_child_samples': 20,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
}

# K-Fold Cross-Validation
NUM_SPLITS = 5
SEED = 42
kf = KFold(n_splits=NUM_SPLITS, shuffle=True, random_state=SEED)
map3_scores = []
models = [] # A list to store trained models

# Initialize an array to store all validation prediction probabilities
# X.shape[0] is the total number of rows in the data, params['num_class'] is the number of classes
all_val_preds_proba = np.zeros((X.shape[0], params['num_class']))

# --- å…¨ä½“å‡¦ç�†æ™‚é–“ã�®è¨ˆæ¸¬é–‹å§‹ ---
total_start_time = time.time()

for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
    print(f"Fold {fold+1}")
    
    # --- å�„ãƒ•ã‚©ãƒ¼ãƒ«ãƒ‰å‡¦ç�†æ™‚é–“ã�®è¨ˆæ¸¬é–‹å§‹ ---
    fold_start_time = time.time()

    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              eval_metric='multi_logloss',
              callbacks=[lgb.early_stopping(100, verbose=False)])

    y_pred_proba = model.predict_proba(X_val)

    # Store the validation prediction probabilities for this fold in the appropriate location in all_val_preds_proba
    all_val_preds_proba[val_index] = y_pred_proba

    # Get the top 3 category IDs based on prediction probabilities for each row (user)
    # y_pred_val will be a list of lists, where each inner list contains the top 3 predicted category IDs for a user
    y_pred_val_ranked = []
    for i in range(len(y_pred_proba)):
        # Get category indices in descending order of probability
        # argsort returns indices in ascending order, so [::-1] reverses it for descending
        top_k_indices = np.argsort(y_pred_proba[i])[::-1][:3]
        y_pred_val_ranked.append(top_k_indices.tolist())
    
    # y_val might need to be converted to a list of lists for each user's true category IDs.
    # If y_val is a single category ID, convert it to a list of lists.
    # For example, if y_val = [0, 1, 2] -> y_val_actual = [[0], [1], [2]]
    # If the Kaggle dataset contains multiple correct categories for each user,
    # prepare y_val_actual to match that format.
    y_val_actual = [[label] for label in y_val] # Conversion assuming y_val is a single category ID

    # Calculate MAP@3
    map3_val = mapk(y_val_actual, y_pred_val_ranked, k=3)
    map3_scores.append(map3_val)
    models.append(model) # Save the trained model

    # --- å�„ãƒ•ã‚©ãƒ¼ãƒ«ãƒ‰å‡¦ç�†æ™‚é–“ã�®è¨ˆæ¸¬çµ‚äº†ã�¨è¡¨ç¤º ---
    fold_end_time = time.time()
    print(f"Fold {fold+1} completed in {fold_end_time - fold_start_time:.2f} seconds.")

# --- å…¨ä½“å‡¦ç�†æ™‚é–“ã�®è¨ˆæ¸¬çµ‚äº†ã�¨è¡¨ç¤º ---
total_end_time = time.time()
print(f"\nTotal cross-validation process completed in {total_end_time - total_start_time:.2f} seconds.")

# Show cross-validation results
print("\nCross-validation MAP@3 scores:", map3_scores)
print(f'Optimized Cross-validated MAP@3 score: {np.mean(map3_scores):.3f} +/- {np.std(map3_scores):.3f}')
print(f'Max MAP@3 score: {np.max(map3_scores):.3f}')
print(f'Min MAP@3 score: {np.min(map3_scores):.3f}')


unique_val_classes = np.sort(np.unique(y_val))
num_class_actual = len(unique_val_classes)
class_labels = [str(c_id) for c_id in unique_val_classes]

fig = plt.figure(figsize=(18, 15))
gs = fig.add_gridspec(3, 2)

# 1. Feature Importance
ax0 = fig.add_subplot(gs[0, 0])
feature_importance = model.booster_.feature_importance(importance_type='gain')
feature_names = X.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
importance_df = importance_df.sort_values(by="Importance", ascending=False)
sns.barplot(x="Importance", y="Feature", data=importance_df.head(15), palette="viridis", ax=ax0)
ax0.set_title("Feature Importance (LightGBM)", fontsize=14)
ax0.set_xlabel("Importance Score", fontsize=12)
ax0.set_ylabel("Features", fontsize=12)

# 2. Confusion Matrix
y_pred_class = np.argmax(y_pred_proba, axis=1)
ax1 = fig.add_subplot(gs[0, 1])
cm = confusion_matrix(y_val, y_pred_class)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
disp.plot(cmap=plt.cm.Blues, ax=ax1, values_format='d')
ax1.set_title("Confusion Matrix", fontsize=14)
plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# 3. Predicted Probability Distribution
ax2 = fig.add_subplot(gs[1, 0])
for i in range(num_class_actual):
    sns.histplot(y_pred_proba[:, unique_val_classes[i]], bins=30, kde=True, label=class_labels[i], alpha=0.5, ax=ax2)
ax2.set_title("Distribution of Predicted Probabilities for Each Class", fontsize=14)
ax2.set_xlabel("Predicted Probability", fontsize=12)
ax2.set_ylabel("Frequency", fontsize=12)
ax2.legend()


# 4. Top-K Accuracy (Hit Rate)
ax3 = fig.add_subplot(gs[1, 1])
k_values = [1, 2, 3]
hit_rates = []
for k_val in k_values:
    hits = 0
    total = len(y_val_actual)
    for i in range(total):
        top_k_pred = np.argsort(y_pred_proba[i])[::-1][:k_val].tolist()
        if any(item in top_k_pred for item in y_val_actual[i]):
            hits += 1
    hit_rates.append(hits / total)

sns.barplot(x=[f'Top-{k}' for k in k_values], y=hit_rates, palette="viridis", ax=ax3)
ax3.set_title("Top-K Hit Rate", fontsize=14)
ax3.set_xlabel("K (Number of Top Predictions)", fontsize=12)
ax3.set_ylabel("Hit Rate (Proportion of Users with Correct Prediction in Top-K)", fontsize=12)
ax3.set_ylim(0, 1)

# 5. Example of individual prediction (to intuitively understand MAP@3)
ax4 = fig.add_subplot(gs[2, :])
num_examples = 5
example_indices = np.random.choice(len(y_val), num_examples, replace=False)

example_data = []
for idx in example_indices:
    actual_label = y_val_actual[idx]
    predicted_probs = y_pred_proba[idx]
    
    # Get categories and their probabilities in descending order
    sorted_indices = np.argsort(predicted_probs)[::-1]
    sorted_probs = predicted_probs[sorted_indices]
    
    # Display top 5 predictions (top 3 are most important for MAP@3, but a broader range can be useful)
    top_n_preds = []
    # Use num_class_actual to ensure the loop range does not exceed the actual number of classes
    for i in range(min(5, num_class_actual)): # Note the length of sorted_indices
        pred_label_id_idx = sorted_indices[i] # Index in y_pred_proba (0, 1, 2...)
        pred_prob = sorted_probs[i]
        
        # pred_label_id_idx is a 0-based index, so class_labels must be created to correspond to this index.
        # Alternatively, a logic might be needed to get the actual category ID using unique_val_classes[pred_label_id_idx]
        # and then find the label corresponding to that ID.
        
        # With the current definition of class_labels, unique_val_classes[i] and class_labels[i] correspond.
        # Therefore, if pred_label_id_idx is a column index in y_pred_proba,
        # class_labels[pred_label_id_idx] should retrieve the correct label name.
        top_n_preds.append(f"{class_labels[pred_label_id_idx]} ({pred_prob:.2f})")
    
    example_data.append({
        'Actual Label(s)': [class_labels[l_id] for l_id in actual_label], # Map IDs in actual_label to class_labels
        'Predicted Top N': ', '.join(top_n_preds[:3]) + ('...' if len(top_n_preds) > 3 else '')
    })

example_df = pd.DataFrame(example_data)
ax4.axis('off')
tbl = ax4.table(cellText=example_df.values, colLabels=example_df.columns, loc='center', cellLoc='left')
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.2, 1.2)
ax4.set_title("Example Predictions (Actual vs. Predicted Top N)", fontsize=14)

plt.tight_layout()
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


# Align array indices with Fertilizer_Name.
fertilizer_mapping_df = df_Fertilizer_Name.drop_duplicates().sort_values('Fertilizer_Name_encoded')
# Mapping dictionary from numerical index to fertilizer name
idx_to_fertilizer_name = dict(zip(fertilizer_mapping_df['Fertilizer_Name_encoded'], fertilizer_mapping_df['Fertilizer Name']))

# If LabelEncoder was used, you can obtain it as follows:
# idx_to_fertilizer_name = {i: name for i, name in enumerate(label_encoder.classes_)}

# 2. Get the top 3 fertilizer names for each test data sample
predicted_fertilizer_names = []
for i in range(len(ensembled_pred_proba)):
    # Get category indices in descending order of probability (sort in descending order)
    # [:3] to get the top 3
    top_k_indices = np.argsort(ensembled_pred_proba[i])[::-1][:3]

    # Convert the top 3 numerical indices to actual fertilizer names
    names = [idx_to_fertilizer_name[idx] for idx in top_k_indices]

    # Join multiple fertilizer names with a space into a single string
    predicted_fertilizer_names.append(" ".join(names))


pd.DataFrame(predicted_fertilizer_names)


# # --- Visualization ---
# fig, axes = plt.subplots(1, 3, figsize=(20, 6)) # Arrange 3 graphs side by side

# # 1. Training Data Target Variable Distribution
# # Since y is a numerical label, it's easier to understand by converting it to actual fertilizer names before counting
# y_fertilizer_names = y.map(idx_to_fertilizer_name)
# sns.countplot(y=y_fertilizer_names, order=y_fertilizer_names.value_counts().index, palette='viridis', ax=axes[0])
# axes[0].set_title("Train Target Variable Distribution")
# axes[0].set_xlabel("Count")
# axes[0].set_ylabel("Fertilizer Name")

# # 2. Validation Prediction Distribution
# flat_y_pred_val_fertilizer_names = flat_y_pred_val_series.map(idx_to_fertilizer_name)
# sns.countplot(y=flat_y_pred_val_fertilizer_names, order=flat_y_pred_val_fertilizer_names.value_counts().index, palette='magma', ax=axes[1])
# axes[1].set_title("Validation Top-3 Predictions Distribution")
# axes[1].set_xlabel("Count")
# axes[1].set_ylabel("Fertilizer Name")

# # 3. Test Prediction Distribution
# sns.countplot(y=flat_predicted_fertilizer_series, order=flat_predicted_fertilizer_series.value_counts().index, palette='cividis', ax=axes[2])
# axes[2].set_title("Test Top-3 Predictions Distribution")
# axes[2].set_xlabel("Count")
# axes[2].set_ylabel("Fertilizer Name")

# plt.tight_layout() # Adjust layout to prevent overlapping elements
# plt.show() # Display the plots





submission = pd.DataFrame({
    'id': test_id,
    'Fertilizer Name': predicted_fertilizer_names
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




