import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from IPython.core.display import HTML
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.metrics import mean_squared_error, log_loss
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline


train_data_path = "/kaggle/input/playground-series-s5e6/train.csv"
test_data_path = "/kaggle/input/playground-series-s5e6/test.csv"


train_df = pd.read_csv(train_data_path)
display(HTML(train_df.head(5).to_html()))


test_df = pd.read_csv(test_data_path)
display(HTML(test_df.head(5).to_html()))


train_df = train_df.drop("id", axis=1)
soil_ids = test_df[["id"]]
test_df = test_df.drop("id", axis=1)


train_missing_cols = train_df.columns[train_df.isnull().any()].to_list()
print(train_missing_cols)
test_missing_cols = test_df.columns[test_df.isnull().any()].to_list()
print(test_missing_cols)


print(len(train_df["Soil Type"].unique()))
print(len(train_df["Crop Type"].unique()))
unique_fertilizer_num = len(train_df["Fertilizer Name"].unique())
print(unique_fertilizer_num)


labelEncoder = LabelEncoder()
train_df["Soil Type"] = labelEncoder.fit_transform(train_df["Soil Type"])
test_df["Soil Type"] = labelEncoder.fit_transform(test_df["Soil Type"])
train_df["Crop Type"] = labelEncoder.fit_transform(train_df["Crop Type"])
test_df["Crop Type"] = labelEncoder.fit_transform(test_df["Crop Type"])
display(HTML(train_df.head(5).to_html()))
display(HTML(test_df.head(5).to_html()))


fertilizer_name = train_df[["Fertilizer Name"]]
train_df = train_df.drop("Fertilizer Name", axis=1)
display(HTML(fertilizer_name.head(5).to_html()))


labelEncoder = LabelEncoder()
fertilizer_name_encoded = labelEncoder.fit_transform(fertilizer_name["Fertilizer Name"])
print(fertilizer_name_encoded[:5])


X_train_np = train_df.values
X_test_np = test_df.values


scaler = MinMaxScaler()
X_train_np = scaler.fit_transform(X_train_np)
# X_cv = scaler.transform(X_cv)
X_test_np = scaler.transform(X_test_np)
print(X_train_np[0])


strat_model = XGBClassifier(random_state=42,
                     objective='multi:softprob',
                     num_class=7,
                    n_estimators=7000,
                    learning_rate=0.02,
                    max_depth=32,
                    gamma=0.20,
                    max_delta_step=4,
                    alpha=5.6,
                    reg_lambda=0.06,
                    min_child_weight=2,
                    subsample=0.8,
                    colsample_bytree=0.3,
                    n_jobs=-1,
                    eval_metric='mlogloss',
                    device='cuda')


fold_scores = []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_index, cv_index) in enumerate(skf.split(X_train_np, fertilizer_name_encoded)):
    print(f"\n===== Fold {fold+1} =====")
    X_train_fold, X_cv_fold = X_train_np[train_index], X_train_np[cv_index]
    Y_train_fold, Y_cv_fold = fertilizer_name_encoded[train_index], fertilizer_name_encoded[cv_index]
    eval_set = [(X_cv_fold, Y_cv_fold)]
    strat_model.fit(X_train_fold, Y_train_fold, eval_set=eval_set, verbose=True)
    val_preds_proba = strat_model.predict_proba(X_cv_fold)
    fold_logloss = log_loss(Y_cv_fold, val_preds_proba)
    fold_scores.append(fold_logloss)
    print(f"LogLoss for Fold {fold+1}: {fold_logloss:.5f}")


model = XGBClassifier(random_state=42,
                     objective='multi:softprob',
                     num_class=7,
                    n_estimators=7000,
                    learning_rate=0.02,
                    max_depth=32,
                    gamma=0.20,
                    max_delta_step=4,
                    alpha=5.6,
                    reg_lambda=0.06,
                    min_child_weight=2,
                    subsample=0.8,
                    colsample_bytree=0.3,
                    n_jobs=-1,
                    eval_metric='mlogloss',
                    device='cuda')
model.fit(X_train_np, fertilizer_name_encoded, verbose=True)


mean_score = np.mean(fold_scores)
std_score = np.std(fold_scores)

print("\n===== Cross-Validation Summary =====")
print(f"Scores for each fold: {[round(score, 5) for score in fold_scores]}")
print(f"Mean LogLoss across all folds: {mean_score:.5f}")
print(f"Standard Deviation of LogLoss: {std_score:.5f}")


importances = model.feature_importances_

# Create a DataFrame for easier handling
importance_df = pd.DataFrame({
    'Feature': train_df.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False) # Sort by importance

# --- Plot using Seaborn ---
plt.figure(figsize=(12, 8)) # Adjust figure size
sns.barplot(x='Importance', y='Feature', data=importance_df.head(20)) # Plot top 20
plt.title('Top 20 Feature Importances from XGBoost Classifier (Gain)')
plt.xlabel('Importance Score (Gain)')
plt.ylabel('Features')
plt.tight_layout()
plt.show()


y_pred = model.predict_proba(X_test_np)


print(y_pred.shape[0])


string_list = []
for row in y_pred:
    sorted_indices = np.argsort(row)
    rev_sorted_indices = sorted_indices[::-1]
    top_3_indices = rev_sorted_indices[:3]
    string_values = labelEncoder.inverse_transform(top_3_indices)
    fertilizer_string_val = string_values[0] + " " + string_values[1] + " " + string_values[2]
    string_list.append(fertilizer_string_val)
# final_np_arr = np.array(string_list)
print(string_list[:5])


print(len(string_list))


df = pd.DataFrame({
    'id': soil_ids.squeeze().values,
    'Fertilizer Name': string_list  # Ensure it's a 1D array
})

# Save to CSV
csv_file_path = '/kaggle/working/predictions_xgboost_6.csv'
df.to_csv(csv_file_path, index=False)

# Output the path where the file is saved
csv_file_path







