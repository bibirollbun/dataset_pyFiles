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


df_train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')


df_train.head()


df_test.head()


df_train.describe()


df_train.info()


df_null = df_train.isnull().sum()>0


df_null


 df_train.isnull().sum()


df_train.sample(5)


print(df_train.shape)
print(df_test.shape)
print(df_train.columns)
print(df_test.columns)


import seaborn as sns
import matplotlib.pyplot as plt
fig = plt.figure()
bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 100]
labels = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+']
df_train['age_group'] = pd.cut(df_train['age'], bins=bins, labels=labels, right=False)

sns.countplot(x='age_group'  , data = df_train ,  palette='coolwarm')
plt.title("AGE DISTRIBUTION")
plt.xlabel('age')
plt.ylabel('count')
plt.show()




df_train['dataset'] = 'train'
df_test['dataset'] = 'test'
df = pd.concat([df_train, df_test], axis=0).reset_index(drop=True)



df


df.shape


df.info()


num = df.select_dtypes(include=['int64','float64']).columns.tolist()
cat = df.select_dtypes(include=['object','bool']).columns.tolist()
print("Numerical Columns:", num)
print("Categorical Columns:", cat)


missing = df.isnull().sum()
percent = (missing/len(df))*100
missing_df = pd.DataFrame({'missing_values':missing,'percentage':percent})
missing_df = missing_df[missing_df['missing_values']>0]
missing_df


df[num].describe()


for col in cat:
    print(f"\nUnique values in '{col}':")
    print(df[col].value_counts())


plt.figure(figsize=(8,6))
sns.countplot(data = df , x = 'y' , palette = 'coolwarm' , edgecolor = 'black')
plt.title("Distribution of Subscription" , fontsize = 10)
plt.xlabel('Subscribed to Term Deposit ' , fontsize= 8)
plt.ylabel('count' , fontsize= 8)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()



num_cols = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']

for col in num_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col], kde=True, color='skyblue', edgecolor='black')
    plt.title(f'Distribution of {col}', fontsize=14)
    plt.xlabel(col, fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    print(f'\nðŸ“Š Descriptive Stats for {col}:\n')
    print(df[col].describe(), '\n' + '-'*40)


cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']

for col in cat_cols:
    sns.countplot(x=col, data=df, palette='coolwarm',
                  order=df[col].value_counts().index, edgecolor='black')
    
    plt.title(f"Distribution of {col}", fontsize=8)
    plt.xlabel(col, fontsize=6)
    plt.ylabel('Count', fontsize=6)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.7, linestyle='--')
    plt.tight_layout()
    plt.show()
    
    print(df[col].value_counts(normalize=True).round(3), '\n' + '-'*40)



plt.figure(figsize = (18,10))
for i,col in enumerate(num_cols):
    plt.subplot(3,3,i+1)
    sns.boxplot(data = df , y= col ,  color = 'black')
    plt.title(f"Boxplot: {col}")
    plt.grid(True)
plt.tight_layout()
plt.show()


cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']

for col in cat_cols:
    plt.figure(figsize = (10 , 4))
    sns.countplot(x=col , 
                  data = df ,
                  edgecolor = 'black' , 
                  order = df[col].value_counts().index  ,palette = "coolwarm"
                 )
    plt.title(f"distribution {col}",fontsize =14)
    plt.xlabel(col , fontsize = 10)
    plt.ylabel('count',fontsize = 10)
    plt.xticks(rotation = 45 , ha = 'right')
    plt.grid(axis = 'y' , linestyle = '--' , alpha = 0.7)
    plt.tight_layout()
    plt.show()
    print(df[col].value_counts(normalize=True).round(3))


cols_to_plot = ['housing', 'loan', 'contact', 'poutcome']
for col in cols_to_plot:
    plt.figure(figsize = (10 , 4))
    sns.countplot(x=col , 
                  data = df ,
                  hue='y',
        palette=['#1F77B4', '#FF7F0E'],
        edgecolor='black'
                 
                 )
    plt.title(f'Distribution of {col} by Subscription (y)', fontsize=14)
    plt.xlabel(f'{col}', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation = 45 , ha = 'right')
    plt.grid(axis = 'y' , linestyle = '--' , alpha = 0.7)
    plt.legend(title="Subscribed to y" , labels = ['No (0)', 'Yes (1)'])
    plt.tight_layout()
    plt.show()
    print(df[col].value_counts(normalize=True).round(3))


plt.figure(figsize = (10,6))
sns.heatmap(data = df[num_cols].corr(), annot = True ,cmap = "coolwarm", fmt = ".2F", linewidth =0.5 , linecolor = 'white',annot_kws = {'size':10}, cbar_kws={'shrink':0.5}  )
plt.title("Correlation Between Numerical Features", fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


plt.figure(figsize = (18,10))
for i,col in enumerate(num_cols):
    plt.subplot(2,4,i+1)
    sns.boxplot(data = df , y= col ,  color = 'black',palette=['#1F77B4', '#FF7F0E'], 
        linewidth=1.2,
        fliersize=4)
    plt.title(f'{col} by Subscription', fontsize=14, fontweight='semibold', color='#2E4057')
    plt.ylabel(col,fontsize=10)
    plt.xlabel('subs',fontsize=10)
    
    plt.grid(axis ='y',linestyle='--',alpha=0.7)
plt.tight_layout()
plt.show()


df['log_balance']  = np.log1p(df['balance'] - df['balance'].min() + 1)
df['log_duration'] = np.log1p(df['duration'])


cat_cols = [col for col in cat_cols if col in df.columns]

for col in cat_cols:
    if df[col].dtype == 'object':
        print(f'{col} â†’ unknowns: {df[col].isin(["unknown"]).sum()}')



for col in cat_cols:
    
    if df[col].dtype == 'object':
        print(f'{col} â†’ unknowns: {df[col].isin(["unknown"]).sum()}')


binary_map = {'yes':1,'no':0}
df['default']= df['default'].map(binary_map)
df['housing'] = df['housing'].map(binary_map)
df['loan'] = df['loan'].map(binary_map)
multi_cat_cols = ['job', 'marital', 'education', 'contact', 'month', 'poutcome']
df = pd.get_dummies(df, columns=multi_cat_cols, drop_first=True)


train_df = df[df['dataset'] == 'train'].drop(columns=['dataset'], errors='ignore')
test_df  = df[df['dataset'] == 'test'].drop(columns=['dataset'], errors='ignore')


train_df = train_df.drop(columns=['id', 'balance', 'duration'], errors='ignore')  
test_df  = test_df.drop(columns=['y', 'balance', 'duration'], errors='ignore')


X = train_df.drop('y', axis=1)

y = train_df['y']


# Fix all remaining object columns
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = X[col].astype(str)  # Just to be safe
        X[col] = pd.factorize(X[col])[0]  # Simple encoding

for col in test_df.columns:
    if test_df[col].dtype == 'object':
        test_df[col] = test_df[col].astype(str)
        test_df[col] = pd.factorize(test_df[col])[0]



import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load the data
train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')

# Drop unnecessary columns (like 'id')
X = train.drop(columns=['id', 'y', 'balance', 'duration'], errors='ignore')  # balance & duration can be removed if skewed
y = train['y']

test_ids = test['id']  # Save for submission
test = test.drop(columns=['id', 'balance', 'duration'], errors='ignore')

# Encode categorical features
label_encoders = {}

# Combine train & test for consistent encoding
combined = pd.concat([X, test], axis=0)

for col in combined.columns:
    if combined[col].dtype == 'object':
        le = LabelEncoder()
        combined[col] = le.fit_transform(combined[col].astype(str))
        label_encoders[col] = le

# Split back into train and test
X = combined.iloc[:len(X)]
test = combined.iloc[len(X):]

# Encode target 'y' (yes/no)
y_le = LabelEncoder()
y = y_le.fit_transform(y)  # yes=1, no=0

# Train/test split (optional for validation)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# model = LogisticRegression(max_iter=1000)
# model.fit(X_train, y_train)

# # Validate (optional)
# y_pred = model.predict(X_val)
# print("Validation Accuracy:", round(accuracy_score(y_val, y_pred) * 100, 2), "%")

# # Predict on test set
# test_preds = model.predict(test)

# # Decode predictions back to 'yes'/'no'
# final_preds = y_le.inverse_transform(test_preds)

# # Save to submission file
# submission['y'] = final_preds
# submission.to_csv('submission.csv', index=False)
# print("Submission file saved as submission.csv")



# from sklearn.ensemble import RandomForestClassifier
# model = RandomForestClassifier(random_state=42)
# model.fit(X_train, y_train)
# y_pred = model.predict(X_val)
# test_preds = model.predict(test)
# accuracy_rf = accuracy_score(y_val, y_pred)
# print("Random Forest Accuracy:", accuracy_rf)





# from sklearn.tree import DecisionTreeClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score

# # --- Decision Tree ---
# dt_model = DecisionTreeClassifier(random_state=42)
# dt_model.fit(X_train, y_train)
# y_pred_dt = dt_model.predict(X_val)
# accuracy_dt = accuracy_score(y_val, y_pred_dt)

# # --- K-Nearest Neighbors ---
# knn_model = KNeighborsClassifier(n_neighbors=5)  # Default is 5
# knn_model.fit(X_train, y_train)
# y_pred_knn = knn_model.predict(X_val)
# accuracy_knn = accuracy_score(y_val, y_pred_knn)


# print("Decision Tree Accuracy:", accuracy_dt)
# print("K-Nearest Neighbors (KNN) Accuracy:", accuracy_knn)



# test_preds_knn = knn_model.predict(test)


# sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")


# sample_submission["target"] = test_preds_knn


# sample_submission.to_csv("submission_knn.csv", index=False)

# print("KNN predictions saved to submission_knn.csv")


# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import accuracy_score
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.neighbors import KNeighborsClassifier

# # Load data
# train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
# test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
# submission = pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')

# # Drop unnecessary columns
# X = train.drop(columns=['id', 'y', 'balance', 'duration'], errors='ignore')
# y = train['y']
# test_ids = test['id']
# test = test.drop(columns=['id', 'balance', 'duration'], errors='ignore')

# # Encode categorical features
# combined = pd.concat([X, test], axis=0)
# label_encoders = {}
# for col in combined.columns:
#     if combined[col].dtype == 'object':
#         le = LabelEncoder()
#         combined[col] = le.fit_transform(combined[col].astype(str))
#         label_encoders[col] = le

# # Split back
# X = combined.iloc[:len(X)]
# test = combined.iloc[len(X):]

# # Encode target
# y_le = LabelEncoder()
# y = y_le.fit_transform(y)  # 'yes' -> 1, 'no' -> 0
# # 
# # Train-validation split
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# # Model definitions
# models = {
#     "Logistic Regression": LogisticRegression(max_iter=1000),
#     "Random Forest": RandomForestClassifier(random_state=42),
#     "Decision Tree": DecisionTreeClassifier(random_state=42),
#     "KNN": KNeighborsClassifier(n_neighbors=5)
# }

# accuracies = {}
# predictions = {}

# # Train & evaluate all models
# for name, model in models.items():
#     model.fit(X_train, y_train)
#     y_pred_val = model.predict(X_val)
#     acc = accuracy_score(y_val, y_pred_val)
#     accuracies[name] = acc
#     predictions[name] = model.predict(test)
#     print(f"{name} Accuracy: {acc:.4f}")

# # Select best model
# best_model_name = max(accuracies, key=accuracies.get)
# print(f"\nâœ… Best Model: {best_model_name} with Accuracy: {accuracies[best_model_name]:.4f}")

# # Get final predictions
# final_test_preds = predictions[best_model_name]
# final_test_preds_label = y_le.inverse_transform(final_test_preds)

# # Create submission
# submission['y'] = final_test_preds_label
# submission.to_csv('submission.csv', index=False)
# print("ðŸŽ‰ Submission file saved as submission.csv")



# from xgboost import XGBClassifier

# # Initialize the model
# xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

# # Fit the model
# xgb_model.fit(X_train, y_train)

# # Validate
# y_pred_xgb = xgb_model.predict(X_val)
# xgb_acc = accuracy_score(y_val, y_pred_xgb)
# print(f"XGBoost Accuracy: {xgb_acc:.4f}")

# # Predict on test set
# xgb_test_preds = xgb_model.predict(test)
# xgb_test_preds_label = y_le.inverse_transform(xgb_test_preds)

# # Save to submission
# submission['y'] = xgb_test_preds_label
# submission.to_csv('xgb_submission.csv', index=False)
# print("ðŸŽ¯ XGBoost submission saved as xgb_submission.csv")



# from lightgbm import LGBMClassifier
# from sklearn.metrics import accuracy_score

# # Initialize the model
# lgbm_model = LGBMClassifier(random_state=42)

# # Fit the model
# lgbm_model.fit(X_train, y_train)

# # Validate
# y_pred_lgbm = lgbm_model.predict(X_val)
# lgbm_acc = accuracy_score(y_val, y_pred_lgbm)
# print(f"LightGBM Accuracy: {lgbm_acc:.4f}")

# # Predict on test set
# lgbm_test_preds = lgbm_model.predict(test)
# lgbm_test_preds_label = le.inverse_transform(lgbm_test_preds)

# # Save to submission
# submission['y'] = lgbm_test_preds_label
# submission.to_csv('lgbm_submission.csv', index=False)
# print("ðŸŽ¯ LightGBM submission saved as lgbm_submission.csv")



from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd

# Assume X, y, test, and submission are already defined
# Also assume you used LabelEncoder on y if it was categorical
# le = LabelEncoder()
# y = le.fit_transform(y)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
accuracies = []
test_preds_proba = []

for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    model = LGBMClassifier(random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Validation accuracy
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    accuracies.append(acc)
    print(f"Fold {fold} Accuracy: {acc:.4f}")

    # Predict on test set (probabilities)
    test_pred = model.predict_proba(test)
    test_preds_proba.append(test_pred)

# Final average CV accuracy
print(f"\nâœ… Average CV Accuracy: {np.mean(accuracies):.4f}")

# Soft voting: average probabilities, then take argmax
test_preds_proba = np.mean(test_preds_proba, axis=0)
final_test_preds = np.argmax(test_preds_proba, axis=1)

# Inverse transform if label encoder was used
# final_test_preds_label = le.inverse_transform(final_test_preds)

# Update submission
submission['y'] = final_test_preds
submission.to_csv('lgbm_cv_submission.csv', index=False)
print("ðŸŽ¯ Submission saved as lgbm_cv_submission.csv")



# from sklearn.model_selection import StratifiedKFold
# from sklearn.metrics import accuracy_score
# from catboost import CatBoostClassifier
# import numpy as np
# import pandas as pd

# # Assuming X, y, test, and submission are already defined
# # If needed: y = LabelEncoder().fit_transform(y)

# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# accuracies = []
# test_preds_proba = []

# for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
#     X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
#     y_train, y_val = y[train_idx], y[val_idx]

#     # CatBoost Model (silent, fast, accurate)
#     model = CatBoostClassifier(
#         iterations=1000,
#         learning_rate=0.05,
#         depth=6,
#         eval_metric='Accuracy',
#         random_seed=42,
#         verbose=0,
#         early_stopping_rounds=50
#     )

#     model.fit(X_train, y_train, eval_set=(X_val, y_val))

#     # Validation accuracy
#     y_pred = model.predict(X_val)
#     acc = accuracy_score(y_val, y_pred)
#     accuracies.append(acc)
#     print(f"Fold {fold} Accuracy: {acc:.4f}")

#     # Predict on test set (probabilities)
#     test_pred = model.predict_proba(test)
#     test_preds_proba.append(test_pred)

# # Final CV accuracy
# print(f"\nâœ… Average CV Accuracy: {np.mean(accuracies):.4f}")

# # Soft voting
# test_preds_proba = np.mean(test_preds_proba, axis=0)
# final_test_preds = np.argmax(test_preds_proba, axis=1)

# # Save to submission
# submission['y'] = final_test_preds
# submission.to_csv('catboost_cv_submission.csv', index=False)
# print("ðŸŽ¯ Submission saved as catboost_cv_submission.csv")
# #




