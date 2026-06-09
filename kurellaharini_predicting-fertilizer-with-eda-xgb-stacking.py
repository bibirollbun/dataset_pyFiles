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
sample_submission=pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')
print(sample_submission.shape)
sample_submission.head()


import pandas as pd

train=pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')

print(train.info())


train.describe()


print(train.shape)
train.head()


print(test.shape)
test.head()


train.isnull().sum()


import seaborn as sns

train['Fertilizer Name'].value_counts()
sns.countplot(data=train, x='Fertilizer Name')


correlation=train['Temparature'].corr(train['Humidity'])
print(f"Correlation between Temperature and Humidity: {correlation:.3f}")


import matplotlib.pyplot as plt

numeric_features=['Temparature', 'Humidity', 'Nitrogen', 'Phosphorous', 'Potassium']
train[numeric_features].hist(figsize=(8,6),bins=20)
plt.tight_layout()


import matplotlib.pyplot as plt
import seaborn as sns

numeric_features=['Temparature', 'Humidity', 'Nitrogen', 'Phosphorous', 'Potassium']
for col in numeric_features:
    sns.boxplot(data=train, x=col)
    plt.title(f"Boxplot of {col}")
    plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

numeric_features=['Temparature', 'Humidity', 'Nitrogen', 'Phosphorous', 'Potassium']
for col in numeric_features:
    plt.figure(figsize=(6,4))
    sns.violinplot(x='Fertilizer Name', y=col, data=train)
    #plt.xticks(rotation=45)
    plt.title(f"{col} vs Fertilizer")
    plt.show()


from sklearn.preprocessing import LabelEncoder

le_soil = LabelEncoder()
le_crop = LabelEncoder()
le_fertilizer = LabelEncoder()

train['Soil Type'] = le_soil.fit_transform(train['Soil Type'])
train['Crop Type'] = le_crop.fit_transform(train['Crop Type'])
train['Fertilizer Name'] = le_fertilizer.fit_transform(train['Fertilizer Name'])
print(train.head())


le_soil1 = LabelEncoder()
le_crop1 = LabelEncoder()
test['Soil Type'] = le_soil.fit_transform(test['Soil Type'])
test['Crop Type'] = le_crop.fit_transform(test['Crop Type'])
print(test.head())


print(train.columns.tolist())


print(test.columns.tolist())


print(train['Fertilizer Name'].value_counts())


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


# Features and Target
X = train.drop(['Fertilizer Name', 'id'], axis=1)
y = train['Fertilizer Name']

target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)

# Train-test split
X_train, X_valid, y_train, y_valid = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and check accuracy
y_pred = model.predict(X_valid)
accuracy = accuracy_score(y_valid, y_pred)
print(f"Accuracy: {accuracy:.2f}")


import matplotlib.pyplot as plt
import seaborn as sns

# After training the model
importances = model.feature_importances_
feature_names = X_train.columns

feat_imp = pd.Series(importances, index=feature_names)
feat_imp = feat_imp.sort_values(ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x=feat_imp, y=feat_imp.index)
plt.title("Feature Importance")
plt.show()


# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score

# model = DecisionTreeClassifier(max_depth=10, random_state=42)
# model.fit(X_train, y_train)
# y_pred = model.predict(X_valid)
# accuracy = accuracy_score(y_valid, y_pred)
# print(f"Decision Tree Accuracy: {accuracy:.2f}")


train.duplicated().sum()


train.columns


train.head()


import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X = train.drop('Fertilizer Name', axis=1)
y = train['Fertilizer Name']

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train)

# Plot feature importance
import matplotlib.pyplot as plt
xgb.plot_importance(model)
plt.show()


# from catboost import CatBoostClassifier
# cat_model = CatBoostClassifier(verbose=0, random_state=42)
# cat_model.fit(X_train, y_train)
# y_pred = cat_model.predict(X_valid)
# print("CatBoost Accuracy:", accuracy_score(y_valid, y_pred))


# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import LabelEncoder

# X = train.drop(['Fertilizer Name', 'id'], axis=1)
# y = train['Fertilizer Name']
# test_ids = test['id']
# X_test = test.drop(['id'], axis=1)

# X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# model = RandomForestClassifier(n_estimators=300, random_state=42)
# model.fit(X_train, y_train)

# probs = model.predict_proba(X_test)

# top3_indices = np.argsort(probs, axis=1)[:, -3:][:, ::-1]

# top3_names = []
# for row in top3_indices:
#     fertilizer_names = le_fertilizer.inverse_transform(row)
#     fertilizer_names_str = [str(name) for name in fertilizer_names]
#     top3_names.append(' '.join(fertilizer_names_str))

# # submission = pd.DataFrame({
# #     'id': test_ids,
# #     'Fertilizer Name': top3_names
# # })
# # submission.to_csv('submission.csv', index=False)

# # print("Submission file created successfully!")
# # print(submission.head())


# import lightgbm as lgb
# lgb_model = lgb.LGBMClassifier(random_state=42)
# lgb_model.fit(X_train, y_train)
# y_pred = lgb_model.predict(X_valid)
# print("LightGBM Accuracy:", accuracy_score(y_valid, y_pred))


# from sklearn.linear_model import LogisticRegression

# model = LogisticRegression(max_iter=1000, random_state=42)
# model.fit(X_train, y_train)
# y_pred = model.predict(X_valid)
# print("Logistic Regression Accuracy:", accuracy_score(y_valid, y_pred))


# from sklearn.neighbors import KNeighborsClassifier

# model = KNeighborsClassifier(n_neighbors=5)
# model.fit(X_train, y_train)
# y_pred = model.predict(X_valid)
# print("KNN Accuracy:", accuracy_score(y_valid, y_pred))


# import xgboost as xgb
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score

# # features and target
# X = train.drop(['Fertilizer Name', 'id'], axis=1)
# y = train['Fertilizer Name']
# test_ids = test['id']
# X_test = test.drop(['id'], axis=1)

# # split for validation
# X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# # train xgb model
# model = xgb.XGBClassifier(
#     n_estimators=500,
#     learning_rate=0.05,
#     max_depth=6,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     random_state=42,
#     use_label_encoder=False,
#     eval_metric='mlogloss'
# )

# model.fit(X_train, y_train)
# y_pred = model.predict(X_valid)
# accuracy = accuracy_score(y_valid, y_pred)
# # print(f"Improved Accuracy: {accuracy:.4f}")

# # Predict probabilities
# probs = model.predict_proba(X_test)

# # Get Top 3 Predictions
# top3_indices = np.argsort(probs, axis=1)[:, -3:][:, ::-1]

# # Inverse transform using existing encoder
# top3_names = []
# for row in top3_indices:
#     actual_names = le_fertilizer.inverse_transform(row)
#     top3_names.append(' '.join(actual_names))

# # Creating submission file
# # submission = pd.DataFrame({'id': test_ids, 'Fertilizer Name': top3_names})
# # submission.to_csv('submission.csv', index=False)

# # print("Submission file with fertilizer names!")
# # print(submission.head())


# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier, VotingClassifier
# from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import accuracy_score

# # Features and Target
# X = train.drop(['Fertilizer Name', 'id'], axis=1)
# y = train['Fertilizer Name']
# test_ids = test['id']
# X_test = test.drop(['id'], axis=1)

# # Train-validation split
# X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# # Models for Ensemble
# rf = RandomForestClassifier(n_estimators=300, random_state=42)
# xgb = XGBClassifier(n_estimators=300, learning_rate=0.05, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
# lgb = LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=42)

# # Voting Classifier 
# ensemble_model = VotingClassifier(estimators=[
#     ('rf', rf),
#     ('xgb', xgb),
#     ('lgb', lgb)
# ], voting='soft')

# ensemble_model.fit(X_train, y_train)

# # Validation Accuracy
# y_valid_pred = ensemble_model.predict(X_valid)
# print(f"Validation Accuracy: {accuracy_score(y_valid, y_valid_pred):.4f}")

# # Predicting probabilities for the test data
# probs = ensemble_model.predict_proba(X_test)

# # Get Top 3 predictions 
# top3_indices = np.argsort(probs, axis=1)[:, -3:][:, ::-1]

# # Convert numeric predictions â†’ Fertilizer Names using inverse_transform()
# top3_names = []
# for row in top3_indices:
#     fertilizer_names = le_fertilizer.inverse_transform(row)
#     fertilizer_names_str = [str(name) for name in fertilizer_names]
#     top3_names.append(' '.join(fertilizer_names_str))

# # submission file
# submission = pd.DataFrame({'id': test_ids, 'Fertilizer Name': top3_names})
# submission.to_csv('submission.csv', index=False)

# print("Submission file created successfully!")
# print(submission.head())


# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# import xgboost as xgb
# from catboost import CatBoostClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import accuracy_score

# # Assuming your data is already numeric and label encoding done
# X = train.drop(['Fertilizer Name', 'id'], axis=1)
# y = train['Fertilizer Name']
# X_test = test.drop(['id'], axis=1)
# test_ids = test['id']

# # Train-test split for validation
# X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# # Base Models
# rf = RandomForestClassifier(n_estimators=200, random_state=42)
# xgb_model = xgb.XGBClassifier(n_estimators=300, random_state=42, eval_metric='mlogloss')
# cat_model = CatBoostClassifier(verbose=0, random_state=42)

# # Train base models
# rf.fit(X_train, y_train)
# xgb_model.fit(X_train, y_train)
# cat_model.fit(X_train, y_train)

# # Get predictions (probabilities) from base models on validation set
# rf_valid = rf.predict_proba(X_valid)
# xgb_valid = xgb_model.predict_proba(X_valid)
# cat_valid = cat_model.predict_proba(X_valid)

# # Stack them horizontally to form new features for meta-model
# stacked_valid = np.hstack((rf_valid, xgb_valid, cat_valid))

# # Meta-model (can try Logistic Regression, XGBoost, etc.)
# meta_model = LogisticRegression(max_iter=1000)
# meta_model.fit(stacked_valid, y_valid)

# # Evaluate on validation data
# rf_test = rf.predict_proba(X_test)
# xgb_test = xgb_model.predict_proba(X_test)
# cat_test = cat_model.predict_proba(X_test)

# # Evaluate Stacking Accuracy
# final_valid_preds = meta_model.predict(stacked_valid)
# accuracy = accuracy_score(y_valid, final_valid_preds)
# #print(f"Stacking Validation Accuracy: {accuracy:.4f}")

# # Stack test predictions for final meta-model
# stacked_test = np.hstack((rf_test, xgb_test, cat_test))

# # Final predictions (top 1 for now)
# final_preds = meta_model.predict(stacked_test)

# # Inverse transform to get actual fertilizer names
# final_preds_labels = le_fertilizer.inverse_transform(final_preds)

# # For submission â†’ Predict Top 3 using XGBoost probabilities (for MAP@3 score)
# probs = xgb_model.predict_proba(X_test)
# top3_indices = np.argsort(probs, axis=1)[:, -3:][:, ::-1]
# top3_names = []
# for row in top3_indices:
#     names = le_fertilizer.inverse_transform(row)
#     top3_names.append(' '.join(names))

# # submission = pd.DataFrame({'id': test_ids, 'Fertilizer Name': top3_names})
# # submission.to_csv('submission.csv', index=False)

# # #print(" Stacking submission created successfully!")
# # print(submission.head())


import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

# Prepare data
X = train.drop(['Fertilizer Name', 'id'], axis=1)
y = train['Fertilizer Name']
X_test = test.drop(['id'], axis=1)
test_ids = test['id']

n_classes = len(np.unique(y))

# 10-Fold Stratified CV
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
oof_preds = np.zeros((len(X), n_classes))
test_preds = np.zeros((len(X_test), n_classes))
fold_accs = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\nðŸŒŸ Fold {fold+1}/10")

    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    model = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.02,
        max_depth=10,
        subsample=0.9,
        colsample_bytree=0.9,
        gamma=0.3,
        min_child_weight=2,
        objective='multi:softprob',
        num_class=n_classes,
        random_state=fold,
        tree_method='hist',
        eval_metric='mlogloss'
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=100,
        verbose=200
    )

    val_pred = model.predict(X_val)
    acc = accuracy_score(y_val, val_pred)
    fold_accs.append(acc)
    print(f"âœ… Fold {fold+1} Accuracy: {acc:.4f}")

    # OOF preds for meta-analysis (optional)
    oof_preds[val_idx] = model.predict_proba(X_val)

    # Average test predictions
    test_preds += model.predict_proba(X_test) / skf.n_splits

mean_acc = np.mean(fold_accs)
print(f"\nðŸŒŸ Average 10-Fold CV Accuracy: {mean_acc:.4f}")

# Predict Top-3 classes from averaged probabilities
top3_indices = np.argsort(test_preds, axis=1)[:, -3:][:, ::-1]

# Join indices to fertilizer names (strings)
top3_names = []
for row in top3_indices:
    actual_names = [str(y.cat.categories[i]) if hasattr(y, 'cat') else str(np.unique(y)[i]) for i in row]
    top3_names.append(' '.join(actual_names))

submission = pd.DataFrame({'id': test_ids, 'Fertilizer Name': top3_names})
submission.to_csv("submission.csv", index=False)
print("\n Submission saved as 'submission.csv'")
print(submission.head())

