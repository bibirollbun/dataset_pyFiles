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


train_data = pd.read_csv(r"/kaggle/input/equity-post-HCT-survival-predictions/train.csv")
test_data = pd.read_csv(r"/kaggle/input/equity-post-HCT-survival-predictions/test.csv")
data_dictionary = pd.read_csv(r"/kaggle/input/equity-post-HCT-survival-predictions/data_dictionary.csv")
sample_submission_data = pd.read_csv(r"/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv")


print("train_data :", train_data.shape)
print("test_data :", test_data.shape)
print("data_dictionary :", data_dictionary.shape)
print("sample_submission_data :", sample_submission_data.shape)


train_data.head()


train_data[['efs','efs_time']]


test_data.head()


data_dictionary.head()


train_data.isnull().sum().sort_values(ascending=False)


# Calculate missing values
missing_values = train_data.isnull().mean() * 100

# Plot
missing_values.plot(kind='bar', figsize=(10, 6), color='skyblue')
plt.title('Percentage of Missing Values by Feature')
plt.ylabel('Percentage')
plt.xlabel('Features')
plt.xticks(rotation=90)
plt.show()


#train_data = train_data.dropna()
train_data = train_data.drop_duplicates()
train_data.shape


train_data.info()


# Identify categorical columns
cat_cols = train_data.select_dtypes(include=["object", "category"]).columns.tolist()

# Define subplot grid (3 rows, auto columns)
num_plots = len(cat_cols)
rows = 10
cols = (num_plots // rows) + (num_plots % rows > 0)  # Adjust columns dynamically

fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
axes = axes.flatten()  # Flatten for easy indexing

# Plot each categorical column
for i, col in enumerate(cat_cols):
    value_counts = train_data[col].value_counts()
    axes[i].pie(value_counts, labels=value_counts.index, autopct="%1.1f%%", startangle=140)
    axes[i].set_title(f"Distribution of {col}")

# Hide unused subplots (if any)
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


# Distribution of the data:
train_data.drop(['ID'],axis=1).hist(figsize=(20,10),color = 'skyblue', edgecolor='black')
plt.show()


y = train_data['efs'].values
plt.hist(y, bins=10, edgecolor="black")
plt.xlabel("efs")
plt.ylabel("Frequency")
plt.title("efs Distribution")
plt.show()


test_data.isnull().sum().sort_values(ascending=False)


set(train_data.columns) - set(test_data.columns)


num_cols = list(train_data.select_dtypes(exclude=['object']).columns.difference(['efs','efs_time']))
cat_cols = list(train_data.select_dtypes(include=['object']).columns)

num_cols_test = list(test_data.select_dtypes(exclude=['object']).columns)
cat_cols_test = list(test_data.select_dtypes(include=['object']).columns)


len(num_cols), len(cat_cols), len(num_cols_test), len(cat_cols_test)


# Fill missing values
train_data[train_data.select_dtypes(include=['number']).columns] = train_data.select_dtypes(include=['number']).apply(lambda x: x.fillna(x.median()))
train_data[train_data.select_dtypes(include=['object']).columns] = train_data.select_dtypes(include=['object']).apply(lambda x: x.fillna('missing'))

# Fill missing values
test_data[test_data.select_dtypes(include=['number']).columns] = test_data.select_dtypes(include=['number']).apply(lambda x: x.fillna(x.median()))
test_data[test_data.select_dtypes(include=['object']).columns] = test_data.select_dtypes(include=['object']).apply(lambda x: x.fillna('missing'))


#  object datatype columns encoding:
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
for col_name in cat_cols:
    train_data[col_name]=labelencoder.fit_transform(train_data[col_name])
    test_data[col_name]=labelencoder.transform(test_data[col_name])


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
train_data[num_cols] = scaler.fit_transform(train_data[num_cols])
test_data[num_cols] = scaler.transform(test_data[num_cols])



# Compute correlation of all features with target
corr_with_target = train_data.corr()["efs"].drop("efs").sort_values(ascending=False)  # Drop self-correlation
print(corr_with_target)


X = train_data.drop(['ID','efs','efs_time'], axis=1)
y = train_data['efs']
test = test_data.drop(['ID'], axis=1)


!pip install lifelines


import xgboost as xgb
from xgboost import XGBClassifier
# Train an XGBoost classifier
model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1)
model.fit(X, y)

# Plot feature importance
xgb.plot_importance(model, importance_type="weight", max_num_features=50)
plt.title("Feature Importance")
plt.show()


xgb_params = {'n_estimators': 296, 'max_depth': 3, 'learning_rate': 0.09144468539793771, 'subsample': 0.7240618773913701, 'colsample_bytree': 0.8176817744555251, 'reg_alpha': 0.006067217165430026, 'reg_lambda': 2.16059692713411}
Best_Parameters = {'n_estimators': 178, 'max_depth': 3, 'learning_rate': 0.18471977114593788, 'subsample': 0.9143855442564214, 'colsample_bytree': 0.9589290352228939, 'reg_alpha': 0.008428005421335278, 'reg_lambda': 7.75233932394396}
#params = {'n_estimators': 264, 'max_depth': 3, 'learning_rate': 0.13491260495358495, 'subsample': 0.6760002264525008, 'colsample_bytree': 0.7367054377437254, 'reg_alpha': 0.4826645307426227, 'reg_lambda': 1.008597250417151e-05}


params = {'n_estimators': 260, 'max_depth': 3, 'learning_rate': 0.15371741505959508, 'subsample': 0.8759888092077012, 'colsample_bytree': 0.7177797461949258, 'reg_alpha': 0.007962228439645077, 'reg_lambda': 9.713261919741738}
#Final C-index: 0.7914


params1 = {'n_estimators': 261, 'max_depth': 3, 'learning_rate': 0.11614002563650597, 'subsample': 0.5548856870570206, 'colsample_bytree': 0.95237231041388, 'reg_alpha': 5.061949964878245, 'reg_lambda': 4.9068659160118597e-05}
#Mean C-index: 0.7548034575222539
params2 = {'n_estimators': 239, 'max_depth': 3, 'learning_rate': 0.12698874892807244, 'subsample': 0.808706235987305, 'colsample_bytree': 0.5004897426722484, 'reg_alpha': 5.805059897617575, 'reg_lambda': 0.002902293690574958}


from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
#from lifelines.utils import concordance_index


# Initialize Stratified K-Fold
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize model
xgb = XGBClassifier(**params2,use_label_encoder=False, eval_metric="logloss")

# Store C-index scores
c_index_scores = []
preds  = []

for train_idx, val_idx in kf.split(X, y):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # Train XGBClassifier
    xgb.fit(X_train, y_train)

    # Predict probabilities for positive class
    y_pred_prob = xgb.predict_proba(X_val)[:, 1]
    pred = xgb.predict_proba(test)[:, 1]
    preds.append(pred)

    # Compute C-index (concordance index)
    #c_index = concordance_index(y_val, y_pred_prob)
    #c_index_scores.append(c_index)

# Print results
#print("C-index Scores for each fold:", c_index_scores)
#print("Mean C-index:", np.mean(c_index_scores))

# Store Predictions in DataFrame
preds = np.mean(preds, axis=0)
submission = pd.DataFrame({'ID': sample_submission_data.ID, 'prediction': preds})
print(submission.head())
submission.to_csv('submission.csv', index=False)


lgbParameters = {'n_estimators': 187, 'max_depth': 3, 'learning_rate': 0.21368222044736684, 'num_leaves': 56, 'subsample': 0.7609033998478479, 'colsample_bytree': 0.5021013894189721, 'reg_alpha': 0.00011035166308219896, 'reg_lambda': 0.0002628277285767593, 'min_child_samples': 44}


from lightgbm import LGBMClassifier
# Initialize Stratified K-Fold
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize model
lgb = LGBMClassifier(**lgbParameters,verbosity=-1,use_label_encoder=False, eval_metric="logloss")

# Store C-index scores
c_index_scores = []
preds  = []

for train_idx, val_idx in kf.split(X, y):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # Train XGBClassifier
    lgb.fit(X_train, y_train)

    # Predict probabilities for positive class
    y_pred_prob = lgb.predict_proba(X_val)[:, 1]
    pred = lgb.predict_proba(test)[:, 1]
    preds.append(pred)
   # Compute C-index (concordance index)
   # c_index = concordance_index(y_val, y_pred_prob)
    #c_index_scores.append(c_index)

# Print results
#print("C-index Scores for each fold:", c_index_scores)
#print("Mean C-index:", np.mean(c_index_scores))

# Store Predictions in DataFrame
preds = np.mean(preds, axis=0)
submission = pd.DataFrame({'ID': sample_submission_data.ID, 'prediction': preds})
print(submission.head())
submission.to_csv('submission.csv', index=False)

