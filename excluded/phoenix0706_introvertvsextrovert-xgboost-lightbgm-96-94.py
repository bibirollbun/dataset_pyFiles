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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings



# Pretty settings
warnings.filterwarnings("ignore")
plt.style.use("ggplot")
sns.set_palette("crest")
pd.set_option("display.max_columns", 100)


# Kaggle paths
TRAIN_PATH = "/kaggle/input/playground-series-s5e7/train.csv"
TEST_PATH  = "/kaggle/input/playground-series-s5e7/test.csv"

train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)



train.head()


train.shape


train.info()


train.describe()


test.shape


test.info()


train.columns


train["Time_spent_Alone"].value_counts()



sns.boxplot(x=train["Time_spent_Alone"])
plt.show()
# we can see 


train["Social_event_attendance"].value_counts()


import seaborn as sns
import matplotlib.pyplot as plt

float_cols = train.select_dtypes(include='float').columns

for col in float_cols:
    plt.figure(figsize=(8, 4))
    sns.violinplot(x=train[col])
    plt.title(f'Violin Plot of {col}')
    plt.show()



sns.countplot(train, x="Stage_fear")
plt.show()


train.columns


sns.countplot(train, x="Drained_after_socializing")
plt.show()


sns.countplot(train, x="Personality")
plt.show()



float_cols = train.select_dtypes(include='float').columns
object_cols = train.select_dtypes(include='object').columns
train[float_cols] = train[float_cols].fillna(train[float_cols].median())
for col in object_cols:
    train[col] = train[col].fillna(train[col].mode()[0])


train.isnull().sum()


for col in ["Stage_fear","Drained_after_socializing"]:
    train[col] = train[col].map({'Yes': 1, 'No': 0})



train.head()


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train["Personality_encoded"] = le.fit_transform(train["Personality"])


train.head()


X = train.drop(['Personality','Personality_encoded'], axis=1)
y = train['Personality_encoded']


from sklearn.model_selection import StratifiedKFold, GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# XGBoost classifier
xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)

# Hyperparameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Randomized Search CV
search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring='f1_weighted',
    cv=skf,
    verbose=2,
    n_jobs=-1,
)

# Fit
search.fit(X, y)

# Best parameters
print("Best Parameters:", search.best_params_)
print("Best Cross-Validation F1 Score:", search.best_score_)

# Final model
best_model = search.best_estimator_




y_pred = best_model.predict(X)

print("Accuracy:", accuracy_score(y, y_pred))
print(classification_report(y, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y, y_pred))



float_cols = test.select_dtypes(include='float').columns
object_cols = test.select_dtypes(include='object').columns
test[float_cols] = test[float_cols].fillna(train[float_cols].median())
for col in object_cols:
    test[col] = test[col].fillna(train[col].mode()[0])
for col in ["Stage_fear","Drained_after_socializing"]:
    test[col] = test[col].map({'Yes': 1, 'No': 0})



# Predict probabilities (if needed) or direct classes
test_preds = best_model.predict(test)

# Map numeric predictions back to original labels
label_map = {1: 'Introvert', 0: 'Extrovert'}
test_preds_labels = pd.Series(test_preds).map(label_map)




submission = pd.DataFrame({
    'id': test['id'],  # from test.csv
    'Personality': test_preds_labels
})


submission.head()


submission.to_csv('submission.csv', index=False)




from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV

lgbm = LGBMClassifier()

param_grid = {
    'num_leaves': [31, 50, 70],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 300, 500],
    'max_depth': [-1, 10, 20],

}

grid = GridSearchCV(estimator=lgbm, param_grid=param_grid, 
                    cv=3, scoring='roc_auc', verbose=1, n_jobs=-1)
grid.fit(X, y)

print("Best parameters found: ", grid.best_params_)



# Evaluate
acc = accuracy_score(y, y_pred)
print(f"Accuracy on validation set: {acc:.4f}")
print(classification_report(y, y_pred))


# Best model (already trained on CV folds)
best_model_lgbm = grid.best_estimator_



float_cols = test.select_dtypes(include='float').columns
object_cols = test.select_dtypes(include='object').columns
test[float_cols] = test[float_cols].fillna(train[float_cols].median())
for col in object_cols:
    test[col] = test[col].fillna(train[col].mode()[0])
for col in ["Stage_fear","Drained_after_socializing"]:
    test[col] = test[col].map({'Yes': 1, 'No': 0})



# Predict probabilities (if needed) or direct classes
test_preds = best_model_lgbm.predict(test)

# Map numeric predictions back to original labels
label_map = {1: 'Introvert', 0: 'Extrovert'}
test_preds_labels = pd.Series(test_preds).map(label_map)
submission = pd.DataFrame({
    'id': test['id'],  # from test.csv
    'Personality': test_preds_labels
})


submission.head()


submission.to_csv('submission.csv', index=False)





