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
from sklearn.linear_model import RidgeClassifierCV, LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, PolynomialFeatures
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from sklearn.ensemble import StackingClassifier
from sklearn.calibration import CalibratedClassifierCV
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings("ignore")

# ğŸ“¦ Load the data
train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
sub = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')

# ğŸ§¹ Encode the target
le = LabelEncoder()
train['Personality'] = le.fit_transform(train['Personality'])

# ğŸ�¯ Features and target
X = train.drop(columns=['id', 'Personality'])
y = train['Personality']
X_test = test.drop(columns=['id'])

# ğŸ�·ï¸� Identify column types
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

# âš™ï¸� Preprocessing
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('poly', PolynomialFeatures(interaction_only=False, include_bias=False)),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_pipeline, numeric_features),
    ('cat', categorical_pipeline, categorical_features)
])

# ğŸ”§ Full preprocessing pipeline
pipeline = Pipeline([
    ('preprocess', preprocessor)
])

X_processed = pipeline.fit_transform(X)
X_test_processed = pipeline.transform(X_test)

# ğŸ§ª Train/Val/Test split
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=42)
for train_idx, test_idx in sss.split(X_processed, y):
    X_train, X_hold = X_processed[train_idx], X_processed[test_idx]
    y_train, y_hold = y.iloc[train_idx], y.iloc[test_idx]

sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
for val_idx, test_idx in sss2.split(X_hold, y_hold):
    X_val, X_final = X_hold[val_idx], X_hold[test_idx]
    y_val, y_final = y_hold.iloc[val_idx], y_hold.iloc[test_idx]

# ğŸ§  Base models
ridge = RidgeClassifierCV(alphas=np.logspace(-4, 4, 20), cv=3)
lgbm = LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=7, random_state=42)
hgb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=31, random_state=42)

# ğŸ§  Meta model with calibration
meta = CalibratedClassifierCV(LogisticRegression(max_iter=2000, solver='lbfgs', C=1.0), cv=3)

# âš¡ Stacking ensemble
stacked_model = StackingClassifier(
    estimators=[
        ('ridge', ridge),
        ('lgbm', lgbm),
        ('hgb', hgb)
    ],
    final_estimator=meta,
    cv=3,
    passthrough=True,
    n_jobs=-1
)

# ğŸš‚ Train
stacked_model.fit(X_train, y_train)

# ğŸ�¯ Validation metrics
y_val_pred = stacked_model.predict(X_val)
print("\nğŸ“Š Validation Report:")
print(classification_report(y_val, y_val_pred, target_names=le.classes_))

# ğŸ�¯ Test metrics
y_test_pred = stacked_model.predict(X_final)
print("\nğŸ“Š Final Holdout Report:")
print(classification_report(y_final, y_test_pred, target_names=le.classes_))

# ğŸ“¤ Predict test.csv for submission
final_preds = le.inverse_transform(stacked_model.predict(X_test_processed))
sub['Personality'] = final_preds
sub.to_csv('robust_model_submission.csv', index=False)
print("\nâœ… Submission file saved as 'robust_model_submission.csv'")





