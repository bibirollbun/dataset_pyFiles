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


from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


data_dir = Path('/kaggle/input/playground-series-s5e12/')
train_data = pd.read_csv(data_dir / "train.csv")
test_data = pd.read_csv(data_dir / "test.csv")


train_data.shape


train_data.head().T


train_data.info()


X_train, X_test, y_train, y_test = train_test_split(train_data.drop("diagnosed_diabetes", axis=1), train_data["diagnosed_diabetes"],
                                                    test_size=.15, random_state=42)
X_train = X_train.drop('id', axis=1)
X_test = X_test.drop('id', axis=1)


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer, make_column_selector

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
cat_pipeline_1 = make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore"))
cat_pipeline_2 = make_pipeline(SimpleImputer(strategy="most_frequent"), OrdinalEncoder(handle_unknown="error"))

preprocessing_pipeline = make_column_transformer(
    (num_pipeline, ['age', 'alcohol_consumption_per_week', 'physical_activity_minutes_per_week', 'diet_score', 'sleep_hours_per_day', 'screen_time_hours_per_day',
                    'bmi', 'waist_to_hip_ratio', 'systolic_bp', 'diastolic_bp', 'heart_rate', 'cholesterol_total', 'hdl_cholesterol', 'ldl_cholesterol', 'triglycerides']),
    (cat_pipeline_1, ['gender', 'ethnicity', 'smoking_status', 'employment_status']),
    (cat_pipeline_2, ['education_level', 'income_level']),
    (SimpleImputer(strategy="most_frequent"), ['family_history_diabetes', 'hypertension_history', 'cardiovascular_history']),
    remainder='drop'
)


preprocessing_pipeline


from sklearn.metrics import recall_score, precision_score, roc_auc_score, accuracy_score, roc_curve
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV


gbrt_classifier = make_pipeline(preprocessing_pipeline, GradientBoostingClassifier())


param_grid = [{
    'gradientboostingclassifier__max_depth': [3, 5, 15],
    'gradientboostingclassifier__max_features': [3, 5, 7],
    'gradientboostingclassifier__n_estimators': [50, 100, 300]
}]
grid_search = GridSearchCV(gbrt_classifier, param_grid=param_grid, cv=3)


grid_search.fit(X_train, y_train)


gbrt_best = grid_search.best_estimator_


def submission_output(estimator, test_data, file_name):
    test_set_predictions = estimator.predict_proba(test_data)[:, 1]
    test_data["diagnosed_diabetes"] = test_set_predictions
    test_data[["id", "diagnosed_diabetes"]].to_csv(file_name, index=False)


submission_output(rf, test_data, "submission.csv")

