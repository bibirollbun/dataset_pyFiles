!pip install /kaggle/input/cibmtr-whl-files-for-installation/ecos-2.0.14-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
!pip install /kaggle/input/cibmtr-whl-files-for-installation/scikit_survival-0.20.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl



!pip install /kaggle/input/pip-install-lifelines/autograd-1.7.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/autograd-gamma-0.5.0.tar.gz
!pip install /kaggle/input/pip-install-lifelines/interface_meta-1.3.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/formulaic-1.0.2-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/lifelines-0.30.0-py3-none-any.whl


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


train_hct = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/train.csv")
test_hct = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv")
sample_submission = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv")


train_hct=train_hct.sample(frac=0.4,random_state=42)


# Check data
print("Train data shape:", train_hct.shape)
print("Train data columns:", train_hct.columns)
print("Missing values:\n", train_hct.isnull().sum())
print("Target variable stats:\n", train_hct['efs'].describe())


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from sklearn.ensemble import VotingClassifier
from catboost import CatBoostClassifier

from sklearn.preprocessing import PowerTransformer

# Convert categorical features to category type
categorical_cols = train_hct.select_dtypes(include=['object']).columns
for col in categorical_cols:
    train_hct[col] = train_hct[col].astype('category')

label_encoder = LabelEncoder()

# Function to handle unknown categories
def handle_unknown_categories(encoder, train_data, test_data, column):
    # Fit the encoder on the training data
    encoder.fit(train_data[column])
    
    # Transform the training data
    train_data[column] = encoder.transform(train_data[column])

    # For the test data, we will handle unknown categories manually
    # Create a mapping for categories in the test data that are not in the training data
    known_categories = encoder.classes_
    test_data[column] = test_data[column].apply(
        lambda x: encoder.transform([x])[0] if x in known_categories else -1
    )







# Apply the custom encoder function to categorical columns
for col in categorical_cols:
    handle_unknown_categories(label_encoder, train_hct, test_hct, col)


# Split data into features (X) and target (y)
X = train_hct.drop(columns=['efs','efs_time','ID']) 
y = train_hct[['efs','efs_time']]

X = X.fillna(X.mean())  # Replace NaNs with column mean (or use a method of choice)
X = X.replace([np.inf, -np.inf], np.nan).fillna(X.ffill())  # Handle infinities

X = X.loc[:, (X.nunique() > 1)]

X = X.clip(lower=X.quantile(0.01), upper=X.quantile(0.99), axis=1)



# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.0868, random_state=42)


# Create structured array for y_train
y_train_structured = np.array(
    [(bool(event), time) for event, time in zip(y_train['efs'], y_train['efs_time'])],
    dtype=[('event', '?'), ('time', '<f8')]
)

# Create structured array for y_val
y_val_structured = np.array(
    [(bool(event), time) for event, time in zip(y_val['efs'], y_val['efs_time'])],
    dtype=[('event', '?'), ('time', '<f8')]
)

# Verify the structured arrays
print("y_train_structured:", y_train_structured[:5])
print("y_val_structured:", y_val_structured[:5])




params={'max_depth': 8,
 'max_features': 0.29110519961044856,
 'min_samples_leaf': 4,
 'min_samples_split': 5,
 'n_estimators': 70}


from sksurv.util import Surv
from sksurv.ensemble import RandomSurvivalForest

rsf = RandomSurvivalForest(**params)
rsf.fit(X_train, y_train_structured)



rsf_score = rsf.score(X_val, y_val_structured)
print("Model score:", rsf_score)


# Convert categorical features to category type
categorical_cols_test = test_hct.select_dtypes(include=['object']).columns
for col in categorical_cols_test:
    test_hct[col] = test_hct[col].astype('category')


test_hct = test_hct.fillna(test_hct.mean())  # Replace NaNs with column mean (or use a method of choice)
test_hct = test_hct.replace([np.inf, -np.inf], np.nan).fillna(test_hct.ffill())  # Handle infinities

# test_hct = test_hct.loc[:, (test_hct.nunique() > 1)]

test_hct_1 = test_hct.clip(lower=test_hct.quantile(0.01), upper=test_hct.quantile(0.99), axis=1)
survival_function_results=rsf.predict_survival_function(test_hct.drop(['ID'],axis=1))


# List to hold final cumulative event probabilities
final_cumulative_event_probabilities = []

# Calculate final cumulative event probability for each individual
for result in survival_function_results:
    final_survival_prob = result.y[-1]  # Last survival probability value
    final_cumulative_event_prob = 1 - final_survival_prob  # Cumulative event probability (1 - survival probability)
    final_cumulative_event_probabilities.append(final_cumulative_event_prob)

# Create a DataFrame with the predictions
df_preds = pd.DataFrame(final_cumulative_event_probabilities, columns=["prediction"])


df=pd.concat([test_hct['ID'],df_preds],axis=1)


df.to_csv('submission.csv',index=False)


df




