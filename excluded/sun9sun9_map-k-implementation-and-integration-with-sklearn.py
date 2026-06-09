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


df_train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv', index_col = ['id'])


from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

# Define the target variable (the label we want to predict)
target = 'Fertilizer Name'

# Define the list of categorical features
X_cat = ['Soil Type', 'Crop Type']

# Define the list of numerical features
X_num = [i for i in df_train.columns if i not in X_cat and i != target]

# Combine categorical and numerical feature lists
X_all = X_cat + X_num

# Create a LabelEncoder for encoding the target variable
le = LabelEncoder()
le.fit(df_train[target])  # Fit the label encoder to the target values

# Create an OrdinalEncoder for encoding the categorical input features
# The output will be a pandas DataFrame
oe = OrdinalEncoder().set_output(transform='pandas')

# Replace categorical features with their ordinal-encoded versions,
# and transform the target variable into label-encoded values
df_train = pd.concat([
    # Drop the original categorical columns and add a new column 'target_l'
    # with the label-encoded target values
    df_train.drop(columns=X_cat).assign(
        target_l=lambda x: le.transform(x[target])
    ),
    # Ordinal-encode the categorical input features
    oe.fit_transform(df_train[X_cat])
], axis=1)


from sklearn.metrics import make_scorer

# Define a custom scoring function for MAP@k using prediction probabilities
def mapk_prob(y_true, y_prob, k=3):
    return (
        # Step 1: argsort(-y_prob) gives indices of predicted classes in descending order of probability
        # Step 2: Compare top-k predictions with the true label (broadcasted across k predictions)
        # Step 3: Assign precision weights: 1/1, 1/2, ..., 1/k for top-k ranks
        # Step 4: Take the dot product to get the weighted score for each observation
        # Step 5: Average over all observations
        (np.argsort(-y_prob, axis=1) == np.expand_dims(y_true, axis=-1))[:, :k].dot(1 / np.arange(1, k + 1))
    ).mean()

# Wrap the custom scoring function into a scikit-learn scorer object
# This allows it to be used in model selection tools like GridSearchCV
map3_scorer = make_scorer(
    mapk_prob,         # The custom MAP@3 function
    needs_proba=True   # Specifies that the model must output class probabilities
)


from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
import lightgbm as lgb

# Create a stratified shuffle split cross-validator with 1 split and a fixed random seed for reproducibility
sss = StratifiedShuffleSplit(n_splits=1, random_state=123)

# Evaluate the model using cross-validation and the custom MAP@3 scorer
cross_val_score(
    lgb.LGBMClassifier(verbose=0),           # LightGBM classifier with no output verbosity
    df_train[X_all],                         # Feature set (both categorical and numerical)
    df_train['target_l'],                    # Label-encoded target values
    cv=sss,                                  # Use stratified shuffle split cross-validation
    scoring=map3_scorer,                     # Use the custom MAP@3 scoring function
    fit_params={                             # Pass categorical feature information to LightGBM
        'categorical_feature': ['Soil Type', 'Crop Type']  # Specify categorical columns by name
    }
)




