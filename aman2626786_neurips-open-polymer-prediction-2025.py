import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# for removing the extra warnings
import warnings
warnings.filterwarnings('ignore')


# pd.read_csv("sample_submission.csv")


train = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train.csv")


test = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/test.csv")


train.sample(10)


train.info()


train.isnull().sum()
# Here we can see that there is a lot of missing values and data is also imbalanced.


train.shape


train.duplicated().sum()
# There is no duplicated values


train.describe()


test.head()
# Only three sample here, very less amount of data.


train.dropna()  # This is bed thing, we need to clerify it.


train['SMILES'][0]


# pip install rdkit


# !pip install xgboost


# Required Libraries
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import (
    RandomForestRegressor, AdaBoostRegressor,
    BaggingRegressor, ExtraTreesRegressor,
    GradientBoostingRegressor
)
from xgboost import XGBRegressor


# Step 1: Feature Extraction from SMILES
def featurize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return {
            'MolWt': Descriptors.MolWt(mol),
            'NumHDonors': Descriptors.NumHDonors(mol),
            'NumHAcceptors': Descriptors.NumHAcceptors(mol),
            'TPSA': Descriptors.TPSA(mol),
            'MolLogP': Descriptors.MolLogP(mol),
            'RingCount': Descriptors.RingCount(mol),
        }
    else:
        return {
            'MolWt': None, 'NumHDonors': None, 'NumHAcceptors': None,
            'TPSA': None, 'MolLogP': None, 'RingCount': None
        }



# Apply feature extraction to train and test
train_features = train['SMILES'].apply(featurize).apply(pd.Series)
test_features = test['SMILES'].apply(featurize).apply(pd.Series)


train_features


test_features


# Combine features with original data
train_full = pd.concat([train, train_features], axis=1)
test_full = pd.concat([test, test_features], axis=1)


train_full


test_full


# Step 2: Train models on label and predict on test
labels = ['FFV', 'Tc', 'Tg', 'Density', 'Rg']
models = {}
predictions = pd.DataFrame()
predictions['id'] = test['id']

for label in labels:
    df = train_full[train_full[label].notnull()]
    X = df[['MolWt', 'NumHDonors', 'NumHAcceptors', 'TPSA', 'MolLogP', 'RingCount']]
    y = df[label]

    if len(df) > 10:  # Only train if there's enough data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        model = BaggingRegressor(n_estimators= 50)
        model.fit(X_train, y_train)
        val_preds = model.predict(X_val)
        val_mae = mean_absolute_error(y_val, val_preds)
        val_r2s = r2_score(y_val, val_preds)
        models[label] = model

        # Predict on test set
        X_test = test_full[['MolWt', 'NumHDonors', 'NumHAcceptors', 'TPSA', 'MolLogP', 'RingCount']]
        test_preds = model.predict(X_test)
        predictions[label] = test_preds
    else:
        predictions[label] = train[label].mean()  # fallback if not enough data


val_mae


val_r2s


predictions.head()


models = {
    'LinearRegression': (LinearRegression(), {}),
    'Ridge': (Ridge(), {'alpha': [0.1, 1.0, 10]}),
    'Lasso': (Lasso(), {'alpha': [0.001, 0.01, 0.1]}),
    'DecisionTree': (DecisionTreeRegressor(), {'max_depth': [5, 10, None]}),
    'KNN': (KNeighborsRegressor(), {'n_neighbors': [3, 5, 7]}),
    'RandomForest': (RandomForestRegressor(), {'n_estimators': [50, 100], 'max_depth': [5, 10, None]}),
    'AdaBoost': (AdaBoostRegressor(), {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1, 1.0]}),
    'Bagging': (BaggingRegressor(), {'n_estimators': [10, 50]}),
    'ExtraTrees': (ExtraTreesRegressor(), {'n_estimators': [50, 100], 'max_depth': [5, 10, None]}),
    'GradientBoosting': (GradientBoostingRegressor(), {'n_estimators': [50, 100], 'learning_rate': [0.05, 0.1]}),
    'XGBoost': (XGBRegressor(), {'n_estimators': [50, 100], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5]})
}



# Define the labels to predict
labels = ['FFV', 'Tc', 'Tg', 'Density', 'Rg']
predictions = pd.DataFrame({'id': test['id']})
results = []

for label in labels:
    print(f"\nğŸ”� Training for label: {label}")
    df = train_full[train_full[label].notnull()]
    X = df[['MolWt', 'NumHDonors', 'NumHAcceptors', 'TPSA', 'MolLogP', 'RingCount']]
    y = df[label]

    if len(df) > 10:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        best_mae = float('inf')
        best_model = None
        best_model_name = ''
        best_params = {}

        for name, (model, param_grid) in models.items():
            print(f"  â†’ Trying {name} ...")
            grid = GridSearchCV(model, param_grid, scoring='neg_mean_absolute_error', cv=3, n_jobs=-1)
            grid.fit(X_train, y_train)

            preds = grid.predict(X_val)
            mae = mean_absolute_error(y_val, preds)
            r2 = r2_score(y_val, preds)

            if mae < best_mae:
                best_mae = mae
                best_model = grid.best_estimator_
                best_model_name = name
                best_params = grid.best_params_

        # Store prediction in submission file
        X_test = test_full[['MolWt', 'NumHDonors', 'NumHAcceptors', 'TPSA', 'MolLogP', 'RingCount']]
        test_preds = best_model.predict(X_test)
        predictions[label] = test_preds

        results.append({
            'Label': label,
            'Best Model': best_model_name,
            'Best Params': best_params,
            'MAE': best_mae,
            'R2 Score': r2_score(y_val, best_model.predict(X_val))
        })

    else:
        # Fallback if insufficient data
        predictions[label] = train[label].mean()
        results.append({
            'Label': label,
            'Best Model': 'None',
            'Best Params': {},
            'MAE': 'Insufficient data',
            'R2 Score': '-'
        })

# Export to submission file
predictions.to_csv("submission.csv", index=False)

# Print model summary
results_df = pd.DataFrame(results)
print("\nğŸ“Š Best Models Summary:")
print(results_df)



predictions

