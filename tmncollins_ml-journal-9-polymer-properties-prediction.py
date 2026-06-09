try:
    import rdkit
except:
    !pip install --quiet /kaggle/input/rdkit-2025-3-3-cp311/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import networkx as nx
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdmolops
from rdkit import Chem
import pickle

import warnings
warnings.filterwarnings('ignore')


train_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
test_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')


def canonical_smiles(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    canonical = Chem.MolToSmiles(molecule, canonical = True)
    return canonical

def compute_all_descriptors(smiles):
	mol = Chem.MolFromSmiles(smiles)
	if mol is None:
		return [None] * (len(Descriptors.descList) - len(useless_cols))
	return [desc[1](mol) for desc in Descriptors.descList]

def preprocessing(df):
    desc_names = [desc[0] for desc in Descriptors.descList]
    descriptors = [compute_all_descriptors(smile) for smile in df['SMILES'].to_list()]

    result = pd.DataFrame(descriptors, columns = desc_names)
    result = result.replace([-np.inf, np.inf], np.nan)
    return result


train_df = preprocessing(train_df)
train_df = train_df.dropna(axis=1)
test_df = preprocessing(test_df)


train_df.to_csv('train_data.csv')


init_train_data = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')

for col in test_df.columns:
    if col not in train_df.columns:
        test_df = test_df.drop(col, axis=1)


import seaborn as sns

#plotting the heatmap for correlation
corr_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
ax = sns.heatmap(corr_df.select_dtypes(include=np.number).drop('id', axis=1).corr(), annot=True)


def make_bins(items, step=1):
    items = items.dropna()
    bins = []
    i = min(items)
    while i < max(items):
        bins.append(i)
        i += step
    return bins

#Index(['id', 'SMILES', 'Tg', 'FFV', 'Tc', 'Density', 'Rg'], dtype='object')

plt.hist(init_train_data['Rg'], bins=make_bins(init_train_data['Rg'], step=0.5), density=False)
plt.ylabel('Frequency')
plt.xlabel('Rg / Å')
plt.show()

print(len(init_train_data))


from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import xgboost as xgb
from sklearn.impute import SimpleImputer

targets = ['FFV', 'Tg', 'Tc', 'Rg', 'Density']
seed = 42
np.random.seed(seed)

xgb_params = {
    'n_estimators': 5000,
    'objective': 'reg:squarederror',
    'learning_rate': 0.01,
    'max_depth': 4,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'reg_alpha': 1,
    'reg_lambda': 1,
    'verbosity': 0,
}

pred_classes = {}

class VerboseEveryN(xgb.callback.TrainingCallback):
    def __init__(self, n):
        self.n = n
    def after_iteration(self, model, epoch, evals_log):
        if epoch % self.n == 0:
            for data_name, metrics in evals_log.items():
                for metric_name, values in metrics.items():
                    print(f"Round {epoch} - {data_name} {metric_name}: {values[-1]: .4f}")
        return False


        
for target in targets:
    print(f"\n\nCurrently working on target: {target}")
    X = train_df
    y = init_train_data[target]
    folds = 3
    kf = KFold(n_splits = folds)
    X = X[y.notnull()].reset_index(drop = True)
    y = y[y.notnull()].reset_index(drop = True)
    
    # clipping data to prevent inf/missing=inf error from xgb could also do (-1e10, 1e10)
    X = X.clip(-1e6, 1e6)
    y = y.clip(-1e6, 1e6)
    
    # Final ensemble prediction
    current_class_pred = np.zeros((1, len(test_df)), dtype = float)

    for fold, (train_idx, test_idx) in enumerate(kf.split(X, y)):
        print(f"\n--- Fold {fold + 1} / {folds} ---")
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

        xgb_callbacks = [VerboseEveryN(2000)]
        model = xgb.XGBRegressor(**xgb_params)
        model.fit(
            X_train, y_train,
            eval_set = [(X_test, y_test)],
            early_stopping_rounds = 200,
            verbose = 0,
            callbacks = xgb_callbacks
        )
        pred = model.predict(test_df)
        current_class_pred += model.predict(test_df)
    current_class_pred /= folds
    pred_classes[target] = current_class_pred


def scatter_plot(pred, true, x_label=None, y_label=None, title=None):
    plt.scatter(pred, true, s=3)
    plt.plot(true, true, c='black')
    if x_label: plt.xlabel(x_label)
    if y_label: plt.ylabel(y_label)
    if title: plt.title(title)
    plt.show()


for target in targets:
    X = train_df
    y = init_train_data[target]
    
    # clipping data to prevent inf/missing=inf error from xgb could also do (-1e10, 1e10)
    X = X[y.notnull()].reset_index(drop = True)
    y = y[y.notnull()].reset_index(drop = True)
    X = X.clip(-1e6, 1e6)
    y = y.clip(-1e6, 1e6)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=seed)
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(
        X_train, y_train,
        eval_set = [(X_val, y_val)],
        early_stopping_rounds = 200,
        verbose = 0
    )

    pred_y = model.predict(X_test)
    scatter_plot(pred_y, y_test, x_label='Predicted', y_label='Experimental', title=target)

    print(f'Target: {target}. Test Score:', r2_score(y_test, pred_y))



for target in targets:
    X = train_df
    y = init_train_data[target]
    
    # clipping data to prevent inf/missing=inf error from xgb could also do (-1e10, 1e10)
    X = X[y.notnull()].reset_index(drop = True)
    y = y[y.notnull()].reset_index(drop = True)
    X = X.clip(-1e6, 1e6)
    y = y.clip(-1e6, 1e6)

    # Kfolds
    folds = 3
    kf = KFold(n_splits = folds)
    
    X_t, X_test, y_t, y_test = train_test_split(X, y, test_size=0.15, random_state=seed)
    
    # Final ensemble prediction
    current_class_pred = np.zeros((1, len(X_test)), dtype = float)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_t, y_t)):
        print(f"\n--- Fold {fold + 1} / {folds} ---")
        X_train, y_train = X_t.iloc[train_idx], y_t.iloc[train_idx]
        X_val, y_val = X_t.iloc[val_idx], y_t.iloc[val_idx]

        xgb_callbacks = [VerboseEveryN(2000)]
        model = xgb.XGBRegressor(**xgb_params)
        model.fit(
            X_train, y_train,
            eval_set = [(X_val, y_val)],
            early_stopping_rounds = 200,
            verbose = 0,
            callbacks = xgb_callbacks
        )
        
        current_class_pred += model.predict(X_test)

    current_class_pred /= folds

    scatter_plot(current_class_pred, y_test, 
                 x_label='Predicted', y_label='Experimental', title=target)

    print(f'Target: {target}. Test Score:', r2_score(y_test, current_class_pred.flatten()))



sample = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/test.csv")

submission = pd.DataFrame({
    "id": sample['id'],
    'Tg': pred_classes['Tg'].squeeze(),
    'FFV': pred_classes['FFV'].squeeze(),
    'Tc': pred_classes['Tc'].squeeze(),
    'Density': pred_classes['Density'].squeeze(),
    'Rg': pred_classes['Rg'].squeeze()
})


submission.to_csv('submission.csv', index = False)


from sklearn.ensemble import BaggingRegressor, VotingRegressor

bag_size = 10
print(f'Bagging {bag_size} XGB Regressors')

for target in targets:
    xgb_estimator = xgb.XGBRegressor(**xgb_params)
    bag_model = BaggingRegressor(xgb_estimator, bag_size, random_state=42)
    
    X = train_df
    y = init_train_data[target]
    
    # clipping data to prevent inf/missing=inf error from xgb could also do (-1e10, 1e10)
    X = X[y.notnull()].reset_index(drop = True)
    y = y[y.notnull()].reset_index(drop = True)
    X = X.clip(-1e6, 1e6)
    y = y.clip(-1e6, 1e6)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    
    bag_model.fit(X_train, y_train)
    print(target, bag_model.score(X_test, y_test))



sample = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/test.csv")

pred_bagging = dict()

bag_size = 10
print(f'Bagging {bag_size} XGB Regressors')

for target in targets:
    print(target)
    xgb_estimator = xgb.XGBRegressor(**xgb_params)
    bag_model = BaggingRegressor(xgb_estimator, bag_size, random_state=42)
    
    X = train_df
    y = init_train_data[target]
    
    # clipping data to prevent inf/missing=inf error from xgb could also do (-1e10, 1e10)
    X = X[y.notnull()].reset_index(drop = True)
    y = y[y.notnull()].reset_index(drop = True)
    X = X.clip(-1e6, 1e6)
    y = y.clip(-1e6, 1e6)
    
    bag_model.fit(X, y)
    pred_bagging[target] = bag_model.predict(test_df)

submission = pd.DataFrame({
    "id": sample['id'],
    'Tg': pred_bagging['Tg'].squeeze(),
    'FFV': pred_bagging['FFV'].squeeze(),
    'Tc': pred_bagging['Tc'].squeeze(),
    'Density': pred_bagging['Density'].squeeze(),
    'Rg': pred_bagging['Rg'].squeeze()
})

submission.to_csv('submission_bagging.csv', index = False)


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.neighbors import KNeighborsRegressor

for target in targets:
    r1 = LinearRegression()
    r2 = RandomForestRegressor(n_estimators=30, random_state=42)
    r3 = KNeighborsRegressor()
    r4 = xgb.XGBRegressor(**xgb_params)

    voter = VotingRegressor([('lr', r1), ('rf', r2), ('kn', r3)])   

    X = train_df
    y = init_train_data[target]
    
    # clipping data to prevent inf/missing=inf error from xgb could also do (-1e10, 1e10)
    X = X[y.notnull()].reset_index(drop = True)
    y = y[y.notnull()].reset_index(drop = True)
    X = X.clip(-1e6, 1e6)
    y = y.clip(-1e6, 1e6)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    
    voter.fit(X_train, y_train)
    print(target, voter.score(X_test, y_test))



import shap

for target in targets:
    X = train_df
    y = init_train_data[target]
    
    # clipping data to prevent inf/missing=inf error from xgb could also do (-1e10, 1e10)
    X = X[y.notnull()].reset_index(drop = True)
    y = y[y.notnull()].reset_index(drop = True)
    X = X.clip(-1e6, 1e6)
    y = y.clip(-1e6, 1e6)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=seed)
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(
        X_train, y_train,
        eval_set = [(X_val, y_val)],
        early_stopping_rounds = 200,
        verbose = 0
    )

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_test)
    shap.initjs()
    print()
    shap.force_plot(explainer.expected_value, shap_values[0], X_test.values[0], feature_names = X_test.columns.tolist(), matplotlib=True)
    plt.show()


shap.summary_plot(shap_values, X_test)

