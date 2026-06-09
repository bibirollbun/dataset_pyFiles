!pip install rdkit


import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error


train_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
test_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')


train_df.head()


print("\nMissing values in training data:")
print(train_df.isnull().sum())


def smiles_to_descriptors(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        descriptor_names = [x[0] for x in Descriptors.descList]
        calculator = Descriptors.MolecularDescriptorCalculator(descriptor_names)
        descriptors = calculator.CalcDescriptors(mol)
        return pd.Series(descriptors, index=descriptor_names)
    except:
        return None


train_descriptors = train_df['SMILES'].apply(smiles_to_descriptors)
test_descriptors = test_df['SMILES'].apply(smiles_to_descriptors)


train_processed = pd.concat([train_df['id'], train_descriptors, train_df.iloc[:, 2:]], axis=1).dropna(subset=['id']).reset_index(drop=True)
test_processed = pd.concat([test_df['id'], test_descriptors], axis=1).dropna(subset=['id']).reset_index(drop=True)


train_processed.replace([np.inf, -np.inf], np.nan, inplace=True)
test_processed.replace([np.inf, -np.inf], np.nan, inplace=True)
train_processed.fillna(train_processed.mean(), inplace=True)
test_processed.fillna(train_processed.mean(), inplace=True) 


target_cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
feature_cols = [col for col in train_processed.columns if col not in ['id'] + target_cols]


submission_df = test_df[['id']].copy()
for col in target_cols:
    submission_df[col] = 0.0 


for target in target_cols:
    print(f"Training model for target: {target}")


for target in target_cols:
    print(f"Training model for target: {target}")
    
    train_data = train_processed.dropna(subset=[target])
    X_train = train_data[feature_cols]
    y_train = train_data[target]

    X_test = test_processed[feature_cols]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
        print(f"  Fold {fold+1}")
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        lgb_params = {
            'objective': 'regression_l1',  
            'metric': 'mae',
            'n_estimators': 1000,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 1,
            'verbose': -1,
            'n_jobs': -1,
            'seed': 42,
            'boosting_type': 'gbdt',
        }

        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(X_fold_train, y_fold_train,
                  eval_set=[(X_fold_val, y_fold_val)],
                  eval_metric='mae',
                  callbacks=[lgb.early_stopping(100, verbose=False)])

        oof_preds[val_idx] = model.predict(X_fold_val)
        test_preds += model.predict(X_test) / kf.n_splits
    
    submission_df[target] = test_preds

    mae_oof = mean_absolute_error(y_train, oof_preds)
    print(f"  OOF MAE for {target}: {mae_oof:.4f}\n")


print("Submission file head:")
print(submission_df.head())

submission_df.to_csv('submission.csv', index=False)

