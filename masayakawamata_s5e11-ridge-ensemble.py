import warnings
warnings.simplefilter('ignore')


import pandas as pd, numpy as np

train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')

train.head(3)


TARGET = 'loan_paid_back'


import glob, os

all_oof_data = []
all_test_data = []

oof_files = glob.glob('/kaggle/input/**/oof_*.csv', recursive=True)
print(f"Found {len(oof_files)} oof files.")

for oof_path in oof_files:
    test_path = oof_path.replace('oof_', 'test_')

    base_name = os.path.basename(oof_path)
    model_name = base_name.replace('oof_', '').replace('.csv', '')

    all_oof_data.append({
        'df': pd.read_csv(oof_path),
        'name': model_name
    })
    all_test_data.append({
        'df': pd.read_csv(test_path),
        'name': model_name
    })

def merge_dataframes_by_id(data_list, id_col='id', feature_col=TARGET):

    first_data = data_list[0]
    merged_df = first_data['df'].rename(columns={
        feature_col: f"{feature_col}_{first_data['name']}"
    })

    for data in data_list[1:]:
        renamed_df = data['df'].rename(columns={
            feature_col: f"{feature_col}_{data['name']}"
        })
        merged_df = pd.merge(merged_df, renamed_df, on=id_col, how='outer')

    return merged_df

oof_df = merge_dataframes_by_id(all_oof_data)
test_df = merge_dataframes_by_id(all_test_data)

oof_df[TARGET] = train[TARGET].values

oof_df.head(3)


FEATURES = [col for col in oof_df.columns if col not in ['id',TARGET]]

X = oof_df[FEATURES]
y = oof_df[TARGET]


from sklearn.model_selection import StratifiedKFold

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score


oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f'---Fold {fold+1}/5---')
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx] 

    X_test = test_df[FEATURES].copy()
    
    model = Ridge()
    model.fit(X_train, y_train)

    val_preds = model.predict(X_val)
    oof_preds[val_idx] += val_preds

    test_preds += model.predict(X_test)

    print(f"Fold {fold+1} AUC: {roc_auc_score(y_val, val_preds)}")

test_preds /= 5
overall_score = roc_auc_score(y, oof_preds)
print(f"Overall OOF AUC: {overall_score}")

coeffs = pd.Series(model.coef_, index=FEATURES)
print("--- Model Coefficients ---")
print(coeffs.sort_values(ascending=False))


print('--- Training on all data ---')
X_test = test_df[FEATURES].copy()

model = Ridge()
model.fit(X, y)

test_preds = model.predict(X_test)

print("Model training and prediction complete.")

coeffs = pd.Series(model.coef_, index=FEATURES)
print("--- Model Coefficients (Full Data) ---")
print(coeffs.sort_values(ascending=False))


pd.DataFrame({'id': train.id, TARGET: oof_preds}).to_csv(f'oof_l2_ridge_{overall_score}.csv', index=False)
pd.DataFrame({'id': test.id, TARGET: test_preds}).to_csv(f'test_l2_ridge_{overall_score}.csv', index=False)




