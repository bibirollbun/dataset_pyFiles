import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import shap
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
import xgboost as xgb



train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv', index_col='id')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv', index_col='id')
original = pd.read_csv('/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv').rename_axis('id', axis='index')

combined = pd.concat([train, test, original])


combined['train'] = ['train' for i in range(0, train.shape[0])] + ['test' for i in range(0, test.shape[0])]  + ['original' for i in range(0, original.shape[0])]


shap_columns = pd.read_csv('/kaggle/input/predicting-fertilizers/shap_cv_importance.csv')


shap_columns = shap_columns.loc[:shap_columns[shap_columns['feature'] == 'random'].index[0] - 1, :]


features = pd.concat([shap_columns['feature'], pd.Series(['train'])], ignore_index=True)


x, y = combined.drop(columns = ['Fertilizer Name']), combined[combined['train']=='train']['Fertilizer Name']

x['soil_crop_type'] = x['Soil Type'] + '_' + x['Crop Type']

x['Nitrogen_Potassium'] = x['Nitrogen'] * x['Potassium']
x['Phosphorous_Potassium'] = x['Phosphorous'] * x['Potassium']
x['Nitrogen_Phosphorous'] = x['Nitrogen'] * x['Phosphorous']
x['N_to_K'] = x['Nitrogen'] / (x['Potassium'] + 1e-5)
x['N_to_P'] = x['Nitrogen'] / (x['Phosphorous'] + 1e-5)
x['Moisture_Temp_Interaction'] = x['Moisture'] * x['Temparature']
x['K_to_P'] = x['Potassium'] / (x['Phosphorous'] + 1e-5)
x['Humidity_Temp_Interaction'] = x['Humidity'] * x['Temparature']
x['P_minus_K'] = x['Phosphorous'] - x['Potassium']
x['NPK_sum'] = x['Nitrogen'] + x['Phosphorous'] + x['Potassium']
x['Humidity_to_Temp'] = x['Humidity'] / (x['Temparature'] + 1e-5)
x['N_minus_P'] = x['Nitrogen'] - x['Phosphorous']
x['Moisture_to_Temp'] = x['Moisture'] / (x['Temparature'] + 1e-5)
x['N_minus_K'] = x['Nitrogen'] - x['Potassium']



x = x[features]


objects = [col for col in x.columns if x[col].dtype == 'O']


x[objects] = x[objects].astype('category')


cats = [col for col in x.columns if x[col].dtype == 'category']
nums = [col for col in x.columns if col not in cats]
del objects


def reduce_numeric_memory(df, nums, verbose=True):
    df = df[nums].copy()
    start_mem = df.memory_usage(deep=True).sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtypes

        if pd.api.types.is_numeric_dtype(col_type):
            c_min = df[col].min()
            c_max = df[col].max()

            if pd.api.types.is_integer_dtype(col_type):
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)

            elif pd.api.types.is_float_dtype(col_type):
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage(deep=True).sum() / 1024**2

    if verbose:
        print(f'Memory usage reduced from {start_mem:.2f} MB to {end_mem:.2f} MB '
              f'({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)')

    return df


x[nums] = reduce_numeric_memory(x, nums)


correlation_matrix = x[nums].corr().abs()
high_corr_pairs = (
    correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    .stack()
    .reset_index()
)
high_corr_pairs.columns = ['feature_1', 'feature_2', 'correlation']
high_corr_pairs = high_corr_pairs[high_corr_pairs['correlation'] > 0.7]


high_corr_pairs


shap_map = shap_columns.set_index('feature')['mean_abs_shap']
high_corr_pairs['shap_1'] = high_corr_pairs['feature_1'].map(shap_map)
high_corr_pairs['shap_2'] = high_corr_pairs['feature_2'].map(shap_map)

high_corr_pairs['drop'] = np.where(high_corr_pairs['shap_1'] > high_corr_pairs['shap_2'],
                                   high_corr_pairs['feature_2'],
                                   high_corr_pairs['feature_1'])

dropping_features = high_corr_pairs['drop'].unique().tolist()


x = x.drop(columns = dropping_features)


best_params = {
    'learning_rate': 0.02329520415647146,
    'max_depth': 13,
    'min_child_weight': 6.069867182685234,
    'gamma': 0.2614297036064851,
    'subsample': 0.7996473562373125,
    'colsample_bytree': 0.3353707236850521,
    'reg_lambda': 3.2913115742063654,
    'reg_alpha': 3.0476063115519225
}


def mapk(actual, predicted, k=3):
    def apk(a, p, k):
        score = 0.0
        for i in range(min(k, len(p))):
            if p[i] == a:
                score += 1.0 / (i + 1)
                break
        return score
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


original = x[x['train'] == 'original']
test = x[x['train'] == 'test']
x = x[x['train'] == 'train']

x = x.drop(columns=['train'])
test = test.drop(columns=['train'])
original = original.drop(columns=['train'])
le = LabelEncoder()


y_original = combined[combined['train']=='original']['Fertilizer Name']

y_encoded = le.fit_transform(y)
num_classes = len(np.unique(y_encoded))

y_encoded_original = le.transform(y_original)

FOLDS = 5
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

oof_preds = np.zeros((len(x), num_classes))
pred_prob = np.zeros((len(test), num_classes))

params = {
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
    "num_class": num_classes,
    "tree_method": "hist",
    "device": "cuda",
}
params.update(best_params)

for fold, (train_idx, valid_idx) in enumerate(skf.split(x, y_encoded)):

    X_train, X_valid = x.iloc[train_idx], x.iloc[valid_idx]
    y_train, y_valid = y_encoded[train_idx], y_encoded[valid_idx]

    X_train = pd.concat([X_train, original], ignore_index=True)
    y_train = np.concatenate((y_train, y_encoded_original))
    
    dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
    dval = xgb.DMatrix(X_valid, label=y_valid, enable_categorical=True)

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=4000,
        evals=[(dval, "validation")],
        early_stopping_rounds=250,
        verbose_eval=False
    )

    proba_valid = model.predict(dval)
    oof_preds[valid_idx] = proba_valid

    top_3_preds = np.argsort(proba_valid, axis=1)[:, -3:][:, ::-1]
    actual = [[label] for label in y_valid]
    map3_score = mapk(actual, top_3_preds)
    print(f"âœ… FOLD {fold+1}: MAP@3 Score: {map3_score:.5f}")

    # Predicting on test
    dtest = xgb.DMatrix(test, enable_categorical=True)
    pred_prob += model.predict(dtest)

top_3_preds_all = np.argsort(oof_preds, axis=1)[:, -3:][:, ::-1]
final_map3 = mapk(y_encoded, top_3_preds_all)
print(f"\nðŸ“Š Final OOF MAP@3: {final_map3:.5f}")


pred_prob


pred_prob /= FOLDS
top_3_test_preds = np.argsort(pred_prob, axis=1)[:, -3:][:, ::-1]


submission = pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')

predictions = [le.inverse_transform(row) for row in top_3_test_preds]

submission['Fertilizer Name'] = predictions

submission['Fertilizer Name'] = submission['Fertilizer Name'].apply(lambda x: " ".join(x))

submission.to_csv("submission.csv", index=False)


submission

