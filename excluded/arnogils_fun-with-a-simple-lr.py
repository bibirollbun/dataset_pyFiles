import warnings

warnings.filterwarnings("ignore")

%load_ext cuml.accel
%load_ext cudf.pandas


import pathlib
import pandas as pd
import warnings

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 25)

warnings.filterwarnings("ignore")

input_path_comp = pathlib.Path('/kaggle/input/playground-series-s5e6')
input_path_orig = pathlib.Path('/kaggle/input/fertilizer-prediction')

train_df = pd.read_csv(input_path_comp / 'train.csv', index_col='id')
train_df.columns = train_df.columns.str.lower()
train_df.columns = train_df.columns.str.split().str.join('_')

test_df = pd.read_csv(input_path_comp / 'test.csv', index_col='id')
test_df.columns = test_df.columns.str.lower()
test_df.columns = test_df.columns.str.split().str.join('_')

orig_df = pd.read_csv(input_path_orig / 'Fertilizer Prediction.csv')
orig_df.columns = orig_df.columns.str.lower()
orig_df.columns = orig_df.columns.str.split().str.join('_')

sample_submission = pd.read_csv(input_path_comp / 'sample_submission.csv')

train_df.shape, test_df.shape, orig_df.shape


train_df = train_df.rename(columns={'crop_type': 'croptype', 'soil_type': 'soiltype'})
test_df = test_df.rename(columns={'crop_type': 'croptype', 'soil_type': 'soiltype'})
orig_df = orig_df.rename(columns={'crop_type': 'croptype', 'soil_type': 'soiltype'})


train_df['const'] = 1
test_df['const'] = 1
orig_df['const'] = 1


from itertools import combinations

num_cols = [
    'temparature',
    'humidity',
    'moisture',
    'nitrogen',
    'potassium',
    'phosphorous'
]

cat_cols = [
    'soiltype',
    'croptype',
]

target = 'fertilizer_name'


def mapk(actual, predicted, k=3):
    def apk(a, p, k):
        p = p[:k]
        score = 0.0
        hits = 0
        seen = set()
        for i, pred in enumerate(p):
            if pred in a and pred not in seen:
                hits += 1
                score += hits / (i + 1.0)
                seen.add(pred)
        return score / min(len(a), k)
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


def augment_data_as_sparse_matrix(X_train, y_train, X_orig, y_orig, weight_orig=1.0):

    X_train_sparse = X_train if isinstance(X_train, csr_matrix) else csr_matrix(X_train)
    X_orig_sparse = X_orig if isinstance(X_orig, csr_matrix) else csr_matrix(X_orig)

    X_augmented = vstack([X_train_sparse, X_orig_sparse])
    y_augmented = np.concatenate([np.asarray(y_train), np.asarray(y_orig)])

    weights_new = np.full(X_train_sparse.shape[0], 1.0)
    weights_orig = np.full(X_orig_sparse.shape[0], weight_orig)
    sample_weights_augmented = np.concatenate([weights_new, weights_orig])

    return X_augmented, y_augmented, sample_weights_augmented


comb3 = ['croptype_nitrogen_phosphorous',
 'nitrogen_potassium_phosphorous',
 'moisture_nitrogen_phosphorous',
 'moisture_croptype_phosphorous',
 'moisture_potassium_phosphorous',
 'moisture_croptype_nitrogen',
 'moisture_nitrogen_potassium',
 'humidity_nitrogen_phosphorous',
 'croptype_potassium_phosphorous',
 'humidity_moisture_phosphorous',
 'temparature_nitrogen_phosphorous',
 'temparature_moisture_phosphorous',
 'croptype_nitrogen_potassium',
 'soiltype_nitrogen_phosphorous',
 'humidity_moisture_nitrogen',
 'moisture_soiltype_phosphorous',
 'humidity_croptype_phosphorous',
 'moisture_croptype_potassium',
 'humidity_potassium_phosphorous',
 'temparature_potassium_phosphorous',
 'temparature_croptype_phosphorous',
 'temparature_moisture_nitrogen',
 'humidity_croptype_nitrogen',
 'humidity_nitrogen_potassium',
 'soiltype_croptype_phosphorous',
 'moisture_soiltype_nitrogen',
 'humidity_moisture_croptype',
 'temparature_nitrogen_potassium',
 'humidity_moisture_potassium',
 'soiltype_potassium_phosphorous',
 'temparature_moisture_croptype',
 'temparature_croptype_nitrogen',
 'soiltype_croptype_nitrogen',
 'temparature_moisture_potassium',
 'nitrogen_phosphorous_const',
 'soiltype_nitrogen_potassium',
 'moisture_phosphorous_const',
 'moisture_soiltype_croptype',
 'temparature_humidity_phosphorous',
 'moisture_soiltype_potassium',
 'humidity_soiltype_phosphorous',
 'humidity_croptype_potassium',
 'temparature_humidity_nitrogen',
 'moisture_nitrogen_const',
 'temparature_humidity_moisture',
 'temparature_croptype_potassium',
 'temparature_soiltype_phosphorous',
 'croptype_phosphorous_const',
 'humidity_soiltype_nitrogen',
 'potassium_phosphorous_const',
 'humidity_moisture_soiltype',
 'soiltype_croptype_potassium',
 'temparature_moisture_soiltype',
 'moisture_croptype_const',
 'croptype_nitrogen_const',
 'temparature_soiltype_nitrogen',
 'nitrogen_potassium_const',
 'temparature_humidity_croptype',
 'moisture_potassium_const',
 'humidity_soiltype_croptype',
 'temparature_humidity_potassium',
 'humidity_phosphorous_const',
 'temparature_soiltype_croptype',
 'humidity_soiltype_potassium',
 'temparature_phosphorous_const',
 'humidity_moisture_const',
 'temparature_soiltype_potassium',
 'humidity_nitrogen_const',
 'temparature_moisture_const',
 'croptype_potassium_const',
 'soiltype_phosphorous_const',
 'moisture_soiltype_const',
 'temparature_nitrogen_const',
 'humidity_croptype_const',
 'temparature_humidity_soiltype',
 'soiltype_nitrogen_const',
 'temparature_croptype_const',
 'humidity_potassium_const',
 'temparature_potassium_const',
 'soiltype_croptype_const',
 'soiltype_potassium_const',
 'temparature_humidity_const',
 'humidity_soiltype_const',
 'temparature_soiltype_const']


import numpy as np
from itertools import combinations
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack, vstack, csr_matrix

X = train_df
X = X.astype(str)

X_o = orig_df
X_o = X_o.astype(str)

X_t = test_df
X_t = X_t.astype(str)

le = LabelEncoder()
y = X.pop('fertilizer_name')
y = le.fit_transform(y)

y_o = X_o.pop('fertilizer_name')
y_o = le.transform(y_o)

X_all = pd.concat([X, X_o, X_t], ignore_index=True).astype(str)

X_all_e = X_all.copy()
for c1, c2 in combinations(X_all.columns, 2):
    X_all_e[c1+'_'+c2] = X_all[c1]+'_'+X_all[c2]

topk = 15
for c1_c2_c3 in comb3[:topk]:
    c1, c2, c3 = c1_c2_c3.split('_')
    X_all_e[c1_c2_c3] = X_all[c1]+'_'+X_all[c2]+'_'+X_all[c3]

X_e = X_all_e.iloc[:len(X)]
X_o_e = X_all_e.iloc[len(X):-len(X_t)]
X_t_e = X_all_e.iloc[-len(X_t):]

cat_cols = X_e.columns.tolist()

test_pred_probas = np.zeros((X_t_e.shape[0], 7))

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(X_e, y)):

    print(f"{'='*20} FOLD {fold + 1} {'='*20}")

    X_train, X_val = X_e.iloc[train_idx], X_e.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse=True, drop='first'), cat_cols)
        ],
        remainder='passthrough'
    )

    X_train = preprocessor.fit_transform(X_train)
    X_val = preprocessor.transform(X_val)
    X_orig = preprocessor.transform(X_o_e)
    X_test = preprocessor.transform(X_t_e)

    X_train, y_train, weights_train = augment_data_as_sparse_matrix(
        X_train, y_train,
        X_orig, y_o,
        weight_orig=6.0
    )

    lr_model = LogisticRegression(C=1e-2, max_iter=1000)
    lr_model.fit(X_train, y_train, sample_weight=weights_train)

    y_val_pred_proba = lr_model.predict_proba(X_val)
    y_tst_pred_proba = lr_model.predict_proba(X_test)

    test_pred_probas += y_tst_pred_proba

    k_map = 3
    sorted_pred_indices = np.argsort(-y_val_pred_proba, axis=1)
    top_k_predicted_labels = [row[:k_map].tolist() for row in sorted_pred_indices]

    actual_labels_for_mapk = [[label] for label in y_val.tolist()] 

    fold_mapk = mapk(actual_labels_for_mapk, top_k_predicted_labels, k=k_map)

    print(f"MAP@{k_map} for FOLD {fold} : {fold_mapk:.5f}")


avg_test_pred_probas = np.argsort(-(test_pred_probas / 5), axis=1)
avg_test_pred_probas


top3_labels_test = [row[:3].tolist() for row in avg_test_pred_probas]
top3_labels_test[:10]


top3_names_test = [le.inverse_transform(row).tolist() for row in top3_labels_test]
top3_names_test[:10]


submission_list = []
for row in top3_names_test:
    submission_list.append(' '.join(row))

sample_submission['Fertilizer Name'] = submission_list


sample_submission.to_csv('submission.csv', index=False)

