import os

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option('display.max_columns', None)
sns.set(style="whitegrid")


print(*os.walk('/kaggle/input'), sep='\n')


train_data = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test_data = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
original_data = pd.read_csv("/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv")
submission_data = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")


print("train_data shape :",train_data.shape)
print("test_data shape :",test_data.shape)
print("original_data shape :",original_data.shape)
print("submission_data shape :",submission_data.shape)


train_data.head()


# test_data.shape, submission_data.shape
print(test_data.head(), submission_data.head(), sep='\n')



train_data = train_data.drop("id", axis=1)
test_data = test_data.drop("id", axis=1)
# cheat: merge original and competition dataset :D
train_data_new = pd.concat([train_data, original_data], ignore_index=True)
train_data_new = train_data_new.drop_duplicates()
print("shape of the data :",train_data_new.shape)


'''
Objective:
To select best fertilizer [Fetilizer Name]

Categorical features:
- soil type
- crop type
- fertilizer name
'''


def pd_one_hot(dataset, col):
    one_hot = pd.get_dummies(dataset[col])
    dataset = dataset.drop(col,axis = 1)
    dataset = dataset.join(one_hot)
    return dataset


def transform_dataset(dataset, cat_cols: list[str]):
    # categorical to onehot
    for col in cat_cols:
        dataset = pd_one_hot(dataset, col)
    
    # normalize (no need for decision trees)
    
    return dataset


print(train_data_new.shape)
train_data_new.head()


from sklearn.preprocessing import PolynomialFeatures

def feature_engineer(df, fe_columns: list[str] = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']):
    df_poly = df.copy()


    df_poly['Temp_Humidity_Interaction'] = df_poly['Temparature'] * df_poly['Humidity']
    df_poly['N_P_Ratio'] = df_poly['Nitrogen'] / (df_poly['Phosphorous'].replace(0, 1e-6))
    df_poly['K_P_Ratio'] = df_poly['Potassium'] / (df_poly['Phosphorous'].replace(0, 1e-6))
    df_poly['N_K_Ratio'] = df_poly['Nitrogen'] / (df_poly['Potassium'].replace(0, 1e-6))
    df_poly['P_K_Ratio'] = df_poly['Phosphorous'].replace(0, 1e-6) / (df_poly['Potassium'].replace(0, 1e-6))
    df_poly['N_P_K_interaction'] = df_poly['Phosphorous'] + df_poly['Potassium'] + df_poly['Nitrogen']

    col_to_return = [
                    'Temp_Humidity_Interaction',
                     'N_P_Ratio',
                     # 'K_P_Ratio',
                     # 'N_K_Ratio',
                     # 'P_K_Ratio',
                     # 'N_P_K_interaction'
                    ]
    
    return df_poly[col_to_return]

# apply feature engineering
columns_feature_engineering = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']

df_fe = feature_engineer(train_data_new, columns_feature_engineering)
df_fe_test = feature_engineer(test_data, columns_feature_engineering)

df_fe


train_data_new = pd.concat([train_data_new, df_fe], ignore_index=False, axis=1)

test_data = pd.concat([test_data, df_fe_test], ignore_index=False, axis=1)


num_cols_train = list(train_data_new.select_dtypes(include='number')\
    .columns\
    .difference(['Fertilizer Name']))
cat_cols_train = list(train_data_new.select_dtypes(exclude='number')\
    .columns\
    .difference(['Fertilizer Name']))
num_cols_test = list(test_data.select_dtypes(include='number')\
    .columns)
cat_cols_test = list(test_data.select_dtypes(exclude='number')\
    .columns)

num_cols_train, cat_cols_train, num_cols_test, cat_cols_test


train_data_copy = train_data_new.copy()
test_data_copy = test_data.copy()

# cat_cols = test_data.select_dtypes(exclude='number').columns.tolist()
# feature_les = {col: LabelEncoder() for col in cat_cols}  # encoder for each categorical feature
target_le = LabelEncoder()


# for col in cat_cols:
#     train_data_copy[col] = feature_les[col].fit_transform(train_data_copy[col])
#     test_data_copy[col] = feature_les[col].transform(test_data_copy[col])

train_data_copy['Fertilizer Name'] = target_le.fit_transform(train_data_copy['Fertilizer Name'])


from sklearn.model_selection import KFold
from category_encoders import LeaveOneOutEncoder

# Sample setup
X = train_data_copy.drop(columns=['Fertilizer Name'])     # Features
y_encoded = train_data_copy['Fertilizer Name']                    # Target
X_test = test_data_copy.copy()             # Test set


# Columns to encode
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

# Prepare result placeholders
X_encoded = pd.DataFrame(index=X.index)
X_test_encoded_list = []

# Define KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Loop over folds
for train_idx, val_idx in kf.split(X):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]

    # Fit encoder on training fold
    loo = LeaveOneOutEncoder(cols=cat_cols, sigma=0.1)
    loo.fit(X_train, y_train)

    # Transform validation fold and test set
    X_val_encoded = loo.transform(X_val)
    X_test_encoded = loo.transform(X_test)

    # Store validation encoding
    X_encoded.loc[val_idx, X_val_encoded.columns] = X_val_encoded

    # Store test encoding
    X_test_encoded_list.append(X_test_encoded)

# Average test encodings across folds
X_test_encoded = sum(X_test_encoded_list) / len(X_test_encoded_list)

# Final encoded data
X_encoded.reset_index(drop=True, inplace=True)
X_test_encoded.reset_index(drop=True, inplace=True)

train_data_copy = X_encoded.copy()
test_data_copy = X_test_encoded.copy()
train_data_copy['Fertilizer Name'] = y_encoded.values.copy()

X_encoded


train_data_copy.head()[cat_cols]


num_classes = len(target_le.classes_)
num_classes


from sklearn.preprocessing import label_binarize
import numpy as np


def map3_score(predicted_top3: np.ndarray,   # shape = (n_val, 3), dtype = object or int
               y_true_fold: np.ndarray,      # shape = (n_val,)
              ) -> float:
    """
    predicted_top5[i] is a lengthâ€�3 array of labels (strings/ints) that your model thinks
    are most likely for sample i, ordered from most confident 3rd most confident.
    y_true_fold[i] is the single true label for sample i.
    We give credit = 1/rank if the true label is at position 'rank' in that topâ€�3 list;
    otherwise 0. Then we average over all i.
    """
    print(type(predicted_top3), type(y_true_fold))
    
    n_val = y_true_fold.shape[0]
    total_score = 0.0

    for i in range(n_val):
        true_label = y_true_fold[i]
        top3_preds = predicted_top3[i].tolist()  # convert row to a Python list

        try:
            # .index(...) returns 0-based position. Add +1 to get 1-based rank.
            rank = top3_preds.index(true_label) + 1
            if rank <= 3:
                total_score += 1.0 / rank
            # If rank > 3, that cannot happen here, because top3_preds has exactly 3 items.
        except ValueError:
            # true_label not in top-3  score += 0
            pass

    return total_score / n_val


import optuna
from optuna.samplers import TPESampler

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score


X = train_data_copy.drop('Fertilizer Name',axis = 1)
y = train_data_copy["Fertilizer Name"]
# test = test_data_copy.copy()

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)


def xgboost_objective(trial):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 13),
        'min_child_weight': trial.suggest_float('min_child_weight', 0, 10),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'alpha': trial.suggest_loguniform('alpha', 1e-3, 10.0),
        'lambda': trial.suggest_loguniform('lambda', 1e-3, 10.0),
        'subsample': trial.suggest_float('subsample', 0, 1),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0, 1),
        'eta': trial.suggest_float('eta', 0, 1),
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    map3_scores = []

    for train_index, val_index in cv.split(X_train, y_train):

        x_train_fold = X_train.iloc[train_index]
        x_val_fold = X_train.iloc[val_index]
        
        y_train_fold = y_train.iloc[train_index]
        y_val_fold = y_train.iloc[val_index]
        
        model = XGBClassifier(
            **params,
            verbosity=0,
            objective='multi:softprob',
            enable_categorical=True,
            # tree_method='hist',
            tree_method="gpu_hist",
            gpu_id=0, 
            predictor="gpu_predictor",
            n_jobs=-1,
            random_seed=42
        )
        
        model.fit(x_train_fold, y_train_fold, eval_set=[(x_val_fold, y_val_fold)],
              early_stopping_rounds=50, verbose=False)

        pred_proba = model.predict_proba(x_val_fold)
        top3_index = np.argsort(pred_proba, axis=1)[:, -3:][:, ::-1]
        class_labs = model.classes_
        top3_labs = class_labs[top3_index]

        fold_map3 = map3_score(top3_labs, y_val_fold.to_numpy())
        map3_scores.append(fold_map3)
        mean_map3 = np.mean(map3_scores)

    return mean_map3

optimize_xgb = False

if optimize_xgb:
    study = optuna.create_study(direction="maximize", study_name = "optimize_xgb", sampler=TPESampler(n_startup_trials=30, seed=42, multivariate=True))
    study.optimize(xgboost_objective, n_trials=50, n_jobs=1)
    print(f"Best trial: {study.best_trial.params}")
    best_trail_params = study.best_trial.params.copy()


test_data_copy.head()


# <class 'numpy.ndarray'> <class 'numpy.ndarray'>
# Best trial:
# {'learning_rate': 0.18181172014492183, 'max_depth': 18, 'min_child_weight': 3.6443012437587052, 'gamma': 0.05940236743027757, 'alpha': 5.163520935045377, 'subsample': 0.8431783710998233, 'colsample_bytree': 0.28703655082980695, 'eta': 0.697099057795741, 'n_estimators': 446, 'lambda': 0.07297117932384822}


params = {'learning_rate': 0.07827209493668666, 'max_depth': 10, 
         'min_child_weight': 7.644459139670194, 'gamma': 0.4158775299466052, 
         'alpha': 0.003161777870766947, 'subsample': 0.7896840093318356, 
         'colsample_bytree': 0.23874892844194234, 'eta': 0.5244371252937642, 
         'n_estimators': 1675, 'lambda': 1.8837428404220584
         }


if optimize_xgb:
    params.update(best_trail_params)

xgb_classifier = XGBClassifier(
    **params, 
    verbosity=0,
    objective='multi:softprob',
    enable_categorical=True,
    # tree_method="gpu_hist",
    # gpu_id=0, 
    # predictor="gpu_predictor",
    n_jobs=-1,
    random_seed=42
)

xgb_classifier.fit(X_train, y_train, eval_set=[(X_valid, y_valid)],
      early_stopping_rounds=50, verbose=False)


pred_proba = xgb_classifier.predict_proba(X_valid)
top3_index = np.argsort(pred_proba, axis=1)[:, -3:][:, ::-1]
class_labs = xgb_classifier.classes_
top3_labs = class_labs[top3_index]

valid_map3 = map3_score(top3_labs, y_val.to_numpy())
print(f"Validation results, MAP@3: {valid_map3}")


pred_proba = xgb_classifier.predict_proba(test_data_copy.to_numpy())
top3_index = np.argsort(pred_proba, axis=1)[:, -3:][:, ::-1]
class_labs = xgb_classifier.classes_
top3_labs = class_labs[top3_index]


top3_result_strings = np.array(list(map(
    lambda x: ' '.join(target_le.inverse_transform(x)), top3_labs)))


top3_result_strings.shape, test_data.shape, submission_data.shape


submission = pd.DataFrame({
    'id': submission_data['id'].values,
    'Fertilizer Name': top3_result_strings
})
submission.to_csv('/kaggle/working/submission.csv',index=False)
submission




