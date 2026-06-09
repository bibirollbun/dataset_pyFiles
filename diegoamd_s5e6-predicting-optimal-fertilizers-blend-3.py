import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score


train_set = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
train_set.head()


X = train_set.drop(columns = ["id", "Fertilizer Name"])
y = train_set["Fertilizer Name"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 19)


numerical_columns = X_train.select_dtypes(include = ["int", "float"]).columns
categorical_columns = X_train.select_dtypes(include = ["object"]).columns


ct = ColumnTransformer([
    ("scaler", StandardScaler(), numerical_columns),
    ("encoder", OneHotEncoder(handle_unknown = "ignore"), categorical_columns)
])


X_train_preprocessed = ct.fit_transform(X_train)
X_test_preprocessed = ct.transform(X_test)


le = LabelEncoder()


y_train_labeled = le.fit_transform(y_train)
y_test_labeled = le.transform(y_test)


def map_at_3(y_true, y_pred_top3):
    """
    Compute MAP@3 score for top-3 encoded label predictions.
    Scoring: 1 point for rank 1, 1/2 for rank 2, 1/3 for rank 3, 0 otherwise.
    Args:
        y_true: List or array of encoded true labels (e.g., [0, 1, ...])
        y_pred_top3: List or array of top-3 predicted labels (e.g., [[0, 1, 2], ...])
    Returns:
        float: Mean Average Precision at 3 (MAP@3)
    """
    scores = []
    for true_label, pred_top3 in zip(y_true, y_pred_top3):
        if len(pred_top3) != 3:
            raise ValueError(f"Prediction must contain exactly 3 labels, got: {pred_top3}")
        if pred_top3[0] == true_label:
            scores.append(1.0)
        elif pred_top3[1] == true_label:
            scores.append(0.5)
        elif pred_top3[2] == true_label:
            scores.append(1/3)
        else:
            scores.append(0.0)
    return np.mean(scores)


cat_params = {
    'iterations': 930,
    'depth': 5,
    'learning_rate': 0.18803333840450714,
    'l2_leaf_reg': 9.16131205596557,
    'border_count': 230,
    'bagging_temperature': 0.08559493603367059,
    'random_strength': 5.470798645890166,
    'task_type': 'GPU',
    'verbose': 0
}


opt_cat = CatBoostClassifier(**cat_params)


xgb_params = {
    'n_estimators': 961,
    'max_depth': 10,
    'learning_rate': 0.024048081015320764,
    'reg_lambda': 6.4456033446395935,
    'subsample': 0.6901125847016257,
    'colsample_bytree': 0.523633407081191,
    'min_child_weight': 6.175088770608872,
    'tree_method': 'hist',
    'device': 'cuda',
    'use_label_encoder': False,
    'eval_metric': 'mlogloss'
}


opt_xgb = XGBClassifier(**xgb_params)


lgbm_params = {
    'n_estimators': 472,
    'max_depth': 9,
    'learning_rate': 0.04941845906481925,
    'reg_lambda': 8.46332116038683,
    'num_leaves': 129,
    'subsample': 0.9457569984486377,
    'colsample_bytree': 0.5437403100437204,
    'device': 'gpu',
    'verbosity': -1
}


opt_lgbm = LGBMClassifier(**lgbm_params)


opt_cat.fit(X_train_preprocessed, y_train_labeled)


opt_xgb.fit(X_train_preprocessed, y_train_labeled)


opt_lgbm.fit(X_train_preprocessed, y_train_labeled)


cat_probs = opt_cat.predict_proba(X_test_preprocessed)
xgb_probs = opt_xgb.predict_proba(X_test_preprocessed)
lgbm_probs = opt_lgbm.predict_proba(X_test_preprocessed)


mean_probs = (cat_probs + xgb_probs + lgbm_probs) / 3
mean_probs


y_pred_top3 = [np.argsort(prob_row)[::-1][:3] for prob_row in mean_probs]
test_score = map_at_3(y_test_labeled, y_pred_top3)
print("Test MAP@3 score:", test_score)


test_set = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
test_set.head()


test_features = test_set.drop(columns = ["id"])
test_features_preprocessed = ct.transform(test_features)


probs1 = opt_cat.predict_proba(test_features_preprocessed)
probs2 = opt_xgb.predict_proba(test_features_preprocessed)
probs3 = opt_lgbm.predict_proba(test_features_preprocessed)
mean_probs = (probs1 + probs2 + probs3) / 3

y_pred_top3 = [np.argsort(prob_row)[::-1][:3] for prob_row in mean_probs]
y_pred_strings = [' '.join([le.inverse_transform([idx])[0] for idx in top3]) for top3 in y_pred_top3]


sub5 = pd.DataFrame({"id": test_set["id"], "Fertilizer Name": y_pred_strings})
sub5.head()


sub5.to_csv("sub5.csv", index = False)


meta_features = np.hstack([cat_probs, xgb_probs, lgbm_probs])


from sklearn.linear_model import LogisticRegression
meta_model = LogisticRegression(multi_class = 'multinomial', max_iter = 1000)
meta_model.fit(meta_features, y_test_labeled)


meta_features2 = np.hstack([probs1, probs2, probs3])


meta_probs = meta_model.predict_proba(meta_features2)

y_pred_top3 = [np.argsort(prob_row)[::-1][:3] for prob_row in meta_probs]
y_pred_strings = [' '.join([le.inverse_transform([idx])[0] for idx in top3]) for top3 in y_pred_top3]


sub6 = pd.DataFrame({"id": test_set["id"], "Fertilizer Name": y_pred_strings})
sub6.head()


sub6.to_csv("sub6.csv", index = False)


meta_model = XGBClassifier(
    tree_method = 'hist',
    device = 'cuda',
    use_label_encoder = False,
    eval_metric = 'mlogloss',
    random_state = 19
)

meta_model.fit(meta_features, y_test_labeled)


meta_probs = meta_model.predict_proba(meta_features2)

y_pred_top3 = [np.argsort(prob_row)[::-1][:3] for prob_row in meta_probs]
y_pred_strings = [' '.join([le.inverse_transform([idx])[0] for idx in top3]) for top3 in y_pred_top3]


sub8 = pd.DataFrame({"id": test_set["id"], "Fertilizer Name": y_pred_strings})
sub8.head()


sub8.to_csv("sub8.csv", index = False)




