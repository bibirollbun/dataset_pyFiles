import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import optuna


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


best_params = {
    'n_estimators': 472,
    'max_depth': 9,
    'learning_rate': 0.04941845906481925,
    'reg_lambda': 8.46332116038683,
    'num_leaves': 129,
    'subsample': 0.9457569984486377,
    'colsample_bytree': 0.5437403100437204,
    'device': 'gpu'
}


opt_lgbm = LGBMClassifier(**best_params)
opt_lgbm.fit(X_train_preprocessed, y_train_labeled)


opt_lgbm.booster_.save_model('/kaggle/working/lightgbm_best_map_at_3.txt')


# Get Test MAP@3
probs = opt_lgbm.predict_proba(X_test_preprocessed)
y_pred_top3 = [np.argsort(prob_row)[::-1][:3] for prob_row in probs]
test_score = map_at_3(y_test_labeled, y_pred_top3)
print("Test MAP@3 score:", test_score)


test_set = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
test_set.head()


test_features = test_set.drop(columns = ["id"])


test_features_preprocessed = ct.transform(test_features)


probs = opt_lgbm.predict_proba(test_features_preprocessed)
y_pred_top3 = [np.argsort(prob_row)[::-1][:3] for prob_row in probs]
y_pred_strings = [' '.join([le.inverse_transform([idx])[0] for idx in top3]) for top3 in y_pred_top3]


submission = pd.DataFrame({"id": test_set["id"], "Fertilizer Name": y_pred_strings})
submission.head()


submission.to_csv("submission.csv", index = False)




