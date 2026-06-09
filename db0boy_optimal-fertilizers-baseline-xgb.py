import random 
from dataclasses import dataclass


import pandas as pd
import numpy as np


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold


from xgboost import XGBClassifier


@dataclass(frozen=True)
class Config:
    train_path = "/kaggle/input/playground-series-s5e6/train.csv"
    test_path = "/kaggle/input/playground-series-s5e6/test.csv"
    random_state = 42
    test_size = 0.2


random.seed(Config.random_state)  # Python
np.random.seed(Config.random_state)  # NumPy


def mapk(y_true, y_pred, k=3):
    score = 0.0
    for true, pred in zip(y_true, y_pred):
        if true in pred:
            rank = pred.index(true) + 1 
            score += 1.0 / rank
    return score / len(y_true)


train = pd.read_csv(Config.train_path)
test = pd.read_csv(Config.test_path)


test_id = test["id"].copy()


def feature_engineering(X: pd.DataFrame):
    X = X.copy()
    
    X = X.drop("id", axis=1)
    
    return X


class TargetOneHotEncoder(OneHotEncoder):
    def fit(self, X, y):
        super().fit(y.values.reshape(-1, 1))
        return self

    def transform(self, X, y):
        return X, super().transform(y.values.reshape(-1, 1))

    def inverse_transform(self, X, y):
        return X, super().inverse_transform(y)

    def fit_transform(self, X, y, **kwargs):
        self.fit(X, y)
        return self.transform(X, y)


cat_cols = ["Soil Type", "Crop Type"]

transformer = ColumnTransformer([
        (
            "cat", 
            OneHotEncoder(
                categories=[
                    sorted(train[el].unique()) 
                    for el in cat_cols
                ], 
                handle_unknown="error",
                sparse_output=False
            ), 
            cat_cols
        )
    ],
    remainder="passthrough",
    
)

transformer.set_output(transform="pandas")


class PipelineStep(BaseEstimator, TransformerMixin):
    @staticmethod
    def split_data(X: pd.DataFrame):
        y = X["Fertilizer Name"]
        X = X.drop("Fertilizer Name", axis=1)

        return X, y
        
    def __init__(self):
        self.data_pipeline_X = Pipeline(
            [
                ("feature_engineering", FunctionTransformer(feature_engineering)),
                ("transform_X", transformer),
            ]
        )
        self.y_pipeline = TargetOneHotEncoder(
            categories=[
                sorted(train["Fertilizer Name"].unique()) 
            ], 
            handle_unknown="error",
            sparse_output=False
        )
        
    def fit(self, X: pd.DataFrame):
        X, y = self.split_data(X)
        
        res = self.data_pipeline_X.fit_transform(X)
        self.y_pipeline.fit(res, y)
        return self
        
    def transform(self, X: pd.DataFrame):
        X, y = self.split_data(X)

        res = self.data_pipeline_X.transform(X)
        return self.y_pipeline.transform(res, y)
        


data_pipeline = PipelineStep()


X, y = data_pipeline.fit_transform(train)
test_data = data_pipeline.data_pipeline_X.transform(test)


y_lab = np.argmax(y, axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=Config.test_size, 
    random_state=Config.random_state, 
    stratify=y_lab
)


model = XGBClassifier(
    objective="multi:softprob",  
    num_class=y.shape[1],
    eval_metric="mlogloss",
    use_label_encoder=False,
    random_state=Config.random_state,
)


model.fit(X_train, np.argmax(y_train, axis=1))

probs = model.predict_proba(X_test)

top3 = np.argsort(-probs, axis=1)[:, :3]

top3_list = [list(row) for row in top3]

map3_score = mapk(np.argmax(y_test, axis=1), top3_list, k=3)
print(f"MAP@3 on test set: {map3_score:.4f}")


from sklearn.model_selection import StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=Config.random_state)


y_pred_proba = cross_val_predict(
    model,
    X,
    y_lab,
    cv=cv,
    method="predict_proba"
)

top3_preds = np.argsort(-y_pred_proba, axis=1)[:, :3]
top3_preds_list = [list(row) for row in top3_preds]

map3_score = mapk(y_lab, top3_preds_list, k=3)
print(f"MAP@3 (cross-validated): {map3_score:.4f}")


probs = model.predict_proba(test_data)  

topk = np.argsort(-probs, axis=1)[:, :3]  # (n_samples, 3)

n_samples, k = topk.shape
n_classes = probs.shape[1]

topk_flat_onehot = np.zeros((n_samples * k, n_classes))
topk_flat_onehot[np.arange(n_samples * k), topk.ravel()] = 1

_, decoded_labels = data_pipeline.y_pipeline.inverse_transform(None, topk_flat_onehot)

decoded_topk = decoded_labels.reshape(n_samples, k)
decoded_topk_strings = [" ".join(row) for row in decoded_topk]

submission = pd.DataFrame({
    "id": test_id,
    "Fertilizer Name": decoded_topk_strings
})

submission.to_csv("submission.csv", index=False)

