
import os
import numpy as np
import pandas as pd

from itertools import combinations
from typing import List, Union, Iterable, Optional, Tuple, Dict, Callable

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier


class BasicMapper(BaseEstimator, TransformerMixin):
    def __init__(self,
                 month_col: str = 'month',
                 education_col: str = 'education',
                 answer_cols: Iterable[str] = ('default', 'housing', 'loan')):
        self.month_col = month_col
        self.education_col = education_col
        self.answer_cols = list(answer_cols)
        self._month_map = {'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12}
        self._answer_map = {'yes': 1, 'no': 0}
        self._education_map = {'unknown': 0, 'primary': 1, 'secondary': 2, 'tertiary': 3}

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        
        if self.month_col in X:
            X[self.month_col] = X[self.month_col].map(self._month_map)
            
        if self.education_col in X:
            X[self.education_col] = X[self.education_col].map(self._education_map)
            
        for col in self.answer_cols:
            if col in X:
                X[col] = X[col].map(self._answer_map
                                   )
        return X


class FeatureCombTransformer(BaseEstimator, TransformerMixin):
    """
    Creates encoded combo features like 'job+marital' by joining values as strings,
    learning categories on train, and mapping to integers.
    Unknown combos at transform time -> 0.
    """
    def __init__(self, columns: List[str], pair_sizes: Union[int, List[int]] = 2, prefix: str = "cmb"):
        self.columns = list(columns)
        self.pair_sizes = pair_sizes if isinstance(pair_sizes, list) else [pair_sizes]
        self.prefix = prefix
        self._combo_categories_: Dict[Tuple[str, ...], Dict[str, int]] = {}
        self._fit_columns_: List[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        self._fit_columns_ = [c for c in self.columns if c in X.columns]
        
        if not self._fit_columns_:
            return self
            
        Xs = X[self._fit_columns_].astype(str).copy()
        self._combo_categories_.clear()
        
        for n in self.pair_sizes:
            for cols in combinations(self._fit_columns_, n):
                joined = Xs[cols[0]].copy()
                
                for c in cols[1:]:
                    joined = joined + "_" + Xs[c]
                    
                cats = pd.Series(joined.unique())
                mapping = {v: i+1 for i, v in enumerate(sorted(cats))}
                self._combo_categories_[cols] = mapping
                
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._fit_columns_:
            return X
            
        Xs = X[self._fit_columns_].astype(str).copy()
        out = pd.DataFrame(index=X.index)
        
        for cols, mapping in self._combo_categories_.items():
            name = f"{self.prefix}_" + "+".join(cols)
            joined = Xs[cols[0]].copy()
            for c in cols[1:]:
                joined = joined + "_" + Xs[c]
            out[name] = joined.map(mapping).fillna(0).astype(int
                                                            )
        return pd.concat([X.reset_index(drop=True), out.reset_index(drop=True)], axis=1)


def build_preprocessor(
    ohe_cols_present: List[str],
    comb_columns_present: Optional[List[str]] = None,
    comb_sizes: Union[int, List[int]] = (2, 3),
    scale_numeric: bool = False
) -> Pipeline:
    steps = []
    steps.append(('map_basic', BasicMapper()))
    if comb_columns_present:
        steps.append(('feat_combos', FeatureCombTransformer(columns=comb_columns_present, pair_sizes=comb_sizes)))

    ohe = OneHotEncoder(handle_unknown='ignore', sparse=False, dtype=int)

    numeric_selector: Callable[[pd.DataFrame], list] = lambda X: X.select_dtypes(include='number').columns.tolist()

    transformers = [('ohe', ohe, ohe_cols_present)]
    if scale_numeric:
        transformers.append(('num_scale', StandardScaler(with_mean=False), numeric_selector))
    else:
        transformers.append(('num_pass', 'passthrough', numeric_selector))

    coltx = ColumnTransformer(transformers, remainder='drop')
    steps.append(('cols', coltx))
    
    return Pipeline(steps)


def cross_val_and_predict(X: pd.DataFrame,
                          y: pd.Series,
                          test_df: pd.DataFrame,
                          pipe: Pipeline,
                          n_splits: int = 3,
                          random_state: int = 42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = []
    test_proba = None
    
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        pipe.fit(X_tr, y_tr)
        preds = pipe.predict(X_va)
        acc = accuracy_score(y_va, preds)
        scores.append(acc)
        print(f"Fold {fold}: accuracy={acc:.4f}")
        proba = pipe.predict_proba(test_df)
        test_proba = proba if test_proba is None else (test_proba + proba)
        
    test_proba /= n_splits
    test_pred = np.argmax(test_proba, axis=1)
    
    print(f"\nMean CV accuracy: {np.mean(scores):.4f}")
    return np.array(scores), test_pred, test_proba


def make_submission(pipe: Pipeline,
                    X_train: pd.DataFrame,
                    y_train: pd.Series,
                    test_proba: np.ndarray,
                    test_df: pd.DataFrame,
                    id_col: str = "id",
                    positive_label=1,
                    out_path: str = "submission.csv") -> pd.DataFrame:
    
    pipe.fit(X_train, y_train)  # lock in classes_
    classes = list(pipe.named_steps["model"].classes_)
    
    if positive_label in classes:
        pos_idx = classes.index(positive_label)
    else:
        try:
            pos_idx = classes.index("yes")
        except ValueError:
            pos_idx = int(np.argmax(classes))
            
    ids = test_df[id_col].values if id_col in test_df.columns else test_df.index.values
    submission = pd.DataFrame({"id": ids, "y": test_proba[:, pos_idx]})
    
    submission.to_csv(out_path, index=False)

    print(f"Saved {out_path}")
    return submission




# MAIN RUN

DATA_DIR = "/kaggle/input/playground-series-s5e8"
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH  = os.path.join(DATA_DIR, "test.csv")
TARGET_COL = "y"
ID_COL = "id"

train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)

y = train_df[TARGET_COL]
if y.dtype == object:
    y = y.map({'yes': 1, 'no': 0}).astype(int)
else:
    y = y.astype(int)

drop_cols = [TARGET_COL] + ([ID_COL] if ID_COL in train_df.columns else [])
X = train_df.drop(columns=drop_cols)

CAT_TO_OHE   = ['job', 'marital', 'contact', 'poutcome']
SIG_FEATURES = ['previous', 'pdays', 'duration', 'balance', 'education', 'marital', 'job', 'age']

ohe_cols_present  = [c for c in CAT_TO_OHE if c in X.columns]
comb_cols_present = [c for c in SIG_FEATURES if c in X.columns]

preproc = build_preprocessor(
    ohe_cols_present=ohe_cols_present,
    comb_columns_present=comb_cols_present,
    comb_sizes=[2, 3],
    scale_numeric=False
)

xgb_clf = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=7,
    subsample=0.8,
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False
)

pipe = Pipeline([('prep', preproc), ('model', xgb_clf)])

scores, test_pred, test_proba = cross_val_and_predict(X, y, test_df, pipe, n_splits=3, random_state=42)

submission = make_submission(
    pipe=pipe,
    X_train=X,
    y_train=y,
    test_proba=test_proba,
    test_df=test_df,
    id_col=ID_COL,
    positive_label=1,
    out_path="submission.csv"
)

print(submission.head())





