import numpy as np 
import pandas as pd
import os
from sklearn.feature_selection import VarianceThreshold
import seaborn as sns
import matplotlib.pyplot as plt
import time
import joblib

from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import log_loss, accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
import umap
from sklearn.impute import SimpleImputer

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping



data = pd.read_csv('/kaggle/input/otto-group-product-classification-challenge/train.csv')
print(data.shape)
data.head()


# Features in hand
X = data.drop(columns=["id", "target"])


# target y in hand
y = data['target']


n_classes = y.unique()


print(data.isna().sum().sum())
(data.isna().mean() * 100).sort_values(ascending=False)


data.dtypes.value_counts()


data.duplicated().sum()


def iqr_outlier_stats(X):
    stats = []

    for col in X.columns:
        Q1 = X[col].quantile(0.25)
        Q3 = X[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outliers = ((X[col] < lower) | (X[col] > upper)).sum()
        outlier_pct = outliers / len(X) * 100

        stats.append({
            "feature": col,
            "outliers_count": outliers,
            "outliers_pct": outlier_pct,
            "lower_bound": lower,
            "upper_bound": upper
        })

    return pd.DataFrame(stats).set_index("feature")


outlier_df = iqr_outlier_stats(X)
outlier_df.sort_values("outliers_pct", ascending=False).head(10)


(outlier_df["outliers_pct"] > 0).mean() * 100


X["feat_60"].hist(bins=50)
plt.title("Raw feature distribution")
plt.show()


# transformed log1
np.log1p(X["feat_60"]).hist(bins=50)
plt.title("Log-transformed distribution")
plt.show()


# ploting everything because why not 
X.hist(
    bins=40,
    figsize=(20, 15),
    layout=(10, 10)
)
plt.tight_layout()
plt.show()


# zeroes precentage
total_zeros = (X == 0).sum().sum()
total_zeros / X.size * 100


skewness = X.skew().sort_values(ascending=False)
skewness.head(10)


# how many classes do we have 
y.unique()


# checking if there is class imbalance
# Percentage of each class
class_percent = y.value_counts(normalize=True) * 100
print(class_percent) 
good_percentage = 100 / len(y.unique())
print(f"Good percentage would be {good_percentage:.1f}%")


# features that barely change across the dataset: Features with almost the same value for every row carry little to no information.
vt = VarianceThreshold(threshold=0.1)
vt.fit(X)


variances = pd.Series(vt.variances_, index=X.columns)
to_remove = variances[variances <= 0.1]
to_keep = variances[variances > 0.1]

print("Total features:", X.shape[1])
print("Features to REMOVE:", len(to_remove))
print("Features to KEEP:", len(to_keep))



# lowest variance
to_remove.sort_values()


variances.describe()


plt.figure(figsize=(12, 7))
plt.hist(variances, bins=50)
plt.axvline(0.01)
plt.xlabel("Feature Variance")
plt.ylabel("Count")
plt.title("Feature Variance Distribution")
plt.show()


# Linear relationships (between features)
pearson_corr = X.corr()

# Monotonic / non-linear (between features)
spearman_corr = X.corr(method="spearman")

# General dependency with target
mi = mutual_info_classif(X, y)


# plot only the heat map of corr > 0.7 between two features with the two corr methods 
corrs = {
    "Pearson": pearson_corr,
    "Spearman": spearman_corr
}

for name, corr in corrs.items():
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        corr,
        cmap="coolwarm",
        center=0,
        mask=abs(corr) < 0.7
    )
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.title(f"Strong Feature Correlations ({name}) | |r| ≥ 0.7")
    plt.show()


# summerizing the high correated features (to delete one later maybe)
threshold = 0.7
corr = pearson_corr.abs()  

upper = corr.where(
    np.triu(np.ones(corr.shape), k=1).astype(bool)
)

high_corr_pairs = (
    upper.stack()
    .reset_index()
)

high_corr_pairs.columns = ["Feature_1", "Feature_2", "Correlation"]

high_corr_pairs = high_corr_pairs[high_corr_pairs["Correlation"] > threshold]

high_corr_pairs.sort_values(by="Correlation", ascending=False)


# ploting the top 20 mi features 
mi_series = pd.Series(mi, index=X.columns)
mi_series = mi_series.sort_values(ascending=False)
plt.figure(figsize=(8, 6))
mi_series.head(20).plot(kind="barh")
plt.xlabel("Mutual Information")
plt.title("Top 20 Features by Mutual Information")
plt.gca().invert_yaxis()
plt.show()


# taking a closer look into mi
mi_series = pd.Series(mi, index=X.columns)


# the features with the lowest mi 
bottom_mi_features = mi_series.sort_values().head(10)
bottom_mi_features


mi_series.describe() # just to see if the lowest ones are that bad


# Brain storming and thoughts

# To Try:
# cross-validation
# maybe reursive feature elemination
# Number of non-zeros elements in each row as a new feature
# Ranking-based features (The heighest value features ) which features are the biggest for this row ? 

# Models to try:
# Xgboost 
# random forest
# logistic regression
# neural networks 
# naive bayes
# model stacking (with KNN, random Forest , logoistic regression ... ) 


# Metrics 
# Logloss, accuarcy, Confution matrix , Classification Report 

# save the models 


!pip install xgboost


import xgboost as xgb


# metrices to use everywhere 
metrics = ['log_loss', 'accuracy_score', 'classification_report', 'confusion_matrix']


# Encode the y 
le = LabelEncoder()
y_encoded = le.fit_transform(y)


# fine Tuning and finding the best parms (with randomized search cross validation )
param_dist = {
    "max_depth": [3, 4, 8, 14],
    "min_child_weight": [1, 3, 5, 7],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "reg_alpha": [0, 0.1, 0.5, 1],
    "reg_lambda": [1, 3, 5, 10]
}

search = RandomizedSearchCV(
    xgb.XGBClassifier(
        learning_rate=0.05,
        n_estimators=300,
        eval_metric="mlogloss",
        random_state=42,   
    ),
    param_distributions=param_dist,
    n_iter=2, # 30
    cv=3,
    scoring="neg_log_loss",
    n_jobs=-1
)

# for 14 Log loss: 0.4544
#Accuracy: 0.8261
search.fit(X, y_encoded)


search.best_params_


# first we split the data to train and test 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


print('X Train shape :', X_train.shape)
print('X Test shape : ' , X_test.shape)


# for xgboost only 
y_encoded_test = le.fit_transform(y_test)
y_encoded_train = le.fit_transform(y_train)


# fiting the model to train dataset
xgb_BM_model = search.best_estimator_.fit(X_train, y_encoded_train)


def evaluate_model(model, X_test, y_test, metrics=metrics):
    """
    Evaluating the model with a list of matrices
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    for m in metrics:
        if m == 'log_loss':
            print(f"Log loss: {log_loss(y_test, y_proba):.4f}")

        elif m == 'accuracy_score':
            print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

        elif m == 'classification_report':
            print("\nClassification report:")
            print(classification_report(y_test, y_pred))

        elif m == 'confusion_matrix':
            cm = confusion_matrix(y_test, y_pred)
            print("\nConfusion matrix:\n", cm)
            ConfusionMatrixDisplay.from_predictions(y_test, y_pred)


print("------------------------- Evaluation Of the XGBoost Model -------------------------")
print('-----------------------------------------------------------------------------------')
evaluate_model(xgb_BM_model, X_test, y_encoded_test, metrics=metrics)


base_models = {
    "lr_log": Pipeline([
        ("log", FunctionTransformer(np.log1p)),
        ("lr", LogisticRegression(
            max_iter=300
        ))
    ]),

    "rf_raw": RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        random_state=42,
        max_features="sqrt"
    ),

    "knn_scaled": Pipeline([
        ("log", FunctionTransformer(np.log1p)),
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(
            n_neighbors=95,
            metric="euclidean",
            weights="distance"
        ))
    ])
}


# out of folds 
def generate_oof_predictions(models, X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    n_classes = len(np.unique(y))
    oof_preds = {
        name: np.zeros((X.shape[0], n_classes))
        for name in models
    }
    
    # train on k-1 folds, predict on 1 fold, k times
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train = y[train_idx]

        for name, model in models.items():
            model.fit(X_train, y_train)
            oof_preds[name][val_idx] = model.predict_proba(X_val)

    return oof_preds



oof_preds = generate_oof_predictions(base_models, X_train, y_encoded_train)
X_meta_train = np.hstack(list(oof_preds.values()))


# to check which base Models contribute the best 
for name, preds in oof_preds.items():
    acc = accuracy_score(y_encoded_train, preds.argmax(axis=1))
    ll = log_loss(y_encoded_train, preds)
    print(f"{name}: accuracy={acc:.4f}, log_loss={ll:.4f}")


meta_model = xgb.XGBClassifier(
    objective="multi:softprob",
    num_class=n_classes,
    n_estimators=300,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss",
    random_state=42
)



meta_model.fit(X_meta_train, y_encoded_train)


# The whole pipline


for model in base_models.values():
    model.fit(X_train, y_encoded_train)



meta_features_test = []

for model in base_models.values():
    preds = model.predict_proba(X_test)
    meta_features_test.append(preds)

X_meta_test = np.hstack(meta_features_test)



meta_test_preds = meta_model.predict_proba(X_meta_test)


print("Final stacked log loss:",
      log_loss(y_encoded_test, meta_test_preds))

print("Final stacked accuracy:",
      accuracy_score(y_encoded_test,
                     meta_test_preds.argmax(axis=1)))
meta_test_preds = meta_model.predict_proba(X_meta_test)


# this is for the Ranking-based features (The heighest value features ) which features are the biggest for this row ? 
def add_rank_features_np(X, k=4):
    """
    Generate ranking-based one-hot features for the top-k values per row.

    For each sample (row), the indices of the top-k largest feature values
    are identified. For each rank position, a one-hot vector of length
    n_features is created, resulting in k * n_features additional features.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Input feature matrix.

    k : int, default=4
        Number of top-ranked features to encode per sample.

    Returns
    -------
    np.ndarray of shape (n_samples, k * n_features)
        One-hot encoded ranking features.
    """
    
    n_samples, n_features = X.shape
    topk_idx = np.argsort(X, axis=1)[:, -k:][:, ::-1]

    rank_feats = np.zeros((n_samples, k * n_features), dtype=np.float32)

    for i in range(n_samples):
        for r in range(k):
            rank_feats[i, r * n_features + topk_idx[i, r]] = 1.0

    return rank_feats
    
# to add to the pipeline   
class RankFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer that appends ranking-based features.

    This transformer computes one-hot encoded features indicating the
    indices of the top-k largest values in each sample and appends them
    to the original feature matrix.

    Parameters
    ----------
    k : int, default=4
        Number of top-ranked features to encode per sample.
    """
    def __init__(self, k=4):
        self.k = k

    def fit(self, X, y=None):
        self.n_features_ = X.shape[1]
        return self

    def transform(self, X):
        X_np = np.asarray(X)
        rank_feats = add_rank_features_np(X_np, k=self.k)
        return np.hstack([X_np, rank_feats])



# To use NN 
def build_nn(input_dim : int, n_classes: int, seed=None) -> Sequential:
    """
    Build and compile a simple feedforward neural network for multiclass classification.

    Architecture:
        Input → Dense(256) → Dropout → Dense(128) → Dropout → Softmax

    Parameters
    ----------
    input_dim : int
        Number of input features.

    n_classes : int
        Number of output classes.

    seed : int or None, default=None
        Random seed for reproducibility.

    Returns
    -------
    tensorflow.keras.Sequential
        Compiled Keras model.
    """
    
    if seed is not None:
        tf.keras.utils.set_random_seed(seed)

    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(256, activation="relu"),
        Dropout(0.3),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(n_classes, activation="softmax")
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model



class PreprocessedKerasNN(BaseEstimator, ClassifierMixin):
    """
    Scikit-learn compatible wrapper for Keras neural networks with preprocessing.

    Supports:
    - Log or square-root feature transformations
    - Optional ranking-based feature augmentation
    - Feature scaling using StandardScaler

    Parameters
    ----------
    build_fn : callable
        Function that builds and returns a compiled Keras model.

    epochs : int, default=25
        Number of training epochs.

    batch_size : int, default=256
        Training batch size.

    seed : int or None, default=None
        Random seed for reproducibility.

    verbose : int, default=0
        Verbosity level for Keras training.

    transform : {"log", "sqrt"}, default="log"
        Feature transformation applied before scaling.

    feature_rank : bool, default=False
        Whether to append ranking-based features.

    rank_k : int, default=4
        Number of top-ranked features used if feature_rank=True.
    """
    
    def __init__(
        self,
        build_fn,
        epochs=100,
        batch_size=256,
        seed=None,
        verbose=0,
        transform="log",
        feature_rank=False,
        rank_k=4,
        early_stopping=True,
        patience=8,
        val_split=0.15
    ):
        self.build_fn = build_fn
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.verbose = verbose
        self.transform = transform
        self.feature_rank = feature_rank
        self.rank_k = rank_k
        self.early_stopping = early_stopping # using early stoping method 
        self.patience = patience
        self.val_split = val_split


    def _transform_X(self, X):
        """
        Apply numeric transformation and optional ranking features.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        np.ndarray
            Transformed feature matrix.
        """
        X_np = np.asarray(X)
        if self.transform == "log":
            X_np = np.log1p(X_np)
        elif self.transform == "sqrt":
            X_np = np.sqrt(X_np + 0.375)

        features = [X_np]
        if self.feature_rank:
            rank_feats = add_rank_features_np(X_np, k=self.rank_k)
            features.append(rank_feats)

        return np.hstack(features)

    def fit(self, X, y):
        """
        Fit the neural network model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        self
        """
        X_t = self._transform_X(X)
        
        self.scaler_ = StandardScaler()
        X_t = self.scaler_.fit_transform(X_t)
        
        self.n_classes_ = len(np.unique(y))
        y_cat = to_categorical(y, self.n_classes_)
        
        self.model_ = self.build_fn(X_t.shape[1], self.n_classes_, self.seed)
        
        callbacks = []
        if self.early_stopping:
            callbacks.append(
                EarlyStopping(
                    monitor="val_loss",
                    patience=self.patience,
                    restore_best_weights=True,
                    verbose=self.verbose
                )
            )
        
        self.model_.fit(
            X_t,
            y_cat,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.val_split,
            callbacks=callbacks,
            verbose=self.verbose
        )
        
        return self


    def predict_proba(self, X):
        """
        Predict class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        np.ndarray of shape (n_samples, n_classes)
        """
        X_t = self._transform_X(X)
        X_t = self.scaler_.transform(X_t)
        return self.model_.predict(X_t, verbose=0)



class AdvancedStackingClassifier:
    """
    Advanced stacking ensemble for multiclass classification.

    Features:
    - Stratified K-fold out-of-fold predictions
    - Optional bagging of base models
    - Weighted ensemble of multiple meta models
    - Optional feature augmentation (nonzero counts, UMAP)

    Parameters
    ----------
    base_models : dict
        Dictionary of base estimators.

    meta_models : list of dict
        Each dict must contain:
        - "model": estimator with predict_proba
        - "coaf": float weight

    n_splits : int, default=5
        Number of CV folds For the OOF

    bagging_dict : dict or None, default=None
        Number of bags per base model.

    use_nonzero_feature : bool, default=True
        Whether to append nonzero-count feature.

    use_umap : bool, default=False
        Whether to append UMAP features.

    umap_model : object or None
        Preconfigured UMAP transformer.

    random_state : int, default=42
        Random seed.
    """
    def __init__(
        self,
        base_models: dict,
        meta_models: list[dict],
        n_splits=5,
        bagging_dict=None,   
        use_nonzero_feature=True,
        use_umap=False,
        umap_model=None,
        random_state=42
    ):
        
        self.base_models = base_models
        self.meta_models = meta_models
        self.n_splits = n_splits
        self.random_state = random_state

        self.bagging_dict = bagging_dict or {}
        self.use_nonzero_feature = use_nonzero_feature
        self.use_umap = use_umap
        self.umap_model = umap_model

        self.fitted_base_models_ = []
        self.n_classes_ = None

        # making sure that the coafs of the meta models add up to 1 
        total = sum(m["coaf"] for m in self.meta_models)
        for m in self.meta_models:
            m["coaf"] /= total

  
    # Feature Engineering
    def _augment_features(self, X):
        """
        Apply optional feature augmentation to input data.

        Augmentations include:
        - Nonzero feature count per sample
        - UMAP-transformed features

        Parameters
        ----------
        X : pandas.DataFrame
            Input feature matrix.

        Returns
        -------
        X_aug : pandas.DataFrame
            Augmented feature matrix.
        """
        X_aug = X.copy()

        if self.use_nonzero_feature:
            nonzero_count = (X_aug != 0).sum(axis=1)
            X_aug["nonzero_count"] = nonzero_count

        if self.use_umap:
            umap_features = self.umap_model.transform(X)
            for i in range(umap_features.shape[1]):
                X_aug[f"umap_{i}"] = umap_features[:, i]

        return X_aug
        
    def normalize_proba(self, p):
        """
        Normalize predicted probabilities so each row sums to 1.

        This is a safety step to correct minor numerical drift when
        blending multiple probabilistic models.

        Parameters
        ----------
        p : np.ndarray of shape (n_samples, n_classes)
            Raw probability predictions.

        Returns
        -------
        np.ndarray
            Row-normalized probability predictions.
        """
        return p / p.sum(axis=1, keepdims=True)


    def fit(self, X, y, verbose=True):
        """
        Fit the stacking ensemble.

        Workflow:
        1. Optionally fit UMAP on training data.
        2. Generate OOF predictions for each base model using stratified CV.
        3. Retrain each base model on the full training set.
        4. Train meta models on concatenated OOF predictions.

        Parameters
        ----------
        X : pandas.DataFrame
            Training features.

        y : array-like of shape (n_samples,)
            Encoded target labels.

        verbose : bool, default=True
            Whether to print training progress and metrics.

        Returns
        -------
        self : AdvancedStackingClassifier
            Fitted instance.
        """
        start_total = time.time()
        
        X = X.reset_index(drop=True)
        y = np.asarray(y)

        if self.use_umap:
            self.umap_model.fit(X)

        X_aug = self._augment_features(X)

        self.n_classes_ = len(np.unique(y))
        n_samples = X.shape[0]

        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state
        )

        oof_meta_features = []

        # looping through base models 
        for name, model in self.base_models.items():
            n_bags = self.bagging_dict.get(name, 1)

            for bag in range(n_bags):
                model_start = time.time()
                if verbose:
                    print(f"\nTraining {name} (bag {bag+1}/{n_bags})")

                oof_preds = np.zeros((n_samples, self.n_classes_))

                for fold, (tr, val) in enumerate(skf.split(X_aug, y), 1):
                    model_clone = clone(model)
                    model_clone.fit(X_aug.iloc[tr], y[tr])
                    oof_preds[val] = model_clone.predict_proba(X_aug.iloc[val])

                oof_meta_features.append(oof_preds)

                model_clone = clone(model)
                model_clone.fit(X_aug, y)
                self.fitted_base_models_.append((name, model_clone))

                if verbose:
                    print(
                        f"OOF logloss: {log_loss(y, oof_preds):.4f}, "
                        f"accuracy: {accuracy_score(y, oof_preds.argmax(1)):.4f}, " 
                        f"Execution time for {name}: {time.time() - model_start:.1f}s"

                    )

        # Training meta model 
        X_meta_train = np.hstack(oof_meta_features)
        self.fitted_meta_models_ = []

        for m in self.meta_models:
            meta = clone(m["model"])
            meta.fit(X_meta_train, y)
            self.fitted_meta_models_.append({
                "model": meta,
                "coaf": m["coaf"]
            })
        
        if verbose:
            meta_preds = np.zeros((len(y), self.n_classes_))
            for m in self.fitted_meta_models_:
                meta_preds += m["coaf"] * m["model"].predict_proba(X_meta_train)
        
            meta_preds = self.normalize_proba(meta_preds)
            print("\nMeta models performance (OOF):")
            print("Meta Accuracy:", accuracy_score(y, meta_preds.argmax(1)))
            print("Meta LogLoss:", log_loss(y, meta_preds))
            print(f"\nTotal training time: {time.time() - start_total:.1f}s")


        return self

    
    def predict_proba(self, X):
        """
        Predict class probabilities for new data.

        Parameters
        ----------
        X : pandas.DataFrame
            Input features.

        Returns
        -------
        np.ndarray of shape (n_samples, n_classes)
            Normalized class probability predictions.
        """
        X = X.reset_index(drop=True)
        X_aug = self._augment_features(X)

        meta_features = []

        for name, model in self.fitted_base_models_:
            meta_features.append(model.predict_proba(X_aug))

        X_meta = np.hstack(meta_features)
        probs = np.zeros((X_meta.shape[0], self.n_classes_))
        for m in self.fitted_meta_models_:
            probs += m["coaf"] * m["model"].predict_proba(X_meta)
        
        return self.normalize_proba(probs)


    def predict(self, X):
        """
        Predict class labels for new data.

        Parameters
        ----------
        X : pandas.DataFrame
            Input features.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Predicted class indices.
        """
        return self.predict_proba(X).argmax(axis=1)



def log1p_clip(X):
    return np.log1p(np.clip(X, 0, None))

# Base Models 
base_models = {
    
    "lr_log": Pipeline([
        #("rank", RankFeatureTransformer(k=4)), #  didn't help 
        ("log", FunctionTransformer(np.log1p)),
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            max_iter=3000,
            solver="lbfgs",
            n_jobs=-1
    ))
    ]),

    "rf_raw": RandomForestClassifier(
        n_estimators=600,
        max_depth=None,
        random_state=42,
        max_features="sqrt"
    ),

    "knn_scaled": Pipeline([
        # ("imputer", SimpleImputer(strategy="median")), # this was for the Umap 
        ("log", FunctionTransformer(log1p_clip, validate=False)),
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(
            n_neighbors=95,
            metric="euclidean",
            weights="distance"
        ))
    ]), 
    
    "xgb": xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=n_classes,
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05
    )
}


for i in range(5):
    base_models[f"nn_sqrt_{i}"] = PreprocessedKerasNN(
        build_fn=build_nn,
        epochs=50, 
        patience=10,
        early_stopping=False,
        batch_size=256,
        seed=100 + i,
        transform="sqrt", 
        feature_rank=False
    )    

for i in range(10):
    base_models[f"nn_log_{i}"] = PreprocessedKerasNN(
        build_fn=build_nn,
        epochs=50,
        patience=10,
        early_stopping=False,
        batch_size=256,
        seed=42 + i,
        transform="log", 
        feature_rank = False
    )

# bagging_dict = {"rf_raw": 5} # decided not to use this because it didn't help 




# Meta Models 
xgb_meta = xgb.XGBClassifier(
    objective="multi:softprob",
    num_class=n_classes,
    n_estimators=350,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss",
    random_state=42
)

lr_meta = Pipeline([
    ("scaler", StandardScaler()), 
    ("lr", LogisticRegression(
        multi_class="multinomial",
        max_iter=3000,
        C=0.5,
        solver="lbfgs"
    )) 
])



meta_models = [{
    "model": xgb_meta, 
    "coaf": 1.0
}] 
# {
#     "model": lr_meta,  # didn't help  ( the coaf was 0.3 )
#     "coaf": 0
# }]



# using Umap 
umap_model = umap.UMAP(
    n_components=10,        
    n_neighbors=15,
    min_dist=0.1,
    metric="euclidean",
    random_state=42
)



stack = AdvancedStackingClassifier(
    base_models=base_models,
    meta_models=meta_models,
    n_splits=5
)

stack.fit(X_train, y_encoded_train)



preds = stack.predict_proba(X_test)
y_pred = preds.argmax(axis=1)

print("Final log loss:", log_loss(y_encoded_test, preds))
print("Final accuracy:", accuracy_score(y_encoded_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_encoded_test, y_pred))
print("\nClassification report:\n",
      classification_report(y_encoded_test, y_pred))

ConfusionMatrixDisplay.from_predictions(y_encoded_test, y_pred)
# The best so far 
# Final log loss: 0.4254230561022298
# Final accuracy: 0.8350840336134454


# Saving the stack  
joblib.dump(stack, "/kaggle/working/stacking_model.pkl")


################ Cross Validation for the whole Pipeline #############

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

loglosses = []
accuracies = []


le = LabelEncoder()
y_encoded = le.fit_transform(y)
n_classes = len(le.classes_) 
cm_total = np.zeros((n_classes, n_classes), dtype=int)

for fold, (tr, val) in enumerate(cv.split(X, y_encoded), 1):
    print(f"\n===== OUTER FOLD {fold} =====")

    X_train, X_val = X.iloc[tr], X.iloc[val]
    y_train, y_val = y_encoded[tr], y_encoded[val]

    stack = AdvancedStackingClassifier(
        base_models=base_models,
        meta_models=meta_models,
        n_splits=5
    )

    stack.fit(X_train, y_train, verbose=False)

    preds = stack.predict_proba(X_val)
    y_pred = preds.argmax(1)

    loglosses.append(log_loss(y_val, preds))
    accuracies.append(accuracy_score(y_val, y_pred))
    cm_total += confusion_matrix(y_val, y_pred, labels=np.arange(n_classes))

print("\n==== CV RESULTS ====")
print(f"LogLoss: {np.mean(loglosses):.4f} ± {np.std(loglosses):.4f}")
print(f"Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")

print("\nAggregated Confusion Matrix:")
print(cm_total)



# loading the stack 
# stack = joblib.load("/kaggle/input/YOUR_DATASET/stack.pkl")


sub_stack = AdvancedStackingClassifier(
    base_models=base_models,
    meta_models=meta_models,
    n_splits=5,
)

sub_stack.fit(X, y_encoded)


joblib.dump(sub_stack, "/kaggle/working/substacking_model.pkl")


test_data = pd.read_csv("/kaggle/input/otto-group-product-classification-challenge/test.csv")
test_data.shape


test_ids = test_data['id']
sub_X = test_data.drop(columns=["id"])


sub_probs = sub_stack.predict_proba(sub_X)

submission = pd.DataFrame(
    sub_probs,
    columns=le.classes_
)

submission.insert(0, "id", test_ids)

submission.to_csv("/kaggle/working/submission.csv", index=False)




