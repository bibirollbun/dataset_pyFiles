!cp /kaggle/input/cibmtr-solution/sol.png .


from IPython.display import HTML

HTML("""
<div style="background-color: #333333; padding: 10px; display: inline-block;">
  <img src="./sol.png" alt="sol.png">
</div>
""")


!pip install -q /kaggle/input/pip-install-lifelines/autograd-1.7.0-py3-none-any.whl
!pip install -q /kaggle/input/pip-install-lifelines/autograd-gamma-0.5.0.tar.gz
!pip install -q /kaggle/input/pip-install-lifelines/interface_meta-1.3.0-py3-none-any.whl
!pip install -q /kaggle/input/pip-install-lifelines/formulaic-1.0.2-py3-none-any.whl
!pip install -q /kaggle/input/pip-install-lifelines/lifelines-0.30.0-py3-none-any.whl

!cp /kaggle/input/tabm-model/tabm_reference.py .
!cp /kaggle/input/tabm-model/rtdl_num_embeddings.py .


import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from typing import Any, Union


class EFSCombinedModel(BaseEstimator, RegressorMixin):
    """
    Combined model that predicts both the probability of an event (p) and the estimated survival time (t).

    This model internally uses a classifier to predict the event probability and a regressor (trained only on samples 
    where the event occurred) to estimate the survival time. The regressor output is log-transformed during training and 
    then inverted in prediction.

    Parameters
    ----------
    classifier : estimator
        A scikit-learn compatible classifier (e.g., CatBoostClassifier).
    regressor : estimator
        A scikit-learn compatible regressor (e.g., CatBoostRegressor).
    """
    def __init__(self, classifier: Any, regressor: Any):
        self.classifier = classifier
        self.regressor = regressor

    def _extract_y(self, y: Union[pd.DataFrame, np.ndarray]) -> (np.ndarray, np.ndarray):
        """
        Validate and extract event indicator and survival time from y.

        Parameters
        ----------
        y : DataFrame or numpy array, shape (n_samples, 2)
            If a DataFrame, it must contain the columns 'efs' and 'efs_time'.
            If an array, it must have two columns corresponding to efs and efs_time.

        Returns
        -------
        y_efs : numpy array
            Binary event indicators.
        y_time : numpy array
            Survival times.
        """
        if isinstance(y, pd.DataFrame):
            if not {"efs", "efs_time"}.issubset(y.columns):
                raise ValueError("DataFrame y must contain 'efs' and 'efs_time' columns.")
            return y["efs"].values, y["efs_time"].values
        elif isinstance(y, np.ndarray) and y.ndim == 2 and y.shape[1] == 2:
            return y[:, 0], y[:, 1]
        else:
            raise ValueError("y must be either a DataFrame with 'efs' and 'efs_time' or a 2D numpy array with 2 columns.")

    def fit(self, X: Any, y: Union[pd.DataFrame, np.ndarray]) -> "EFSCombinedModel":
        """
        Fit the two-step model on training data.

        Parameters
        ----------
        X : array-like
            Feature matrix.
        y : DataFrame or numpy array, shape (n_samples, 2)
            Contains binary event indicator in the first column ('efs')
            and event time in the second column ('efs_time').

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Extract event indicator and survival time.
        y_efs, y_time = self._extract_y(y)
        
        # Fit the classifier on the entire dataset.
        self.classifier_ = self.classifier.fit(X, y_efs)
        
        # For the regressor, only fit on samples where efs == 1.
        mask = (y_efs == 1)
        self.regressor_ = self.regressor.fit(X[mask], np.log1p(y_time[mask]))
        return self

    def predict(self, X: Any) -> np.ndarray:
        """
        Predict event probability and estimated survival time.

        Parameters
        ----------
        X : array-like
            Input samples.

        Returns
        -------
        predictions : numpy array of shape (n_samples, 2)
            First column: probability of event occurrence (efs = 1).
            Second column: predicted survival time conditional on efs = 1
        """
        check_is_fitted(self, ["classifier_", "regressor_"])
        
        # Predict probability of event occurrence.
        efs_probs = self.classifier_.predict_proba(X)[:, 1]
        # Predict survival time and invert the log transformation.
        reg_preds = np.expm1(self.regressor_.predict(X))
        
        return np.stack([efs_probs, reg_preds], axis=1)

    def get_params(self, deep: bool = True) -> dict:
        """
        Get parameters for this estimator, including sub-estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        params = {"classifier": self.classifier, "regressor": self.regressor}
        if deep:
            if hasattr(self.classifier, "get_params"):
                for key, value in self.classifier.get_params(deep=True).items():
                    params[f"classifier__{key}"] = value
            if hasattr(self.regressor, "get_params"):
                for key, value in self.regressor.get_params(deep=True).items():
                    params[f"regressor__{key}"] = value
        return params

    def set_params(self, **params) -> "EFSCombinedModel":
        """
        Set parameters for this estimator and its sub-estimators.

        Returns
        -------
        self : object
            Estimator instance with updated parameters.
        """
        classifier_params = {}
        regressor_params = {}
        for key, value in params.items():
            if key.startswith("classifier__"):
                classifier_params[key.replace("classifier__", "")] = value
            elif key.startswith("regressor__"):
                regressor_params[key.replace("regressor__", "")] = value
            else:
                setattr(self, key, value)
        if classifier_params and self.classifier is not None:
            self.classifier.set_params(**classifier_params)
        if regressor_params and self.regressor is not None:
            self.regressor.set_params(**regressor_params)
        return self

def make_efs_pipeline(classifier: Any, regressor: Any, categorical_cols: list, numeric_cols: list) -> Pipeline:
    """
    Create a scikit-learn pipeline that preprocesses data and applies the custom EFSCombinedModel.

    Parameters
    ----------
    classifier : estimator
        Classifier for predicting event occurrence.
    regressor : estimator
        Regressor for predicting survival time.
    categorical_cols : list
        List of categorical column names.
    numeric_cols : list
        List of numeric column names.

    Returns
    -------
    pipeline : Pipeline
        A pipeline that first preprocesses the data and then applies the custom model.
    """
    # Preprocessor for numerical and categorical features.
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="mean"), numeric_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="NaN")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical_cols),
        ],
        remainder="passthrough"
    )
    
    # Initialize the custom model.
    efs_model = EFSCombinedModel(classifier=classifier, regressor=regressor)
    
    # Combine preprocessing and the model into a single pipeline.
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("efs_model", efs_model),
    ])
    
    return pipeline



import torch
import torch.nn.functional as F

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

def _to_float32(X):
    return X.astype("float32")


def make_nn_numerical_preprocessor():
    return Pipeline([
        ("imputer", SimpleImputer(strategy='mean')),
        ("scaler", StandardScaler()),
        ("to_float32", FunctionTransformer(_to_float32))
    ])


def make_nn_categorical_preprocessor():
    return Pipeline([
        ("imputer", SimpleImputer(strategy='constant', fill_value='NaN')),
        ("encoder", OrdinalEncoder(dtype=np.int64))
    ])


def replace_relu_with_gelu(module):
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU):
            setattr(module, name, nn.GELU())
        else:
            replace_relu_with_gelu(child)


def step_lr_with_min(optimizer, init_lr=1e-2, step_size=100, gamma=0.25, min_lr=1e-5):
    def lr_lambda(epoch):
        stepped_lr = init_lr * (gamma ** (epoch // step_size))
        effective_lr = max(stepped_lr, min_lr)
        return effective_lr / init_lr

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


import pytorch_lightning as pl


class ConcordanceIndexApproxModel(pl.LightningModule):
    def __init__(
        self,
        model,
        lr=1e-3,
        alpha=1,
        lr_step_size=1000,
        weight_decay=0
    ):
        super().__init__()
        # Save all hyperparameters for logging/checkpointing, etc.
        self.save_hyperparameters(ignore=["model"])
    
        self.model = model

        self.lr = lr
        self.lr_step_size = lr_step_size
        self.alpha = alpha
        self.weight_decay = weight_decay

        self.validation_step_outputs = []

    def forward(self, x_num, x_cat):
        return self.model(x_num, x_cat)
    
    def cindex_approx(self, preds, times, events):
        event_idx = torch.where(events == 1)[0]
        t_event = times[event_idx]
        p_event = preds[event_idx]

        time_order_mask = (t_event.unsqueeze(1) < times.unsqueeze(0))

        preds_diff = p_event.unsqueeze(1) - preds.unsqueeze(0)
        valid_diffs = preds_diff[time_order_mask]

        return torch.sigmoid(self.alpha * valid_diffs).mean(dim=0)

    def cindex_approx_loss_by_group(self, preds, times, events, groups):
        cindices = []
        
        for g in groups.unique():
            mask = (groups == g)
            g_preds = preds[mask]
            g_times = times[mask]
            g_events = events[mask]

            g_cindex = self.cindex_approx(g_preds, g_times, g_events)
            cindices.append(g_cindex)
        
        cindices_t = torch.stack(cindices)
        
        c_mean = cindices_t.mean(dim=0)
        c_std = cindices_t.std(dim=0, correction=0) if len(cindices) > 1 else 0.0
        
        score = c_mean - c_std
        loss = - score
        return torch.mean(loss)

    def training_step(self, batch, batch_idx):
        x_num, x_cat, times, events, groups = batch
        preds = self.forward(x_num, x_cat).squeeze(-1)

        loss = self.cindex_approx_loss_by_group(preds, times, events, groups)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def on_validation_epoch_start(self):
        self.validation_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        x_num, x_cat, times, events, groups = batch
        preds = self.forward(x_num, x_cat).squeeze(-1).mean(dim=-1)

        val_loss = self.cindex_approx_loss_by_group(preds, times, events, groups)
        self.log("val_loss", val_loss, prog_bar=True, on_epoch=True)
        
        self.validation_step_outputs.append({
            "preds": preds.detach(),
            "times": times.detach(),
            "events": events.detach(),
            "groups": groups.detach()
        })
    
    def on_validation_epoch_end(self):
        all_preds = torch.cat([o["preds"] for o in self.validation_step_outputs], dim=0)
        all_times = torch.cat([o["times"] for o in self.validation_step_outputs], dim=0)
        all_events = torch.cat([o["events"] for o in self.validation_step_outputs], dim=0)
        all_groups = torch.cat([o["groups"] for o in self.validation_step_outputs], dim=0)

        sub = pd.DataFrame({
            "ID": np.arange(len(all_preds)), 
            "efs_time": all_times.cpu().numpy(),
            "efs": all_events.cpu().numpy(),
            "race_group": all_groups.cpu().numpy(),
        })
        sol = pd.DataFrame({
            "ID": np.arange(len(all_preds)), 
            "prediction": all_preds.cpu().numpy().squeeze(),
        })

        val_cindex, val_std = score(sol, sub, "ID")

        self.log("val_cindex", val_cindex)
        self.log("val_std", val_std)
        self.log("val_score", val_cindex - val_std, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = step_lr_with_min(optimizer, init_lr=self.lr, step_size=self.lr_step_size, gamma=0.1, min_lr=1e-5)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': "step",
                'frequency': 1,
            }
        }


from lifelines.utils import concordance_index

from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger

from tabm_reference import Model
import rtdl_num_embeddings


class NNSurvivalEstimator(BaseEstimator, RegressorMixin):
    
    def __init__(
        self, 
        backbone,
        alpha=1, 
        batch_size=256, 
        lr=3e-3, 
        lr_step_size=1000, 
        max_steps=5000, 
        enable_progress_bar=True, 
        weight_decay=0, 
        k=128, 
        numerical_embeddings=True,
    ):
        self.lr = lr
        self.lr_step_size = lr_step_size
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.alpha = alpha
        self.weight_decay = weight_decay
        self.enable_progress_bar = enable_progress_bar
        self.backbone = backbone
        self.k = k
        self.numerical_embeddings = numerical_embeddings

    def fit(self, X_num_train, X_cat_train, y_train, groups_train, X_num_val=None, X_cat_val=None, y_val=None, groups_val=None):
        validate = X_num_val is not None

        if validate:
            assert all(x is not None for x in (X_num_val, X_cat_val, y_val, groups_val))
        
        self.num_preprocessor_ = make_nn_numerical_preprocessor()
        self.cat_preprocessor_ = make_nn_categorical_preprocessor()
        
        X_num_train_pre = self.num_preprocessor_.fit_transform(X_num_train)
        X_cat_train_pre = self.cat_preprocessor_.fit_transform(X_cat_train)
        X_num_val_pre = self.num_preprocessor_.transform(X_num_val) if validate else None
        X_cat_val_pre = self.cat_preprocessor_.transform(X_cat_val) if validate else None

        if self.numerical_embeddings:
            bins=rtdl_num_embeddings.compute_bins(torch.tensor(X_num_train_pre))
            num_embeddings={
                'type': 'PiecewiseLinearEmbeddings',
                'd_embedding': 16,
                'activation': False,
                'version': 'B',
            }
        else:
            bins = None
            num_embeddings = None
            
        cat_cardinalities = [np.unique(X_cat_train_pre[:, i]).size for i in range(X_cat_train_pre.shape[1])]
        n_num_features = X_num_train_pre.shape[1]
        
        tabm_model = Model(
            n_num_features=n_num_features,
            cat_cardinalities=cat_cardinalities,
            n_classes=1,
            backbone=self.backbone,
            bins=bins,
            num_embeddings=num_embeddings,
            arch_type='tabm-mini',
            k=self.k,
        )
        replace_relu_with_gelu(tabm_model)
        
        self.model_ = ConcordanceIndexApproxModel(
            tabm_model, 
            lr=self.lr, 
            lr_step_size=self.lr_step_size, 
            alpha=self.alpha, 
            weight_decay=self.weight_decay
        )
        
        train_dataset = TensorDataset(
            torch.as_tensor(X_num_train_pre), 
            torch.as_tensor(X_cat_train_pre), 
            torch.as_tensor(y_train["efs_time"].values),
            torch.as_tensor(y_train["efs"].values), 
            torch.as_tensor(groups_train),
        )
        if validate:
            val_dataset = TensorDataset(
                torch.as_tensor(X_num_val_pre), 
                torch.as_tensor(X_cat_val_pre), 
                torch.as_tensor(y_val["efs_time"].values),
                torch.as_tensor(y_val["efs"].values), 
                torch.as_tensor(groups_val),
            )
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        if validate:
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
    
        logger = CSVLogger("logs")
        lr_monitor = LearningRateMonitor(logging_interval='step')
    
        trainer = pl.Trainer(
            max_steps=self.max_steps,
            deterministic=True,
            enable_progress_bar=self.enable_progress_bar,
            check_val_every_n_epoch=10,
            logger=logger,
            callbacks=[lr_monitor],
            log_every_n_steps=8,
        )
        
        trainer.fit(
            self.model_, 
            train_loader, 
            val_loader if validate else None
        )
    
        return self

    def predict(self, X_num, X_cat):
        X_num_pre = torch.as_tensor(self.num_preprocessor_.transform(X_num))
        X_cat_pre = torch.as_tensor(self.cat_preprocessor_.transform(X_cat))
        self.model_.eval()
        with torch.no_grad():
            preds = self.model_(X_num_pre, X_cat_pre)
        return preds.cpu().numpy().squeeze().mean(axis=-1)


# The training set has only one value for gvhd_proph='FK+- others(not MMF,MTX)'. Replaced for stability.
df = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/train.csv", index_col=0)
df.loc[df["gvhd_proph"] == "FK+- others(not MMF,MTX)", "gvhd_proph"] = "FK+ MMF +- others"

categorical_cols = list(df.select_dtypes(object).columns)
numerical_cols = list(set(df.select_dtypes(exclude=object).columns) - {"efs", "efs_time"})

test_df_raw = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv", index_col=0)
test_df_raw.loc[test_df_raw["gvhd_proph"] == "FK+- others(not MMF,MTX)", "gvhd_proph"] = "FK+ MMF +- others"


X_num_train = df[numerical_cols]
X_cat_train = df[categorical_cols]
X_train = pd.concat([X_num_train, X_cat_train], axis=1)
y_train = df[["efs", "efs_time"]]

group_encoder = OrdinalEncoder(dtype="int64")
groups_train = group_encoder.fit_transform(X_cat_train[["race_group"]]).reshape(-1)

X_num_test = test_df_raw[numerical_cols]
X_cat_test = test_df_raw[categorical_cols]
X_test = pd.concat([X_num_test, X_cat_test], axis=1)


import xgboost as xgb
import lightgbm as lgb
import catboost as cat
from sklearn.ensemble import VotingClassifier, VotingRegressor


# The hyperparameters and the weights were determined through cross-validation with the help of Optuna
lgb_classifier_parameters = {'n_estimators': 1561, 'num_leaves': 7, 'learning_rate': 0.04197382808868415, 'bagging_fraction': 0.4825829095492543, 'min_child_samples': 16, 'feature_fraction': 0.11989767778924523, 'reg_alpha': 0.11658835968075622, 'reg_lambda': 9.645132853176758, 'max_depth': 8, 'bagging_freq': 1, 'min_split_gain': 0.17458276774601306, 'max_bin': 72}
lgb_regressor_parameters = {'n_estimators': 1920, 'num_leaves': 40, 'learning_rate': 0.029691717940376332, 'bagging_fraction': 0.8089272981629148, 'min_child_samples': 11, 'feature_fraction': 0.7274730305060586, 'reg_alpha': 0.00017045890342554733, 'reg_lambda': 0.00026484398517239707, 'max_depth': 10, 'bagging_freq': 4, 'min_split_gain': 0.0002923026044546345, 'max_bin': 130}

xgb_classifier_parameters = {'n_estimators': 1890, 'max_leaves': 4, 'learning_rate': 0.08787836471900472, 'subsample': 0.6805182045042094, 'min_child_weight': 8, 'colsample_bytree': 0.17117372871474534, 'reg_alpha': 9.992999524174493, 'reg_lambda': 0.007475499603743433, 'max_depth': 3}
xgb_regressor_parameters = {'n_estimators': 1834, 'max_leaves': 49, 'learning_rate': 0.03903983985345222, 'subsample': 0.8021313955005485, 'min_child_weight': 9, 'colsample_bytree': 0.9239367079009665, 'reg_alpha': 4.607045182855083e-08, 'reg_lambda': 0.0014737933676175492, 'max_depth': 8}

cat_classifier_parameters = {'iterations': 2600, 'depth': 4, 'l2_leaf_reg': 1.9192995536417459, 'bagging_temperature': 0.4266735107533081, 'rsm': 0.3364948287139534, 'subsample': 0.6889463250333477, 'random_strength': 3.3398644672174207, 'min_data_in_leaf': 4}
cat_regressor_parameters = {'iterations': 2800, 'depth': 8, 'l2_leaf_reg': 0.27044725223469857, 'bagging_temperature': 0.3133188471464441, 'rsm': 0.9690615328996377, 'subsample': 0.9683103562002902, 'random_strength': 5.5099387091081375, 'min_data_in_leaf': 17}


classifier = VotingClassifier([
    ("xgb",xgb.XGBClassifier(random_state=52, n_jobs=-1, **xgb_classifier_parameters)),
    ("lgb",lgb.LGBMClassifier(random_state=52, n_jobs=-1, verbose=-1, **lgb_classifier_parameters)),
    ("cat", cat.CatBoostClassifier(random_seed=52, thread_count=-1, verbose=0, **cat_classifier_parameters)),
], weights=[0.5, 0.25, 0.25], voting="soft")

regressor = VotingRegressor([
    ("xgb", xgb.XGBRegressor(random_state=52, n_jobs=-1, **xgb_regressor_parameters)),
    ("lgb", lgb.LGBMRegressor(random_state=52, n_jobs=-1, verbose=-1, **lgb_regressor_parameters)),
    ("cat", cat.CatBoostRegressor(random_seed=52, thread_count=-1, verbose=0, **cat_regressor_parameters)),
], weights=[0.45, 0.25, 0.3])

pipeline = make_efs_pipeline(classifier, regressor, categorical_cols, numerical_cols)
pipeline.fit(X_train, y_train)

efs_pipeline_pred = pipeline.predict(X_test)
pred_p = efs_pipeline_pred[:, 0]
pred_t = efs_pipeline_pred[:, 1]


pl.seed_everything(52)

# The hyperparameters were hand-picked based on cross-validation score
nn_model = NNSurvivalEstimator(
    backbone={
        'type': "MLP",
        "n_blocks": 2, 
        "d_block": 192, 
        "dropout": 0.2, 
    },
    k=64,
    max_steps=900,
    batch_size=4800,
    lr=6e-3,
    lr_step_size=300,
    alpha=10,
    weight_decay=1e-6,
    enable_progress_bar=False,
)
nn_model.fit(X_num_train, X_cat_train, y_train, groups_train)
pred_r = nn_model.predict(X_num_test, X_cat_test)


from sklearn.preprocessing import PolynomialFeatures

# These weights come from the second-step neural network
weights = {
    'p': 1.405158281326294, 
    'p t': -0.7868764400482178, 
    'p^2': -0.7010354995727539, 
    'r': 0.336719274520874, 
    't': -0.8060367107391357, 
    't^2': 1.0415241718292236
}

pred_df = pd.DataFrame({
    "p": pred_p, 
    "t": (pred_t - pred_t.min()) / (pred_t.max() - pred_t.min()),
    "r": (pred_r - pred_r.min()) / (pred_r.max() - pred_r.min()),
})

pf = PolynomialFeatures(degree=2, include_bias=False)
pred_df_poly = pd.DataFrame(pf.fit_transform(pred_df), columns=pf.get_feature_names_out())

preds = np.zeros(len(pred_df))

for col, w in weights.items():
    preds += pred_df_poly[col] * w

sub = pd.DataFrame({"ID": X_test.index, "prediction": preds})
sub


sub.to_csv("submission.csv", index=False)










