# Libraries
import os

import pandas as pd
import numpy as np
import polars as pl

import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from metric import score, CompetitionMetric  # metric given by the competition

import xgboost as xgb

import wandb
from wandb.integration.xgboost import WandbCallback
import os
from kaggle_secrets import UserSecretsClient

# OS environment
os.environ["WANDB_SILENT"] = "true"


# CONFIG
SWEEP = 0
SAVE_MODEL = 1


# Getting Kaggle secrets
user_secrets = UserSecretsClient()

# Getting wandb api key
api_key = user_secrets.get_secret("wandb_api_key")

# Logging into wandb
wandb.login(key = api_key)  # prints 'True' if api key works


# Importing training data
train = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv")
train_demographics = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv")


# Combining train with train_demographics ... should have 348 columns if successful
train_merged = pd.merge(train, train_demographics, on = "subject", how = "left")
train_merged.shape[1]  # Num of columns


# Getting orientation, thermopile, and time-of-flight column names
acc_cols = [f'acc_{axis}' for axis in ['x', 'y', 'z']]
rot_cols = [f'rot_{axis}' for axis in ['w', 'x', 'y', 'z']]
thm_cols = [f'thm_{i}' for i in range(1, 6)]
tof_cols = [f'tof_{i}_v{j}' for i in range(1, 6) for j in range(64)]

# Combine all numeric columns into the final list
numerical_cols = ["age", "height_cm", "shoulder_to_wrist_cm", "elbow_to_wrist_cm"] + acc_cols + rot_cols + thm_cols + tof_cols

# Getting already labeled column names
already_labeled_categorical_cols = ["adult_child", "sex", "handedness"]


# Grouping by sequence_id and then merging remaining columns
cols = ['sequence_id'] + acc_cols + rot_cols + thm_cols + tof_cols

summary = (train_merged[cols].groupby('sequence_id').agg(['mean', 'std', 'min', 'max', 'median']))

summary.columns = ['_'.join(col).strip() for col in summary.columns.values]

summary = summary.reset_index()

summary.head()


# Adding summary statistics back to whole grouped dataset
remaining = train_merged.drop_duplicates('sequence_id')

train_df = pd.merge(remaining, summary, on = "sequence_id", how = "left")
train_df.head()


# Getting new numeric columns
exclude_cols = set(already_labeled_categorical_cols) | {"sequence_id"}

numerical_cols = [
    col for col in train_df.select_dtypes(include=["number"]).columns
    if col not in exclude_cols
]


# Label encoding the target
le = LabelEncoder()
train_df["gesture"] = le.fit_transform(train_df["gesture"])


# Storing target gestures
target_classes = ["Above ear - pull hair", "Cheek - pinch skin", "Eyebrow - pull hair", "Eyelash - pull hair",
                  "Forehead - pull hairline", "Forehead - scratch", "Neck - pinch skin", "Neck - scratch"]
target_class_indices = [np.where(le.classes_ == cls)[0][0] for cls in target_classes]


# Column Transformer
num = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer([
    ("num", num, numerical_cols),
    ("cat_pass", "passthrough", already_labeled_categorical_cols)
], remainder = "drop")


# Getting X and y
X = train_df.drop("gesture", axis = 1)
y = train_df["gesture"]


if SWEEP:
    def main():
        with wandb.init() as run:
    
            # Split dataset
            X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42,stratify=y_temp)
            # 60% train, 20% val, 20% test
        
            # Preprocessing
            X_train_trans = preprocessor.fit_transform(X_train)
            X_val_trans = preprocessor.transform(X_val)
            X_test_trans = preprocessor.transform(X_test)
        
            # Initialize XGBoost with wandb.config hyperparameters
            xgboost = xgb.XGBClassifier(
                subsample=wandb.config.subsample,
                reg_lambda=wandb.config.reg_lambda,
                reg_alpha=wandb.config.reg_alpha,
                min_child_weight=wandb.config.min_child_weight,
                max_depth=wandb.config.max_depth,
                gamma=wandb.config.gamma,
                eta=wandb.config.eta,
                colsample_bytree=wandb.config.colsample_bytree,
                n_estimators=1000,
                random_state=42,
                verbosity=0,
                tree_method='hist',
                device='cuda',
                objective="multi:softprob",
                eval_metric="mlogloss",
                callbacks=[WandbCallback()],
                early_stopping_rounds = 15
            )
        
            # Fit with validation set, no early stopping, use wandb callback to log metrics every iteration
            xgboost.fit(
                X_train_trans, y_train,
                eval_set=[(X_train_trans, y_train), (X_val_trans, y_val)],
                verbose = 0
            )
        
            # Predict and evaluate final test metric
            y_pred = xgboost.predict(X_test_trans)
            y_test_labels = le.inverse_transform(y_test)
            y_pred_labels = le.inverse_transform(y_pred)
        
            solution_df = pd.DataFrame({'id': range(len(y_test_labels)), 'gesture': y_test_labels})
            submission_df = pd.DataFrame({'id': range(len(y_pred_labels)), 'gesture': y_pred_labels})
        
            metric = CompetitionMetric()
            hierarchical_f1 = metric.calculate_hierarchical_f1(solution_df, submission_df)
        
            wandb.log({"hierarchical_f1": hierarchical_f1,
                      "best_iteration": xgboost.best_iteration})
    
    # Sweep config (adjust metric to the validation metric logged by callback)
    sweep_config = dict(
        method="bayes",
        name="bayes-optimization-xgboost",
        metric={'goal': 'minimize', 'name': 'validation_1-mlogloss'},  # or 'hierarchical_f1' if you prefer
        early_terminate=dict(
            type='hyperband',
            min_iter=20,
            eta=2
        ),
        parameters=dict(
            subsample=dict(min=0.5, max=1.0, distribution='uniform'),
            reg_lambda=dict(min=0.1, max=50.0, distribution='uniform'),
            reg_alpha=dict(min=0.0, max=10.0, distribution='uniform'),
            min_child_weight=dict(min=1, max=10, distribution='int_uniform'),
            max_depth=dict(min=4, max=16, distribution='int_uniform'),
            gamma=dict(min=0.0, max=5.0, distribution='uniform'),
            eta=dict(min=0.01, max=0.3, distribution='uniform'),
            colsample_bytree=dict(min=0.5, max=1.0, distribution='uniform')
        )
    )
        
    # Run the sweep
    sweep_id = wandb.sweep(sweep=sweep_config, entity='coreymichaud-projects', project='CMI 2025 Kaggle')
    wandb.agent(sweep_id, function=main, count=500)
    # ^ this may have runs that give errors due to running out of memory. in that case, the run ends but the rest keep going, so the code
    # will keep running :)


# if SWEEP:
#     def main():
#         run = wandb.init()
        
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
#         xgboost = xgb.XGBClassifier(
#             subsample=wandb.config.subsample,
#             reg_lambda=wandb.config.reg_lambda,
#             reg_alpha=wandb.config.reg_alpha,
#             min_child_weight=wandb.config.min_child_weight,
#             max_depth=wandb.config.max_depth,
#             gamma=wandb.config.gamma,
#             eta=wandb.config.eta,
#             colsample_bytree=wandb.config.colsample_bytree,
#             n_estimators=wandb.config.n_estimators,
#             random_state=42,
#             verbosity=0,
#             tree_method='hist',
#             device='cuda',
#             objective="multi:softprob"
#         )
    
#         # Pipeline
#         pipe = Pipeline([
#             ("preprocessor", preprocessor),
#             ("xgboost", xgboost)
#         ])
    
#         # Fit model with early stopping
#         pipe.fit(X_train, y_train)
    
#         # Predict and evaluate
#         y_pred = pipe.predict(X_test)
#         y_test_labels = le.inverse_transform(y_test)
#         y_pred_labels = le.inverse_transform(y_pred)
    
#         solution_df = pd.DataFrame({'id': range(len(y_test_labels)), 'gesture': y_test_labels})
#         submission_df = pd.DataFrame({'id': range(len(y_pred_labels)), 'gesture': y_pred_labels})
    
#         metric = CompetitionMetric()
#         hierarchical_f1 = metric.calculate_hierarchical_f1(solution_df, submission_df)
    
#         # Log metrics
#         wandb.log({
#             "hierarchical_f1": hierarchical_f1
#         })
    
#         run.finish()
    
#     # Sweep configuration
#     sweep_config = dict(
#         method="bayes",
#         name="bayes-optimization-xgboost",
#         metric={'goal': 'maximize', 'name': 'hierarchical_f1'},
#         parameters=dict(
#             subsample=dict(min=0.5, max=1.0, distribution='uniform'),
#             reg_lambda=dict(min=0.1, max=10.0, distribution='log_uniform'),
#             reg_alpha=dict(min=0.0, max=5.0, distribution='log_uniform'),
#             min_child_weight=dict(min=1, max=10, distribution='int_uniform'),
#             max_depth=dict(min=4, max=16, distribution='int_uniform'),
#             gamma=dict(min=0.0, max=5.0, distribution='uniform'),
#             eta=dict(min=0.01, max=0.3, distribution='log_uniform'),
#             colsample_bytree=dict(min=0.5, max=1.0, distribution='uniform'),
#             n_estimators=dict(min=50, max=350, distribution='int_uniform')
#         )
#     )
    
#     # Run the sweep
#     sweep_id = wandb.sweep(sweep=sweep_config, entity='coreymichaud-projects', project='CMI 2025 Kaggle')
#     wandb.agent(sweep_id, function=main, count=250)


if SAVE_MODEL:
    
    # Model parameters
    params = dict(
        subsample = 0.7879940858696578,
        reg_lambda = 23.324112198930496,
        reg_alpha = 0.5523929411331463,
        n_estimators = 375,
        min_child_weight = 8,
        max_depth = 9,
        gamma = 0.508831354869273,
        eta = 0.15668085003381013,
        colsample_bytree = 0.5252982328734005,
        random_state = 42,
        tree_method='hist',
        device='cuda',
        objective="multi:softprob"
    )
    
    # Opening wandb run
    with wandb.init(entity = "coreymichaud-projects", project = "CMI 2025 Kaggle", config = params) as run:
        
        # Model with best parameters
        xgboost = xgb.XGBClassifier(**params)
        
        # Pipeline
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("xgboost", xgboost)
        ])
        
        # Fitting model on whole training set
        inference_pipeline = pipe.fit(X,y)

        # Function to save model as artifact for model logging
        def save_artifact(model, model_name, filename):
            joblib.dump(model, filename)
            artifact = wandb.Artifact(name = model_name, type = "model")
            artifact.add_file(filename)
            run.log_artifact(artifact)
        
        save_artifact(inference_pipeline, "xgbclassifier", "xgbclassifier.joblib")  # PyTorch has model.save and the file would be a .pth file

        # Saving model and label encoder within this notebook so I can use in the inference notebook
        joblib.dump(inference_pipeline, 'pipeline.pkl')  # this is being saved... again... because the other nb has .pkl set up already
        joblib.dump(le, 'label_encoder.pkl')

