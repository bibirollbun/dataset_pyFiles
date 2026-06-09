# Basic imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Disable warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
pd.set_option("display.max_columns", 500)

from warnings import filterwarnings
filterwarnings("ignore", category=UserWarning, module="matplotlib")


df_train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv", index_col="id")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv", index_col="id")

print(f"df_train: {df_train.shape}, df_test: {df_test.shape}\n")
df_train.head()


missing_train = df_train.isna().sum().sum()
missing_test = df_test.isna().sum().sum()

print(f"df_train missing values: {missing_train}")
print(f"df_test missing values: {missing_test}")


colors = sns.color_palette("coolwarm")
colors_2 = [colors[0], colors[-1]]

sns.set_style("whitegrid", {'axes.grid' : False})
sns.pairplot(df_train.sample(frac=1e-3, random_state=2025), 
             hue="Sex", palette=colors_2,
             height=1.2, corner=True)

plt.show()


corr_matrix = df_train.corr(method="spearman", numeric_only=True)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, cmap="coolwarm", annot=True)
plt.show()


# Calculate the BMI and take a subset for plotting
df_train["BMI"] = df_train["Weight"] / (df_train["Height"]/100)**2
df_subset = df_train.sample(frac=1e-2, random_state=2025)

# Initialize figure
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.set_style("whitegrid", {'axes.grid' : False})

# 1. Calories vs BMI
sns.scatterplot(x=df_subset["BMI"], y=df_subset["Calories"], 
                hue=df_train["Sex"], palette=colors_2,
                ax=axes[0])

# 2. BMI by sex
sns.histplot(x=df_subset["BMI"], 
             hue=df_subset["Sex"], palette=colors_2,
             ax=axes[1],
             kde=True,
             stat="probability")
axes[1].set_xlim(18, 32)

plt.tight_layout()
plt.show()


from sklearn.base import BaseEstimator, TransformerMixin


class AttributeAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_duration_interactions):
        self.add_duration_interactions = add_duration_interactions

    def fit(self, X, y=None):
        X_copy = X.copy()

        # Calculate BMI (mean and sd for normalization)
        X_copy["BMI"] = X_copy["Weight"] / (X_copy["Height"]/100)**2
        self.bmi_stats = X_copy.groupby("Sex")["BMI"].agg(["mean", "std"]).to_dict("index")
        
        return self

    def transform(self, X):
        X_copy = X.copy()

        # BMI normalized
        X_copy["BMI"] = X_copy["Weight"] / (X_copy["Height"]/100)**2
        X_copy["BMI_z"] = X_copy.apply(self.normalize_bmi, axis=1)

        # Age groups
        X_copy['Age_Group'] = pd.cut(X_copy['Age'], bins=[0, 20, 35, 50, 100], labels=[0, 1, 2, 3])

        # Interaction pairs
        X_copy["Body_Temp_x_Heart_Rate"] = X_copy["Body_Temp"] * X_copy["Heart_Rate"]
        X_copy["Age_x_Heart_Rate"] = X_copy["Age"] * X_copy["Heart_Rate"]
        X_copy["Height_x_Weight"] = X_copy["Height"] * X_copy["Weight"]
        X_copy["Age_x_BMI"] = X_copy["Age"] * X_copy["BMI"]
        if self.add_duration_interactions:
            X_copy["Duration_x_Heart_Rate"] = X_copy["Duration"] * X_copy["Heart_Rate"]
            X_copy["Duration_x_Body_Temp"] = X_copy["Duration"] * X_copy["Body_Temp"]
            X_copy["Duration_x_BMI"] = X_copy["Duration"] * X_copy["BMI"]
            X_copy["Duration_x_Weight"] = X_copy["Duration"] * X_copy["Weight"]
            
        return X_copy

    def normalize_bmi(self, row):
        stats = self.bmi_stats.get(row["Sex"])
        if stats is not None and stats["std"] != 0:
            return (row["BMI"] - stats["mean"]) / stats["std"]
        else:
            return 0 # No misisng values but anyway
        


class GeneralPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.num_cols = None
        self.cat_cols = None

    def fit(self, X, y=None):
        X_copy = X.copy()

        # Identify column types
        self.num_cols = X_copy.select_dtypes(include=["number"]).columns.tolist()
        self.cat_cols = X_copy.select_dtypes(include=["object", "category"]).columns.tolist()

        return self

    def transform(self, X):
        X_copy = X.copy()

        # Set as category
        X_copy[self.cat_cols] = X_copy[self.cat_cols].astype("category")

        return X_copy

        


class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, features_to_remove):
        self.features_to_remove = features_to_remove

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        features_to_keep = [col for col in X_copy.columns if col not in self.features_to_remove]
        return X_copy[features_to_keep]        


from sklearn.pipeline import Pipeline

features_to_remove = ["id", "Calories", "BMI", "Age_Group"]

preprocessor = Pipeline(steps = [
    ("feature_engineering", AttributeAdder(add_duration_interactions=True)),
    ("general_processing", GeneralPreprocessor()),
    ("feature_selection", FeatureSelector(features_to_remove))
])


# Test data processing
df_train_transformed = preprocessor.fit_transform(df_train)
df_train_transformed.head()


from sklearn.metrics import mean_squared_log_error

def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, y_pred))


# Learning rate scheduler
import xgboost as xgb

class CustomLRScheduler(xgb.callback.TrainingCallback):
    def __init__(self, factor=0.5, patience=20, min_lr=1e-3, start_lr=0.1, metric="rmse"):
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.start_lr = start_lr
        self.wait = 0
        self.best_score = float("inf")
        self.current_lr = start_lr
        self.metric = metric

    def before_training(self, model):
        model.set_param("learning_rate", self.current_lr)
        return model

    def after_iteration(self, model, epoch, evals_log):
        score = evals_log["validation_0"][self.metric][-1]
        if score < self.best_score:
            self.best_score = score
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                new_lr = max(self.current_lr * self.factor, self.min_lr)
                if new_lr < self.current_lr:
                    print(f"Reducing learning rate to {new_lr:.6f}")
                    self.current_lr = new_lr
                    model.set_param("learning_rate", self.current_lr)
                self.wait = 0
        return False


# Set up KFold and load data
from sklearn.model_selection import KFold

FOLDS = 3
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=2025)

# Data
df_train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")


# Training function
def fit_predict_KFold(model, kf, train_df, test_df):

    # Initialize 
    oof = np.zeros(len(train_df)) # Out-of_fold predictions
    log_preds = [] # predictions on test set

    # Extract X and y
    X = train_df.drop(columns = ["Calories", "id"]).copy()
    y = train_df["Calories"].copy()
    
    # Main loop
    for i, (train_index, val_index) in enumerate(kf.split(train_df)):

        print("#"*100)
        print(f"### Fold {i+1}")
        print("#"*100)
    
        # Training data
        X_train = X.loc[train_index, :]
        X_train_transformed = preprocessor.fit_transform(X_train)
        y_train = y.loc[train_index]

        # Validation data
        X_val = X.loc[val_index, :]
        X_val_transformed = preprocessor.transform(X_val)
        y_val = y.loc[val_index]
        
        # Test data
        X_test_transformed = preprocessor.transform(test_df)

        # Set callbacks
        model.set_params(callbacks=[
            EarlyStopping(rounds=100, save_best=True),
            CustomLRScheduler(factor=0.5, patience=20, min_lr=1e-3, start_lr=0.1, metric="rmsle"),
        ])
        
        # Fit model
        model.fit(
            X_train_transformed, y_train,
            eval_set=[(X_val_transformed, y_val)],
            verbose=200,
        )

        # Inner oof (out-of-fold predictions)
        oof[val_index] += model.predict(X_val_transformed)
        # Inner test (test predictions)
        preds = model.predict(X_test_transformed)
        log_preds_k = [np.log1p(p) for p in preds] # Transform to log(1+p) space
        log_preds.append(log_preds_k) # Inplace operation returns None!

        
    # Calculate average predictions on test set
    log_preds_stack = np.stack(log_preds, axis=0)
    avg_log_preds = np.average(log_preds_stack, axis=0)
    final_preds = np.expm1(avg_log_preds)

    return oof, final_preds


from xgboost.callback import EarlyStopping
import xgboost as xgb

# Model
params = {'max_depth': 7, 
          'learning_rate': 0.1, 
          'n_estimators': 2000, 
          'subsample': 0.8, 
          'colsample_bytree': 0.6,
          'min_child_weight': 40, 
          'enable_categorical': True,
          'eval_metric':'rmsle'}

model_xgb = xgb.XGBRegressor(**params)


oof_xgb, pred_xgb = fit_predict_KFold(model_xgb, kf, df_train, df_test)  


# Training function
def fit_predict_KFold_log(model, kf, train_df, test_df):

    # Initialize 
    oof = np.zeros(len(train_df)) # Out-of_fold predictions
    log_preds = [] # predictions on test set

    # Extract X and y
    X = train_df.drop(columns = ["Calories", "id"]).copy()
    y = train_df["Calories"].copy()
    
    # Main loop
    for i, (train_index, val_index) in enumerate(kf.split(train_df)):

        print("#"*100)
        print(f"### Fold {i+1}")
        print("#"*100)
    
        # Training data
        X_train = X.loc[train_index, :]
        X_train_transformed = preprocessor.fit_transform(X_train)
        y_train = np.log1p(y.loc[train_index])

        # Validation data
        X_val = X.loc[val_index, :]
        X_val_transformed = preprocessor.transform(X_val)
        y_val = np.log1p(y.loc[val_index])
        
        # Test data
        X_test_transformed = preprocessor.transform(test_df)

        # Set callbacks
        model.set_params(callbacks=[
            EarlyStopping(rounds=100, save_best=True),
            CustomLRScheduler(factor=0.5, patience=20, min_lr=1e-3, start_lr=0.1, metric="rmse"),
        ])
        
        # Fit model
        model.fit(
            X_train_transformed, y_train,
            eval_set=[(X_val_transformed, y_val)],
            verbose=200,
        )

        # Inner oof (out-of-fold predictions)
        oof[val_index] += model.predict(X_val_transformed)
        # Inner test (test predictions)
        preds = model.predict(X_test_transformed)
        log_preds.append(preds) # Inplace operation returns None!


    # Calculate average predictions on test set
    log_preds_stack = np.stack(log_preds, axis=0)
    avg_log_preds = np.average(log_preds_stack, axis=0)
    final_preds = np.expm1(avg_log_preds)

    # Inverse transform oof predictions
    final_oof = np.expm1(oof)

    return final_oof, final_preds


# Model
params = {'max_depth': 6, 
          'learning_rate': 0.1, 
          'n_estimators': 2000, 
          'subsample': 0.8, 
          'colsample_bytree': 0.8,
          'min_child_weight': 40, 
          'enable_categorical': True,
          'eval_metric':'rmse'}

model_xgb = xgb.XGBRegressor(**params)


oof_log, pred_log = fit_predict_KFold_log(model_xgb, kf, df_train, df_test)  


# Training function
def fit_predict_KFold_calmin(model, kf, train_df, test_df):

    # Initialize 
    oof = np.zeros(len(train_df)) # Out-of_fold predictions
    log_preds = [] # predictions on test set

    # Extract X and y
    X = train_df.drop(columns = ["Duration", "Calories", "id"]).copy()
    y = train_df["Calories"].copy() / train_df["Duration"].copy()

    preprocessor = Pipeline(steps = [
        ("feature_engineering", AttributeAdder(add_duration_interactions=False)),
        ("general_processing", GeneralPreprocessor()),
        ("feature_selection", FeatureSelector(features_to_remove))
    ])
   
    # Main loop
    for i, (train_index, val_index) in enumerate(kf.split(train_df)):

        print("#"*100)
        print(f"### Fold {i+1}")
        print("#"*100)
    
        # Training data
        X_train = X.loc[train_index, :]
        X_train_transformed = preprocessor.fit_transform(X_train)
        y_train = y.loc[train_index]

        # Validation data
        X_val = X.loc[val_index, :]
        X_val_transformed = preprocessor.transform(X_val)
        y_val = y.loc[val_index]
        
        # Test data
        X_test_transformed = preprocessor.transform(test_df.drop(columns = ["Duration"]).copy())

        # Set callbacks
        model.set_params(callbacks=[
            EarlyStopping(rounds=100, save_best=True),
            CustomLRScheduler(factor=0.5, patience=20, min_lr=1e-3, start_lr=0.1, metric="rmsle"),
        ])
        
        # Fit model
        model.fit(
            X_train_transformed, y_train,
            eval_set=[(X_val_transformed, y_val)],
            verbose=200,
        )

        # Inner oof (out-of-fold predictions)
        oof[val_index] += model.predict(X_val_transformed) * train_df["Duration"].loc[val_index]
        # Inner test (test predictions)
        preds = model.predict(X_test_transformed) * test_df["Duration"]
        
        log_preds_k = [np.log1p(p) for p in preds] # Transform to log(1+p) space
        log_preds.append(log_preds_k) # Inplace operation returns None!

        
    # Calculate average predictions on test set
    log_preds_stack = np.stack(log_preds, axis=0)
    avg_log_preds = np.average(log_preds_stack, axis=0)
    final_preds = np.expm1(avg_log_preds)

    return oof, final_preds


from xgboost.callback import EarlyStopping
import xgboost as xgb

# Model
params = {'max_depth': 6, 
          'learning_rate': 0.1, 
          'n_estimators': 2000, 
          'subsample': 0.8, 
          'colsample_bytree': 0.8,
          'min_child_weight': 40, 
          'enable_categorical': True,
          'eval_metric':'rmsle'}

model_xgb = xgb.XGBRegressor(**params)


oof_calmin, pred_calmin = fit_predict_KFold_calmin(model_xgb, kf, df_train, df_test) 


y_true = df_train["Calories"].copy()

oof_ens = [oof_xgb, oof_log, oof_calmin]
log_preds = [np.log1p(p) for p in oof_ens]
log_preds_stack = np.stack(log_preds, axis=0)
avg_log_preds = np.average(log_preds_stack, axis=0)
final_oof = np.expm1(avg_log_preds)

print(f"XGBoost basic: {rmsle(y_true, oof_xgb)}")
print(f"XGBoost log: {rmsle(y_true, oof_log)}")
print(f"XGBoost cals/min: {rmsle(y_true, oof_calmin)}")
print(f"XGBoost ensemble: {rmsle(y_true, final_oof)}")


pred_ens = [pred_xgb, pred_log, pred_calmin]
log_preds = [np.log1p(p) for p in pred_ens]
log_preds_stack = np.stack(log_preds, axis=0)
avg_log_preds = np.average(log_preds_stack, axis=0)
final_pred = np.expm1(avg_log_preds)
final_pred = np.clip(final_pred, 0, max(final_pred))

result = pd.DataFrame({
    "id": df_test.id,
    "Calories": final_pred
})

result.to_csv('submission.csv', index=False)




