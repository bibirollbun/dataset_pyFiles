MODEL = "ENSEMBLE"
V = 11


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from warnings import filterwarnings

filterwarnings("ignore")


train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")         
test  = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")

display(train.head(4))
display(test.head(4))


train.drop(columns = ['id'],inplace=True)
test.drop(columns = ['id'],inplace=True)


original = pd.read_csv('/kaggle/input/rainfall-prediction-using-machine-learning/Rainfall.csv')
original = original.rename(columns={'pressure ': 'pressure', 'humidity ': 'humidity', 'cloud ': 'cloud','         winddirection':'winddirection'})
original = original[train.columns].replace({'yes': 1, 'no': 0})
train = pd.concat([train, original], axis=0, ignore_index=True)
print(train.shape)
display(train.head(3))


def FE(data):
    df = data.copy()

    # Temperature Features
    df['temp_gap'] = df['maxtemp'] - df['mintemp']
    #df['temp_avg'] = (df['maxtemp'] + df['mintemp']) / 2
    df['dewpoint_depression'] = df['temparature'] - df['dewpoint']

    # Wind Features
    df['wind_power'] = df['windspeed'] ** 3
    
    # Cloud & Humidity Features
    df['cloud_rolling_avg'] = df['cloud'].rolling(window=3, min_periods=1).mean().fillna(method='bfill')
    df['humidity_cloud_interaction'] = df['humidity'] * df['cloud']
    #df['dewpoint_humidity_interaction'] = df['dewpoint'] * df['humidity']
    
    # Extreme Weather Flags
    #df['high_wind_flag'] = (df['windspeed'] >= 40).astype(int)

    # Drop Unimportant Columns
    df.drop(columns=['day', 'winddirection', 'pressure', 'mintemp', 'temparature'], inplace=True)
    
    return df


train = FE(train)
test  = FE(test)


num_cols = [col for col in train.columns if train[col].dtype in ['float64','int64']]
cat_cols = [col for col in train.columns if train[col].dtype in ['O','category']]


for col in cat_cols:
    merged_df = pd.concat([train[col],test[col]],axis=0)
    val,_ = pd.factorize(merged_df)
    train[col] = val[:len(train)]
    test[col] = val[len(train):]


display(train.head(3))
display(test.head(3))


from data_analysis import load_data


train_loader = load_data(file_df = train)
test_loader = load_data(file_df = test)


train_loader.summarize()


train_loader.feature_target_dependence(target_col = 'rainfall')


train_loader.impute_columns()
test_loader.impute_columns()


train = train_loader.get_df()
test = test_loader.get_df()


target =  "rainfall"
Features = [col for col in train.columns if col!=target]
print(f"We have {len(Features)} features")
print(Features)


import optuna
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


target = 'rainfall'

X = train[Features].copy()  
y = train[target]

# Convert categorical columns to category dtype
cat_cols = X.select_dtypes(include=['object']).columns
X[cat_cols] = X[cat_cols].astype("category")

# Define objective function for Optuna
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 5000, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0001, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0001, 1.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "random_state": 42,
        "enable_categorical": True  # Ensure categorical support
    }

    # Stratified K-Fold Cross Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(X))
    
    for train_idx, valid_idx in skf.split(X, y):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        
        model = XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            early_stopping_rounds=100,
            verbose=False
        )
        
        oof[valid_idx] = model.predict_proba(X_valid)[:, 1]

    # Compute mean ROC AUC score
    score = roc_auc_score(y, oof)
    return score

# Run Optuna optimization
study = optuna.create_study(direction="maximize")  # Maximize AUC Score
study.optimize(objective, n_trials=50)  # Run 50 trials

# Best hyperparameters
print("Best AUC Score:", study.best_value)
print("Best Hyperparameters:", study.best_params)


import optuna
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

target = 'rainfall'

X = train[Features].copy()  
y = train[target]

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 2000, step=100),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "bootstrap": True,  # Required for `predict_proba` to work
        "random_state": 42,
        "n_jobs": -1
    }

    # Stratified K-Fold Cross Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(X))
    
    for train_idx, valid_idx in skf.split(X, y):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        
        model = ExtraTreesClassifier(**params)
        model.fit(X_train, y_train)
        
        oof[valid_idx] = model.predict_proba(X_valid)[:, 1]

    # Compute mean ROC AUC score
    score = roc_auc_score(y, oof)
    return score

# Run Optuna optimization
study = optuna.create_study(direction="maximize")  # Maximize AUC Score
study.optimize(objective, n_trials=10)  # Run 10 trials

# Best hyperparameters
print("Best AUC Score:", study.best_value)
print("Best Hyperparameters:", study.best_params)


from xgboost import XGBClassifier
from sklearn.metrics import auc,roc_auc_score
from sklearn.ensemble import  ExtraTreesClassifier
from tabpfn import TabPFNClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression


STATS = ["mean","std","count","nunique","median","min","max","skew"]


def train_model_with_cv(model_name, params, train, test, target, features, folds=5,model=None):
    """
    Trains a model using nested Stratified K-Fold CV with target encoding.
    
    Parameters:
    - model_name: str -> Model to train ("xgb", "lgbm", "catboost", "extra_trees", "tabpfn")
    - params: dict -> Model hyperparameters
    - train: pd.DataFrame -> Training dataset
    - test: pd.DataFrame -> Test dataset
    - target: str -> Target column name
    - features: list -> List of feature column names
    - folds: int -> Number of folds for StratifiedKFold (default: 5)

    Returns:
    - oof: np.array -> Out-of-fold predictions
    - preds: np.array -> Averaged test predictions
    """

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    
    # Initialize OOF predictions and test predictions
    oof = np.zeros(len(train))
    preds = np.zeros(len(test))
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(train[features], train[target])):
        print(f"####### Outer Fold : {fold + 1} #######")

        # Split train/val sets
        X_train, X_val = train.loc[train_idx, features + [target]].reset_index(drop=True), train.loc[test_idx, features].reset_index(drop=True)
        y_train, y_val = train.loc[train_idx, target], train.loc[test_idx, target]
        X_test = test.copy()

        # Inner Stratified K-Fold for target encoding
        skf2 = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        for foldj, (train_idx_2, test_idx_2) in enumerate(skf2.split(X_train[features], X_train[target])):
            print(f"Inner Fold : {foldj + 1}")

            # Inner train/val split
            X_train_2, X_val_2 = X_train.loc[train_idx_2, features + [target]], X_train.loc[test_idx_2, features]
            y_train_2, y_val_2 = X_train.loc[train_idx_2, target], X_train.loc[test_idx_2, target]

            # Target Encoding
            for col in features:
                tmp = X_train_2.groupby(col)[target].agg(['mean', 'std'])
                tmp.columns = [f"TE_{col}_mean", f"TE_{col}_std"]
                X_val_2 = X_val_2.merge(tmp, on=col, how="left")
                for c in tmp.columns:
                    X_train.loc[test_idx_2, c] = X_val_2[c].values

        # Apply Target Encoding to validation and test sets
        for col in features:
            tmp = X_train.groupby(col)[target].agg(['mean', 'std'])
            tmp.columns = [f"TE_{col}_mean", f"TE_{col}_std"]
            X_val = X_val.merge(tmp, on=col, how="left")
            X_test = X_test.merge(tmp, on=col, how="left")

        # Drop target from training data
        X_train = X_train.drop(columns=[target])

        # Select model
        if model_name == "xgb":
            model = XGBClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            
        elif model_name == "lgbm":
            model = LGBMClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        
        elif model_name == "catboost":
            model = CatBoostClassifier(**params)
            model.fit(X_train, y_train)
        
        elif model_name == "extra_trees":
            model = ExtraTreesClassifier(**params)
            model.fit(X_train, y_train)
        
        elif model_name == "tabpfn":
            model = TabPFNClassifier(**params)
            model.fit(X_train, y_train)
        
        else:
            if not model:
                print(f"Enter the model atleast")
            else:
                model.fit(X_train,y_train)

        # Store predictions
        oof[test_idx] = model.predict_proba(X_val)[:, 1]
        preds += model.predict_proba(X_test)[:, 1] / folds  # Averaging over folds

        score = roc_auc_score(y_val, oof[test_idx])
        print(f"Fold ROC Score: {score}")

    # Compute final ROC AUC score
    score_t = roc_auc_score(train[target], oof)
    print(f"Overall ROC Score: {score_t}")

    return oof, preds


params_xgb = {
    'n_estimators': 800,  # Reduced from 3300 to prevent overfitting
    'learning_rate': 0.05,  # Lowered slightly for more stable training
    'max_depth': 2,  # Reduced complexity to avoid overfitting
    'subsample': 0.75,  # Slightly lower to introduce more randomness
    'colsample_bytree': 0.7,  # Reduced to avoid over-reliance on specific features
    'reg_alpha': 0.01,  # Stronger L1 regularization for sparsity
    'reg_lambda': 0.1,  # Increased L2 regularization to prevent overfitting
    'min_child_weight': 20,  # Increased to prevent small noisy splits
    'gamma': 3,  # Increased to make splits more conservative
    'enable_categorical': True  # Keep categorical handling enabled
}


oof_xgb, preds_xgb = train_model_with_cv(
    model_name="xgb",
    params=params_xgb,
    train=train,
    test=test,
    target="rainfall",
    features=Features,
    folds=5
)


params_lgbm = {
    "n_estimators": 1000,       # Reduce for small datasets (500-2000)
    "learning_rate": 0.05,      # Small step size for better generalization
    "max_depth": -1,            # Let the model decide depth
    "num_leaves": 31,           # Default is 31, increase for complex data
    "min_data_in_leaf": 20,     # Minimum data points per leaf (avoid overfitting)
    "feature_fraction": 0.8,    # Random feature selection (helps regularization)
    "bagging_fraction": 0.8,    # Row sampling to add randomness
    "bagging_freq": 5,          # Perform bagging every 5 iterations
    "reg_alpha": 0.1,           # L1 regularization
    "reg_lambda": 0.1,          # L2 regularization
    "objective": "binary",      # Binary classification task
    "metric": "auc",            # Evaluation metric
    "random_state": 42,
    'early_stopping_rounds':100,
    'verbosity':-1
}


oof_lgb, preds_lgb = train_model_with_cv(
    model_name="lgbm",
    params=params_lgbm,
    train=train,
    test=test,
    target="rainfall",
    features=Features,
    folds=5
)


params_catboost = {
    "iterations": 1000,        # Reduced to prevent overfitting (500-2000 range)
    "learning_rate": 0.05,     # Small step size to generalize well
    "depth": 6,                # Avoid too deep trees (default is 6-8)
    "l2_leaf_reg": 10,         # Strong L2 regularization
    "random_strength": 2,      # Randomness to prevent overfitting
    "border_count": 64,        # Number of binarization splits
    "subsample": 0.8,          # Adds randomness for better generalization
    "colsample_bylevel": 0.8,  # Feature sampling
    "loss_function": "Logloss", # Binary classification
    "eval_metric": "AUC",      # AUC metric
    "random_seed": 42,
    "verbose": 500,
    "early_stopping_rounds": 100
}


oof_cat, preds_cat = train_model_with_cv(
    model_name="catboost",
    params=params_catboost,
    train=train,
    test=test,
    target="rainfall",
    features=Features,
    folds=5
)


params_etc = {
    'n_estimators': 300,  # Reduced from 1000 to prevent overfitting
    'max_depth': None,  # Let the tree expand naturally, limiting via `min_samples_split`
    'min_samples_split': 10,  # Increased to prevent too many small splits
    'min_samples_leaf': 4,  # Increased to ensure each leaf has more samples
    'max_features': 0.5,  # Reduced feature usage to encourage diversity
    'bootstrap': False,  # ExtraTrees performs better without bootstrapping
    'random_state': 42,  # Keep for reproducibility
    'n_jobs': -1  # Utilize all cores for faster training
}


oof_cat, preds_cat = train_model_with_cv(
    model_name="extra_trees",
    params=params_etc,
    train=train,
    test=test,
    target="rainfall",
    features=Features,
    folds=5
)


tab_pfn_params = {'device':'cuda'}

oof_tab, preds_tab = train_model_with_cv(
    model_name="tabpfn",
    params=tab_pfn_params,
    train=train,
    test=test,
    target="rainfall",
    features=Features,
    folds=5
)


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


from sklearn.preprocessing import StandardScaler

train_nn = train.copy()
test_nn = test.copy()

scaler = StandardScaler() 
train_nn_scaled = scaler.fit_transform(train_nn[Features])
test_nn_scaled = scaler.transform(test_nn[Features])


X = train_nn_scaled
y = train[target]
X_test = test_nn_scaled


skf = StratifiedKFold(n_splits=FOLDS, random_state=42, shuffle=True)

oof_nn = np.zeros(len(y)) 
preds_nn = np.zeros(len(X_test))

for train_idx, val_idx in skf.split(X,y):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    model = Sequential([
        Dense(units = 256,activation='relu',kernel_initializer='he_normal',input_shape=(X.shape[1],)),
        Dropout(0.3),
        Dense(units = 128,activation='relu',kernel_initializer='he_normal'),
        Dropout(0.3),
        Dense(units = 64,activation='relu'),
        Dropout(0.3),
        Dense(units = 32,activation='relu'),
        Dropout(0.3),
        Dense(units = 16,activation='relu'),
        Dropout(0.3),
        Dense(units = 1,activation='sigmoid')
         ])
    
    earlyStopping = EarlyStopping(monitor="val_loss",patience=20,restore_best_weights=True)
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_val, y_val), 
              callbacks=[earlyStopping], verbose=1)

    oof_nn[val_idx] = model.predict(X_val).flatten() 
    preds_nn += model.predict(X_test).flatten()

preds_nn /= 5


score_f = roc_auc_score(train[target],oof_nn)
print(f"FINAL AUC : {score_f}")


from hillclimbers import climb_hill, partial


oof_df = pd.DataFrame()
preds_df = pd.DataFrame()

oof_df['xgb'] = oof_xgb
oof_df['cat'] = oof_cat
oof_df['lgbm'] = oof_lgb
oof_df['tabpfn'] = oof_tab



preds_df['xgb'] = preds_xgb
preds_df['cat'] = preds_cat
preds_df['lgbm'] = preds_lgb
preds_df['tabpfn'] = preds_tab


hc_test_pred_probs, hc_oof_pred_probs = climb_hill(
    train = train,
    oof_pred_df=oof_df, 
    test_pred_df=preds_df,
    target="rainfall",
    objective='maximize', 
    eval_metric=partial(roc_auc_score), 
    negative_weights=True, 
    precision=0.0001, 
    plot_hill=False, 
    plot_hist=False,
    return_oof_preds=True
)


sub_df = pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")
sub_df['rainfall'] = preds_tab
sub_df.head()


sub_df.to_csv(f"{MODEL}_{V}.csv",index=False)

