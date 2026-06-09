import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
import os
import warnings
warnings.filterwarnings("ignore")


df_train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
df_train.head()


df_train.describe().T


df_train.isna().sum()


df_tr = df_train.drop(columns=["id"])


numeric_df = df_tr.select_dtypes(include=['number'])
categorical_df = df_tr.select_dtypes(include=['object'])


numeric_df.head()


numeric_df.nunique()


print(numeric_df["alcohol_consumption_per_week"].value_counts())
print(numeric_df["family_history_diabetes"].value_counts())
print(numeric_df["hypertension_history"].value_counts())
print(numeric_df["cardiovascular_history"].value_counts())
print(numeric_df["diagnosed_diabetes"].value_counts())


categorical_df.head()


categorical_df.nunique()


print(categorical_df["gender"].value_counts())
print(categorical_df["ethnicity"].value_counts())
print(categorical_df["education_level"].value_counts())
print(categorical_df["income_level"].value_counts())
print(categorical_df["smoking_status"].value_counts())
print(categorical_df["employment_status"].value_counts())


df_test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
df_te = df_test.drop(columns=["id"])
df_te.head()


edu_map = {
    'No formal': 0,
    'Highschool': 1,
    'Graduate': 2,
    'Postgraduate': 3
}

income_map = {
    'Low': 0,
    'Lower-Middle': 1,
    'Middle': 2,
    'Upper-Middle': 3,
    'High': 4
}

smoke_map = {
    'Never': 0,
    'Former': 1,
    'Current': 2
}

df_tr['education_level'] = df_tr['education_level'].map(edu_map)
df_tr['income_level'] = df_tr['income_level'].map(income_map)
df_tr['smoking_status'] = df_tr['smoking_status'].map(smoke_map)

df_te['education_level'] = df_te['education_level'].map(edu_map)
df_te['income_level'] = df_te['income_level'].map(income_map)
df_te['smoking_status'] = df_te['smoking_status'].map(smoke_map)


from sklearn.preprocessing import OneHotEncoder

nominal_cols = ['gender', 'ethnicity', 'employment_status']

ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
ohe.fit(df_tr[nominal_cols])

X_tr_ohe = ohe.transform(df_tr[nominal_cols])
X_te_ohe = ohe.transform(df_te[nominal_cols])

ohe_cols = ohe.get_feature_names_out(nominal_cols)
df_tr_ohe = pd.DataFrame(X_tr_ohe, columns=ohe_cols, index=df_tr.index)
df_te_ohe = pd.DataFrame(X_te_ohe, columns=ohe_cols, index=df_te.index)

df_tr = df_tr.drop(columns=nominal_cols)
df_te = df_te.drop(columns=nominal_cols)

df_tr = pd.concat([df_tr, df_tr_ohe], axis=1)
df_te = pd.concat([df_te, df_te_ohe], axis=1)

print("Train and Test OHE applied and aligned.")


print(df_tr.shape)
print(df_te.shape)


df_tr['diagnosed_diabetes'] = df_tr['diagnosed_diabetes'].astype(int)


df_tr['diagnosed_diabetes']


X = df_tr.drop(columns=["diagnosed_diabetes"])
y = df_tr["diagnosed_diabetes"]


from sklearn.model_selection import StratifiedKFold, train_test_split
SEED=42
X_dev, X_holdout, y_dev, y_holdout = train_test_split(
    X, y, 
    test_size=0.2, 
    stratify=y, 
    random_state=SEED
)


import scipy.stats as stats

numerical_features = [
    'age',
    'physical_activity_minutes_per_week', 'diet_score',
    'sleep_hours_per_day', 'screen_time_hours_per_day', 'bmi',
    'waist_to_hip_ratio', 'systolic_bp', 'diastolic_bp', 'heart_rate',
    'cholesterol_total', 'hdl_cholesterol', 'ldl_cholesterol',
    'triglycerides'
]

# Make sure plots don't overlap
plt.style.use("seaborn-v0_8")

for feature in numerical_features:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # --- Left: KDE Plot ---
    sns.kdeplot(
        data=X,
        x=feature,
        fill=True,
        alpha=0.6,
        color="blue",
        ax=axes[0]
    )
    axes[0].set_title(f"KDE Plot of {feature}")

    # --- Right: Q-Q Plot ---
    stats.probplot(X[feature].dropna(), dist="norm", plot=axes[1])
    axes[1].set_title(f"QQ Plot of {feature}")

    plt.tight_layout()
    plt.show()


import math

num_feats = numerical_features
n_feats = len(num_feats)

plots_per_page = 16
pages = math.ceil(n_feats / plots_per_page)

for p in range(pages):
    start = p * plots_per_page
    end = min(start + plots_per_page, n_feats)
    feats_subset = num_feats[start:end]

    fig, axes = plt.subplots(4, 4, figsize=(18, 16))
    axes = axes.flatten()

    for ax, feature in zip(axes, feats_subset):
        sns.boxplot(
            x=y, 
            y=X[feature], 
            ax=ax, 
            palette="Set2"
        )
        ax.set_title(feature)
        ax.set_xlabel("diagnosed_diabetes (0 or 1)")
        ax.set_ylabel(feature)

    # Hide empty subplots (if number < 16)
    for k in range(len(feats_subset), 16):
        fig.delaxes(axes[k])

    fig.suptitle(f"Boxplots: Features {start+1} to {end}", fontsize=16)
    plt.tight_layout()
    plt.show()


from sklearn.preprocessing import PowerTransformer, StandardScaler

# 1) Features requiring Yeo-Johnson
yeo_features = [
    'physical_activity_minutes_per_week',
    'triglycerides'
]

# 2) Features requiring Winsorization
winsor_features = [
    'physical_activity_minutes_per_week',
    'screen_time_hours_per_day',
    'systolic_bp',
    'diastolic_bp',
    'cholesterol_total',
    'ldl_cholesterol',
    'triglycerides'
]

# 3) Binary or ordinal (no transform and no scale also no winsorization)
binary_features = [
    'family_history_diabetes',
    'hypertension_history',
    'cardiovascular_history',
    'alcohol_consumption_per_week'
]

# 4) Everything else gets normal StandardScaler
continuous_features = [
    'age', 'sleep_hours_per_day', 'bmi', 'waist_to_hip_ratio', 
    'heart_rate', 'hdl_cholesterol','diet_score'
]

# Combine for scaling later
all_scale_features = (
    yeo_features 
    + winsor_features 
    + continuous_features
)


import tensorflow as tf
import optuna
from tensorflow import keras
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score


def build_model(trial, input_dim):

    # Hyperparameter search
    n_layers = trial.suggest_int("n_layers", 1, 2)
    n_units = trial.suggest_int("n_units", 64, 512)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    l2_reg = trial.suggest_float("l2_reg", 1e-5, 1e-2, log=True)
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)

    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(input_dim,)))

    for _ in range(n_layers):
        model.add(keras.layers.Dense(
            n_units,
            activation="relu",
            kernel_regularizer=keras.regularizers.l2(l2_reg)
        ))
        model.add(keras.layers.Dropout(dropout))

    # Final output layer
    model.add(keras.layers.Dense(1, activation="sigmoid"))

    # Compile
    optimizer = keras.optimizers.Adam(learning_rate=lr)

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")]
    )

    return model


def objective(trial):
    batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])
    epochs = trial.suggest_int("epochs", 20, 30, 50)

    oof_preds = np.zeros(len(X_dev))
    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

    for train_idx, val_idx in kf.split(X_dev, y_dev):
        
        X_tr = X_dev.iloc[train_idx]
        X_val = X_dev.iloc[val_idx]
        y_tr = y_dev.iloc[train_idx]
        y_val = y_dev.iloc[val_idx]

        # winsorization
        winsor_limits = {}
        for f in winsor_features:
            lower = X_tr[f].quantile(0.01)
            upper = X_tr[f].quantile(0.99)
            winsor_limits[f] = (lower, upper)
            X_tr[f] = np.clip(X_tr[f], lower, upper)

        for f in winsor_features:
            lower, upper = winsor_limits[f]
            X_val[f] = np.clip(X_val[f], lower, upper)
        
        # yeo jhonson transformation
        pt = PowerTransformer(method='yeo-johnson')
        X_tr[yeo_features] = pt.fit_transform(X_tr[yeo_features])
        X_val[yeo_features] = pt.transform(X_val[yeo_features])

        # standard scaler
        scaler = StandardScaler()
        X_tr[all_scale_features] = scaler.fit_transform(X_tr[all_scale_features])
        X_val[all_scale_features] = scaler.transform(X_val[all_scale_features])

        # convert to  np array
        X_tr = X_tr.values.astype("float32")
        X_val = X_val.values.astype("float32")
        y_tr = y_tr.values.astype("float32")
        y_val = y_val.values.astype("float32")

        model = build_model(trial, X_tr.shape[1])
        es = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor="val_auc", mode="max")
        model.fit(X_tr, y_tr, validation_data=(X_val, y_val),
                  batch_size=batch_size, epochs=epochs, verbose=0, callbacks=[es])
        oof_preds[val_idx] = model.predict(X_val, batch_size=2048).ravel()

    return roc_auc_score(y_dev, oof_preds)


study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=30)

print("Best params:", study.best_params)


winsor_limits = {}
for f in winsor_features:
    lower = X_dev[f].quantile(0.01)
    upper = X_dev[f].quantile(0.99)
    winsor_limits[f] = (lower, upper)
    X_dev[f] = np.clip(X_dev[f], lower, upper)
for f in winsor_features:
    lower, upper = winsor_limits[f]
    X_holdout[f] = np.clip(X_holdout[f], lower, upper)


pt = PowerTransformer(method='yeo-johnson')
X_dev[yeo_features] = pt.fit_transform(X_dev[yeo_features])
X_holdout[yeo_features] = pt.transform(X_holdout[yeo_features])


scaler = StandardScaler()
X_dev[all_scale_features] = scaler.fit_transform(X_dev[all_scale_features])
X_holdout[all_scale_features] = scaler.transform(X_holdout[all_scale_features])


X_dev_np = X_dev.values.astype(np.float32)
X_holdout_np = X_holdout.values.astype(np.float32)


best_trial = study.best_trial
best_params = study.best_params

model_final_holdout = build_model(best_trial, X_dev_np.shape[1])
es = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor="val_auc", mode="max")
model_final_holdout.fit(X_dev_np, y_dev.values, validation_data=(X_holdout_np, y_holdout.values),
                        batch_size=best_params["batch_size"], epochs=best_params["epochs"],
                        verbose=0, callbacks=[es])

holdout_pred = model_final_holdout.predict(X_holdout_np, batch_size=2048).ravel()
print("Holdout AUC:", roc_auc_score(y_holdout, holdout_pred))


oof_preds_full = np.zeros(len(X))
test_preds_full = np.zeros(len(df_te))

kf_full = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

for fold, (train_idx, val_idx) in enumerate(kf_full.split(X, y)):
    print(f"Fold {fold+1}...")
    X_tr = X.iloc[train_idx]
    X_val = X.iloc[val_idx]
    y_tr = y.iloc[train_idx]
    y_val = y.iloc[val_idx]

    # winsorization
    winsor_limits = {}
    for f in winsor_features:
        lower = X_tr[f].quantile(0.01)
        upper = X_tr[f].quantile(0.99)
        winsor_limits[f] = (lower, upper)
        X_tr[f] = np.clip(X_tr[f], lower, upper)

    for f in winsor_features:
        lower, upper = winsor_limits[f]
        X_val[f] = np.clip(X_val[f], lower, upper)
        
    # yeo jhonson transformation
    pt = PowerTransformer(method='yeo-johnson')
    X_tr[yeo_features] = pt.fit_transform(X_tr[yeo_features])
    X_val[yeo_features] = pt.transform(X_val[yeo_features])

    # standard scaler
    scaler = StandardScaler()
    X_tr[all_scale_features] = scaler.fit_transform(X_tr[all_scale_features])
    X_val[all_scale_features] = scaler.transform(X_val[all_scale_features])

    # convert to  np array
    X_tr = X_tr.values.astype("float32")
    X_val = X_val.values.astype("float32")
    y_tr = y_tr.values.astype("float32")
    y_val = y_val.values.astype("float32")

    df_te_fold = df_te.copy()
    # winsorization on df_te
    for f in winsor_features:
        lower, upper = winsor_limits[f]
        df_te_fold[f] = np.clip(df_te_fold[f], lower, upper)

    # yeo jhonson transformation on df_te
    df_te_fold[yeo_features] = pt.transform(df_te_fold[yeo_features])

    # Standard scaling on df_te
    df_te_fold[all_scale_features] = scaler.transform(df_te_fold[all_scale_features])
    
    df_te_np = df_te_fold.values.astype(np.float32)
    
    model = build_model(best_trial, X_tr.shape[1])
    es = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor="val_auc", mode="max")
    model.fit(X_tr, y_tr, validation_data=(X_val, y_val),
              batch_size=best_params["batch_size"], epochs=best_params["epochs"],
              verbose=0, callbacks=[es])

    oof_preds_full[val_idx] = model.predict(X_val, batch_size=2048).ravel()
    test_preds_full += model.predict(df_te_np, batch_size=2048).ravel() / 5

print("Final OOF AUC:", roc_auc_score(y, oof_preds_full))


df_oof = pd.DataFrame({
    "id": df_train["id"],
    "diagnosed_diabetes": y,
    "nn_pred": oof_preds_full
})
df_oof.to_csv("oof_nn.csv", index=False)

df_test_pred = pd.DataFrame({
    "id": df_test["id"],
    "nn_pred": test_preds_full
})
df_test_pred.to_csv("test_nn.csv", index=False)

print("Saved oof_nn.csv and test_nn.csv")


subm = pd.read_csv("/kaggle/working/test_nn.csv")
subm.head()


subm = subm.rename(columns={'nn_pred': 'diagnosed_diabetes'})
subm.to_csv('submission.csv', index=False)




