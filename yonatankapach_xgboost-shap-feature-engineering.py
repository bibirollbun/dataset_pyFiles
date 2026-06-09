import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

import optuna
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split


sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')
df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
df_sub = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


warnings.filterwarnings('ignore', category=FutureWarning)

# Check target distribution
sns.countplot(data=df, x='Personality')
plt.title("Target Class Distribution"); plt.show()

# Correlation matrix of numerical features
sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation"); plt.show()

# Pairplot for selected features
sns.pairplot(df, hue='Personality', vars=df.select_dtypes('number').columns[:4])


# Split
X = df.drop(['id','Personality'], axis=1)
y_raw = df['Personality']
le = LabelEncoder()
y = le.fit_transform(y_raw)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# Preprocessing
num_feats = X.select_dtypes('number').columns.tolist()
cat_feats = X.select_dtypes('object').columns.tolist()


# Split columns
X_num = df[num_feats]
X_cat = df[cat_feats]

# Define missing masks for validation (only on observed entries)
mask_num_obs = ~X_num.isna()
mask_cat_obs = ~X_cat.isna()

def random_mask(X, mask_obs, frac=0.1, seed=0):
    rng = np.random.RandomState(seed)
    mask = mask_obs & (rng.rand(*X.shape) < frac)
    X_masked = X.copy()
    X_masked[mask] = np.nan
    return X_masked, mask


def objective(trial):
    # --- Hyperparameters ---
    max_iter = trial.suggest_int("max_iter", 5, 20)
    tol = trial.suggest_float("tol", 1e-2, 1e-1, log=True)

    rf_reg = RandomForestRegressor(
        n_estimators=trial.suggest_int("reg_n_est", 20, 80),
        max_depth=trial.suggest_int("reg_max_depth", 3, 6),
        min_samples_leaf=trial.suggest_int("reg_min_leaf", 1, 5),
        random_state=42, n_jobs=-1
    )

    rf_clf = RandomForestClassifier(
        n_estimators=trial.suggest_int("clf_n_est", 20, 80),
        max_depth=trial.suggest_int("clf_max_depth", 3, 6),
        random_state=42, n_jobs=-1
    )

    # --- Mask for validation ---
    X_num_masked, mask_num_val = random_mask(X_num, mask_num_obs, frac=0.1, seed=trial.number)
    X_cat_masked, mask_cat_val = random_mask(X_cat, mask_cat_obs, frac=0.1, seed=trial.number)

    # --- Encode categorical before imputing ---
    cat_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_cat_enc   = cat_encoder.fit_transform(X_cat_masked)
    X_cat_true  = cat_encoder.transform(X_cat)

    # --- Fit imputers ---
    imp_num = IterativeImputer(estimator=rf_reg, max_iter=max_iter, tol=tol, random_state=42)
    imp_cat = IterativeImputer(estimator=rf_clf, max_iter=max_iter, tol=tol, random_state=42)

    X_num_imp = imp_num.fit_transform(X_num_masked.to_numpy())
    X_cat_imp = imp_cat.fit_transform(X_cat_enc)

    # --- Extract matched entries ---
    val_idx_num = np.where(mask_num_val.to_numpy())
    val_idx_cat = np.where(mask_cat_val.to_numpy())

    y_num_true = X_num.to_numpy()[val_idx_num]
    y_num_pred = X_num_imp[val_idx_num]
    y_cat_true = X_cat_true[val_idx_cat]
    y_cat_pred = X_cat_imp[val_idx_cat]

    # --- Evaluation ---
    label_scaler = StandardScaler()
    y_num_true_nrm = label_scaler.fit_transform(y_num_true.reshape(-1,1))
    y_num_pred_nrm = label_scaler.fit_transform(y_num_pred.reshape(-1,1))
    rmse = np.sqrt(mean_squared_error(y_num_true_nrm, y_num_pred_nrm))
    acc = accuracy_score(y_cat_true.ravel(), y_cat_pred.ravel())

    return -acc + rmse  # combined objective


sampler = optuna.samplers.TPESampler(
    seed=42,                 # for reproducibility
    n_startup_trials=5     # pure random trials before TPE kicks in
)

pruner = optuna.pruners.MedianPruner(
    n_startup_trials=5,      # let early trials complete before pruning kicks in
    n_warmup_steps=5         # wait for enough steps (if using intermediate reports)
)

study = optuna.create_study(direction='minimize', sampler=sampler, pruner=pruner)
study.optimize(objective, n_trials=50, timeout=600)

print("Best score:", study.best_value)
print("Best parameters:", study.best_params)


best = study.best_params

rf_reg = RandomForestRegressor(
    n_estimators=best["reg_n_est"],
    max_depth=best["reg_max_depth"],
    min_samples_leaf=best["reg_min_leaf"],
    random_state=42, n_jobs=-1
)

rf_clf = RandomForestClassifier(
    n_estimators=best["clf_n_est"],
    max_depth=best["clf_max_depth"],
    random_state=42, n_jobs=-1
)

imp_num = IterativeImputer(estimator=rf_reg, max_iter=best["max_iter"], tol=best["tol"], random_state=42)
imp_cat = IterativeImputer(estimator=rf_clf, max_iter=best["max_iter"], tol=best["tol"], random_state=42)

# Build pipelines for each feature type
num_pipe = Pipeline([
    ('impute', imp_num),
    ('scale', StandardScaler())
])

cat_pipe = Pipeline([
    ('encode', OrdinalEncoder(handle_unknown='use_encoded_value',
                              unknown_value=-1)),
    ('impute', imp_cat)
])

# Combine using ColumnTransformer
preprocessor = ColumnTransformer([
    ('num', num_pipe, num_feats),
    ('cat', cat_pipe, cat_feats)
])


Xp = preprocessor.fit_transform(X)
Xp_train, Xp_test, y_train, y_test = train_test_split(Xp, y, stratify=y, random_state=52)


def objective(trial):
    params = {
        'n_estimators':      trial.suggest_int('n_estimators', 50, 500),
        'max_depth':         trial.suggest_int('max_depth', 3, 10),
        'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample':         trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha':         trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        'reg_lambda':        trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        'objective':         'binary:logistic',
        'use_label_encoder': False,
        'eval_metric':       'logloss',
        'random_state':      42
    }

    model = XGBClassifier(**params)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, Xp_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
    return scores.mean()

study = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner)
study.optimize(objective, n_trials=50)


xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, **study.best_params)
xgb_model.fit(Xp_train, y_train)
test_acc = xgb_model.score(Xp_test, y_test)
print(f"Hold-out Test Accuracy: {test_acc:.4f}")


import shap

Xp_test_df = pd.DataFrame(Xp_test, columns=X.columns)
Xp_train_df = pd.DataFrame(Xp_train, columns=X.columns)

# Initialize SHAP explainer
explainer = shap.Explainer(xgb_model, Xp_train_df)
shap_values = explainer(Xp_test_df)

# ðŸ”¥ Global feature impact
shap.plots.bar(shap_values)

# ðŸŒˆ Summary plot: distribution & impact
shap.summary_plot(shap_values, Xp_test_df, plot_type="dot")


# Pick a few test samples
for i in range(3):
    shap.plots.waterfall(shap_values[i], max_display=10)


# See interaction between top 2 impactful features
shap.plots.scatter(shap_values[:, 4], color=shap_values[:, 1])


from sklearn.manifold import TSNE

shap_matrix = np.array(shap_values.values)
shap_tsne = TSNE(n_components=2, perplexity=8, random_state=42).fit_transform(shap_matrix)

plt.figure(figsize=(8,6))
plt.scatter(shap_tsne[:,0], shap_tsne[:,1], c=y_test, cmap='coolwarm', alpha=0.7)
plt.title("SHAP-based clustering of X_test samples")
plt.xlabel("t-SNE 1"); plt.ylabel("t-SNE 2")
plt.colorbar(label='True class'); plt.show()


interaction_values = shap.TreeExplainer(xgb_model).shap_interaction_values(Xp_test_df)

# View the top interacting pairs
mean_interactions = np.abs(interaction_values).mean(axis=0)
interaction_df = pd.DataFrame(mean_interactions, columns=Xp_test_df.columns, index=Xp_test_df.columns)
interaction_df.sort_values(by=interaction_df.columns[0], ascending=False)


# Check interactions
shap.dependence_plot("Time_spent_Alone", shap_values.values, Xp_test_df, interaction_index="Friends_circle_size")
shap.dependence_plot("Time_spent_Alone", shap_values.values, Xp_test_df, interaction_index="Post_frequency")
shap.dependence_plot("Friends_circle_size", shap_values.values, Xp_test_df, interaction_index="Going_outside")


# Get top 5 pairs
top_pairs = (
    interaction_df.where(~np.eye(len(interaction_df), dtype=bool))
    .stack()
    .sort_values(ascending=False)
    .head(10)
)
print("Top interacting pairs:\n", top_pairs)


Xp2_train_df = Xp_train_df.copy()
Xp2_test_df = Xp_test_df.copy()

Xp2_train_df["Time_x_Post"] = Xp_train_df["Time_spent_Alone"] * Xp_train_df["Post_frequency"]
Xp2_train_df["Going_x_Post"] = Xp_train_df["Going_outside"] * Xp_train_df["Post_frequency"]
Xp2_train_df["Friends_x_Time"] = Xp_train_df["Friends_circle_size"] * Xp_train_df["Time_spent_Alone"]
Xp2_train_df["Time_x_Stage"] = Xp_train_df["Time_spent_Alone"] * Xp_train_df["Stage_fear"]
Xp2_train_df["Going_x_Time"] = Xp_train_df["Going_outside"] * Xp_train_df["Time_spent_Alone"]

Xp2_test_df["Time_x_Post"] = Xp_test_df["Time_spent_Alone"] * Xp_test_df["Post_frequency"]
Xp2_test_df["Going_x_Post"] = Xp_test_df["Going_outside"] * Xp_test_df["Post_frequency"]
Xp2_test_df["Friends_x_Time"] = Xp_test_df["Friends_circle_size"] * Xp_test_df["Time_spent_Alone"]
Xp2_test_df["Time_x_Stage"] = Xp_test_df["Time_spent_Alone"] * Xp_test_df["Stage_fear"]
Xp2_test_df["Going_x_Time"] = Xp_test_df["Going_outside"] * Xp_test_df["Time_spent_Alone"]


xgb_model_extended = XGBClassifier(**study.best_params)
xgb_model_extended.fit(Xp2_train_df, y_train)  # or re-run with Xp_train extended


explainer_ext = shap.Explainer(xgb_model_extended, Xp2_train_df)
shap_vals_ext = explainer_ext(Xp2_train_df)
shap.summary_plot(shap_vals_ext, Xp2_train_df)


X_sub = df_sub.drop(['id'], axis=1)
X_sub = preprocessor.fit_transform(X_sub)
X_sub = pd.DataFrame(X_sub, columns=X.columns)
X_sub["Time_x_Post"] = X_sub["Time_spent_Alone"] * X_sub["Post_frequency"]
X_sub["Going_x_Post"] = X_sub["Going_outside"] * X_sub["Post_frequency"]
X_sub["Friends_x_Time"] = X_sub["Friends_circle_size"] * X_sub["Time_spent_Alone"]
X_sub["Time_x_Stage"] = X_sub["Time_spent_Alone"] * X_sub["Stage_fear"]
X_sub["Going_x_Time"] = X_sub["Going_outside"] * X_sub["Time_spent_Alone"]
y_sub = xgb_model_extended.predict(X_sub)

submission = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')
submission['Personality'] = le.inverse_transform(y_sub)
submission.to_csv('submission.csv', index=False)

