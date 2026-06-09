import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import optuna
import shap

from sklearn.cluster import KMeans
from sklearn.metrics import auc, confusion_matrix, roc_auc_score as ras, roc_curve, silhouette_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import RobustScaler


data_folder = 'playground-series-s5e8'
# Identifying Kaggle vs localhost
data_dir = f"/kaggle/input/{data_folder}/" if (os.environ.get('KAGGLE_KERNEL_RUN_TYPE', 'Localhost') in [ 'Batch', 'Interactive' ]) else './'

train_df = pd.read_csv(f'{data_dir}train.csv', index_col = 'id')
test_df = pd.read_csv(f'{data_dir}test.csv', index_col = 'id')
sample = pd.read_csv(f'{data_dir}sample_submission.csv')


print(f"{train_df[train_df.duplicated(keep = False)]}\n")
train_df.head()


train_df.describe()


for col in train_df.select_dtypes(["object"]):
    print(f'{col}: {train_df[col].unique()}')


def corr_mat(df):
    c_mat = df.corr(numeric_only = True)
    mask = np.triu(np.ones_like(c_mat, dtype = bool))
    sns.heatmap(c_mat, mask = mask, vmin = -1, vmax = 1, cmap = 'coolwarm', annot = True, square = True);

corr_mat(train_df)


# look at the numeric columns
def num_plotter(data, target = None, showmeans = False):
    for col in data.select_dtypes(["int", "float"]):
        if col != target:
            plt.figure(figsize = (5,1))
            sns.boxplot(data = data, x = col, y = target, showmeans = showmeans,
                       meanprops = {
                           'marker': 'o',
                           'markerfacecolor': 'white',
                           'markeredgecolor': 'black',
                           'markersize': '5'
                       })
            plt.show();

num_plotter(train_df.assign(y = train_df['y'].map({0: 'no', 1: 'yes'})), target = 'y', showmeans = True)


# look at the non-numeric columns
def cat_bar_plotter(df, normalize = False):
    for col in df.select_dtypes("object").columns:
        plt.figure(figsize = (6,2))
        df[col].value_counts(normalize = normalize, dropna = False).plot.bar()
        plt.show();

cat_bar_plotter(train_df.assign(y = train_df['y'].map({0: 'no', 1: 'yes'})), normalize = True)


obj_columns = ['job', 'marital', 'education', 'contact', 'poutcome']
num_columns = ['age', 'balance', 'duration', 'campaign', 'previous']

def preprocess(df):
    df = df.assign(
        default = (df['default'] == 'yes').astype(int),
        housing = (df['housing'] == 'yes').astype(int),
        loan = (df['loan'] == 'yes').astype(int),
        month = df['month'].map({'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}),
    )
    df = pd.get_dummies(df, columns = ['job', 'marital', 'education', 'contact', 'poutcome'], drop_first = True, dtype = 'int')
    return df

train_df = preprocess(train_df)
test_df = preprocess(test_df)
train_df.head()


def inertia_silhouette_plot(df_scaled, max_k, n_init = 'auto', max_iter = 100, random_state = None):
    fig, ax = plt.subplots(1, 2, figsize = (10, 3))
    inertia_val = []
    silhouette_val = []
    for i in range(2, max_k + 1):
        kmeans = KMeans(n_clusters = i, n_init = n_init, max_iter = max_iter, random_state = random_state).fit(df_scaled)
        inertia_val.append(kmeans.inertia_)
        silhouette_val.append(silhouette_score(df_scaled, kmeans.labels_, sample_size = int(df_scaled.shape[0] / 30), random_state = 42))
    inertia_series = pd.Series(inertia_val, index = range(2, max_k + 1))
    silhouette_series = pd.Series(silhouette_val, index = range(2, max_k + 1))

    inertia_series.plot(marker = 'o', ax = ax[0])
    ax[0].set_xlabel('Number of Clusters (k)')
    ax[0].set_ylabel('Inertia')
    ax[0].set_xlim(0, max_k + 1)
    ax[0].set_title('Number of clusters vs inertia')

    silhouette_series.plot(marker = 'o', ax = ax[1])
    ax[1].set_xlabel('Number of Clusters (k)')
    ax[1].set_ylabel('Silhouette')
    ax[1].set_xlim(0, max_k + 1)
    ax[1].set_title('Number of clusters vs silhouette')

scaler = RobustScaler()
train_scaled = pd.DataFrame(scaler.fit_transform(train_df[num_columns]), columns = num_columns)
test_scaled = pd.DataFrame(scaler.transform(test_df[num_columns]), columns = num_columns)
inertia_silhouette_plot(train_scaled, 10, n_init = 10, random_state = 42);


kmeans_model = KMeans(n_clusters = 2, n_init = 10, max_iter = 300, random_state = 42)
train_df['cluster'] = kmeans_model.fit_predict(train_scaled)
test_df['cluster'] = kmeans_model.predict(test_scaled)

# Cluster statistics
def make_cluster_features(df):
    new_data = {}
    for feature in num_columns:
            new_data[f'{feature}_cluster_mean'] = df.groupby('cluster')[feature].transform('mean')
            new_data[f'{feature}_cluster_std'] = df.groupby('cluster')[feature].transform('std')
            new_data[f'{feature}_cluster_rank'] = df.groupby('cluster')[feature].rank(pct = True)
    return pd.DataFrame(new_data, index = df.index)

train_df = train_df.join(make_cluster_features(train_df))
test_df  = test_df.join(make_cluster_features(test_df))


X = train_df.drop(columns = ['y'])
y = train_df['y']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)

params = {
    'metric': 'auc',    
    'verbosity': -1,
    'class_weight': {0:1,1:3},
    'random_state': 42
}

params_tuned = {
    'n_estimators': 10_000,
    'learning_rate': 0.02,
    'num_leaves': 512,
    'colsample_bytree': 0.7,
    'max_bin': 1023
}

params_optuna = { # Optuna hyperparameters
    'n_estimators': 10000,
    'learning_rate': 0.026205110621468197,
    'num_leaves': 114,
    'colsample_bytree': 0.267412032352763,
    'max_bin': 1015,
    'min_data_in_leaf': 66
}

fit_params = {'callbacks': [lgb.early_stopping(400), lgb.log_evaluation(1000)]}


# Optuna hyperparameter tuning
def objective(trial):
    params = {
        "metric": "auc",
        "verbosity": -1,
        'class_weight': {0:1,1:3},
        "n_estimators": 10000,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log = True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 2**10),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 1.0),
        "max_bin": trial.suggest_int("max_bin", 2**8 - 1, 2**10 - 1),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
    }

    model = lgb.LGBMClassifier(**params).fit(
        X_train, y_train, eval_set = [(X_val, y_val)], callbacks = [lgb.early_stopping(400)]
    )
    auc = ras(y_val, model.predict_proba(X_val)[:, 1])
    return auc

# optuna.logging.set_verbosity(optuna.logging.WARNING)
# study = optuna.create_study(direction = 'maximize')
# study.optimize(objective, n_trials = 30)
# print('Best hyperparameters:', study.best_params)
# print('Best AUC:', study.best_value)


model_base = lgb.LGBMClassifier(**params).fit(X = X_train, y = y_train)
preds_base = model_base.predict(X_val)
y_probs_base = model_base.predict_proba(X_val)[:, 1]

model_tuned = lgb.LGBMClassifier(**params, **params_tuned).fit(**fit_params, X = X_train, y = y_train, eval_set = [(X_val, y_val)])
preds_tuned = model_tuned.predict(X_val)
y_probs_tuned = model_tuned.predict_proba(X_val)[:, 1]

model_optuna = lgb.LGBMClassifier(**params, **params_optuna).fit(**fit_params, X = X_train, y = y_train, eval_set = [(X_val, y_val)])
preds_optuna = model_optuna.predict(X_val)
y_probs_optuna = model_optuna.predict_proba(X_val)[:, 1]


fig, axes = plt.subplots(1, 3, figsize=(12, 1.5))

sns.heatmap(ax = axes[0], data = confusion_matrix(y_val, preds_base), annot = True, cmap = 'Blues', fmt = 'g', square = True
           ).set(title = 'Base', xlabel = 'Predicted', ylabel = 'Actual')
sns.heatmap(ax = axes[1], data = confusion_matrix(y_val, preds_tuned), annot = True, cmap = 'Blues', fmt = 'g', square = True
           ).set(title = 'Tuned', xlabel = 'Predicted', ylabel = 'Actual')
sns.heatmap(ax = axes[2], data = confusion_matrix(y_val, preds_optuna), annot = True, cmap = 'Blues', fmt = 'g', square = True
           ).set(title = 'Optuna', xlabel = 'Predicted', ylabel = 'Actual')
plt.show();

def roc_plot(yvals, probs, names):
    plt.figure(figsize = (6, 4))
    fpr = np.zeros(len(probs), dtype = object)
    tpr = np.zeros(len(probs), dtype = object)
    thresholds = np.zeros(len(probs), dtype = object)
    auc_val = np.zeros(len(probs), dtype = object)
    for i, j in enumerate(probs):
        fpr[i], tpr[i], thresholds[i] = roc_curve(yvals, probs[i])
        auc_val[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], label = f'{names[i]} (AUC = {auc_val[i]:.4f})')

    plt.plot([0, 1], [0, 1], 'k--', label = 'Random Guess (AUC = 0.50)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Different LightGBM Models')
    plt.legend()
    plt.show()

roc_plot(y_val, [y_probs_base, y_probs_tuned, y_probs_optuna], ['Base', 'Tuned', 'Optuna'])


def cross_val(X, y, n_splits):
    kf = StratifiedKFold(n_splits = n_splits, shuffle = True, random_state = 42)
    y_pred = np.zeros(len(sample))
    cv_ras = []
    
    no_classes = len(np.unique(y))
    actual_classes = np.empty([0], dtype=int)
    predicted_classes = np.empty([0], dtype=int)
    predicted_proba = np.empty([0, no_classes])
    all_shap = []

    for train_ndx, val_ndx in kf.split(X, y):
        # Subset data based on CV folds
        X_train, y_train = X.iloc[train_ndx], y.iloc[train_ndx]
        X_val, y_val = X.iloc[val_ndx], y.iloc[val_ndx]
        actual_classes = np.append(actual_classes, y_val)
        
        # Fit the Model on fold's training data
        model = lgb.LGBMClassifier(**params, **params_tuned).fit(**fit_params, X = X_train, y = y_train, eval_set = [(X_val, y_val)])        
        cv_ras.append(ras(y_val, model.predict_proba(X_val)[:, 1]))
        y_pred += model.predict_proba(test_df)[:, 1]
        predicted_classes = np.append(predicted_classes, model.predict(X_val))
        try:
            predicted_proba = np.append(predicted_proba, model.predict_proba(X_val), axis=0)
        except:
            predicted_proba = np.append(predicted_proba, np.zeros((len(X_val), no_classes), dtype=float), axis=0)

        explainer = shap.Explainer(model, X_val)
        shap_values = explainer(X_val.sample(100, random_state = 42))
        all_shap.append(shap_values)

    print(f"All Validation AUC: {[round(x, 5) for x in cv_ras]}")
    print(f"Cross Val AUC: {np.mean(cv_ras):.5f} +/- {np.std(cv_ras):.5f}")

    return actual_classes, predicted_classes, predicted_proba, y_pred / n_splits, all_shap

# actual_classes and predicted_classes are used for the correlation matrix
# actual_classes and predicted_proba are used for the ROC plot
# final_pred is the final predictions
# all_shap is the combined shap.Explainer for the end plot
actual_classes, predicted_classes, predicted_proba, final_pred, all_shap = cross_val(X, y, 5)


plt.figure(figsize = (6,2))
sns.heatmap(data = confusion_matrix(actual_classes, predicted_classes), annot = True, cmap = 'Blues', fmt = 'g', square = True
           ).set(title = 'CV', xlabel = 'Predicted', ylabel = 'Actual')
plt.show();

roc_plot(actual_classes, [predicted_proba[:, 1]], ['CV'])

shap_values_combined = shap.Explanation(
    values = np.vstack([sv.values for sv in all_shap]),
    base_values = np.concatenate([sv.base_values for sv in all_shap]),
    data = np.vstack([sv.data for sv in all_shap]),
    feature_names = X.columns
)
plt.title('SHAP values after cross-validation')
shap.plots.beeswarm(shap_values_combined)


sample['y'] = final_pred
sample.head(10)
sample.to_csv('submission.csv',index = False)

