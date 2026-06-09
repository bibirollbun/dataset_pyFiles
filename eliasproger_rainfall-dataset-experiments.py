import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, RandomizedSearchCV, KFold, TimeSeriesSplit, train_test_split
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer, OneHotEncoder, PolynomialFeatures, StandardScaler, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectFromModel, RFECV
from sklearn.decomposition import PCA
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


TRAIN_DATASET_PATH = "/kaggle/input/playground-series-s5e3/train.csv"
TEST_DATASET_PATH = "/kaggle/input/playground-series-s5e3/test.csv"
SUBMISSION = True


import warnings
warnings.filterwarnings(
    action='ignore',
    category=DeprecationWarning,
)
warnings.filterwarnings(
    action='ignore',
    category=FutureWarning,
)
warnings.filterwarnings(
    action='ignore',
    category=RuntimeWarning,
)


dataset = pd.read_csv(TRAIN_DATASET_PATH)


dataset.head()


dataset.describe()


dataset.info()


num_features = dataset.columns
print(f"Numeric features amount: {len(num_features)}")
print(f"Numeric features names:")
print(*num_features, sep=", ")


data_counts = dataset["rainfall"].value_counts().sort_values()
plt.pie([data_counts.get(0, 0), data_counts.get(1, 0)], labels=["No RainFall", "RainFall"], autopct='%.0f%%')
plt.title("Target value distribution")
plt.show()


fig, axes = plt.subplots(5, 3, figsize=(10, 10), dpi=500, constrained_layout=True)
plt.suptitle("Features distribution plots", fontsize=16, y=1.03)
for ax, feature in zip(axes.flat, num_features):
    sns.histplot(dataset, x=feature, kde=True, ax=ax)
    ax.set_title(feature)
fig.show()


data = dataset.copy()
fig, axes = plt.subplots(5, 3, figsize=(10, 10), dpi=500, constrained_layout=True)
plt.suptitle("Relationship between features and target feature", fontsize=16, y=1.03)
for ax, feature in zip(axes.flat, num_features):
    sns.violinplot(data, x="rainfall", y=feature, ax=ax, inner='quartile')
    ax.set_title(feature)
    ax.set_xlabel("Rainfall (0/1)", fontsize=10)
    ax.set_ylabel(feature, fontsize=10)
fig.show()


fig = plt.figure(figsize=(12,10))
sns.heatmap(dataset.corr(), annot=True, cmap="RdYlGn")


def day_to_datetime(df):
    data = df.copy()
    data['is_new_year'] = (df['day'] == 1) & (df['day'].shift() == 365)
    data['year'] = 2015 + data['is_new_year'].cumsum()

    df['datetime'] = pd.to_datetime(
        data['year'].astype(str) + '-' + df['day'].astype(str), 
        format='%Y-%j'
    )
    return df
    
dataset = day_to_datetime(dataset)


data = dataset.copy()
fig, axes = plt.subplots(5, 3, figsize=(10, 10), dpi=500, constrained_layout=True)
plt.suptitle("Relationship between features and datetime", fontsize=16, y=1.03)
for ax, feature in zip(axes.flat, num_features):
    sns.lineplot(data, x="datetime", y=feature, ax=ax)
    ax.set_title(feature)
fig.show()


data = dataset.copy().groupby(dataset['datetime'].dt.weekday).mean()
fig, axes = plt.subplots(5, 3, figsize=(10, 10), dpi=500, constrained_layout=True)
plt.suptitle("Relationship between features and week day", fontsize=16, y=1.03)
for ax, feature in zip(axes.flat, num_features):
    sns.lineplot(data, x=data.index, y=feature, ax=ax)
    ax.set_title(feature)
fig.show()


data = dataset.copy().groupby(dataset['datetime'].dt.month).mean()
fig, axes = plt.subplots(5, 3, figsize=(10, 10), dpi=500, constrained_layout=True)
plt.suptitle("Relationship between features and month", fontsize=16, y=1.03)
for ax, feature in zip(axes.flat, num_features):
    sns.lineplot(data, x=data.index, y=feature, ax=ax)
    ax.set_title(feature)
fig.show()


data = dataset.copy().groupby(dataset['datetime'].dt.dayofyear).mean()
fig, axes = plt.subplots(5, 3, figsize=(10, 10), dpi=500, constrained_layout=True)
plt.suptitle("Relationship between features and day of the year", fontsize=16, y=1.03)
for ax, feature in zip(axes.flat, num_features):
    sns.lineplot(data, x=data.index, y=feature, ax=ax)
    ax.set_title(feature)
fig.show()


def feature_engeneering(df):
    df = df.copy()
    df = day_to_datetime(df)
    df['season'] = (df.datetime.dt.quarter-1).astype(int)
    df['month'] = (df.datetime.dt.month-1).astype(int)

    df['pressure_change'] = df['pressure'].diff().fillna(0)
    df['pressure_change_speed'] = df['pressure_change'].diff().fillna(0)
    df['temp_change'] = df['temparature'].diff().fillna(0)
    df['temp_range'] = df["maxtemp"]-df["mintemp"]
    df['cloud_sunshine_ratio'] = df['cloud'] / df['sunshine'].clip(lower=0.1)
    df['dewpoint_depression'] = df['temparature'] - df['dewpoint']
    df['dewpoint_humidity_interaction'] = df['dewpoint'] * df['humidity']
    df['temp_trend_3d'] = df['temparature'].diff(3).fillna(0)
    df['cloud_trend_3d'] = df['cloud'].diff(3).fillna(0)
    df['humidity_tredn_3d'] = df["humidity"].diff(3).fillna(0)
    df['cloud_sunshine_interaction'] = df['cloud'] * df['sunshine']
    
    for window in [3, 7, 14]:
        df[f'temparature_rolling_{window}d'] = df['temparature'].rolling(window=window, min_periods=1).mean()
        df[f'humidity_rolling_{window}d'] = df['humidity'].rolling(window=window, min_periods=1).mean()
        df[f'cloud_rolling_{window}d'] = df['cloud'].rolling(window=window, min_periods=1).mean()
        
    df['heat_index'] = 0.5 * df['temparature'] + 0.5 * df['dewpoint'] + 0.1 * df['humidity']
    df['wind_vectors'] = (df['windspeed'] * 
                         np.sin(np.radians(df['winddirection'])) + 
                         df['windspeed'] * np.cos(np.radians(df['winddirection'])))
    
    df['pressure_tendency_24h'] = df['pressure'].diff(periods=2)
    df['pressure_oscillation'] = (df['pressure'] - df['pressure'].rolling(3).mean()).abs()
    
    df['temp_dewpoint_convergence'] = (df['temparature'].diff(3) - 
                                      df['dewpoint'].diff(3)).rolling(3).mean()
    
    df['equivalent_potential_temp'] = df['temparature'] + 2.5 * df['dewpoint']
    
    df['storm_index'] = (df['humidity']/100 * 
                        (df['temparature'] - df['mintemp']) * 
                        df['cloud_rolling_3d']/100)
    for window in [7, 14]:
        df[f'temp_std_{window}d'] = df['temparature'].rolling(window=window, min_periods=4).std().fillna(0)
        df[f'pressure_std_{window}d'] = df['pressure'].rolling(window=window, min_periods=4).std().fillna(0)
        df[f'humidity_std_{window}d'] = df['humidity'].rolling(window=window, min_periods=4).std().fillna(0)

    df = df.drop('datetime', axis=1)
    return df

fe_transformer = FunctionTransformer(feature_engeneering)
new_num_features = [
    "pressure_change", "pressure_change_speed", "temp_change", 
    "temp_range", "cloud_sunshine_ratio", "dewpoint_depression",
    "dewpoint_humidity_interaction", "temp_trend_3d", "cloud_sunshine_interaction",
    "temparature_rolling_3d", "humidity_rolling_3d", "cloud_rolling_3d",
    "temparature_rolling_7d", "humidity_rolling_7d", "cloud_rolling_7d",
    "temparature_rolling_14d", "humidity_rolling_14d", "cloud_rolling_14d",
    "heat_index", "wind_vectors", "pressure_tendency_24h", "storm_index",
    "pressure_oscillation", "temp_dewpoint_convergence", "equivalent_potential_temp",
    "temp_std_7d", "humidity_std_7d", "pressure_std_7d",
    "temp_std_14d", "humidity_std_14d", "pressure_std_14d",
]


numerical_features = num_features.drop(["rainfall", "id"])
numerical_features = numerical_features.join(pd.Index(new_num_features), how="outer")
num_features_pipeline = Pipeline([
    ('numerical imputer', SimpleImputer(strategy="mean")),
    ('mm scaller', StandardScaler()),
])
X = dataset.drop(["rainfall", "id"], axis=1)
y = dataset["rainfall"]


X_processed_for_fs = fe_transformer.fit_transform(X, y)


def feature_importance(model_type, X_train, y_train, plot=True):
    selector = None
    if model_type == "xgb":
        selector = XGBClassifier(
            objective="binary:logistic", tree_method="hist", 
            eval_metric="logloss", verbosity=0, n_jobs=-1
        )
    elif model_type == "lgbm":
        selector = LGBMClassifier(
            objective="binary", metric="logloss",
            boosting_type="gbdt", device="cpu", verbose=-1
        )
    elif model_type == "cat":
        selector = CatBoostClassifier(
            grow_policy='Depthwise',  bootstrap_type='Bayesian', od_type="Iter",
            eval_metric='AUC', loss_function="Logloss", task_type="CPU"
        )
    else:
        raise ValueError(f"Not correct model type. You use {model_type}, but available only xgb, lgbm, cat")

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = []
    feature_importances_list = []
    
    for train_idx, val_idx in kfold.split(X_train):
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
        if "lgbm" not in model_type:
            selector.fit(X_train_fold, y_train_fold, verbose=False)
        else:
            selector.fit(X_train_fold, y_train_fold)
            
        
        y_pred = selector.predict_proba(X_val_fold)[:,1]
        auc_scores.append(roc_auc_score(y_val_fold, y_pred))
        feature_importances_list.append(selector.feature_importances_)

    auc = np.mean(auc_scores)
    feature_importances = np.mean(feature_importances_list, axis=0)
    feature_importance_df = pd.DataFrame({"Feature": X_processed_for_fs.columns, "Importance Score": feature_importances})
    top_10_features_df = feature_importance_df.sort_values(by="Importance Score", ascending=False).iloc[:10]
    top_10_feature_names = top_10_features_df["Feature"]
    top_10_feature_importance_score = top_10_features_df["Importance Score"]
    if plot:
        plt.figure(figsize=(10, 6))
        sns.barplot(x="Importance Score", y="Feature", data=top_10_features_df)
        for inx, value in enumerate(top_10_feature_importance_score):
            plt.text(value + 0.005, inx, f'{value:.3f}', fontsize=12, va='center')
        plt.title(f"Top 10 of {len(X_train.columns)} Feature Importances with ROC AUC score {auc:.2%}")
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        plt.show()
    return auc, feature_importances, top_10_feature_names.to_list(), top_10_feature_importance_score


_, _, top_10_feature_names_xgb, _ = feature_importance("xgb", X_processed_for_fs, y)
_, _, top_10_feature_names_lgbm, _ = feature_importance("lgbm", X_processed_for_fs, y)
_, _, top_10_feature_names_cat, _ = feature_importance("cat", X_processed_for_fs, y)


fs_result_features = list(set(top_10_feature_names_xgb+top_10_feature_names_lgbm+top_10_feature_names_cat))
print("Final features count:", len(fs_result_features))
print("Final features:", end=" ")
print(*fs_result_features, sep=", ")

fs_data_pipeline = Pipeline([
    ("fe transformer", fe_transformer),
    ("fs transformer", FunctionTransformer(lambda x: x[fs_result_features])),
    ("data pipeline", num_features_pipeline)
])
X_processed = fs_data_pipeline.fit_transform(X, y)


X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


if SUBMISSION:
    lr = LogisticRegression(max_iter=1010)
    lr_scores = cross_val_score(lr, X_processed, y, cv=TimeSeriesSplit(n_splits=5), scoring='roc_auc')
    lr.fit(X_processed, y)
    print(f"Model train performance: {lr_scores.mean():.2%}")
else:
    lr = LogisticRegression(max_iter=1010)
    lr_scores = cross_val_score(lr, X_train, y_train, cv=TimeSeriesSplit(n_splits=5), scoring='roc_auc')
    lr.fit(X_train, y_train)
    lr_y_pred = lr.predict_proba(X_test)[:, 1]
    print(f"Model train performance: {lr_scores.mean():.2%}")
    print(f"Model test performance: {roc_auc_score(y_test, lr_y_pred):.2%}")


if SUBMISSION:
    svc = SVC(kernel="linear", probability=True)
    svc_scores = cross_val_score(svc, X_processed, y, cv=TimeSeriesSplit(n_splits=5), scoring='roc_auc')
    svc.fit(X_processed, y)
    print(f"Model train performance: {svc_scores.mean():.2%}")
else:
    svc = SVC(kernel="linear", probability=True)
    svc_scores = cross_val_score(svc, X_train, y_train, cv=TimeSeriesSplit(n_splits=5), scoring='roc_auc')
    svc.fit(X_train, y_train)
    svc_y_pred = svc.predict_proba(X_test)[:, 1]
    print(f"Model train performance: {svc_scores.mean():.2%}")
    print(f"Model test performance: {roc_auc_score(y_test, svc_y_pred):.2%}")


if SUBMISSION:
    rf = RandomForestClassifier()
    rf_scores = cross_val_score(rf, X_processed, y, cv=TimeSeriesSplit(n_splits=5), scoring='roc_auc')
    rf.fit(X_processed, y)
    print(f"Model train performance: {rf_scores.mean():.2%}")
else:
    rf = RandomForestClassifier()
    rf_scores = cross_val_score(rf, X_train, y_train, cv=TimeSeriesSplit(n_splits=5), scoring='roc_auc')
    rf.fit(X_train, y_train)
    rf_y_pred = rf.predict_proba(X_test)[:, 1]
    print(f"Model train performance: {rf_scores.mean():.2%}")
    print(f"Model test performance: {roc_auc_score(y_test, rf_y_pred):.2%}")


if SUBMISSION:
    xgb = XGBClassifier(
        objective="binary:logistic", tree_method="hist", 
        eval_metric="logloss", verbosity=0, n_jobs=-1
    )
    xgb_scores = cross_val_score(xgb, X_processed, y, cv=TimeSeriesSplit(n_splits=5), scoring='roc_auc')
    xgb.fit(X_processed, y)
    print(f"Model train performance: {xgb_scores.mean():.2%}")
else:
    xgb = XGBClassifier(
        objective="binary:logistic", tree_method="hist", 
        eval_metric="logloss", verbosity=0, n_jobs=-1
    )
    xgb_scores = cross_val_score(xgb, X_train, y_train, cv=TimeSeriesSplit(n_splits=5), scoring='roc_auc')
    xgb.fit(X_train, y_train)
    xgb_y_pred = xgb.predict_proba(X_test)[:, 1]
    print(f"Model train performance: {xgb_scores.mean():.2%}")
    print(f"Model test performance: {roc_auc_score(y_test, xgb_y_pred):.2%}")


if SUBMISSION:
    lrt = LogisticRegression(max_iter=1010, C=1.0, penalty='l1', solver='liblinear')
    lrt_scores = cross_val_score(lrt, X_processed, y, cv=TimeSeriesSplit(n_splits=5), scoring='roc_auc')
    lrt.fit(X_processed, y)
    print(f"Model train performance: {lrt_scores.mean():.2%}")
else:
    lrt = LogisticRegression(max_iter=1010, C=1.0, penalty='l1', solver='liblinear')
    lrt_scores = cross_val_score(lrt, X_train, y_train, cv=TimeSeriesSplit(n_splits=5), scoring='roc_auc')
    lrt.fit(X_train, y_train)
    lrt_y_pred = lrt.predict_proba(X_test)[:, 1]
    print(f"Model train performance: {lrt_scores.mean():.2%}")
    print(f"Model test performance: {roc_auc_score(y_test, lrt_y_pred):.2%}")


if SUBMISSION:
    svct = SVC(kernel="linear", C=0.1, gamma=0.001, probability=True)
    svct_scores = cross_val_score(svct, X_processed, y, cv=TimeSeriesSplit(n_splits=5), scoring='roc_auc')
    svct.fit(X_processed, y)
    print(f"Model train performance: {svct_scores.mean():.2%}")
else:
    svct = SVC(kernel="linear", C=0.1, gamma=0.001, probability=True)
    svct_scores = cross_val_score(svct, X_train, y_train, cv=TimeSeriesSplit(n_splits=5), scoring='roc_auc')
    svct.fit(X_train, y_train)
    svct_y_pred = svct.predict_proba(X_test)[:, 1]
    print(f"Model train performance: {svct_scores.mean():.2%}")
    print(f"Model test performance: {roc_auc_score(y_test, svct_y_pred):.2%}")


if SUBMISSION:
    rft = RandomForestClassifier(
        max_depth=None,
        min_samples_leaf=3,
        min_samples_split=2,
        n_estimators=200,
        random_state=42
    )
    rft_scores = cross_val_score(rft, X_processed, y, cv=TimeSeriesSplit(n_splits=5), scoring='roc_auc')
    rft.fit(X_processed, y)
    print(f"Model train performance: {rft_scores.mean():.2%}")
else:
    rft = RandomForestClassifier(
        max_depth=None,
        min_samples_leaf=3,
        min_samples_split=2,
        n_estimators=200,
        random_state=42
    )
    rft_scores = cross_val_score(rft, X_train, y_train, cv=TimeSeriesSplit(n_splits=5), scoring='roc_auc')
    rft.fit(X_train, y_train)
    rft_y_pred = rft.predict_proba(X_test)[:, 1]
    print(f"Model train performance: {rft_scores.mean():.2%}")
    print(f"Model test performance: {roc_auc_score(y_test, rft_y_pred):.2%}")


if SUBMISSION:
    xgbt = XGBClassifier(
        subsample=0.7, scale_pos_weight=3, reg_lambda=0.5, reg_alpha=1, 
        n_estimators=800, max_depth=2, learning_rate=0.01, gamma=2, 
        colsample_bytree=0.5
    )
    
    xgbt_scores = cross_val_score(svct, X_processed, y, cv=TimeSeriesSplit(n_splits=5), scoring='roc_auc')
    xgbt.fit(X_processed, y, verbose=False)
    print(f"Model train performance: {xgbt_scores.mean():.2%}")
else:
    xgbt = XGBClassifier(
        subsample=0.7, scale_pos_weight=3, reg_lambda=0.5, reg_alpha=1, 
        n_estimators=800, max_depth=2, learning_rate=0.01, gamma=2, 
        colsample_bytree=0.5, eval_metric='auc'
    )
    
    xgbt.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=200)


voting_estimators = [
    ("logistic regression", LogisticRegression(max_iter=1010)),
    ("svc", SVC(kernel="linear", probability=True)),
    ("xgb clf", 
     XGBClassifier(
        subsample=0.7, scale_pos_weight=3, reg_lambda=0.5, reg_alpha=1, 
        n_estimators=800, max_depth=2, learning_rate=0.01, gamma=2, 
        colsample_bytree=0.5, eval_metric='auc'
     )
    )
]
if SUBMISSION:
    voting_clf = VotingClassifier(voting_estimators, n_jobs=-1, voting='soft')
    voting_clf_scores = cross_val_score(voting_clf, X_processed, y, cv=TimeSeriesSplit(n_splits=5), scoring='roc_auc')
    voting_clf.fit(X_processed, y)
    print(f"Model train performance: {voting_clf_scores.mean():.2%}")
else:
    voting_clf = VotingClassifier(voting_estimators, n_jobs=-1, voting='soft')
    voting_clf_scores = cross_val_score(voting_clf, X_train, y_train, cv=TimeSeriesSplit(n_splits=5), scoring='roc_auc')
    voting_clf.fit(X_train, y_train)
    voting_clf_y_pred = voting_clf.predict_proba(X_test)[:, 1]
    print(f"Model train performance: {voting_clf_scores.mean():.2%}")
    print(f"Model test performance: {roc_auc_score(y_test, voting_clf_y_pred):.2%}")


stacking_estimators = [
    ("logistic regression", LogisticRegression(max_iter=1010)),
    ("svc", SVC(kernel="linear", probability=True)),
    ("xgb clf", 
     XGBClassifier(
        subsample=0.7, scale_pos_weight=3, reg_lambda=0.5, reg_alpha=1, 
        n_estimators=800, max_depth=2, learning_rate=0.01, gamma=2, 
        colsample_bytree=0.5, eval_metric='auc'
     )
    )
]

meta_model = Pipeline([
    ('scaler', StandardScaler()),
    ('calibration', CalibratedClassifierCV(
        MLPClassifier(hidden_layer_sizes=(20,10), max_iter=500),
        method='isotonic',
        cv=TimeSeriesSplit(n_splits=5)
    ))
])
if SUBMISSION:
    stacking_clf = StackingClassifier(stacking_estimators, meta_model)
    stacking_clf_scores = cross_val_score(voting_clf, X_processed, y, cv=TimeSeriesSplit(n_splits=5), scoring='roc_auc')
    stacking_clf.fit(X_processed, y)
    print(f"Model best score: {stacking_clf_scores.mean():.2%}")
else:
    stacking_clf = StackingClassifier(stacking_estimators, meta_model)
    stacking_clf_scores = cross_val_score(voting_clf, X_train, y_train, cv=TimeSeriesSplit(n_splits=5), scoring='roc_auc')
    stacking_clf.fit(X_train, y_train)
    stacking_clf_y_pred = stacking_clf.predict_proba(X_test)[:, 1]
    
    print(f"Model best score: {stacking_clf_scores.mean():.2%}")
    print(f"Model test performance: {roc_auc_score(y_test, stacking_clf_y_pred):.2%}")


test_dataset = pd.read_csv(TEST_DATASET_PATH)
X_submission = fs_data_pipeline.transform(test_dataset.drop(["id"], axis=1))


stacking_clf_preds = stacking_clf.predict_proba(X_submission)[:, 1]
pd.DataFrame({"id": test_dataset["id"], "rainfall": stacking_clf_preds}).to_csv("stacking_clf_submission.csv", index=False)


svc_preds = svc.predict_proba(X_submission)[:, 1]
pd.DataFrame({"id": test_dataset["id"], "rainfall": svc_preds}).to_csv("svc_submission.csv", index=False)


lr_preds = lr.predict_proba(X_submission)[:, 1]
pd.DataFrame({"id": test_dataset["id"], "rainfall": lr_preds}).to_csv("lr_submission.csv", index=False)

