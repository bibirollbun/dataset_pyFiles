pip install powershap -qq


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew, kurtosis
import contextlib
import io
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import TwoSlopeNorm
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import PolynomialFeatures
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.linear_model import LogisticRegression
import itertools
import shap
from powershap import PowerShap
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import PartialDependenceDisplay
import warnings


warnings.filterwarnings("ignore")


data = pd.read_parquet('../input/amex-data-integer-dtypes-parquet-format/train.parquet')
target = pd.read_csv('../input/amex-default-prediction/train_labels.csv')
data = data.groupby("customer_ID").last()
data = data.merge(target, on = 'customer_ID', how = 'left')
del target


data


# 獲取X和y
def get_X_and_y(df): return df.drop(columns=['target', 'customer_ID', 'S_2']), df['target']

# 計算違義率
def caculate_default_rate(df): print(f"Default Rate: {df.mean():.4%}")

# 拆分訓練集與測試集
def separate_train_and_test(X, y): return train_test_split(X, y, test_size=0.2, random_state=42)

# one-hot編碼
def one_hot_encode(df):
    for feature in df.select_dtypes(include=["object"]):
        df[feature] = df[feature].fillna("miss")
        for value in df[feature].unique(): df[f'{feature}_{value}'] = (df[feature] == value).astype(int)
        df.drop(columns=[feature], inplace=True)
    return df

# 繪製PCA
def plot_pca(X_train, X_test, y, filename):
    pca = PCA(n_components=2).fit_transform(pd.concat([X_train, X_test], axis=0).fillna(pd.concat([X_train, X_test], axis=0).mean()).reset_index(drop=True))
    df_pca = pd.DataFrame(pca, columns=['PC1', 'PC2']).assign(TARGET=y.loc[X_train.index.union(X_test.index)].reset_index(drop=True))
    plt.figure(figsize=(10, 6))
    for t, c in zip([0, 1], ['blue', 'red']):
        plt.scatter(df_pca.loc[df_pca['TARGET'] == t, 'PC1'], df_pca.loc[df_pca['TARGET'] == t, 'PC2'], color=c, alpha=0.5, label=f'TARGET={t}')
    plt.xlabel('PC1'), plt.ylabel('PC2'), plt.grid(), plt.legend(title='TARGET', loc='upper right')
    plt.savefig(filename, transparent=True, dpi=300), plt.show()

# log轉換
def log_transformate(df):
    for feature in df.select_dtypes(include=['number']).columns[~df.isin([0, 1]).all()]:
        skewed = skew(df[feature].dropna())
        if skewed > 1:
            min_val = df[feature].min(skipna=True)
            if min_val > 1:
                df[feature] = np.log(df[feature]).where(df[feature] > 0, np.nan)
            elif 0 < min_val <= 1:
                df[feature] = np.log(df[feature] + 1).where(df[feature] + 1 > 0, np.nan)
            else:
                df[feature] = np.log(df[feature] + abs(min_val) + 1).where(df[feature] + abs(min_val) + 1 > 0, np.nan)
    return df

def standardscaler(df):
    for feature in df.select_dtypes(include=['number']).columns[~df.isin([0, 1]).all()]:
        mean = df[feature].mean(skipna=True)
        std = df[feature].std(skipna=True)
        if pd.notna(std) and std > 1e-8:
            df[feature] = (df[feature] - mean) / std
        else:
            df[feature] = 0
    return df

# 填充缺失值
def missing_value_impute(df):
    for feature in df.select_dtypes(include=['number']).columns[~df.isin([0, 1]).all()]:
        df[f"{feature}_missing"] = df[feature].isnull().astype(int)
        df[feature].fillna(0, inplace=True)
    return df

# 對齊特徵數
def align_features(X_train, X_test):
    for col in set(X_train.columns) - set(X_test.columns): X_test[col] = 0
    for col in set(X_test.columns) - set(X_train.columns): X_train[col] = 0
    return X_train[sorted(X_train.columns)], X_test[sorted(X_test.columns)]


X, y = get_X_and_y(data)


caculate_default_rate(y)


X_train, X_test, y_train, y_test = separate_train_and_test(X, y)


X_train = one_hot_encode(X_train)
X_test = one_hot_encode(X_test)


plot_pca(X_train, X_test, y, 'original_pca')


X_train = log_transformate(X_train)
X_test = log_transformate(X_test)


X_train = standardscaler(X_train)
X_test = standardscaler(X_test)


X_train = missing_value_impute(X_train)
X_test = missing_value_impute(X_test)


X_train, X_test = align_features(X_train, X_test)


plot_pca(X_train, X_test, y, 'preprocessed_pca')


# 比較不同前處理步驟的LR結果
def compare_preprocessing_strategies(X, y):
    def apply_pipeline(X, steps):
        X = X.copy()
        for step in steps:
            X = step(X)
        return X

    strategies = [
        ("one-hot + miss",               [one_hot_encode, missing_value_impute]),
        ("one-hot + log + miss",     [one_hot_encode, log_transformate, missing_value_impute]),
        ("one-hot + Z-std + miss",     [one_hot_encode, standardscaler, missing_value_impute]),
        ("one-hot + log + Z-std + miss", [one_hot_encode, log_transformate, standardscaler, missing_value_impute]),
        ("one-hot + Z-std + log + miss", [one_hot_encode, standardscaler, log_transformate, missing_value_impute]),
    ]

    results = []

    for name, pipeline in strategies:
        # 切分資料
        X_train_raw, X_test_raw, y_train, y_test = separate_train_and_test(X, y)

        # 分別應用前處理流程
        X_train = apply_pipeline(X_train_raw, pipeline)
        X_test = apply_pipeline(X_test_raw, pipeline)

        # 對齊特徵
        X_train, X_test = align_features(X_train, X_test)

        # 建立模型並訓練
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        # 計算 AUC
        y_train_pred = model.predict_proba(X_train)[:, 1]
        y_test_pred = model.predict_proba(X_test)[:, 1]
        train_auc = roc_auc_score(y_train, y_train_pred)
        test_auc = roc_auc_score(y_test, y_test_pred)

        results.append({
            "preprocess": name,
            "train auc": train_auc,
            "test auc": test_auc
        })

    return pd.DataFrame(results)

def plot_preprocessed_roc_and_auc_df(X_train, y_train, X_test, y_test):
    models = {
        "xgb": lambda: XGBClassifier(random_state=42),
        "lr": lambda: LogisticRegression(max_iter=1000),
        "rf": lambda: RandomForestClassifier(n_estimators=100, random_state=42),
        "nn": lambda: MLPClassifier(hidden_layer_sizes=(32,), activation='relu', solver='adam', max_iter=500, random_state=42)
    }

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    mean_fpr = np.linspace(0, 1, 100)

    # 改為存儲四種 AUC
    auc_scores = {
        "cv_train": {},
        "cv_validation": {},
        "full_train": {},
        "test": {}
    }

    for model_name, model_fn in models.items():
        train_aucs = []
        val_aucs = []
        plt.figure(figsize=(8, 6))

        for train_idx, val_idx in kf.split(X_train):
            X_kf_train, X_kf_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_kf_train, y_kf_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            model = model_fn()
            model.fit(X_kf_train, y_kf_train)

            # 計算 train AUC for current fold
            y_train_pred = model.predict_proba(X_kf_train)[:, 1]
            train_auc = roc_auc_score(y_kf_train, y_train_pred)
            train_aucs.append(train_auc)

            # 計算 validation AUC for current fold
            y_val_pred = model.predict_proba(X_kf_val)[:, 1]
            fpr, tpr, _ = roc_curve(y_kf_val, y_val_pred)
            val_auc = auc(fpr, tpr)
            val_aucs.append(val_auc)

            tpr_interp = np.interp(mean_fpr, fpr, tpr)
            tpr_interp[0] = 0.0
            plt.plot(mean_fpr, tpr_interp, alpha=0.3, label=f'{model_name} Fold {len(val_aucs)}')

        # 儲存交叉驗證結果
        auc_scores["cv_train"][model_name] = np.mean(train_aucs)
        auc_scores["cv_validation"][model_name] = np.mean(val_aucs)

        # 繪製平均 ROC 曲線
        tprs = []
        for train_idx, val_idx in kf.split(X_train):
            model = model_fn()
            model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
            y_val_score = model.predict_proba(X_train.iloc[val_idx])[:, 1]
            fpr, tpr, _ = roc_curve(y_train.iloc[val_idx], y_val_score)
            tpr_interp = np.interp(mean_fpr, fpr, tpr)
            tpr_interp[0] = 0.0
            tprs.append(tpr_interp)
        
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        
        mean_auc = auc(mean_fpr, mean_tpr)
        plt.plot(mean_fpr, mean_tpr, label=f'Mean ROC (AUC = {mean_auc:.4f})')

        # 整體 train fit（不分折）
        model = model_fn()
        model.fit(X_train, y_train)
        y_full_train_pred = model.predict_proba(X_train)[:, 1]
        full_train_auc = roc_auc_score(y_train, y_full_train_pred)
        auc_scores["full_train"][model_name] = full_train_auc

        # Test AUC
        y_test_pred = model.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, y_test_pred)
        auc_scores["test"][model_name] = test_auc

        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')
        plt.savefig(f'{model_name}_roc_curve.png', transparent=True, dpi=300)
        plt.show()

    # 結果轉成 DataFrame
    proprocessed_auc_df = pd.DataFrame(auc_scores).reset_index()
    proprocessed_auc_df.rename(columns={"index": "model"}, inplace=True)
    return proprocessed_auc_df

def plot_preprocessed_roc_and_auc_df(X_train, y_train, X_test, y_test):
    models = {
        "xgb": lambda: XGBClassifier(random_state=42),
        "lr": lambda: LogisticRegression(max_iter=1000),
        "rf": lambda: RandomForestClassifier(n_estimators=100, random_state=42),
        "nn": lambda: MLPClassifier(hidden_layer_sizes=(32,), activation='relu', solver='adam', max_iter=500, random_state=42)
    }

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    mean_fpr = np.linspace(0, 1, 100)

    # 改為存儲四種 AUC
    auc_scores = {
        "cv_train": {},
        "cv_validation": {},
        "full_train": {},
        "test": {}
    }

    for model_name, model_fn in models.items():
        train_aucs = []
        val_aucs = []
        plt.figure(figsize=(8, 6))

        for train_idx, val_idx in kf.split(X_train):
            X_kf_train, X_kf_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_kf_train, y_kf_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            model = model_fn()
            model.fit(X_kf_train, y_kf_train)

            # 計算 train AUC for current fold
            y_train_pred = model.predict_proba(X_kf_train)[:, 1]
            train_auc = roc_auc_score(y_kf_train, y_train_pred)
            train_aucs.append(train_auc)

            # 計算 validation AUC for current fold
            y_val_pred = model.predict_proba(X_kf_val)[:, 1]
            fpr, tpr, _ = roc_curve(y_kf_val, y_val_pred)
            val_auc = auc(fpr, tpr)
            val_aucs.append(val_auc)

            tpr_interp = np.interp(mean_fpr, fpr, tpr)
            tpr_interp[0] = 0.0
            plt.plot(mean_fpr, tpr_interp, alpha=0.3, label=f'{model_name} Fold {len(val_aucs)}')

        # 儲存交叉驗證結果
        auc_scores["cv_train"][model_name] = np.mean(train_aucs)
        auc_scores["cv_validation"][model_name] = np.mean(val_aucs)

        # 繪製平均 ROC 曲線
        tprs = []
        for train_idx, val_idx in kf.split(X_train):
            model = model_fn()
            model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
            y_val_score = model.predict_proba(X_train.iloc[val_idx])[:, 1]
            fpr, tpr, _ = roc_curve(y_train.iloc[val_idx], y_val_score)
            tpr_interp = np.interp(mean_fpr, fpr, tpr)
            tpr_interp[0] = 0.0
            tprs.append(tpr_interp)
        
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        
        mean_auc = auc(mean_fpr, mean_tpr)
        plt.plot(mean_fpr, mean_tpr, label=f'Mean ROC (AUC = {mean_auc:.4f})')

        # 整體 train fit（不分折）
        model = model_fn()
        model.fit(X_train, y_train)
        y_full_train_pred = model.predict_proba(X_train)[:, 1]
        full_train_auc = roc_auc_score(y_train, y_full_train_pred)
        auc_scores["full_train"][model_name] = full_train_auc

        # Test AUC
        y_test_pred = model.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, y_test_pred)
        auc_scores["test"][model_name] = test_auc

        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')
        plt.savefig(f'{model_name}_roc_curve.png', transparent=True, dpi=300)
        plt.show()

    # 結果轉成 DataFrame
    proprocessed_auc_df = pd.DataFrame(auc_scores).reset_index()
    proprocessed_auc_df.rename(columns={"index": "model"}, inplace=True)
    return proprocessed_auc_df


# compare_preprocessing_df = compare_preprocessing_strategies(X, y)
# compare_preprocessing_df.to_csv("compare_preprocessing_df.csv", index=False)


# proprocessed_auc_df = plot_preprocessed_roc_and_auc_df(X_train, y_train, X_test, y_test)
# proprocessed_auc_df.to_csv("proprocessed_auc_df.csv", index=False)


# 四種方法的特徵選擇
def select_features(X_train, y_train):
    def get_feature_importance(model, X, method='coef'):
        if method == 'coef':
            importance = np.abs(model.coef_).flatten()
        elif method == 'shap':
            explainer = shap.Explainer(model, X)
            importance = np.abs(explainer(X).values).mean(axis=0)
        else:
            importance = model.feature_importances_
        return pd.DataFrame({'feature': X.columns, 'importance': importance}).sort_values(by='importance', ascending=False)
    def generate_feature_groups(importance_df):
        return {i: importance_df.head(i)['feature'].tolist() for i in range(2, len(importance_df) + 1)}
    xgb_model = XGBClassifier(random_state=42)
    xgb_model.fit(X_train, y_train)
    xgb_importance_df = get_feature_importance(xgb_model, X_train, method='xgb')
    feature_groups_xgb = generate_feature_groups(xgb_importance_df)
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    lr_importance_df = get_feature_importance(lr_model, X_train, method='coef')
    feature_groups_lr = generate_feature_groups(lr_importance_df)
    shap_xgb_importance_df = get_feature_importance(xgb_model, X_train, method='shap')
    feature_groups_shap_xgb = generate_feature_groups(shap_xgb_importance_df)
    shap_lr_importance_df = get_feature_importance(lr_model, X_train, method='shap')
    feature_groups_shap_lr = generate_feature_groups(shap_lr_importance_df)
    return {
        'xgb': feature_groups_xgb,
        'lr': feature_groups_lr,
        'shap_xgb': feature_groups_shap_xgb,
        'shap_lr': feature_groups_shap_lr
    }

# 創建 only selection dataframe
def make_only_selection_df(select_result, X_train, X_test, y_train, y_test):
    def train_and_evaluate(model, feature_subset):
        X_train_selected = X_train[feature_subset]
        X_test_selected = X_test[feature_subset]
        model.fit(X_train_selected, y_train)
        y_pred = model.predict_proba(X_test_selected)[:, 1]
        return roc_auc_score(y_test, y_pred)
    results_lr = {key: [] for key in select_result.keys()}
    results_xgb = {key: [] for key in select_result.keys()}
    feature_counts = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]
    for num_features in feature_counts:
        for key, feature_dict in select_result.items():
            selected_features = feature_dict[num_features]
            lr_model = LogisticRegression(max_iter=1000, random_state=42)
            xgb_model = XGBClassifier(random_state=42)
            results_lr[key].append(train_and_evaluate(lr_model, selected_features))
            results_xgb[key].append(train_and_evaluate(xgb_model, selected_features))
    df_results_lr = pd.DataFrame(results_lr, index=feature_counts)
    df_results_xgb = pd.DataFrame(results_xgb, index=feature_counts)
    return df_results_lr, df_results_xgb

# 繪製 only selection auc
def plot_only_selection_auc(lr_df, xgb_df):
    plt.figure(figsize=(18, 18))
    x = lr_df.index
    methods = {
        "xgb": {"color": "blue", "marker": "o"},
        "lr": {"color": "red", "marker": "s"},
        "shap_xgb": {"color": "green", "marker": "D"},
        "shap_lr": {"color": "purple", "marker": "^"},
    }
    for method, style in methods.items():
        plt.plot(x, lr_df[method], label=f'lr: {method}_selection', color=style["color"], marker=style["marker"], linestyle="-")
        plt.plot(x, xgb_df[method], label=f'xgb: {method}_selection', color=style["color"], marker=style["marker"], linestyle="--")
    plt.plot(x, [0.9566] * len(x), label="lr: benchmark", color="black", linewidth=2.5, linestyle="-")
    plt.plot(x, [0.9554] * len(x), label="xgb: benchmark", color="black", linewidth=2.5, linestyle="--")
    plt.xlabel("Number of features", fontsize=20)
    plt.ylabel("AUC", fontsize=20)
    plt.ylim(0.953, 0.958)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    legend = plt.legend(fontsize=18, title="Raw features", title_fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.savefig(f'only_selection_auc.png', transparent=True, dpi=300)
    plt.show()

# 生成多項式特徵
def generate_polynomial_dataset(df, features, degree):
    interactions = pd.DataFrame(index=df.index)
    for d in range(2, degree + 1):
        for combination in itertools.combinations_with_replacement(features, d):
            new_feature_name = "_x_".join(combination)
            interactions[new_feature_name] = df[list(combination)].prod(axis=1)
    return interactions

# 創建 select than interact dataframe
def make_select_than_interact_df(model_type, degree, max_features, select_result, X_train, X_test, y_train, y_test):
    results = {}
    for method, feature_groups in select_result.items():
        for k, features in itertools.islice(feature_groups.items(), max_features):
            interaction_features_train = generate_polynomial_dataset(X_train, features, degree)
            interaction_features_test = generate_polynomial_dataset(X_test, features, degree)
            X_train_augmented = pd.concat([X_train, interaction_features_train], axis=1)
            X_test_augmented = pd.concat([X_test, interaction_features_test], axis=1)
            if model_type == 'xgb':
                model = XGBClassifier(random_state=42)
            elif model_type == 'lr':
                model = LogisticRegression(max_iter=1000, random_state=42)
            model.fit(X_train_augmented, y_train)
            y_pred_prob = model.predict_proba(X_test_augmented)[:, 1]
            roc_auc = roc_auc_score(y_test, y_pred_prob)
            results[f"{method}_{k}"] = roc_auc
            del interaction_features_train, interaction_features_test, X_train_augmented, X_test_augmented
    data = {method: {int(k.split('_')[-1]): v for k, v in results.items() if k.startswith(method + "_")} 
            for method in select_result.keys()}
    df = pd.DataFrame(data)
    return df

# 繪製 select than interact auc
def plot_select_than_interact_auc(lr_df, xgb_df, degree):
    plt.figure(figsize=(18, 18))
    base_x = lr_df.index
    methods = {
        "xgb": {"color": "blue", "marker": "o"},
        "lr": {"color": "red", "marker": "s"},
        "shap_xgb": {"color": "green", "marker": "D"},
        "shap_lr": {"color": "purple", "marker": "^"},
    }
    for method, style in methods.items():
        if degree == 2:    
            x = 481 + base_x * (base_x + 1) / 2
        elif degree == 3:
            x = 481 + (base_x * (base_x + 1) / 2) + (base_x * (base_x + 1) * (base_x + 2) / 6)
        plt.plot(x, lr_df[method], label=f'lr: {method}_selection', color=style["color"], marker=style["marker"], linestyle="-")
        plt.plot(x, xgb_df[method], label=f'xgb: {method}_selection', color=style["color"], marker=style["marker"], linestyle="--")
    plt.plot(x, [0.9566] * len(x), label="lr: benchmark", color="black", linewidth=2.5, linestyle="-")
    plt.plot(x, [0.9554] * len(x), label="xgb: benchmark", color="black", linewidth=2.5, linestyle="--")
    plt.xlabel("Number of features", fontsize=20)
    plt.ylabel("AUC", fontsize=20)
    plt.ylim(0.953, 0.958)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    legend = plt.legend(fontsize=20, title="Raw features", title_fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    if degree == 2:    
        legend = plt.legend(fontsize=20, title="Quadratic augment features", title_fontsize=20)
        plt.savefig(f'quadratic_select_than_interact_auc.png', transparent=True, dpi=300)
    elif degree == 3:
        legend = plt.legend(fontsize=20, title="Cubic augment features", title_fontsize=20)
        plt.savefig(f'cubic_select_than_interact_auc.png', transparent=True, dpi=300)
    plt.show()


# select_result = select_features(X_train, y_train)
# print("前 15 個特徵:", select_result['shap_xgb'][15])
# print("前 30 個特徵:", select_result['shap_xgb'][30])


# df_results_lr, df_results_xgb = make_only_selection_df(select_result, X_train, X_test, y_train, y_test)
# df_results_lr.to_csv("lr_only_selection_df.csv")
# df_results_xgb.to_csv("xgb_only_selection_df.csv")


# plot_only_selection_auc(df_results_lr, df_results_xgb)


# lr_degree2_df = make_select_than_interact_df('lr', 2, 29, select_result, X_train, X_test, y_train, y_test)
# lr_degree2_df.to_csv("lr_degree2_df.csv")
# xgb_degree2_df = make_select_than_interact_df('xgb', 2, 29, select_result, X_train, X_test, y_train, y_test)
# xgb_degree2_df.to_csv("xgb_degree2_df.csv")
# lr_degree3_df = make_select_than_interact_df('lr', 3, 14, select_result, X_train, X_test, y_train, y_test)
# lr_degree3_df.to_csv("lr_degree3_df.csv")
# xgb_degree3_df = make_select_than_interact_df('xgb', 3, 14, select_result, X_train, X_test, y_train, y_test)
# xgb_degree3_df.to_csv("xgb_degree3_df.csv")


# plot_select_than_interact_auc(lr_degree2_df, xgb_degree2_df, 2)
# plot_select_than_interact_auc(lr_degree3_df, xgb_degree3_df, 3)


#製作 lasso 與 powershap 的 dataframe
def make_lasso_and_powershap_df(X_train, y_train, X_test, y_test, degree, features, lambda_values):
    auc_result = []
    if features:
        interaction_train = generate_polynomial_dataset(X_train, features, degree)
        interaction_test = generate_polynomial_dataset(X_test, features, degree)
        X_train = pd.concat([X_train, interaction_train], axis=1)
        X_test = pd.concat([X_test, interaction_test], axis=1)
    for lambda_val in lambda_values:
        model = LogisticRegression(penalty='l1', C=1/lambda_val, solver='liblinear', random_state=42)
        model.fit(X_train, y_train)
        selected = np.where(model.coef_[0] != 0)[0]
        train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
        test_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        auc_result.append({'model': f'lasso (λ={lambda_val})', 'p': len(selected), 'train': train_auc, 'test': test_auc})
    with contextlib.redirect_stdout(io.StringIO()):
        powershap = PowerShap(model=XGBClassifier(random_state=42), power_iterations=10, alpha=0.05, power_req_iterations=1)
        powershap.fit(X_train, y_train)
    selected_cols = X_train.columns[powershap.get_support()]
    X_train = X_train[selected_cols]
    X_test = X_test[selected_cols]
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
    test_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    auc_result.append({'model': f'powershap (α=0.5)', 'p': len(selected_cols), 'train': train_auc, 'test': test_auc})
    return pd.DataFrame(auc_result)
    
#繪製 lasso 和 powershap 的 auc
def plot_lasso_vs_powershap_auc(auc_df, degree):
    lasso_df = auc_df[auc_df['model'].str.contains('lasso')].copy()
    powershap_df = auc_df[auc_df['model'].str.contains('powershap')].copy()
    lasso_df = lasso_df.sort_values(by='p')
    plt.figure(figsize=(10, 6))
    plt.plot(lasso_df['p'], lasso_df['train'], marker='o', linestyle='-', label='LR: lasso (train)', color='blue')
    plt.plot(lasso_df['p'], lasso_df['test'], marker='o', linestyle='--', label='LR: lasso (test)', color='blue')
    plt.scatter(powershap_df['p'], powershap_df['train'], color='red', marker='D', label='LR: powershap (train)')
    plt.scatter(powershap_df['p'], powershap_df['test'], color='red', marker='X', label='LR: powershap (test)')
    plt.xlabel('Number of features')
    plt.ylabel('AUC')
    # plt.ylim(0.72, 0.76)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    if degree == 1:    
        legend = plt.legend(title="Raw features")
        plt.savefig(f'raw_lasso_vs_powershap_auc.png', transparent=True, dpi=300)
    elif degree == 2:    
        legend = plt.legend(title="Quadratic augment features")
        plt.savefig(f'quadratic_lasso_vs_powershap_auc.png', transparent=True, dpi=300)
    elif degree == 3:
        legend = plt.legend(title="Cubic augment features")
        plt.savefig(f'cubic_lasso_vs_powershap_auc.png', transparent=True, dpi=300)
    plt.show()


raw_features = []
raw_lambda = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000]
raw_df = make_lasso_and_powershap_df(X_train, y_train, X_test, y_test, 1, raw_features, raw_lambda)
raw_df.to_csv("raw_lasso_vs_powershap_df.csv")
plot_lasso_vs_powershap_auc(raw_df, 1)


quadratic_features = ['P_2', 'B_1', 'B_3', 'B_4', 'D_43', 'D_45', 'S_3', 'D_39', 'D_48', 'B_7', 'B_5', 'D_46', 'D_112', 'D_47', 'D_42', 'D_129', 'D_62', 'B_2', 'D_121', 'R_1', 'S_8', 'S_5', 'D_41', 'B_16', 'S_26', 'D_49', 'B_28', 'B_17', 'D_52', 'D_59']
quadratic_lambda = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000]
quadratic_df = make_lasso_and_powershap_df(X_train, y_train, X_test, y_test, 2, quadratic_features, quadratic_lambda)
quadratic_df.to_csv("quadratic_lasso_vs_powershap_df.csv")
plot_lasso_vs_powershap_auc(quadratic_df, 2)


cubic_features = ['P_2', 'B_1', 'B_3', 'B_4', 'D_43', 'D_45', 'S_3', 'D_39', 'D_48', 'B_7', 'B_5', 'D_46', 'D_112', 'D_47', 'D_42']
cubic_lambda = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000]
cubic_df = make_lasso_and_powershap_df(X_train, y_train, X_test, y_test, 3, cubic_features, cubic_lambda)
cubic_df.to_csv("cubic_lasso_vs_powershap_df.csv")
plot_lasso_vs_powershap_auc(cubic_df, 3)

