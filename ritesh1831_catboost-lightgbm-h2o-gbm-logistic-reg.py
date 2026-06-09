# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv", index_col='id')
test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv", index_col='id')


train.head(5)


test.head(5)


train.info()


test.info()


train = train.drop_duplicates()



def custom_describe(df):
    df_ = df.select_dtypes(include=np.number)
    des = df_.describe().T
    des['skewness'] = df_.skew()
    des['kurtosis'] = df_.kurtosis()
    des['count'] = des['count'].astype('int')
    return des


features = train.columns.tolist()
print(features)


numerical_features = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']
categorical_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'day', 'month', 'poutcome']
target = 'y'


fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.countplot(x=target, data=train, ax=axes[0])
axes[0].set_title('Distribution of Target Variable (Subscribed)', fontweight='bold', size=20)
axes[0].set_xticks(ticks=[0, 1],labels=['No', 'Yes'])
axes[0].set_xlabel("subscribed")

train[target].value_counts().plot(kind='pie', ax=axes[1], explode=(0.0, 0.1), autopct="%.2f%%", labels=['No', 'Yes'], pctdistance=0.75)
axes[1].add_artist(plt.Circle((0, 0), 0.5, fc='w'))
axes[1].set_title('Pie Chart of Target Variable', fontweight='bold', size=20)
axes[1].set_ylabel("")

plt.tight_layout()
plt.show()


custom_describe(train[numerical_features])



def numerical_features_plot(df, feature, target):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Analysis of {feature}', fontsize=16, fontweight='bold')

    # Boxplot
    sns.boxplot(data=df, x=feature, y=target, hue=target, orient='h', ax=axes[0])
    axes[0].set_title(f'Boxplot of {feature}')
    axes[0].legend_.remove()  # Turn off legend

    # Violinplot
    sns.violinplot(data=df, x=feature, y=target, hue=target, orient='h', ax=axes[1])
    axes[1].set_title(f'Violinplot of {feature}')
    axes[1].legend_.remove()  # Turn off legend

    # Histogram with KDE
    sns.histplot(data=df, x=feature, hue=target, kde=True, ax=axes[2], alpha=0.6)
    axes[2].set_title(f'Distribution of {feature}')

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()


for feature in numerical_features:
    numerical_features_plot(train, feature, target)


def categorical_features_plot(df, feature):
    value_counts = df[feature].value_counts()
    
    top_n = min(10, len(value_counts))
    top_categories = value_counts.nlargest(top_n)
    
    df_plot = df[df[feature].isin(top_categories.index)]

    top_percentages = (top_categories / len(df)) * 100

    plt.figure(figsize=(25, 6))

    plt.subplot(1, 2, 1)
    sns.countplot(df_plot, x=feature, palette=sns.color_palette('viridis'))
    plt.title(f"Count Plot of{(' Top ' + str(top_n)) if len(value_counts) > 10 else ''} Categories of {feature}", size=16, fontweight='bold')

    plt.subplot(1, 2, 2)
    plt.pie(
        top_percentages,
        labels=top_percentages.index,
        autopct=lambda pct: f"{pct:.2f}%",
        pctdistance=0.75
    )
    plt.gca().add_artist(plt.Circle((0, 0), 0.5, fc='w'))  # Donut hole
    plt.title(
        f"{('Top ' + str(top_n)) if len(value_counts) > 10 else ''} {feature} Categories as % of Full Dataset",
        size=16,
        fontweight='bold'
    )
    plt.ylabel("")

    plt.tight_layout()
    plt.show()

    plt.tight_layout()
    plt.show()


for feature in categorical_features:
    if feature == 'day':
        continue
    categorical_features_plot(train, feature)


df = train.copy()

g = sns.FacetGrid(df[df['job'].isin(df['job'].value_counts().head(6).index)], 
                  col='job', col_wrap=3, height=4, aspect=1.2)
g.map_dataframe(sns.boxplot, x='y', y='age', palette='viridis')
g.set_titles("{col_name}")
g.fig.suptitle('Age Distribution by Job and Subscription Status', y=1.05)
plt.show()


def plot_categorical_heatmap(feature1, feature2):
    cross_tab = pd.crosstab(df[feature1], df[feature2], normalize='index') * 100
    plt.figure(figsize=(10, 6))
    sns.heatmap(cross_tab, annot=True, fmt='.1f', cmap='YlGnBu', linewidths=.5)
    plt.title(f'Percentage of Subscription by {feature1} and {feature2}')
    plt.ylabel(feature1)
    plt.xlabel(feature2)
    plt.show()


plot_categorical_heatmap('job', 'education')


plot_categorical_heatmap('marital', 'education')



plot_categorical_heatmap('poutcome', 'contact')



from sklearn.model_selection import StratifiedKFold

X = train.drop('y', axis=1)
y = train['y']

categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

test_str = test.copy()

X[categorical_features] = X[categorical_features].astype(str)
test_str[categorical_features] = test_str[categorical_features].astype(str)


!pip install catboost lightgbm


!pip install -U h2o


from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import numpy as np

# --- Data already prepared ---
# X, y, categorical_features, test_str are already defined

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Store out-of-fold predictions for meta learner
oof_cat = np.zeros((X.shape[0],))
oof_lgb = np.zeros((X.shape[0],))
oof_h2o = np.zeros((X.shape[0],))

test_cat = np.zeros((test_str.shape[0], n_splits))
test_lgb = np.zeros((test_str.shape[0], n_splits))
test_h2o = np.zeros((test_str.shape[0], n_splits))

roc_scores = []

for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
    print(f"\n===== Fold {fold} =====")
    X_train, X_valid = X.iloc[train_idx].copy(), X.iloc[valid_idx].copy()
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    # --- ensure categorical dtypes for both train/valid/test ---
    for col in categorical_features:
        X_train[col] = X_train[col].astype('category')
        X_valid[col] = X_valid[col].astype('category')
        test_str[col] = test_str[col].astype('category')

    # --- CatBoost ---
    from catboost import CatBoostClassifier
    cat_model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        eval_metric='AUC',
        random_seed=42,
        class_weights=[1, 7]
    )
    cat_model.fit(X_train, y_train, cat_features=categorical_features,
                  eval_set=(X_valid, y_valid), use_best_model=True)
    oof_cat[valid_idx] = cat_model.predict_proba(X_valid)[:, 1]
    test_cat[:, fold] = cat_model.predict_proba(test_str)[:, 1]

    # --- LightGBM (use pandas with categorical_feature param) ---
    import lightgbm as lgb
    lgb_train = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_features)
    lgb_valid = lgb.Dataset(X_valid, label=y_valid, reference=lgb_train, categorical_feature=categorical_features)
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'seed': 42,
        'verbose': -1,
        'scale_pos_weight': 7
    }
    lgb_model = lgb.train(params, lgb_train, num_boost_round=1000,
                          valid_sets=[lgb_train, lgb_valid])
    oof_lgb[valid_idx] = lgb_model.predict(X_valid, num_iteration=lgb_model.best_iteration)
    test_lgb[:, fold] = lgb_model.predict(test_str, num_iteration=lgb_model.best_iteration)

    # --- H2O GBM ---
    import h2o
    from h2o.estimators import H2OGradientBoostingEstimator
    # initialize H2O if not already
    try:
        h2o.cluster_info()
    except:
        h2o.init(nthreads=-1, max_mem_size='6G')
    
    # reset indices to avoid mismatch
    X_train = X_train.reset_index(drop=True)
    X_valid = X_valid.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_valid = y_valid.reset_index(drop=True)
    
    # define target column name
    target_col = y_train.name if hasattr(y_train, "name") and y_train.name is not None else "target"
    
    # prepare H2O frames
    h2o_train = h2o.H2OFrame(pd.concat([X_train, y_train.rename(target_col)], axis=1))
    h2o_valid = h2o.H2OFrame(pd.concat([X_valid, y_valid.rename(target_col)], axis=1))
    h2o_test = h2o.H2OFrame(test_str.copy())
    
    # set categorical columns as factors
    for col in categorical_features:
        if col in h2o_train.columns:
            h2o_train[col] = h2o_train[col].asfactor()
        if col in h2o_valid.columns:
            h2o_valid[col] = h2o_valid[col].asfactor()
        if col in h2o_test.columns:
            h2o_test[col] = h2o_test[col].asfactor()
    
    # ensure target is factor
    h2o_train[target_col] = h2o_train[target_col].asfactor()
    h2o_valid[target_col] = h2o_valid[target_col].asfactor()
    
    # initialize arrays (avoid NameError)
    oof_h2o = np.zeros(len(X_train) + len(X_valid))
    test_h2o = np.zeros((len(h2o_test), n_splits))  # assuming CV with n_splits
    roc_scores = []
    
    # H2O GBM model
    h2o_gbm = H2OGradientBoostingEstimator(
        ntrees=1000,
        learn_rate=0.05,
        max_depth=6,
        seed=42,
        nfolds=0,
        class_sampling_factors=[1.0, 7.0]  # optional, only if you want custom weights
    )
    
    feature_cols = X_train.columns.tolist()
    h2o_gbm.train(x=feature_cols, y=target_col, training_frame=h2o_train, validation_frame=h2o_valid)
    
    # predict and extract probability for class '1'
    pred_valid_h2o = h2o_gbm.predict(h2o_valid).as_data_frame(use_pandas=True)
    pred_test_h2o = h2o_gbm.predict(h2o_test).as_data_frame(use_pandas=True)
    
    # store OOF predictions
    oof_h2o[valid_idx] = pred_valid_h2o['p1'].values
    test_h2o[:, fold] = pred_test_h2o['p1'].values
    
    # --- Evaluate stacking input performance ---
    blended_valid = (oof_cat[valid_idx] + oof_lgb[valid_idx] + oof_h2o[valid_idx]) / 3.0
    score = roc_auc_score(y_valid, blended_valid)
    roc_scores.append(score)
    print(f"Fold {fold} -> ROC-AUC (blend of 3): {score:.5f}")



print("\nMean ROC-AUC across folds:", np.mean(roc_scores))

# --- Prepare meta features for Logistic Regression ---
X_meta = np.column_stack((oof_cat, oof_lgb, oof_h2o))
X_test_meta = np.column_stack((test_cat.mean(axis=1), test_lgb.mean(axis=1), test_h2o.mean(axis=1)))

# --- Logistic Regression meta-learner ---
meta_model = LogisticRegression(solver="lbfgs", random_state=42)
meta_model.fit(X_meta, y)
stacked_preds = meta_model.predict_proba(X_test_meta)[:, 1]



# Submission File

submission = pd.DataFrame({
    "id": test.index,     
    "y": stacked_preds
})

submission.to_csv("submission.csv", index=False)
print("✅ submission.csv file saved!")


import os
os.listdir("/kaggle/working")


import os, shutil

# Make sure the outputs directory exists
os.makedirs("/kaggle/outputs", exist_ok=True)

# Copy the file
shutil.copy("/kaggle/working/submission.csv", "/kaggle/outputs/submission.csv")




