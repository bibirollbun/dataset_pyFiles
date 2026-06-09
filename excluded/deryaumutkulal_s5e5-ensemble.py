# !pip install -q ipyplot

import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns

from io import BytesIO
#from ipyplot import show_images
from PIL import Image
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor

import warnings
warnings.filterwarnings("ignore")

sns.set(style = "whitegrid", context = "notebook")


train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")


train.shape, test.shape


train.head()


test.head()


def quick_overview(df, name="df"):
    print(f"{name.capitalize()} - Basic Info")
    display(df.info())
    display(df.describe(include="all").T)

quick_overview(train, "train")


quick_overview(test, "test")


msno.matrix(train, figsize=(10, 4))
plt.title("Missing Values Matrix - Train Dataset");


msno.matrix(test, figsize=(10, 4))
plt.title("Missing Values Matrix - Test Dataset");


msno.bar(train, figsize=(10, 4))
plt.title("Missing Count per Column - Train Dataset");


fig, axis = plt.subplots(1, 2, figsize = (14,4))
sns.histplot(train["Calories"], bins=60, kde=True, ax=axis[0])
axis[0].set_title("Calories (Raw Scale)")

sns.histplot(np.log1p(train["Calories"]), bins=60, kde=True, ax=axis[1], color="red")
axis[1].set_title("Calories (log1p Scale");


print("Skewness :", train["Calories"].skew().round(3))
print("Kurtosis :", train["Calories"].kurt().round(3))


num_cols = train.select_dtypes(include = ["int64", "float64"]).columns.tolist()
num_cols.remove("Calories")
num_cols.remove("id")

fig, axis = plt.subplots(2, 3, figsize=(15, 4 * len(num_cols) // 3))

for i, col in enumerate(num_cols):
    r, c = divmod(i, 3)
    sns.histplot(train[col], kde = True, ax = axis[r][c], color = "steelblue")
    axis[r][c].set_title(f"Train Dataset - {col}")

plt.tight_layout();


fig, axis = plt.subplots(2, 3, figsize=(15, 4 * len(num_cols) // 3))

for i, col in enumerate(num_cols):
    r, c = divmod(i, 3)
    sns.histplot(test[col], kde = True, ax = axis[r][c], color = "red")
    axis[r][c].set_title(f"Test Dataset - {col}")

plt.tight_layout()
plt.show()


plt.figure(figsize=(4, 3))
sns.countplot(y=train['Sex'], palette="muted")
plt.title("Sex Distribution in Train Dataset");


plt.figure(figsize=(4, 3))
sns.countplot(y=test['Sex'], palette="muted")
plt.title("Sex Distribution in Test Dataset");


corr = train[num_cols + ["Calories"]].corr(method="spearman")

plt.figure(figsize=(10, 7))
sns.heatmap(corr, cmap="RdBu_r", center=0, annot=True, fmt=".2f")
plt.title("Spearman Correlations")


for col in num_cols:
    plt.figure(figsize=(4, 3))
    sns.scatterplot(x=train[col], y=train["Calories"], alpha=0.2)
    sns.regplot(x=train[col], y=train["Calories"], scatter=False, color="red")
    plt.title(f"{col} vs Calories");


plt.figure(figsize=(4, 3))
sns.violinplot(x="Sex", y="Calories", data=train, palette="pastel", inner="quartile")
plt.title("Sex vs Calories");


pair_cols = ["Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp", "Calories"]
sns.pairplot(train[pair_cols], corner=True, diag_kind="kde", hue=None)
plt.suptitle("Pairwise scatter", y=1.02);


def iqr_outliers(series):
    q1, q3 = series.quantile([0.25, 0.75])
    iqr   = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return ((series < lower) | (series > upper)).sum()

print("\nOutlier counts (IQR rule):")

for col in num_cols + ["Calories"]:
    print(f"{col:<12} : {iqr_outliers(train[col])}")


X_vif = train[num_cols].assign(constant=1)
vif_df = pd.DataFrame({
    "feature": num_cols,
    "VIF"    : [variance_inflation_factor(X_vif.values, i) for i in range(len(num_cols))]
})
display(vif_df.sort_values("VIF", ascending=False).style.background_gradient(cmap="Reds"))


scaled = (train[num_cols] - train[num_cols].mean()) / train[num_cols].std()
pca = PCA(n_components=2, random_state=42).fit_transform(scaled)
plt.figure(figsize=(6, 5))
sns.scatterplot(x=pca[:,0], y=pca[:,1],
                hue=pd.qcut(train["Calories"], 5, labels=False), palette="viridis",
                alpha = 0.3, s = 10)
plt.title("PCA – coloured by Calories quintile")
plt.legend(title="Quintile", bbox_to_anchor=(1.05,1));


compare_cols = num_cols + ["Sex"]
train_tag = train.assign(dataset="train")[compare_cols + ["dataset"]]
test_tag  = test.assign(dataset="test")[compare_cols + ["dataset"]]
combo = pd.concat([train_tag, test_tag], axis = 0)

for col in compare_cols:
    if col is "Sex":
        ct = pd.crosstab(combo[col], combo["dataset"], normalize="columns") * 100
        ct.plot.barh(figsize = (6, 4), stacked=False, title=f"{col} – train vs test %")
        plt.show()
    else:
        sns.kdeplot(data=combo, x=col, hue="dataset", fill=True, common_norm=False, alpha=0.4)
        plt.title(f"{col} – train vs test distribution"); plt.show()


import pandas as pd
import numpy as np
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
import catboost as ctb
warnings.filterwarnings('ignore')


# Histogram matching for making the same distributions for original data
def to_distribution(mask, sample):
    sorted_mask = sorted(mask.values)

    mask_quantiles = np.linspace(0, 1, len(mask))
    sample_quantiles = np.argsort(np.argsort(sample)) / (len(sample) - 1)
    
    result = np.interp(sample_quantiles, mask_quantiles, sorted_mask)
    return result


# Making OOF-preds for all GBDTs
def trees_training(models, cv, X_train, y_train, X_test, y_test):
    results = {mod: {"oof_preds": np.zeros(len(X_train)),
                    "y_preds": np.zeros(len(X_test)),
                    "oof_scores": np.zeros(cv),
                    "y_scores": np.zeros(cv)
                    } 
               for mod in models}
    kf = KFold(n_splits=cv, shuffle=True)
    
    for fold_num, (idx_train, idx_test) in enumerate(kf.split(X_train, y_train)):
        X_fold_train, y_fold_train = X_train.iloc[idx_train], y_train.iloc[idx_train]
        X_fold_test, y_fold_test = X_train.iloc[idx_test], y_train.iloc[idx_test]

        X_fold_train, X_fold_val, y_fold_train, y_fold_val = train_test_split(X_fold_train, y_fold_train, test_size = 0.1)

        X_test_copy = X_test.copy()

        for name, model in models.items():
            if name == 'xgb':
                model.fit(X_fold_train, y_fold_train,
                          eval_set = [(X_fold_val, y_fold_val)],
                          early_stopping_rounds = 150,
                          verbose = False)
            elif name == 'lgb':
                model.fit(X_fold_train, y_fold_train, 
                          eval_set = [(X_fold_val, y_fold_val)],
                          callbacks=[lgb.early_stopping(stopping_rounds=150),
                                    lgb.log_evaluation(period=0)]
                         )
                         
            elif name == 'ctb':
                model.fit(X_fold_train, y_fold_train,
                          eval_set = [(X_fold_val, y_fold_val)]
                         )

            oof_preds = model.predict(X_fold_test)
            y_preds = model.predict(X_test_copy)
            oof_score = np.sqrt(mean_squared_error(y_fold_test, oof_preds))
            y_score = np.sqrt(mean_squared_error(y_test, y_preds))

            results[name]['oof_preds'][idx_test] = oof_preds
            results[name]['y_preds'] += y_preds / cv
            results[name]['oof_scores'][fold_num] = oof_score
            results[name]['y_scores'][fold_num] = y_score

    return results


df_train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
df_orig = pd.read_csv('/kaggle/input/fors5e5/calories (1).csv')

df_orig.columns = df_train.columns

df_train['Sex'] = df_train['Sex'].apply(lambda sex: 0 if sex == 'female' else 1)
df_orig['Sex'] = df_orig['Sex'].apply(lambda sex: 0 if sex == 'female' else 1)

df_train['Intensity'] = df_train['Heart_Rate'] / df_train['Duration']
df_orig['Intensity'] = df_train['Heart_Rate'] / df_train['Duration']

df_train.drop(columns = ['id'], inplace = True)
df_orig.drop(columns = ['id'], inplace = True)

for col in df_orig.columns:
    df_orig[col] = to_distribution(df_train[col], df_orig[col])


df_train.shape, df_orig.shape


X = df_train.drop(columns=['Calories'])
y = np.log1p(df_train['Calories'])
df_orig['Calories'] = np.log1p(df_orig['Calories'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


X_train = pd.concat([X_train, df_orig.drop(columns=['Calories'])]).reset_index(drop=True)
y_train = pd.concat([y_train, df_orig['Calories']]).reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)


xgbmodel = xgb.XGBRegressor(
    n_estimators = 7500,
    learning_rate = 0.01,
    max_depth = 6,
    subsample = 0.8,
    colsample_bytree = 0.8,
    alpha = 0,
    verbosity = 0,
    reg_lambda = 0.1,
    # device = 'gpu'
)

lgbmodel = lgb.LGBMRegressor(n_estimators = 7500,
                               learning_rate = 0.01,
                               # device='gpu',
                               verbosity = -1,
                               num_leaves = 31,
                               max_depth = -1,
                               min_child_samples = 20,
                               subsample = 0.8,
                               colsample_bytree = 0.8,
                               reg_alpha = 0.1,
                               reg_lambda = 0.1,
                               verbose = -1
                          )

catboostmodel = ctb.CatBoostRegressor(
    iterations = 9000,
    learning_rate = 0.01,
    early_stopping_rounds = 150,
    # task_type = 'GPU',
    depth = 6,
    l2_leaf_reg = 3,
    bootstrap_type = 'Bayesian',
    bagging_temperature = 1.0,
    train_dir = None,
    logging_level='Silent'
)


models = {
          "xgb": xgbmodel,
          "lgb": lgbmodel,
          "ctb": catboostmodel
         }


results = trees_training(models, 5, X_train, y_train, X_test, y_test)


X_final_train, X_val, y_final_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)


xgbmodel.fit(X_final_train, y_final_train,
              eval_set = [(X_val, y_val)],
              early_stopping_rounds = 150,
              verbose = False)

lgbmodel.fit(X_final_train, y_final_train, 
                          eval_set = [(X_val, y_val)],
                          callbacks=[lgb.early_stopping(stopping_rounds=150),
                                    lgb.log_evaluation(period=0)],
                         )

catboostmodel.fit(X_final_train, y_final_train,
                          eval_set = [(X_val, y_val)]
                 )


X_train_stack = pd.DataFrame([results[model]['oof_preds'] for model in models]).transpose()
X_test_stack = pd.DataFrame([results[model]['y_preds'] for model in models]).transpose()


ridgemodel = RidgeCV(alphas=np.linspace(0.01, 50, 200), scoring='neg_root_mean_squared_error', cv=5)
ridgemodel.fit(X_train_stack, y_train)
preds = ridgemodel.predict(X_test_stack)


ridgemodel.alpha_, ridgemodel.coef_


np.sqrt(mean_squared_error(y_test, preds))


df_test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')


df_test['Sex'] = df_test['Sex'].apply(lambda sex: 0 if sex == 'female' else 1)
df_test['Intensity'] = df_test['Heart_Rate'] / df_test['Duration']


df_test.drop(columns=['id'], inplace=True)


xgb_preds = xgbmodel.predict(df_test)
lgb_preds = lgbmodel.predict(df_test)
ctb_preds = catboostmodel.predict(df_test)


X_preds = pd.DataFrame([xgb_preds, lgb_preds, ctb_preds]).transpose()


pred = ridgemodel.predict(X_preds)


submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')


submission['Calories'] = np.expm1(pred)
submission.to_csv('submission.csv', index=False)


submission




