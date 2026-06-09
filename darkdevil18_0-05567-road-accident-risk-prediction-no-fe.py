import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from rich.console import Console
from rich.table import Table

warnings.filterwarnings("ignore")

plt.style.use("ggplot")
sns.set(font_scale=1.1)


train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv", index_col='id')
test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv", index_col='id')


orig = []
for k in [2,10,100]:
    df = pd.read_csv(f"/kaggle/input/simulated-roads-accident-data/synthetic_road_accidents_{k}k.csv")
    orig.append(df)
orig = pd.concat(orig,axis=0, ignore_index=True)
orig = orig[train.columns]


train = pd.concat([train, orig], axis=0, ignore_index=True)


train.head()


def custom_describe(df, categorical=False):

    if not categorical:
        df = df.select_dtypes(include=np.number)
        
    des = df.describe().T.round(2)
    des['count'] = des['count'].astype('int')

    if not categorical:
        des['skewness'] = df.skew().round(2)
        des['kurtosis'] = df.kurtosis().round(2)

    return des


def df_summary(df, label="Train"):
    console = Console()
    console.rule(f"[bold blue]{label} DataFrame Description[/bold blue]")
    
    console.print(f"[bold]Shape:[/bold] {df.shape}\n")

    # Numeric summary
    numeric_cols = df.select_dtypes(include=np.number).columns
    if len(numeric_cols) > 0:
        console.print("[bold blue]Numerical Columns:[/bold blue]")
        num_stats = custom_describe(df[numeric_cols], categorical=False)
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Column")
        for col in num_stats.columns:
            table.add_column(col)
        for idx, row in num_stats.iterrows():
            table.add_row(idx, *[f"{val:.2f}" if isinstance(val, (float, np.float64)) else str(val) for val in row])
        console.print(table)
        console.print("\n")
    
    # Categorical summary
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(cat_cols) > 0:
        console.print("[bold blue]Categorical Columns:[/bold blue]")
        cat_stats = df[cat_cols].describe().T
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Column")
        for col in cat_stats.columns:
            table.add_column(col)
        for idx, row in cat_stats.iterrows():
            table.add_row(idx, *[str(val) for val in row])
        console.print(table)
        console.print("\n")

df_summary(train)
df_summary(test, "Test")



target = "accident_risk"
features = test.columns.to_list()
numerical_features = train[features].select_dtypes(include=np.number).columns.to_list()
categorical_features = train[features].select_dtypes(exclude=np.number).columns.to_list()


pd.DataFrame({
    'Columns': train.columns.to_list(),
    '# Null': train.isna().sum().values
})


num_duplicates = train.duplicated().sum()
print(f"Number of duplicate rows: {num_duplicates}")

if num_duplicates > 0:
    train = train.drop_duplicates()


fig, axes = plt.subplots(2, 2, figsize=(14, 8))
axes = axes.flatten()

for i, feature in enumerate(numerical_features):
    if pd.api.types.is_integer_dtype(train[feature]):
        sns.histplot(data=train, x=feature, ax=axes[i], kde=True, color='steelblue')
        axes[i].set_title(f"Histogram of {feature}", fontweight="bold", fontsize=14)
    else:
        sns.kdeplot(data=train, x=feature, ax=axes[i], fill=True, color='seagreen')
        axes[i].set_title(f"KDE Plot of {feature}", fontweight="bold", fontsize=14)

    axes[i].set_xlabel(feature, fontsize=12, fontweight="bold")
    axes[i].set_ylabel("Frequency", fontsize=12, fontweight="bold")

plt.suptitle("Distribution of Numerical Features", fontsize=20, fontweight="bold", color="red")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


fig, axes = plt.subplots(2, 4, figsize=(18, 10))
axes = axes.flatten()

for i, feature in enumerate(categorical_features):  # Limit to 8 per page
    sns.countplot(data=train, x=feature, ax=axes[i], palette='viridis')
    axes[i].set_title(f"Countplot of {feature}", fontweight="bold", fontsize=13)
    axes[i].set_xlabel(feature, fontsize=11, fontweight="bold")
    axes[i].set_ylabel("Count", fontsize=11, fontweight="bold")
    axes[i].tick_params(axis='x', rotation=30)

plt.suptitle("Distribution of Categorical Features", fontsize=20, fontweight="bold", color="red")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


fig, axes = plt.subplot_mosaic([['A', 'B'], ['C', 'C']], figsize=(14, 10))
sns.boxplot(y=target, data=train, ax=axes['A'], color='goldenrod')
axes['A'].set_title(f"Boxplot of {target}", fontweight="bold", fontsize=14)

sns.violinplot(y=target, data=train, ax=axes['B'], color='lightgreen')
axes['B'].set_title(f"Violin Plot of {target}", fontweight="bold", fontsize=14)

sns.histplot(train[target], kde=True, ax=axes['C'], color='royalblue')
axes['C'].set_title(f"Histogram of {target}", fontweight="bold", fontsize=14)
axes['C'].set_xlabel(target, fontsize=12, fontweight="bold")

plt.suptitle(f"Distribution of Target Variable: {target}", fontsize=22, fontweight="bold", color='red')
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()


corr = train[numerical_features + [target]].corr()


mask = np.triu(np.ones_like(corr, dtype=bool))

plt.figure(figsize=(10, 8))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Matrix", fontsize=18, fontweight="bold", color="red")
plt.show()


top_corr_features = corr[target].abs().sort_values(ascending=False)[1:5].index.tolist()

sns.pairplot(train, vars=top_corr_features + [target], kind='scatter', corner=True)
plt.suptitle("Pairwise Scatter Plots of Top Features vs Target", 
             fontsize=20, fontweight="bold", color="red", y=1.02)
plt.show()


plt.figure(figsize=(12, 6))
sns.barplot(x='road_type', y=target, hue='lighting', data=train, ci=None, palette='viridis')
plt.title("Average Accident Risk by Road Type and Lighting", fontsize=16, fontweight='bold')
plt.xlabel("Road Type", fontsize=12, fontweight='bold')
plt.ylabel("Average Accident Risk", fontsize=12, fontweight='bold')
plt.xticks(rotation=30)
plt.legend(title="Lighting")
plt.show()


plt.figure(figsize=(12, 4))
sns.barplot(x='holiday', y=target, hue='school_season', data=train, ci=None, palette='magma')
plt.title("Effect of Holidays and School Season on Accident Risk", fontsize=16, fontweight='bold')
plt.xlabel("Holiday (1 = Yes, 0 = No)", fontsize=12, fontweight='bold')
plt.ylabel("Average Accident Risk", fontsize=12, fontweight='bold')
plt.legend(title="School Season")
plt.show()


plt.figure(figsize=(10, 6))
sns.barplot(x='speed_limit', y=target, data=train, ci=None, palette='coolwarm')
plt.title("Average Accident Risk by Speed Limit", fontsize=16, fontweight='bold')
plt.xlabel("Speed Limit (km/h)", fontsize=12, fontweight='bold')
plt.ylabel("Average Accident Risk", fontsize=12, fontweight='bold')
plt.show()


plt.figure(figsize=(16, 12))

for i, feature in enumerate(numerical_features, 1):
    plt.subplot(2, 2, i)
    sns.regplot(x=feature, y='accident_risk', data=train, scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
    plt.title(f"Regression Plot: {feature} vs Accident Risk", fontsize=14, fontweight='bold')
    plt.xlabel(feature, fontsize=12, fontweight='bold')
    plt.ylabel("Accident Risk", fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold

from sklearn.ensemble import RandomForestRegressor


import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

import lightgbm as lgb

def model_trainer(model, X, y, test=None, nsplits=5, random_state=42, log_target=False, verbose=0, model_name=None):
    
    kfold = KFold(n_splits=nsplits, shuffle=True, random_state=random_state)

    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()
    y_array = y.to_numpy() if isinstance(y, pd.Series) else y.copy()

    if log_target:
        y_array = np.log1p(y_array)

    oof_train_preds = np.zeros(len(y_array))
    if test is not None:
        if isinstance(test, pd.DataFrame):
            test = test.to_numpy()
        oof_test_preds = np.zeros(len(test))

    oof_rmse = []
    oof_r2 = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y_array[train_idx], y_array[val_idx]

        
        if model_name == 'xgb':
            model.fit(X_train, y_train, 
                      eval_set=[(X_val, y_val)], 
                      verbose=verbose)
            if hasattr(model, 'best_iteration') and model.best_iteration is not None:
                y_pred = model.predict(X_val, iteration_range=(0, model.best_iteration + 1))
            else:
                y_pred = model.predict(X_val)

        elif model_name == 'lgb':
            model.fit(X_train, y_train, 
                      eval_set=[(X_val, y_val)], 
                      callbacks=[lgb.early_stopping(200, verbose=verbose)])
            y_pred = model.predict(X_val) 

        elif model_name == 'cat':
            model.fit(X_train, y_train, 
                      eval_set=[(X_val, y_val)], 
                      early_stopping_rounds=200, 
                      verbose=verbose)
            y_pred = model.predict(X_val) 
        
        elif model_name == 'hgb':
            model.fit(X_train, y_train, )
            y_pred = model.predict(X_val)

        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

        
        if log_target:
            y_val_exp = np.expm1(y_val)
            y_pred_exp = np.expm1(y_pred)
        else:
            y_val_exp = y_val
            y_pred_exp = y_pred

        rmse = mean_squared_error(y_val_exp, y_pred_exp, squared=False)
        r2 = r2_score(y_val_exp, y_pred_exp)
        oof_rmse.append(rmse)
        oof_r2.append(r2)
        
        oof_train_preds[val_idx] = y_pred

        
        if test is not None:
            test_pred = None
            if model_name == 'xgb':
                if hasattr(model, 'best_iteration') and model.best_iteration is not None:
                    test_pred = model.predict(test, iteration_range=(0, model.best_iteration + 1))
                else:
                    test_pred = model.predict(test)
            else:
                test_pred = model.predict(test)
                
            if log_target:
                test_pred = np.expm1(test_pred)
                
            oof_test_preds += test_pred / nsplits

        print(f"Fold {fold} â†’ RMSE: {rmse:.4f}, R2: {r2:.4f}")

    print(f"\nAverage Fold RMSE Score: {np.mean(oof_rmse):.4f} Â± {np.std(oof_rmse):.4f}")
    print(f"Average Fold R2 Score: {np.mean(oof_r2):.4f} Â± {np.std(oof_r2):.4f}")

    if test is not None:
        return oof_train_preds, oof_test_preds

    return oof_train_preds


numeric_transformer = MinMaxScaler()
categorical_transformer = OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)


X = train.copy()
y = X.pop(target)
y = np.log1p(y)


comb = pd.concat([X, test], axis=0)

preprocessor.fit(comb)

X_processed = preprocessor.transform(X)
test_processed = preprocessor.transform(test)


oof_train_preds = pd.DataFrame()
oof_test_preds = pd.DataFrame()


from xgboost import XGBRFRegressor
rf = XGBRFRegressor()
_ = model_trainer(rf, X_processed, y, log_target=True)


from xgboost import XGBRegressor

xgb_reg = XGBRegressor( # source: https://www.kaggle.com/code/cdeotte/xgb-boosting-over-residuals-cv-0-05595
    objective="reg:squarederror",
    eval_metric="rmse",
    device='cuda',
    tree_method="gpu_hist",
    early_stopping_rounds=200,
    n_estimators=5000,
    eta=0.01,
    max_depth=6,
    colsample_bytree=0.6,
    subsample=0.9,
    seed=42
    
)
oof_train_preds['xgb'], oof_test_preds['xgb'] = model_trainer(xgb_reg, X_processed, y, test_processed, 
                                                              nsplits=7, log_target=True, model_name='xgb', verbose=0)


from catboost import CatBoostRegressor

cat_reg = CatBoostRegressor(
    task_type="GPU",
    loss_function="RMSE",
    n_estimators=10000,
    learning_rate=0.01,
    depth=8,
    random_seed=42,
    allow_writing_files=False
)

oof_train_preds['cat'], oof_test_preds['cat'] = model_trainer(cat_reg, X_processed, y, test_processed, nsplits=7,
                                                              log_target=True, verbose=0, model_name='cat')


from lightgbm import LGBMRegressor

lgb_reg = LGBMRegressor(
    device="GPU",
    objective="regression",
    metric="rmse",
    n_estimators=10000,
    learning_rate=0.01,
    max_depth=8,
    random_state=42,
    verbosity=-1
)

oof_train_preds['lgb'], oof_test_preds['lgb'] = model_trainer(lgb_reg, X_processed, y, test_processed, log_target=True,
                                                              nsplits=7,
                                                              model_name='lgb', verbose=False
                                                             )


from sklearn.ensemble import HistGradientBoostingRegressor

hgb = HistGradientBoostingRegressor(max_iter=3000, learning_rate=0.01, early_stopping=True, n_iter_no_change=200)
oof_train_preds['hgb'], oof_test_preds['hgb'] = model_trainer(hgb, X_processed, y, test_processed, log_target=True, nsplits=7, model_name='hgb')


from sklearn.linear_model import Ridge

lr = Ridge(positive=True)

_, test_preds = model_trainer(lr, oof_train_preds, y, oof_test_preds, nsplits=5, log_target=True, random_state=101)


sub = pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")
sub[target] = test_preds
sub.to_csv("submission.csv", index=False)
sub.head()

