import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import iqr

from sklearn import metrics
from sklearn.model_selection import KFold, cross_val_score
from sklearn.base import clone
from sklearn import preprocessing as prepro
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import RandomizedSearchCV

import category_encoders as ce

from lightgbm import LGBMClassifier

import optuna
from optuna.samplers import TPESampler

import warnings
warnings.filterwarnings('ignore')

# Define your custom colors
set_colors = ['steelblue', 'orange', '#00FFFF']
target_colors = ['#808080', '#A52A2A']
# Create a seaborn palette
set_palette = sns.color_palette(set_colors)
target_palette = sns.color_palette(target_colors)


# # Set Seaborn theme with dark grid and brighter palette
# sns.set_theme(style="darkgrid", font_scale=0.9)

# Update matplotlib parameters for brighter dark theme
plt.rcParams.update({
    'axes.facecolor': 'black',       # Slightly lighter than #222222
    'figure.facecolor': 'black',
    'text.color': '#FFA500',           # Bright gold for better contrast
    'axes.labelcolor': 'gray',      # Softer mint green
    'xtick.color': 'gray',
    'ytick.color': 'gray',
    'grid.color': 'grey',           # Lighter grid lines
    'axes.edgecolor': '#dddddd',        # Light gray edges
    'axes.grid': False,
})

seed=42

# verify the versions
print(f'pandas version: {pd.__version__}')
print(f'numpy version: {np.__version__}')
print(f'seaborn version: {sns.__version__}')


tr_00 = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv', index_col='id')
ts_00 = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv', index_col='id')
sb_00 = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')

# external dataset
or_00 = pd.read_csv('/kaggle/input/d/mohankrishnathalla/diabetes-health-indicators-dataset/diabetes_dataset.csv')[tr_00.columns]

# or_00 = pd.read_csv('')

target = 'diagnosed_diabetes'

tr_00.head(5)


or_00.head()


print(f'shape of train set: {tr_00.shape}\
\nshape of test set: {ts_00.shape}\
\nshape of external set: {or_00.shape}')


tr_00.info()


for df in [tr_00]:
    df[target] = df[target].astype('int').astype('category')


tr_00.describe().T


tr_00.describe(exclude='number').T


num_feats = [feat for feat in ts_00.select_dtypes(include='number').columns.tolist() if ts_00[feat].nunique()>5]
cat_feats = ts_00.select_dtypes(exclude='number').columns.tolist()
bool_feats = [feat for feat in ts_00.select_dtypes('number') if ts_00[feat].nunique()==2]


def plot_num_distribution_grid(df, num_feats=num_feats, grouper=target, palette=target_palette, bins=50, ncols=3):
    """
    Plot histograms + boxplots for numerical features in a grid layout.
    Each feature gets two stacked subplots (hist + box) with boxplot 1/3 height of histplot.
    Both plots share the same x-axis.
    """
    df[target] = df[target].astype('category')
    nrows = int(np.ceil(len(num_feats) / ncols))

    fig = plt.figure(figsize=(ncols*6, nrows*4))
    gs = fig.add_gridspec(nrows=nrows, ncols=ncols, hspace=0.6)

    for i, feat in enumerate(num_feats):
        row = i // ncols
        col = i % ncols

        # Sub-grid with height ratio 3:1 and shared x-axis
        sub_gs = gs[row, col].subgridspec(2, 1, height_ratios=[3, 1])
        ax_hist = fig.add_subplot(sub_gs[0])
        ax_box = fig.add_subplot(sub_gs[1], sharex=ax_hist)

        # Histogram
        # sns.histplot(data=df, x=feat, hue=grouper, bins=bins, kde=True,
        #              palette=palette, multiple="dodge", ax=ax_hist)
        sns.kdeplot(data=df, x=feat, hue=grouper, palette=palette, fill=True, ax=ax_hist)
        ax_hist.set_title(f"{feat} Distribution", fontsize=10, color='white', backgroundcolor='black')
        ax_hist.set_xlabel('')  # Remove x-label from histogram
        ax_hist.tick_params(axis='x', labelbottom=False)  # Hide x-ticks on histogram

        # Boxplot (horizontal to match x-axis)
        sns.boxplot(data=df, x=feat, y=grouper, palette=palette,
                    ax=ax_box,
                    flierprops=dict(marker='o', markerfacecolor='red',
                                    markeredgecolor='black', markersize=3))
        plt.ylabel('')

    plt.tight_layout()
    plt.show()


tr_r = tr_00.copy()
ts_r = ts_00.copy()
or_r = or_00.copy()

tr_r['dataset'] = 'train'
ts_r['dataset'] = 'test'
or_r['dataset'] = 'external'

tr_all_sets = pd.concat([tr_r, ts_r, or_r], ignore_index=True)

plot_num_distribution_grid(tr_all_sets, grouper='dataset', palette=set_palette)


for c, col in enumerate(num_feats, 1):
    plt.subplot(5, 3, c)
    ax = tr_00[col].plot(title=col, figsize=(10, 12), color=set_colors[0], alpha=0.6)
    ts_00[col].plot(color=set_colors[1], ax=ax, alpha=0.6)
    or_00[col].plot(color=set_colors[2], ax=ax, alpha=0.6)
    plt.xticks([])
    if c not in [13, 14, 15]:
        plt.xlabel('')
plt.tight_layout()
plt.show()


plt.figure(figsize=(7, 6))
plt.subplot(221)
tr_00[target].value_counts().plot.pie(
    radius=1.3,
    autopct='%.2f%%',
    startangle=90,
    wedgeprops={'width': 0.8},
    explode=[0.02, 0.02],
    colors=target_colors,
    title='Proportion of target classes in train data'
)
plt.ylabel('')
plt.subplot(222)
ax1 = sns.countplot(tr_00, x=target, palette=target_colors)
for count in ax1.containers:
    ax1.bar_label(count, label_type='center')
plt.subplot(223)
or_00[target].value_counts().plot.pie(
    radius=1.2,
    autopct='%.2f%%',
    startangle=90,
    wedgeprops={'width': 0.8},
    explode=[0.02, 0.02],
    colors=target_colors,
    title='Proportion of target classes in extermal data'
)
plt.ylabel('')
plt.subplot(224)
ax2=sns.countplot(or_00, x=target, palette=target_colors)
for count in ax2.containers:
    ax2.bar_label(count, label_type='center')
    
plt.tight_layout(pad=2, h_pad=2, w_pad=2)
plt.show()


plot_num_distribution_grid(tr_00, grouper=target)


use_cat_mapping = False


# Your mapping dictionaries
gender__dico = {'Male': -1, "Female": 1, 'Other': 0}
education_level__dico = {'No formal': 0, 'Highschool': 1, 'Graduate': 2, 'Postgraduate': 3}
smoking_status__dico = {'Never': 0, 'Former': 1, 'Current': 2}  
income_level__dico = {'Low': 0, 'Lower-Middle': 1, 'Middle': 2, 'Upper-Middle': 3, 'High': 4}
employment_status__dico = {'Unemployed': 0, 'Student': 1, 'Employed': 2, 'Retired': 3}

# Collect mappings in one place keyed by column name
cat_mappings = {
    'gender': gender__dico,
    'education_level': education_level__dico,
    'smoking_status': smoking_status__dico,    # be careful with name consistency
    'income_level': income_level__dico,
    'employment_status': employment_status__dico,
}


# Apply to each dataframe
for df in [tr_00, ts_00, or_00]:
    df['pysicaL_activity_*_sleep_hoursðŸ§®'] = df['physical_activity_minutes_per_week']/df['sleep_hours_per_day']
    df['sleep_hours_per_day_*_sleep_hoursðŸ§®'] = df['sleep_hours_per_day']/df['screen_time_hours_per_day']
    df['bmi_*_diet_scoreðŸ§®'] = df['bmi']/df['diet_score']
    df['diastolic_*_sistolicðŸ§®'] = df['diastolic_bp']/df['systolic_bp']
    if use_cat_mapping:
        for col, mapping in cat_mappings.items():
            # Replace with numeric codes; keep any unknowns as-is
            df[col] = df[col].replace(mapping)
    
            # Optional: convert to nullable integer dtype (handles any non-mapped leftovers)
            # If you are sure everything maps to integers, you can use int instead.
            df[col] = df[col].astype('Int64')  # nullable integer dtype
    else:
        pass

tr_00.head(3)





# Define function to handle outliers
def remove_outliers(df):
    df = df.copy()
    for col in num_feats:
        if df[col].nunique()>20:
            IQR = iqr(df[col])  # calculate the interquartile range
            df[col] = np.clip(df[col], 
                              (np.quantile(df[col], 0.25) - 1.5*IQR), 
                              (np.quantile(df[col], 0.75) + 1.5*IQR)
                             ) # clip the outliers in the range (25, 75)quantile -or+ 1.5 IQ
    return df

# Remove outliers from the various datasets
tr_01 = remove_outliers(tr_00)
ts_01 = remove_outliers(ts_00)
or_01 = remove_outliers(or_00)


X = tr_01.copy()

y = X.pop(target)

# X = pd.get_dummies(X)
# ts_01 = pd.get_dummies(ts_01)


from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=seed)


# The preprocessor used to handle cat_features for xgb and voting classifiers
preprocessor = ColumnTransformer(
    transformers=[
        # ('encoder', prepro.OneHotEncoder(), cat_feats)
        # ('encoder', ce.TargetEncoder(), cat_feats)
        ('encoder', ce.CountEncoder(), cat_feats)
    ],  
    remainder= 'passthrough',
    n_jobs=-1
)


# # The preprocessor used to handle cat_features for xgb and voting classifiers
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('encoder', prepro.OneHotEncoder(), cat_feats)

#     ],  
#     remainder= 'passthrough',
#     n_jobs=-1
# )


def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 1.0),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "num_leaves": trial.suggest_int("num_leaves", 4, 256),
        # "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
        # "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        # "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 1.0),
        # "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt", "dart"]),
        # "objective": "regression",
        # "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.5),
        # "max_bin": trial.suggest_int("max_bin", 100, 255),
        # "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.5, 2.0),
        # "importance_type": trial.suggest_categorical("importance_type", ["split", "gain"]),
        "device": "gpu"  # Uncomment if GPU is available
    }

    # the model pipeline
    model = Pipeline(
        [('preprocessor', preprocessor), 
         ('estimator', LGBMClassifier(**params, verbose=-1))]
    )

    if cv_scorer:
        # Cross-validation (recommended)
        scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
        return scores.mean()
    else:
        # Alternatively
        model.fit(X_train, y_train)
        
        preds = model.predict_proba(X_valid)[:, 1]
        score = metrics.roc_auc_score(y_valid, preds)
       
        return score


cv_scorer=False

def Run_Pass_lgbm_study(n_trials=1):
    if n_trials > 1:
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, timeout=72000, show_progress_bar=True)
        best_study_params = study.best_params

        print(f"Number of finished trials: {len(study.trials)}")
        trial = study.best_trial
        print(f"Best trial RMSE score: {trial.value:.6f}")
    else:
        print("No need to run Optuna, we will use the parameters obtained earlier.")
        
        # best_study_params = {'n_estimators': 470, 
        #                      'learning_rate': 0.09,
        #                      'max_depth': 5, 
        #                      'num_leaves': 256, 
        #                      # 'min_child_samples': 28, 
        #                      # 'colsample_bytree': 0.6478895973288086,
        #                      # 'subsample': 0.528816037125001, 
        #                      'reg_alpha': 1.7035182824513545, 
        #                      'reg_lambda': 0.3866741017392513, 
        #                      # 'boosting_type': 'dart', 
        #                      # 'min_split_gain': 0.4840318270120275, 
        #                      # 'max_bin': 206, 
        #                      # 'scale_pos_weight': 1.2679193621456322, 
        #                      # 'importance_type': 'split'
        #                     }
        # best_study_params = {'n_estimators': 890, 
        #                      'learning_rate': 0.6350991469818366, 
        #                      'max_depth': 2, 
        #                      'num_leaves': 98, 
        #                      'reg_alpha': 0.8291831960823974,
        #                      'reg_lambda': 0.5865654353948012
        #                     }

        best_study_params = {'n_estimators': 960, 
                             'learning_rate': 0.5019675140772736, 
                             'max_depth': 2, 
                             'num_leaves': 52,
                             'reg_alpha': 0.9945293630381706, 
                             'reg_lambda': 0.9849067535604036
                            }
    
    print(f"\nBest parameters: {best_study_params}")
    return best_study_params


best_params = Run_Pass_lgbm_study(n_trials=40)


# The estimator
lgb = LGBMClassifier(**best_params, verbose=-1)

# the model pipeline
lgb_pipe = Pipeline(
    [('preprocessor', preprocessor), 
     ('estimator', lgb)]
)


lgb_pipe.fit( X, y)


# Train model
lgb_pipe.fit(X_train, y_train)
#
lgb = lgb_pipe.named_steps['estimator']
# Get feature importance
importances_split = lgb.feature_importances_

# For gain importance:
importances_gain = lgb.booster_.feature_importance(importance_type='gain')


# Get the fetures names after preprocessing
feature_names = lgb_pipe[:-1].get_feature_names_out()

# Build importance dataframe
importaces = pd.DataFrame(
    {'SplitImportances': importances_split, 
     'GainImportances': importances_gain}, 
    index=feature_names).sort_values(by='GainImportances')

importaces


importaces.iloc[:, 1].plot.barh(figsize=(12, 10), title=importaces.columns[0])


importaces.iloc[:, 0].sort_values().plot.barh(figsize=(12, 10), title=importaces.columns[1])


seed=1087
n_splits = 6
spliter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

# Store out-of-fold predictions
oof_preds = []
oof_true = []

plt.figure(figsize=(6, 5))      
for f, (tr_ind, va_ind) in enumerate(spliter.split(X, y), 1):
    print(f'ðŸš¨ðŸš¨ðŸš¨ Working on fold_{f} of {n_splits}')
    X_tr, X_va = X.iloc[tr_ind], X.iloc[va_ind]
    y_tr, y_va = y.iloc[tr_ind], y.iloc[va_ind]

    # Clone the model before fitting
    clf = clone(lgb_pipe)
    clf.fit(X_tr, y_tr)

    preds = clf.predict_proba(X_va)[:, 1]

    # Save for overall ROC
    oof_preds.extend(preds)
    oof_true.extend(y_va)

    # Per-fold AUC
    score = metrics.roc_auc_score(y_va, preds)
    print(30*' ',f'â€¢â€¢â€¢â€¢â€¢> Fold_{f} AUC: {score:.6f} âœ…\n')

    # Per-fold ROC curve
    fpr, tpr, _ = metrics.roc_curve(y_va, preds)
    plt.plot(fpr, tpr, label=f'Fold {f} AUC  = {score:.5f}')

# Overall ROC curve
overall_auc = metrics.roc_auc_score(oof_true, oof_preds)
fpr, tpr, _ = metrics.roc_curve(oof_true, oof_preds)
plt.plot(fpr, tpr, color='black', linewidth=2,
         label=f'Overall AUC = {overall_auc:.5f}')

# Diagonal baseline
plt.plot([0, 1], [0, 1], color='maroon', linestyle='--')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.title('ROC Curves of All Folds + Overall', 
          color='maroon', fontsize=11, weight='bold')
plt.tight_layout()
plt.show()


sb_00[target] = lgb_pipe.predict_proba(ts_01)[:,1]

sb_00.head()


threshold = 0.54

plt.figure(figsize=(10, 3))
plt.subplot(121)
sns.histplot(sb_00, x=target, bins=40)
plt.axvline(x=sb_00[target].median(), color='r', linestyle='--', linewidth=2)
plt.title('Histogram of predicted_proba')
plt.subplot(122)
(sb_00[target]>threshold).value_counts().plot.pie(
    radius=1.25,
    autopct='%.2f%%',
    startangle=70,
    wedgeprops={'width': 0.8},
    explode=[0.02, 0.02],
    colors=target_colors,
    title=f'Proportions with threshold of {threshold }'
)
plt.ylabel('')
plt.show()


sb_00.to_csv('submission.csv', index=False)
print('The file is ready for submission âœ…')


tr_or = pd.concat([tr_00, or_00], ignore_index=True)

