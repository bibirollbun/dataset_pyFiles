import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import iqr
from itertools import combinations

from sklearn import metrics
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.base import clone

from xgboost import XGBClassifier
from catboost import CatBoostClassifier, Pool

import shap

import warnings
warnings.filterwarnings('ignore')

# Define your custom colors
target_colors = ['#fa8072', '#fdd5b1']
set_colors = ['#ED6AFF', '#c6ff00', '#26c6da']
# Create a seaborn palette
set_palette = sns.color_palette(set_colors)
target_palette = sns.color_palette(target_colors)

# Define a gradient from blue â†’ white â†’ red
my_cmap = "Blues"

# # Set Seaborn theme with dark grid and brighter palette
# sns.set_theme(style="darkgrid", font_scale=0.9)

# Update matplotlib parameters for brighter dark theme
plt.rcParams.update({
    'axes.facecolor': '#fff0f5',  
    'figure.facecolor': '#C6DEFF',
    'text.color': '#607d8b',  
    'axes.labelcolor': '#607d8b', 
    'xtick.color': '#607d8b',
    'ytick.color': '#607d8b',
    'grid.color': '#E12AFB',
    'axes.edgecolor': '#607d8b', 
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

or_00 = pd.read_csv('/kaggle/input/diabetes-health-indicators-dataset/diabetes_dataset.csv')[tr_00.columns]

# or_00 = pd.read_csv('')

target = 'diagnosed_diabetes'

tr_00.head(3)


or_00.head()


print(f'shape of train set: {tr_00.shape}\nshape of test set: {ts_00.shape}\nshape of original set: {or_00.shape}')


tr_00.info()


for df in [tr_00]:
    df[target] = df[target].astype('category')


tr_00.describe().T


tr_00.describe(exclude='number').T


num_feats = [feat for feat in ts_00.select_dtypes(include='number').columns.tolist() if ts_00[feat].nunique()>2]
cat_feats = ts_00.select_dtypes(exclude='number').columns.tolist()


bool_feats = [feat for feat in ts_00.select_dtypes('number') if ts_00[feat].nunique()==2]


tr_comb = pd.concat([tr_00, or_00], ignore_index=True)


# Define a function to perform the adversarial validation of two datasets
def adversarial_validation(df_1, df_2, name_1, name_2):
    adv_df_1, adv_df_2 = df_1[num_features].copy(), df_2[num_features].copy()

    # label the test and train data with 0 and 1 (it doesn't really matter which is which)
    adv_df_1, adv_df_2 = adv_df_1.assign(adv=1),  adv_df_2.assign(adv=0)

    # combine the training and test data into one big dataset
    combined = pd.concat([adv_df_1, adv_df_2], axis=0)

    # Shuffle
    combined = combined.sample(frac=1, random_state=64)

    # perform the binary classification, for example using XGboost
    X_combined = combined.drop('adv', axis=1)
    y_combined = combined.adv
    
    # Define the classifier
    clf = CatBoostClassifier(verbose=0, n_estimators=50)

    # Get the cross validation scores
    adv_scores = []
    X_train, X_valid, y_train, y_valid = train_test_split(X_combined, 
                                                          y_combined, 
                                                          test_size=0.3)
    clf.fit(X_train, y_train)
    y_pred = clf.predict_proba(X_valid)[:,1]
    score = metrics.roc_auc_score(y_valid, y_pred)
    adv_scores.append(score)

    #Plot the roc_curve
    mean_auc = np.mean(adv_scores)
    fpr, tpr, _ = metrics.roc_curve(y_valid, y_pred)
    plt.plot(fpr, tpr, label = 'roc_curve (AUC = %0.4f)' % mean_auc)
    plt.plot([0,1], [0,1], linestyle = '--', color = '#ff6d00', label = 'Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'roc_curve {name_1} vs {name_2}', weight='bold')
    plt.legend()


num_features = list(ts_00.select_dtypes('number'))
plt.figure(figsize=(18,5))
plt.subplot(1,3,1)
adversarial_validation(tr_00, ts_00, 'train', 'test')
plt.subplot(1,3,2)
adversarial_validation(ts_00, or_00, 'train', 'original')
plt.subplot(1,3,3)
adversarial_validation(tr_comb, ts_00, 'train_comb', 'test')


ax = tr_00[target].value_counts().plot.pie(labels=['1', '0'], 
                                             autopct='%1.f%%',
                                             startangle=90,
                                             pctdistance=0.85,
                                             # explode=[0.01, 0.02],
                                             colors=target_colors, 
                                             radius=1.2,
                                             wedgeprops={'width': 0.5},
                                             # title='Distribution of proba > 0.5'
                                             )

plt.ylabel('')
or_00[target].value_counts().plot.pie(
    labels=['', ''],  
    startangle=90, 
    pctdistance=0.54, 
    # explode=[0.01, 0.02],
    colors=target_colors, 
    radius=0.68,wedgeprops={'width': 0.5},# title='Distribution of proba > 0.5',ax=ax
                                             )

plt.suptitle('%tages of the target classes in train and original sets')
plt.ylabel('')
plt.show()


def plot_num_distribution_grid(df, num_feats=num_feats, grouper=target, palette=target_palette, bins=50, ncols=3):
    """
    Plot histograms + boxplots for numerical features in a grid layout.
    Each feature gets two stacked subplots (hist + box) with boxplot 1/3 height of histplot.
    Both plots share the same x-axis.
    """
    df[target] = df[target].astype('category')
    nrows = int(np.ceil(len(num_feats) / ncols))

    fig = plt.figure(figsize=(ncols*6, nrows*6))
    gs = fig.add_gridspec(nrows=nrows, ncols=ncols, hspace=0.4)

    for i, feat in enumerate(num_feats):
        row = i // ncols
        col = i % ncols

        # Sub-grid with height ratio 3:1 and shared x-axis
        sub_gs = gs[row, col].subgridspec(2, 1, height_ratios=[3, 1])
        ax_hist = fig.add_subplot(sub_gs[0])
        ax_box = fig.add_subplot(sub_gs[1], sharex=ax_hist)

        # Histogram
        sns.histplot(data=df, x=feat, hue=grouper, bins=bins,
                     palette=palette, multiple="dodge", ax=ax_hist, kde=True)
        ax_hist.set_title(f"{feat}", fontsize=11.5)
        ax_hist.set_xlabel('')  # Remove x-label from histogram
        ax_hist.tick_params(axis='x', labelbottom=False)  # Hide x-ticks on histogram

        # Boxplot (horizontal to match x-axis)
        sns.boxplot(data=df, x=feat, y=grouper, palette=palette,
                    ax=ax_box,
                    flierprops=dict(marker='o', markerfacecolor='red',
                                    markeredgecolor='gold', markersize=3))
        plt.ylabel('')

    plt.tight_layout(pad=1, h_pad=0.01, w_pad=2)
    plt.show()


tr_r = tr_00.copy()
ts_r = ts_00.copy()
or_r = or_00.copy()

tr_r['dataset'] = 'train'
ts_r['dataset'] = 'test'
or_r['dataset'] = 'original'

tr_all_sets = pd.concat([tr_r, ts_r, or_r], ignore_index=True)

plot_num_distribution_grid(tr_all_sets, grouper='dataset', palette=set_palette)


plot_num_distribution_grid(tr_00, grouper=target)


plt.figure(figsize=(10, 15))
for f, feat in enumerate(cat_feats, 1):
    plt.subplot(3, 3, f)
    pd.Series({' ': 1}).plot.pie(colors=['k'], radius=0.2, shadow=False)
    tr_00[feat].value_counts().plot.pie(autopct='%.1f%%', radius=1, pctdistance=0.54, shadow=False, 
                                         textprops={'color':'black', 'rotation':True, 'weight':'bold', 'size': 8},
                                         startangle=90 , rotatelabels=True, cmap='Set2',
                                         labeldistance=0.8, wedgeprops={'width':0.7}, frame=True)
    plt.ylabel('')
    plt.title(f'%tages {feat}', color='maroon', fontsize=12, weight='bold')
    plt.tight_layout(pad=0.1, h_pad=0.01, w_pad=0.01)


import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))

for f, feat in enumerate(cat_feats, 1):
    plt.subplot(3, 2, f)
    
    # Create counts of target per category
    counts = tr_00.groupby([feat, target]).size().unstack(fill_value=0)
    
    # Horizontal bar plot
    ax = counts.plot.barh(
        stacked=True, ax=plt.gca(), legend=True, color=target_colors)
    ax.set_title(f'Count of {feat}')
    
    # Add labels inside bars
    for container in ax.containers:
        ax.bar_label(container, label_type='center')

plt.tight_layout(pad=1, h_pad=2, w_pad=1)
plt.show()


n_feats = len(cat_feats)
n_rows = (n_feats + 1) // 2   # auto rows for 2 columns
fig, axes = plt.subplots(n_rows, 2, figsize=(12, 8), sharex=True)

axes = axes.flatten()  # flatten for easy indexing

for f, feat in enumerate(cat_feats):
    ax = axes[f]
    
    # Compute counts of target per category
    counts = tr_00.groupby([feat, target]).size().unstack(fill_value=0)
    
    # Convert to percentages row-wise
    percentages = counts.div(counts.sum(axis=1), axis=0) * 100
    
    # Horizontal stacked bar plot with custom colors
    percentages.plot.barh(
        stacked=True, 
        ax=ax, 
        legend=(f==0),   # show legend only on first subplot
        color=target_colors
    )
    
    ax.set_title(f'Percentage of {feat}')
    ax.set_xlim(0, 100)  # shared axis ensures all bars span 0â€“100
    ax.set_xlabel("Percentage (%)")
    
    # Add percentage labels inside bars
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", label_type='center')

# Hide any unused subplot slots
for j in range(f+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout(pad=1, h_pad=2, w_pad=1)
plt.show()


tr_corr = np.abs(tr_00.corr(numeric_only=True))
mask = np.triu(np.ones_like(tr_corr, dtype=bool))
plt.figure(figsize=(9, 9))
sns.heatmap(tr_corr, annot=True, fmt='.1f', cmap=my_cmap, mask=mask, cbar=False)
plt.show()


# sns.pairplot(tr_00.sample(500), hue=target, palette=target_palette)
# plt.show()


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
    for col, mapping in cat_mappings.items():
        # Replace with numeric codes; keep any unknowns as-is
        df[col] = df[col].replace(mapping)

        # Optional: convert to nullable integer dtype (handles any non-mapped leftovers)
        # If you are sure everything maps to integers, you can use int instead.
        df[col] = df[col].astype('Int64')  # nullable integer dtype


tr_00.head()


# Define function to handle outliers
def remove_outliers(df):
    df = df.copy()
    for col in num_feats:
        if df[col].nunique()>20:
            IQR = iqr(df[col])  # calculate the interquartile range
            df[col] = np.clip(df[col], (np.quantile(df[col], 0.25) - 1.51*IQR), (np.quantile(df[col], 0.75) + 1.51*IQR)) # clip the outliers in the range (25, 75)quantile -or+ 1.5 IQ
    return df

# Remove outliers from the various datasets
tr_01 = remove_outliers(tr_00)
ts_01 = remove_outliers(ts_00)
or_01 = remove_outliers(or_00)
tr_comb_01 = remove_outliers(tr_comb)


plot_num_distribution_grid(tr_01, grouper=target)


run_feat_eng = True
cross_combine_cat_feats = True

if run_feat_eng:
    # create new feature
    for df in [tr_01, ts_01, or_01, tr_comb_01]:
        df['diasto_to_systo'] = np.divide(df['diastolic_bp'], df['systolic_bp'])
        df['hdl_cholesterol_to_total'] = np.divide(df['hdl_cholesterol'], df['cholesterol_total'])
        df['ldl_cholesterol_to_total'] = np.divide(df['ldl_cholesterol'], df['cholesterol_total'])
        df['ldl_to_hld'] = np.divide(df['ldl_cholesterol'], df['hdl_cholesterol'])
        df['waistHip_to_bmi'] = np.divide(df['waist_to_hip_ratio'], df['bmi'])
        df['heartRate_to_bmi'] = np.divide(df['heart_rate'], df['bmi'])
        df['triglycerides_to_cholesterol'] = np.divide(df['triglycerides'], df['cholesterol_total'])
        df['histories'] = df['family_history_diabetes'] + df['hypertension_history'] + df['cardiovascular_history']
        df['cholesterol_*_systo'] = df['cholesterol_total']*df['systolic_bp']/100
        # df['bmi_class'] = pd.cut(df['bmi'], [0, 18.5, 25, 30, 35, 40, 100], labels=[1, 2, 3, 4, 5, 6]).astype('int')
        # df['triglycerides_class'] = pd.cut(df['triglycerides'], [0, 150, 199, 1000], labels=[1, 2, 3]).astype('int')
        # df['age_group'] = pd.cut(df['age'], [0, 18, 25, 35, 45, 60, 100], labels=[1, 2, 3, 4, 5, 6]).astype('int')
        # df.drop(columns=['bmi', 'age'], inplace=True)
        if cross_combine_cat_feats:
            # Generate unique pairs without repetition
            for feat_1, feat_2 in combinations(cat_feats, 2):
                # Concatenate as strings to create interaction feature
                df[f'{feat_1}_{feat_2}'] = df[feat_1].astype(str) + '_' + df[feat_2].astype(str)
            else:
                pass
            
    else:
        pass
    
tr_00.head(3)


include_external = False
if include_external:
    X = tr_comb_01.copy()
else:
    X = tr_01.copy()
y = X.pop(target)


import os

# Check if GPU is available (CatBoost uses CUDA)
def gpu_available():
    try:
        # CatBoost has a helper function
        return catboost.get_gpu_device_count() > 0
        print(f"Training with {task_type}")
    except Exception:
        return False

task_type = "GPU" if gpu_available() else "CPU"


# model = CatBoostClassifier(
#     n_estimators=12000,
#     depth=3,
#     learning_rate=0.1,
#     eval_fraction=0.2,
#     eval_metric="AUC",
#     random_seed=seed,
#     use_best_model=True,
#     verbose=200,
#     early_stopping_rounds=40,
#     # task_type="GPU"
# )

model = CatBoostClassifier(
    iterations=3000,
    learning_rate=0.02,
    depth=8,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=42,
    l2_leaf_reg=5,
    early_stopping_rounds=40,
    eval_fraction=0.2,
    bagging_temperature=0.3,
    random_strength=1.5,
    # od_wait=80,
    task_type="GPU",
    verbose=200
)



cat_features = X.select_dtypes(exclude='number').columns.tolist()


model.fit(X, y, cat_features=cat_features)


importances = pd.DataFrame(
    {'feat_importance':model.get_feature_importance()}, 
    index=X.columns).sort_values(by='feat_importance')

importances.plot.barh(figsize=(10, 8), color='green')
plt.title('Feature importance for CatBoostClassifier')
plt.show()


# Let's select n best features by feature_importance
top = 28

top_features = importances.tail(top).index.tolist()
r = [f'{c} : {feature}' for c, feature in enumerate(reversed(top_features), 1)]
print(f'Here are the {len(top_features)} features that are selected base on their importances\n')
display(r)

# Reduced train and test data to just the selected features
X_reduced = X[top_features]
ts_reduced = ts_01[top_features]


final_model = clone(model)

final_model.fit(X_reduced, y, cat_features=X_reduced.select_dtypes(exclude='number').columns.tolist())


expl = shap.Explainer(final_model, feature_names=X_reduced.columns.tolist())

shap_val = expl(X_reduced)

shap.plots.beeswarm(shap_val, max_display=20)


for n in [456, -4, -2]:
    shap.plots.waterfall(shap_val[n], max_display=15, show=True)


# for feat in X_reduced.columns:
#     shap.plots.scatter(shap_val[:, feat], color=shap_val)


n_splits = 6
spliter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

# Store out-of-fold predictions
oof_preds = []
oof_true = []

plt.figure(figsize=(6, 5))      
for f, (tr_ind, va_ind) in enumerate(spliter.split(X_reduced, y), 1):
    print(12*'ðŸš¨',  f' Fitting Fold {f} of  {n_splits} ', 12*'ðŸš¨')
    X_tr, X_va = X.iloc[tr_ind], X.iloc[va_ind]
    y_tr, y_va = y.iloc[tr_ind], y.iloc[va_ind]

    # Clone the model before fitting
    clf = clone(model)
    clf.fit(X_tr, y_tr, cat_features=cat_features)

    preds = clf.predict_proba(X_va)[:, 1]

    # Save for overall ROC
    oof_preds.extend(preds)
    oof_true.extend(y_va)

    # Per-fold AUC
    score = metrics.roc_auc_score(y_va, preds)
    print(52*' ' + f' â€¢â€¢â€¢> Fold_{f} AUC: {score:.6f} âœ…\n\n')

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


ts_proba = final_model.predict_proba(ts_reduced)[:,1]

sb_00[target] = ts_proba


sb_00.head()


threshold = 0.54
median_val = sb_00[target].median()

plt.subplot(121)
sb_00[target].plot.hist(bins=50, color='#a0785a', 
                        figsize=(10, 4), edgecolor='lightgrey', 
                        title='Hist of predicted_proba in test set')
plt.xlabel('Predicted Proba')
plt.axvline(x=sb_00[target].median(), color='r', linestyle='--', linewidth=1)
plt.text(median_val-0.2, plt.ylim()[1]*0.93, f'Median = {median_val:.2f}',
         color='#d9004c', ha='center', va='bottom', fontsize=12)

plt.subplot(122)
(sb_00[target] > threshold).value_counts().plot.pie(labels=['1', '0'], 
                                             autopct='%1.1f%%',
                                             startangle=90,
                                             explode=[0.01, 0.02],
                                             colors=target_colors, 
                                             radius=1.2,
                                             wedgeprops={'width': 0.7},
                                             title=f'Distribution of proba: threshold of {threshold}'
                                             )

plt.ylabel('')
plt.show()


sb_00.to_csv('submission.csv', index=False)
print('âœ¨âœ¨âœ¨ Ready to submit! âœ¨âœ¨âœ¨')

