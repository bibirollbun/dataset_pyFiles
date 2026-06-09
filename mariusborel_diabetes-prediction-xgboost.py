import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import iqr

from sklearn import metrics
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import (MaxAbsScaler, MinMaxScaler, Normalizer,
                                   PowerTransformer, QuantileTransformer, LabelEncoder,
                                   RobustScaler, StandardScaler, minmax_scale,
                                   OneHotEncoder, FunctionTransformer)
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.base import clone

from itertools import combinations

import xgboost as xgb

import warnings
warnings.filterwarnings('ignore')

# Define your custom colors
target_colors = ['orange', '#fc5a8d']
set_colors = ['#ED6AFF', '#c6ff00', '#26c6da']
# Create a seaborn palette
set_palette = sns.color_palette(set_colors)
target_palette = sns.color_palette(target_colors)


# Define a gradient from blue â†’ white â†’ red
colors = ["#eceff1", "#fce4ec", "#f8bbd0", "#f48fb1", "#f06292", "#ec407a"]
my_cmap = LinearSegmentedColormap.from_list("heatmap gradient cmap", colors)


# # Set Seaborn theme with dark grid and brighter palette
# sns.set_theme(style="darkgrid", font_scale=0.9)

# Update matplotlib parameters for brighter dark theme
plt.rcParams.update({
    'axes.facecolor': 'white',  
    'figure.facecolor': 'white',
    'text.color': '#1a237e',  
    'axes.labelcolor': '#1a237e', 
    'xtick.color': '#1a237e',
    'ytick.color': '#1a237e',
    'grid.color': 'lightgrey',
    'axes.edgecolor': '#1a237e', 
    'axes.grid': True,
})

seed=68

# verify the versions
print(f'pandas version: {pd.__version__}')
print(f'numpy version: {np.__version__}')
print(f'seaborn version: {sns.__version__}')


my_cmap


print("datasets colors")
set_palette


print("targets colors")
target_palette


tr_00 = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv', index_col='id')
ts_00 = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv', index_col='id')
sb_00 = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')

or_00 = pd.read_csv('/kaggle/input/diabetes-health-indicators-dataset/diabetes_dataset.csv')[tr_00.columns]

# or_00 = pd.read_csv('')

target = 'diagnosed_diabetes'



tr_00.head(5)


or_00.head()


print(f'shape of train set: {tr_00.shape}\nshape of test set: {ts_00.shape}\nshape of original set: {or_00.shape}')


for df in [tr_00, ts_00, or_00]:
    print(df.isna().sum().sum())
print('None of the three datasets is missing values!')


tr_00.info()


for df in [tr_00]:
    df[target] = df[target].astype('category')


tr_00.describe().T.style.background_gradient(cmap=my_cmap)


tr_00.describe(exclude='number').T


num_feats = [feat for feat in ts_00.select_dtypes(include='number').columns.tolist() if ts_00[feat].nunique()>2]


bool_feats = [feat for feat in ts_00.select_dtypes('number') if ts_00[feat].nunique()==2]
cat_feats = ts_00.select_dtypes(exclude='number').columns.tolist()

cat_and_bool_feat = cat_feats + bool_feats


plt.figure(figsize=(10, 4))

plt.subplot(121)
ax = sns.countplot(tr_00, x=target, palette=target_colors)
ax.grid(False)
for count in ax.containers:
    ax.bar_label(count, label_type='center')

plt.subplot(122)
tr_00[target].value_counts().sort_values(ascending=True).plot.pie(labels=['1', '0'], 
                                     autopct='%1.1f%%', \
                                     explode=[0.05, 0.05],
                                     colors=target_colors,
                                     startangle=90,
                                     radius=1.1,
                                     wedgeprops={'width': 0.7},
                                     # title='Proportion of classes in train data'
                                     )
plt.ylabel('')

plt.suptitle('Count of classes in train data', color='maroon', fontsize=14)

plt.show()


tr_comb = pd.concat([tr_00, or_00], ignore_index=True)
print('Combine the train and original datasets.')


# Define a function to perform the adversarial validation of two datasets
def adversarial_validation(df_1, df_2, name_1, name_2):
    adv_df_1 = df_1[num_features].copy()
    adv_df_2 = df_2[num_features].copy()


    # label the test and train data with 0 and 1 (it doesn't really matter which is which)
    adv_df_1 = adv_df_1.assign(adv=1)
    adv_df_2 = adv_df_2.assign(adv=0)

    # combine the training and test data into one big dataset
    combined = pd.concat([adv_df_1, adv_df_2], axis=0)

    # Shuffle
    combined = combined.sample(frac=1, random_state=64)

    # perform the binary classification, for example using XGboost
    X_combined = combined.drop('adv', axis=1)
    y_combined = combined.adv
    
    # Define the classifier
    clf = xgb.XGBClassifier(verbose=0, n_estimators=50)

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
    plt.plot([0,1], [0,1], linestyle = '--', color = '#d84315', label = 'Random Guess')
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
        ax_hist.set_title(f"{feat} distribution", fontsize=11.5, color='maroon')
        ax_hist.set_xlabel('')  # Remove x-label from histogram
        ax_hist.tick_params(axis='x', labelbottom=False)  # Hide x-ticks on histogram

        # Boxplot (horizontal to match x-axis)
        sns.boxplot(data=df, x=feat, y=grouper, palette=palette,
                    ax=ax_box,
                    flierprops=dict(marker='o', markerfacecolor='red',
                                    markeredgecolor='gold', markersize=3))
        plt.ylabel('')

    plt.tight_layout(pad=1, h_pad=0.2, w_pad=2)
    plt.show()


# Save copies of the datasets
tr_r = tr_00.copy()
ts_r = ts_00.copy()
or_r = or_00.copy()

# Add an identification columns to each features
tr_r['dataset'] = 'train'
ts_r['dataset'] = 'test'
or_r['dataset'] = 'original'

# Combine the datasets
tr_all_sets = pd.concat([tr_r, ts_r, or_r], ignore_index=True)

# Plot the distribution chart for each dataset
plot_num_distribution_grid(tr_all_sets, grouper='dataset', palette=set_palette)


plot_num_distribution_grid(tr_00, grouper=target)


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
    ax.grid(False)
    
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
    ax.grid(False)
    
    # Add percentage labels inside bars
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", label_type='center')

# Hide any unused subplot slots
for j in range(f+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout(pad=1, h_pad=2, w_pad=1)
plt.show()


tr_corr_abs = np.abs(tr_00.corr(numeric_only=True))
mask = np.triu(np.ones_like(tr_corr_abs, dtype=bool)) | (tr_corr_abs<0.01) 
plt.figure(figsize=(12, 12))
ax = sns.heatmap(tr_corr_abs, annot=True, fmt='.1f', cmap=my_cmap, cbar=False, mask=mask)
ax.grid(False)
plt.show()


# sns.pairplot(tr_00.sample(600), hue=target, palette=target_palette, aspect=0.8)
# plt.show()


# Define function to handle outliers
def remove_outliers(df):
    df = df.copy()
    for col in num_feats:
        if df[col].nunique()>20:
            IQR = iqr(df[col])  # calculate the interquartile range
            df[col] = np.clip(df[col], 
                              (np.quantile(df[col], 0.25) - 1.51*IQR), 
                              (np.quantile(df[col], 0.75) + 1.51*IQR)
                             ) # clip the outliers in the range (25, 75)quantile -or+ 1.5 IQ
    return df

# Remove outliers from the various datasets
tr_01 = remove_outliers(tr_00)
ts_01 = remove_outliers(ts_00)
or_01 = remove_outliers(or_00)
tr_comb_01 = remove_outliers(tr_comb)


run_feat_eng = True
cross_combine_cat_feats = True

if run_feat_eng:
    # create new feature
    for df in [tr_01, ts_01, or_01, tr_comb_01]:
        df['diasto_to_systo'] = np.divide(df['diastolic_bp'], df['systolic_bp'])
        df['hdl_cholesterol_to_total'] = np.divide(df['hdl_cholesterol'], df['cholesterol_total'])
        df['ldl_cholesterol_to_total'] = np.divide(df['ldl_cholesterol'], df['cholesterol_total'])
        df['waistHip_to_bmi'] = np.divide(df['waist_to_hip_ratio'], df['bmi'])
        df['heartRate_to_bmi'] = np.divide(df['heart_rate'], df['bmi'])
        df['triglycerides_to_cholesterol'] = np.divide(df['triglycerides'], df['cholesterol_total'])
        df['histories'] = df['family_history_diabetes'] + df['hypertension_history'] + df['cardiovascular_history']
        df['bmi_class'] = pd.cut(df['bmi'], [0, 18.5, 25, 30, 35, 40, 100], labels=[1, 2, 3, 4, 5, 6])
        df.drop(columns=['bmi'], inplace=True)
        if cross_combine_cat_feats:
            # Generate unique pairs without repetition
            for feat_1, feat_2 in combinations(cat_and_bool_feat, 2):
                # Concatenate as strings to create interaction feature
                df[f'{feat_1}_{feat_2}'] = df[feat_1].astype(str) + '_' + df[feat_2].astype(str)
            else:
                pass
            
    else:
        pass
    
tr_00.head(3)


# Get the list of cat features from the engineered data
cat_feats_eng = ts_01.select_dtypes(exclude='number').columns.tolist()


for df in [tr_01, ts_01, or_01, tr_comb_01]:
    for feat in cat_feats_eng:
        df[feat] = df[feat].astype('category')


ts_00.info()


# Decide if the external data should be included to the train set
include_external = True

# Split the train set into data and target
if include_external:
    print("The original dataset is combined to the train set.")
    X = tr_comb_01.copy() # use the train_orig combined dataset
else:
    print("The train set is used alone without including the original data.")
    X = tr_01.copy() # use only the train set
y = X.pop(target)

# Prepare a separate original data and target
X_or = or_01.copy()
y_or = X_or.pop(target)


# Define parameters
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'eta': 0.02,
    'max_depth': 3,
    'gamma': 2,
    # 'learning_rate': 0.02,
    'min_child_weight': 3,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'lambda': 4,   # L2 regularization
    'alpha': 1,    # L1 regularization
    'grow_policy': 'depthwise',
    # 'tree_method': 'gpu_hist',
    'random_state': seed,
}

print("Here are the hyperparameters that are used:")
params


n_splits = 6
spliter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

# Store test predictions
test_pred_proba = pd.DataFrame()
# Store out-of-fold predictions
oof_preds = []
oof_true = []

plt.figure(figsize=(4.5, 4))
for f, (tr_ind, va_ind) in enumerate(spliter.split(X, y), 1):
    print(f'ğŸš¨ğŸš¨ğŸš¨ğŸš¨ğŸš¨ğŸš¨   Fitting Fold {f} of  {n_splits}  ğŸš¨ğŸš¨ğŸš¨ğŸš¨ğŸš¨ğŸš¨')

    # Split train/validation
    X_tr, X_va = X.iloc[tr_ind], X.iloc[va_ind]
    y_tr, y_va = y.iloc[tr_ind], y.iloc[va_ind]

    # Convert to DMatrix
    dtrain = xgb.DMatrix(X_tr, label=y_tr, enable_categorical=True)
    dval   = xgb.DMatrix(X_va, label=y_va, enable_categorical=True)
    dtest  = xgb.DMatrix(ts_01, enable_categorical=True)
    dorig = xgb.DMatrix(X_or, label=y_or, enable_categorical=True)

    # Train with early stopping
    evals = [(dtrain, 'train'), (dval, 'validation')]
   
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=4000,
        evals=evals,
        early_stopping_rounds=20,
        verbose_eval=200
    )


    # Predict on validation with best iteration
    pred_va = model.predict(dval, iteration_range=(0, model.best_iteration))
    score_va = metrics.roc_auc_score(y_va, pred_va)
    print(f'\nFold_{f} ==> auc: {score_va:.6f} on validation data âœ…')

    # Save for overall ROC
    oof_preds.extend(pred_va)
    oof_true.extend(y_va)

    # Predict on the original data with best iteration
    pred_or = model.predict(dorig, iteration_range=(0, model.best_iteration))
    score_or = metrics.roc_auc_score(y_or, pred_or)
    print(f'Fold_{f} ==> auc: {score_or:.6f} on original data   âš ï¸�\n')

    # Predict on test set with best iteration
    test_pred_proba[f'pred_proba_oof_{f}'] = model.predict(dtest, iteration_range=(0, model.best_iteration))

    # Plot ROC curve for each of the folds
    fpr, tpr, _ = metrics.roc_curve(y_va, pred_va)
    plt.plot(fpr, tpr, label=f'Fold {f} AUC  = {score_va:.5f}')
    
# Overall ROC curve
overall_auc = metrics.roc_auc_score(oof_true, oof_preds)
fpr, tpr, _ = metrics.roc_curve(oof_true, oof_preds)
plt.plot(fpr, tpr, color='green', linewidth=2,
         label=f'Overall AUC = {overall_auc:.5f}')
# Diagonal baseline
plt.plot([0, 1], [0, 1], color='grey', linestyle='--')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.title('ROC Curves of All Folds + Overall', 
          color='maroon', fontsize=11, weight='bold')
plt.tight_layout()
plt.show()


dtrain = xgb.DMatrix(X, label=y, enable_categorical=True)

# Fit the model on the entire train data
model = xgb.train(
    params,
    dtrain,
    num_boost_round=4000,
    evals=evals,
    early_stopping_rounds=20,
    verbose_eval=200
)


pred_proba = model.predict(dtest, iteration_range=(0, model.best_iteration))


test_pred_proba['average_proba'] = test_pred_proba.mean(axis=1)
test_pred_proba['prediction'] = pred_proba

test_pred_proba.head()


use_one_fold = True

fold = 1 # Which fold should be used

if use_one_fold:
    final_prediction = f'pred_proba_oof_{fold}' # use prediction from one of the folds
else:
    final_prediction = 'average_proba' # use the average of folds predictions 

sb_00[target] = test_pred_proba[final_prediction]

print(f"We are using {final_prediction} as the final prediction for aour test data")


sb_00.head()


# The threshold for assigning the class from predicted probabilities
threshold = 0.54
median_val = sb_00[target].median()

plt.subplot(131)
ax = sb_00[target].plot.hist(bins=40, color='gray', 
                        figsize=(15, 4), edgecolor='k', 
                        title='Hist of predicted_proba in test set')
ax.grid(False)
plt.xlabel('Predicted Proba')
plt.axvline(x=median_val, color='r', linestyle='--', linewidth=2)
plt.text(median_val-0.2, plt.ylim()[1]*0.93, f'Median = {median_val:.2f}',
         color='#d9004c', ha='center', va='bottom', fontsize=12)

plt.subplot(132)
(sb_00[target] > threshold).value_counts().sort_values(ascending=True).plot.pie(labels=['1', '0'], 
                                             autopct='%1.1f%%', 
                                             explode=[0.05, 0.05],
                                             colors=target_colors,
                                             startangle=90,
                                             radius=1.2,
                                             wedgeprops={'width': 0.7},
                                             title=f'Target proportion in test: threshold > {threshold}'
                                             )
plt.ylabel('')

plt.subplot(133)
tr_00[target].value_counts().sort_values(ascending=True).plot.pie(labels=['1', '0'], 
                                             autopct='%1.1f%%', 
                                             explode=[0.05, 0.05],
                                             colors=target_colors, 
                                             startangle=90,
                                             radius=1.2,
                                             wedgeprops={'width': 0.7},
                                             title='Target proportion in train data'
                                             )
plt.ylabel('')
plt.show()


sb_00.to_csv('submission.csv', index=False)
print('âœ¨âœ¨âœ¨ Ready to submit! âœ¨âœ¨âœ¨')

