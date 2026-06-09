# !pip install /kaggle/input/pip-install-lifelines/autograd-1.7.0-py3-none-any.whl
# !pip install /kaggle/input/pip-install-lifelines/autograd-gamma-0.5.0.tar.gz
# !pip install /kaggle/input/pip-install-lifelines/interface_meta-1.3.0-py3-none-any.whl
# !pip install /kaggle/input/pip-install-lifelines/formulaic-1.0.2-py3-none-any.whl
# !pip install /kaggle/input/pip-install-lifelines/lifelines-0.30.0-py3-none-any.whl


import pandas as pd
import numpy as np

import lifelines
from lifelines import KaplanMeierFitter, NelsonAalenFitter
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import KFold, StratifiedKFold
from lifelines.utils import concordance_index
import lightgbm as lgb
from termcolor import colored
from scipy.stats import skew, kurtosis

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
plt.rcParams['axes.facecolor'] = '#e0f7fa'

import warnings
warnings.filterwarnings('ignore')


# Load the data
submission_df = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv", index_col="ID")
data_dictionary_df = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/data_dictionary.csv")
train_df = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/train.csv", index_col="ID")
test_df = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv", index_col="ID")


submission_df


data_dictionary_df


train_df.head()


train_df.info()


class EDAPlots:
        
    def plot_missing_values(self, data, name):

        # Calculate the percentage of missing values for each column
        missing_percentage = data.isnull().mean() * 100
            
        
        # Plotting the missing values percentage
        plt.figure(figsize=(14, 10))
        
        # Horizontal bar plot with custom color and edgecolor
        ax = missing_percentage.plot(kind='barh', color='#00796B', edgecolor='black')
        
        for i, v in enumerate(missing_percentage):
            ax.text(v +1, i, f"{v:.2f}%", va="center", ha="left", fontsize=8, color='#070808')
        
        # Adding labels and title
        plt.title(f'Percentage of Missing Values (%) per Column in {name} Dataset', fontsize=16, color='black')
        plt.xlabel('Percentage of Missing Values (%)', fontsize=14)
        plt.ylabel('Columns', fontsize=14)
        
        # Set background color for the plot
        plt.gca().set_facecolor('#e0f7fa')
        
        # Enable grid with customized style
        # plt.grid(True, color='#00796B', linestyle='--', linewidth=0.8)
        
        # Rotate x-axis labels to make them more readable
        plt.xticks(rotation=90, fontsize=12)
        
        # Adjust layout for better presentation
        plt.tight_layout()
        
        # Show the plot
        plt.show()

    def plot_categorical_distribution(self, data, column_names):
        # Ensure you have an even number of subplots
        num_columns = len(column_names)
        num_rows = (num_columns // 2) + (num_columns % 2)  # Number of rows, add an extra row if odd
        num_cols = 2  # Fixed 2 columns per row
        
        # Set up the figure and subplots (dynamically based on the number of columns)
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, num_rows * 5), facecolor="#e0f7fa")
        
        # Flatten axes array for easier indexing (in case of multiple rows)
        axes = axes.flatten()
    
        # Loop through the column names and create a subplot for each
        for i, column_name in enumerate(column_names):
            # Generate a custom color palette with more distinct colors
            num_categories = len(data[column_name].unique())
            colors = sns.color_palette("viridis", num_categories)  # Using 'viridis' for better distinction
    
            # Calculate absolute counts and sort them in descending order
            value_counts = data[column_name].value_counts()  # Absolute counts
            total = value_counts.sum()  # Total count for the column
            unique_values = value_counts.index.tolist()
            percentages = ((value_counts / total) * 100).round(2).tolist()  # Percentages
            
            # Sort the unique values by their counts in descending order
            sorted_order = value_counts.index.tolist()
    
            # Bar plot for categorical distribution with the custom color palette
            ax = axes[i]  # Use the appropriate subplot axis
            sns.countplot(
                y=column_name, 
                data=data, 
                palette=colors, 
                order=sorted_order,  # Sort bars in descending order
                ax=ax, 
                width=0.6
            )  # Reduce the bar width
    
            # Adding title and labels
            ax.set_title(f'Distribution of {column_name}', fontsize=12)
            ax.set_xlabel('Count', fontsize=10)
            ax.set_ylabel(column_name, fontsize=10)
    
            # Ensure bars are annotated in the correct order based on sorted categories
            for p, category in zip(ax.patches, sorted_order):
                count = value_counts[category]
                percent = (count / total) * 100
                ax.annotate(f'{count} ({percent:.2f}%)',
                            (p.get_width() + 0.1, p.get_y() + p.get_height() / 2),
                            ha='left', va='center', fontsize=10, color='black')
    
            # Remove default spines
            sns.despine(left=True, bottom=True, ax=ax)
    
            # Remove y-ticks
            ax.get_yaxis().set_visible(False)  # Make y-axis invisible
    
            # Create custom handles and labels for the legend
            handles = [
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], markersize=8)
                for i in range(len(unique_values))
            ]
            labels = [f"{unique_values[i]} ({percentages[i]}%)" for i in range(len(unique_values))]
    
            # Place legend outside the plot on the right
            ax.legend(handles=handles, labels=labels, title=column_name, loc='upper left', fontsize=10, bbox_to_anchor=(1.15, 1))
    
            # Set background color for the plot
            ax.set_facecolor('#e0f7fa')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(2)
            ax.spines['bottom'].set_linewidth(2)
            ax.grid(visible=True)
    
        # Remove unused axes if any (in case of an odd number of columns)
        for j in range(num_columns, len(axes)):
            axes[j].axis('off')
        
        # Tight layout to avoid overlap
        plt.tight_layout()
        plt.show()
  
    def plot_heatmap(self, data, columns, palette, data_name, mask=False, figsize=(14, 12)):
        # Compute the correlation matrix
        corr_matrix = data[columns].corr()
        if mask:
            # Create a mask for the upper triangle:
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        # Set up figure
        plt.figure(figsize=figsize)

        # Draw the heatmap
        sns.heatmap(corr_matrix, cmap=palette, mask=mask, annot=True, fmt=".2f",
                   square=True, cbar_kws={"shrink": .6}, linewidths=.5)
        plt.title(f"Heatmap for {data_name} data")
        plt.show()


    def plot_kde_distribution(self, preds, title):
        plt.figure(figsize=(10,6))
        for key, value in preds.items():
            label = f"{key}, Mean={np.mean(value):.2f}, Std={np.std(value):.2f}, Skew={skew(value):.2f}, Kurt={kurtosis(value):.2f}"
            ax = sns.kdeplot(value, label=label, alpha=0.4, shade=True)
        plt.title(f'{title}')
        plt.xlabel('Prediction Value')
        plt.ylabel('Density')
        legend = plt.legend() # bbox_to_anchor=(1.75, 1.0)
        # Set alpha for the legend background
        legend.get_frame().set_alpha(0.5)  # 0.5 for 50% transparency
        plt.show()


eda = EDAPlots()


eda.plot_missing_values(data=train_df, name="Train")


eda.plot_missing_values(data=test_df, name="Test")


CAT_FEATURES = train_df.select_dtypes(include=["object", "category"]).columns.values


train_df[CAT_FEATURES].describe().T.style.background_gradient(cmap='GnBu', subset=['unique', 'freq'])


eda.plot_categorical_distribution(data=train_df, column_names=CAT_FEATURES)


NUM_FEATURES = train_df.select_dtypes(include=["number"]).columns.values



print(f"Number of numerical features: {len(NUM_FEATURES)}")


train_df[NUM_FEATURES].describe().drop("year_hct", axis=1).T.drop('count', axis=1) \
                        .style.background_gradient(cmap='GnBu', subset=['mean', 'std'])


NUM_FEATURES


eda.plot_heatmap(data=train_df,
                  columns=NUM_FEATURES,
                  palette="GnBu",
                  data_name="Train",
                  mask=True)


eda.plot_heatmap(data=test_df,
                  columns=NUM_FEATURES[:-2],
                  palette="GnBu",
                  data_name="Test",
                  mask=True)


TARGETS = ['efs', 'efs_time', 'naf_label', 'km_label']


# Create Cumulative hazard at times label
naf = NelsonAalenFitter()
naf.fit(train_df['efs_time'], train_df['efs'])
train_df['naf_label'] = -naf.cumulative_hazard_at_times(train_df['efs_time']).values
train_df.loc[train_df['efs'] == 0, "naf_label"] -= 0.1


# Create Survival Function at Times label
kmf = KaplanMeierFitter()
kmf.fit(train_df['efs_time'], train_df['efs'])
train_df["km_label"] = kmf.survival_function_at_times(train_df["efs_time"]).values
train_df.loc[train_df["efs"] == 0, "km_label"] -= 0.1


TARGETS[1:]


from sklearn.preprocessing import QuantileTransformer

fig, axes = plt.subplots(3, 3, figsize=(15, 15))

target_plots_for_kde = TARGETS[1:]

# Transform Log1p
train_df['efs_time_log1p'] = np.log1p(train_df['efs_time'])
train_df['naf_label_log1p'] = np.log1p(train_df['naf_label'])
train_df['km_label_log1p'] = np.log1p(train_df['km_label'])

TARGETS += ['efs_time_log1p', 'naf_label_log1p', 'km_label_log1p']
# Quantile Transformation
qt = QuantileTransformer(output_distribution='normal')
train_df['efs_time_quantile'] = qt.fit_transform(train_df['efs_time'].values.reshape(-1, 1))
train_df['naf_label_quantile'] = qt.fit_transform(train_df['naf_label'].values.reshape(-1, 1))
train_df['km_label_quantile'] = qt.fit_transform(train_df['km_label'].values.reshape(-1, 1))

TARGETS += ['efs_time_quantile', 'naf_label_quantile', 'km_label_quantile']

# First row: Original KDE plots
for i, label in enumerate(['efs_time', 'naf_label', 'km_label']):
    sns.kdeplot(data=train_df, x=label, hue='efs', ax=axes[0, i], shade=True)
    axes[0, i].set_title(f"Original Target: {label} ")

# Second row: Log1p KDE plots
for i, label in enumerate(['efs_time_log1p', 'naf_label_log1p', 'km_label_log1p']):
    sns.kdeplot(data=train_df, x=label, hue='efs', ax=axes[1, i], shade=True)
    axes[1, i].set_title(f"Log1p Transformed: {label}")

# Third row: Quantile KDE plots
for i, label in enumerate(['efs_time_quantile', 'naf_label_quantile', 'km_label_quantile']):
    sns.kdeplot(data=train_df, x=label, hue='efs', ax=axes[2, i], shade=True)
    axes[2, i].set_title(f"Quantile Transformed: {label}")

plt.tight_layout()
plt.show()


# Separate train and test sets
train_df["split"] = 1
test_df['split'] = 0

# Concatenate train and test
concat_df = pd.concat([train_df, test_df], axis=0)

# Remove Split columns from original dataframes
train_df = train_df.drop("split", axis=1)
test_df = test_df.drop("split", axis=1)


## Comment it out only for speeding running notebook
import pandas as pd

# Compute the correlation matrix
all_num_cols = list(NUM_FEATURES) + ["naf_label", "km_label"]
# drop_columns=['hla_low_res_6', 'hla_low_res_8', 'hla_low_res_10', 'hla_high_res_6']
corr_matrix = concat_df[all_num_cols].corr()

# Print the entire correlation matrix
print("Correlation Matrix:")
# print(corr_matrix)

# Optionally, if you only want to print values greater than a threshold (e.g., |0.7|) to spot strong correlations:
threshold = 0.90
high_corr_pairs = corr_matrix.unstack().sort_values(ascending=False)

# Filter pairs that have correlation greater than the threshold, excluding self-correlations (diagonal)
high_corr_pairs = high_corr_pairs[(abs(high_corr_pairs) > threshold) & (high_corr_pairs != 1)]

# Print high correlation pairs
print(f"\nHighly Correlated Pairs (Correlation > {threshold}):")
high_corr_pairs


# Age Difference Between Donor and Recipient
concat_df['age_diff'] = concat_df['donor_age'] - concat_df['age_at_hct']
concat_df['age_ratio'] = concat_df['donor_age'] / concat_df['age_at_hct']


eda.plot_kde_distribution(preds={'hla_match_a_high': concat_df['hla_match_a_high'].dropna(),
                                 'hla_match_b_high': concat_df['hla_match_b_high'].dropna(),
                                 'hla_match_c_high': concat_df['hla_match_c_high'].dropna()},
                         title="HLA Match Quality columns")


concat_df['hla_match_quality'] = (
    concat_df[['hla_match_a_high', 'hla_match_b_high', 'hla_match_c_high']].sum(axis=1))


eda.plot_kde_distribution(preds={'hla_match_quality': concat_df['hla_match_quality'].dropna()},
                          title='High HLA Match Quality Combined')


# concat_df = concat_df.drop(['hla_match_a_high', 'hla_match_b_high', 'hla_match_c_high'], axis=1)


concat_df['sex_match'].astype(str) + "_" + pd.qcut(concat_df['age_at_hct'], 4).astype(str)


eda.plot_kde_distribution(preds={'hla_match_a_low': concat_df['hla_match_a_low'].dropna(),
                                 'hla_match_b_low': concat_df['hla_match_b_low'].dropna(),
                                 'hla_match_c_low': concat_df['hla_match_c_low'].dropna()},
                         title="Low HLA Match Quality columns")


concat_df['hla_low_match_quality'] = (concat_df['hla_match_a_low'] +
                                      concat_df['hla_match_b_low'] +
                                      concat_df['hla_match_c_low'])


eda.plot_kde_distribution(preds={'hla_low_match_quality': concat_df['hla_low_match_quality'].dropna()},
                          title='Low HLA Match Quality Combined')


# concat_df = concat_df.drop(['hla_match_a_low', 'hla_match_b_low', 'hla_match_c_low'], axis=1)


concat_df['hla_low_res_all'] = (concat_df['hla_low_res_6'] + concat_df['hla_low_res_8'] + concat_df['hla_low_res_10'])


# concat_df = concat_df.drop(['hla_low_res_6', 'hla_low_res_8', 'hla_low_res_10'], axis=1)


concat_df['hla_high_res_all'] = (concat_df['hla_high_res_6'] +
                                  concat_df['hla_high_res_8'] +
                                  concat_df['hla_high_res_10'])


# concat_df = concat_df.drop(['hla_high_res_6', 'hla_high_res_8', 'hla_high_res_10'], axis=1)


all_num_cols =  list(concat_df.select_dtypes(include=["float64", "float32"]).columns)


all_num_cols = [col for col in all_num_cols if col not in TARGETS]


train_concat_df = concat_df[concat_df['split'] == 1].drop('split', axis=1).copy()
test_concat_df = concat_df[concat_df['split'] == 0].drop('split', axis=1).drop(TARGETS, axis=1).copy()


train_new_feat_df = concat_df[concat_df['split'] == 1].drop('split', axis=1).copy()
test_new_feat_df = concat_df[concat_df['split'] == 0].drop('split', axis=1).drop(TARGETS, axis=1).copy()


train_new_feat_df.groupby('race_group')['donor_age'].median().to_dict()


train_new_feat_df.shape, test_new_feat_df.shape


for col in all_num_cols[:-2]:
    col_median = train_new_feat_df.groupby('race_group')[col].median()

    # Impute missing values in the train set
    train_new_feat_df[col] = train_new_feat_df.groupby('race_group')[col].transform(lambda x:x.fillna(col_median[x.name]))
    test_new_feat_df[col] = test_new_feat_df['race_group'].map(col_median).fillna(test_new_feat_df[col])

    # Mark missing values and create new columns
    train_new_feat_df[f'missing{col}'] = train_new_feat_df[col].isna().astype(int)
    test_new_feat_df[f'missing{col}'] = test_new_feat_df[col].isna().astype(int)


train_new_feat_df.groupby('race_group')['conditioning_intensity'].apply(lambda x: x.mode()[0])


from sklearn.impute import SimpleImputer

cat_imputer = SimpleImputer(strategy='constant', fill_value="UNKNOWN")
train_new_feat_df[CAT_FEATURES] = cat_imputer.fit_transform(train_new_feat_df[CAT_FEATURES])
test_new_feat_df[CAT_FEATURES] = cat_imputer.transform(test_new_feat_df[CAT_FEATURES])


train_new_feat_df.isna().sum()


train_new_feat_df.groupby('race_group')["cyto_score"].value_counts(dropna=False).unstack()


## Comment it out only for speeding running notebook
import pandas as pd

# Compute the correlation matrix
num_cols_for_corr = all_num_cols + ["naf_label", "km_label"]
# drop_columns=['hla_low_res_6', 'hla_low_res_8', 'hla_low_res_10', 'hla_high_res_6']
corr_matrix = train_new_feat_df[num_cols_for_corr].corr()

# Print the entire correlation matrix
print("Correlation Matrix:")
# print(corr_matrix)

# Optionally, if you only want to print values greater than a threshold (e.g., |0.7|) to spot strong correlations:
threshold = 0.90
high_corr_pairs = corr_matrix.unstack().sort_values(ascending=False)

# Filter pairs that have correlation greater than the threshold, excluding self-correlations (diagonal)
high_corr_pairs = high_corr_pairs[(abs(high_corr_pairs) > threshold) & (high_corr_pairs != 1)]  # & (high_corr_pairs != 1)

# Print high correlation pairs
print(f"\nHighly Correlated Pairs (Correlation > {threshold}):")
high_corr_pairs


eda.plot_heatmap(data=train_new_feat_df,
                  columns=num_cols_for_corr,
                  palette="GnBu",
                  data_name="Train",
                  mask=True, figsize=(18, 14))


class CFG:

    # XGBoost Parameters

    xgb_naf_params = {'objective': 'reg:squarederror',
                      'learning_rate': 0.006075528825176012,
                      'n_estimators': 8758,
                      'max_depth': 5,
                      'reg_lambda': 1.1659262508549852,
                      'reg_alpha': 0.09428525210028153,
                      'colsample_bytree': 0.783786480524185,
                      'subsample': 0.7611558002889776,
                      'min_child_weight': 3.0254087779108443,
                      'gamma': 0.3722306093870793,
                      'random_state': 42,
                      'enable_categorical': True}


    xgb_km_params = {'objective': 'reg:squarederror',
                            'learning_rate': 0.00751112202236412,
                            'n_estimators': 2421,
                            'max_depth': 15,
                            'reg_lambda': 0.07121810017836573,
                            'reg_alpha': 4.403159327413998,
                            'colsample_bytree': 0.7511651344246066,
                            'subsample': 0.7882100358955284,
                            'min_child_weight': 8.993578175386705,
                            'gamma': 0.02162341983515147,
                            'enable_categorical': True,
                            'random_state': 42,}


    # optuna_best for 'naf_label'
    lgbm_naf_params = {'objective': 'regression',
                       'metric': 'rmse',
                       'n_estimators': 6000,
                       'learning_rate': 0.0029359045972924963,
                       'num_leaves': 41,
                       'max_depth': 18,
                       'min_child_samples': 13,
                       'reg_alpha': 0.3226105105322365,
                       'reg_lambda': 0.001612040777485183,
                       'bagging_fraction': 0.9327623358738671,
                       'feature_fraction': 0.5008374482772939,
                       'max_bin': 182,
                       'device': 'cpu',
                       'verbose': 0,
                       'seed': 42}

    lgbm_km_params = {'objective': 'regression',
                     'metric': 'rmse',
                     'n_estimators': 6000,
                     'learning_rate': 0.003381110658716912,
                     'num_leaves': 34,
                     'max_depth': 13, 
                     'min_child_samples': 46, 
                     'reg_alpha': 0.00300168812118655,
                     'reg_lambda': 0.28600765641683357,
                     'bagging_fraction': 0.5465959188526881,
                     'feature_fraction': 0.5479214061797635,
                     'max_bin': 171,
                     'device': 'cpu',
                     'verbose': 0,
                     'seed': 42}


def xgb_model(params, xtrain, ytrain, xval, yval):
    xgb_model = xgb.XGBRegressor(**params)
    return xgb_model.fit(xtrain,
                         ytrain,
                         eval_set=[(xval, yval)],
                         eval_metric="rmse",
                         early_stopping_rounds=100,
                         verbose=500)


def lgbm_model(params, boost_params, xtrain, xval, ytrain, yval, cat_cols):
    # Dataset for naf_label
    lgb_train_dataset = lgb.Dataset(xtrain,
                                    label=ytrain,
                                    categorical_feature=cat_cols,
                                    free_raw_data=False)
    lgb_val_dataset = lgb.Dataset(xval,
                                 label=yval,
                                 categorical_feature=cat_cols,
                                 free_raw_data=False)

    # Train the first model for naf_label
    best_lgb_model = lgb.train(params=params,
                               train_set=lgb_train_dataset,
                               valid_sets=[lgb_train_dataset,lgb_val_dataset],
                               num_boost_round=2000)

    print("Boosting LGBM model ...")
    er_callbacks = [lgb.early_stopping(stopping_rounds=100)]
    boost_lgb_model = lgb.train(params=boost_params,
                               train_set=lgb_train_dataset,
                               valid_sets=[lgb_train_dataset, lgb_val_dataset],
                               num_boost_round=1500,
                               init_model=best_lgb_model,
                               callbacks=er_callbacks)

    return boost_lgb_model


cfg = CFG()

np.random.seed(42)

# target_cols = ["efs", "efs_time", "km_label", "naf_label", "cox_risk_score", "survival_prob_30"]
target_cols = TARGETS
cat_cols = list(CAT_FEATURES)
oof_preds = [] # np.zeros(len(train_df))
all_efs = []
all_efs_time = []
scores = []
test_sub_oof = np.zeros(len(test_concat_df)) 

# Initialize StratifiedKFold
skf = StratifiedKFold(n_splits=5)

for fold, (train_idx, val_idx) in enumerate(skf.split(train_concat_df, train_concat_df['race_group'])):
    train_data = train_concat_df.iloc[train_idx]
    val_data = train_concat_df.iloc[val_idx]

    train_data[cat_cols] = train_data[cat_cols].astype("category")
    val_data[cat_cols] = val_data[cat_cols].astype("category")
    test_concat_df[cat_cols] = test_concat_df[cat_cols].astype("category")
    
    all_efs += list(val_data['efs'].values)
    all_efs_time += list(val_data['efs_time'].values)

    print(f"************* TRAINING FOLD #{fold+1} *****************")

    # LGBM model with 'naf_label'
    print("Train LGBM model with 'naf_label'.....")
    lgb_naf_model = lgbm_model(params=cfg.lgbm_naf_params,
                               boost_params=cfg.lgbm_naf_params,
                               xtrain=train_data.drop(columns=target_cols),
                               xval=val_data.drop(columns=target_cols),
                               ytrain=train_data["naf_label"],
                               yval=val_data["naf_label"],
                               cat_cols=cat_cols)
    
    # XGBoost model with 'naf_label'
    print("Traning XBOOST with 'naf_label ...'")
    xgb_naf_model = xgb_model(cfg.xgb_naf_params,
                              xtrain=train_data.drop(columns=target_cols),
                              ytrain=train_data['naf_label'],
                              xval=val_data.drop(columns=target_cols),
                              yval=val_data['km_label'])

    # LGBM model with 'km_label'
    print("Train LGBM model with 'km_label'.....")
    lgb_km_model = lgbm_model(params=cfg.lgbm_km_params,
                              boost_params=cfg.lgbm_km_params,
                              xtrain=train_data.drop(columns=target_cols),
                              xval=val_data.drop(columns=target_cols),
                              ytrain=train_data["km_label"],
                              yval=val_data["km_label"],
                              cat_cols=cat_cols)

    # XGBoost model with 'km_label'
    print("Traning XGBOOST with 'km_label ...'")
    xgb_km_model = xgb_model(cfg.xgb_km_params,
                             xtrain=train_data.drop(columns=target_cols),
                             ytrain=train_data['km_label'],
                             xval=val_data.drop(columns=target_cols),
                             yval=val_data['km_label'])

    # Predictions
    preds_lgb_naf = lgb_naf_model.predict(val_data.drop(columns=target_cols))
    preds_lgb_km = lgb_km_model.predict(val_data.drop(columns=target_cols))
    preds_xgb_naf = xgb_naf_model.predict(val_data.drop(columns=target_cols))
    preds_xgb_km = xgb_km_model.predict(val_data.drop(columns=target_cols))


    preds_dict = {
        "LGBM-NelsonAF": preds_lgb_naf,
        "XGB-NelsonAF": preds_xgb_naf,
        "LGBM-KaplanMF": preds_lgb_km,
        "XGB-KaplanMF": preds_xgb_km,
        "True NelsonAF": val_data["naf_label"],
        "True KaplanMF": val_data["km_label"]}

    # Plot KDE of predictions and True labels in Fold
    eda.plot_kde_distribution(preds_dict, 'Prediction Distributions')
    
    preds = (preds_lgb_naf + preds_lgb_km + preds_xgb_naf + preds_xgb_km) / 4
    oof_preds += list(preds)

    score = concordance_index(val_data['efs_time'], -preds, val_data['efs'])
    scores.append(score)
    
    test_preds_xgb_naf = xgb_naf_model.predict(test_concat_df)
    test_preds_xgb_km = xgb_km_model.predict(test_concat_df)
    test_preds_lgb_naf = lgb_naf_model.predict(test_concat_df)
    test_preds_lgb_km = lgb_km_model.predict(test_concat_df)
    test_sub_oof += (test_preds_xgb_naf + test_preds_xgb_km + test_preds_lgb_naf + test_preds_lgb_km) / 4

    print(f"******* Fold #{fold+1} C-index: {score} *********")
    print(f"Cumulative Test prediction: {test_sub_oof}\n")
   
print(f"Mean C-index: {sum(scores) / skf.n_splits}\tFull C-index: \
                      {concordance_index(np.array(all_efs_time), -np.array(oof_preds),np.array(all_efs))}")


submission_df['prediction'] = test_sub_oof
submission_df.reset_index(inplace=True)
submission_df.to_csv('submission.csv', index=False)


submission_df


from metric import score

y_true = train_new_feat_df.reset_index()[["ID", "efs","efs_time","race_group"]].copy()
y_pred = train_new_feat_df.reset_index()[["ID"]].copy()
y_pred["prediction"] = oof_preds
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for Ensemble =",m)


xgb_km_feature_imp = xgb_km_model.get_booster().get_score(importance_type='weight')
xgb_naf_feature_imp = xgb_naf_model.get_booster().get_score(importance_type='weight')


importance_df = pd.DataFrame({
    'lgb_km_features': lgb_km_model.feature_name(),
    'lgb_km_importance': lgb_km_model.feature_importance(),
    # 'lgb_naf_features': lgb_naf_model.feature_name(),
    'lgb_naf_importance': lgb_naf_model.feature_importance(),
    # 'xgb_km_features': list(xgb_km_feature_imp.keys()),
    'xgb_km_importance': [int(score) for score in list(xgb_km_feature_imp.values())],
    # 'xgb_naf_features': list(xgb_naf_feature_imp.keys()),
    'xgb_naf_importance': [int(score) for score in list(xgb_naf_feature_imp.values())] 
}).sort_values(by="lgb_naf_importance", ascending=False).reset_index(drop=True)


importance_df.head(15).set_index("lgb_km_features") \
                        .style.background_gradient(cmap='GnBu', subset=['lgb_km_importance',
                                                                        'xgb_km_importance',
                                                                        'xgb_naf_importance',
                                                                        'lgb_naf_importance'])


importance_df.tail(15).style.background_gradient(cmap='GnBu', subset=['lgb_km_importance',
                                                                      'xgb_km_importance',
                                                                      'xgb_naf_importance',
                                                                      'lgb_naf_importance'])


importance_df["lgb_km_features"][:10].tolist()


# def stop_after_n_trials(study, trials):
#     if len(study.trials) >= trials and not study.best_trial:
#         raise optuna.exceptions.OptuneError(f"Stopping trails after {trails}  trials without improvements")


# pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "rmse")


# import optuna
# from sklearn.model_selection import cross_val_score
# from sklearn.metrics import mean_squared_error

# # Objective function for Optuna optimisation with cross-validation
# def objective(trial):
#     params = {
#         'objective': 'regression',
#         # 'metric': 'rmse',
#         'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-1),
#         'num_leaves': trial.suggest_int('num_leaves', 10, 50),
#         'max_depth': trial.suggest_int('max_depth', 3, 20),
#         'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
#         'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 10),
#         'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 10),
#         'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.5, 1.0),
#         'feature_fraction': trial.suggest_uniform('feature_fraction', 0.5, 1.0),
#         'max_bin': trial.suggest_int('max_bin', 64, 255),
#         'n_estimators': 6000, # Fixed number of estimators, Try to find n_estimators
#         'device': 'cpu',
#         'verbose': -1,
#         'seed': 42
#         }

#     train_optuna_df = train_df.copy()
#     # Shuffle the DataFrame
#     train_optuna_df = train_optuna_df.sample(frac=1, random_state=42).reset_index(drop=True)
#     train_optuna_df[CAT_FEATURES] = train_optuna_df[CAT_FEATURES].astype('category')

#     # Use cross-validation for evaluation (5-folds)
#     model = lgb.LGBMRegressor(**params)

#     # Perform 5-fold cross-validation with RMSE
#     cv_scores = cross_val_score(model,
#                                 X=train_optuna_df.drop(target_cols, axis=1),
#                                 y=train_optuna_df["km_label"],
#                                 scoring="neg_root_mean_squared_error", error_score='raise')

#     # Return the negative mean of RMSE scores from cross-validation
#     return -cv_scores.mean()

# # Optuna study setup
# study = optuna.create_study(direction='minimize') # Minimize RMSE
# study.optimize(objective, n_trials=30)

# print(f"Best trail: {study.best_trial}")
# print(f"Best parameters: {study.best_trial.params}")


# import optuna
# from sklearn.model_selection import cross_val_score
# from sklearn.metrics import mean_squared_error

# # Objective function for Optuna optimisation with cross-validation
# def objective(trial):
#     params = {
#         'objective': 'reg:squarederror',
#         'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),
#         'n_estimators': trial.suggest_int('n_estimators', 100, 10000),
#         'max_depth': trial.suggest_int('max_depth', 3, 15),
#         'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
#         'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
#         'colsample': trial.suggest_float('colsample_bytree', 0.5, 1.0),
#         'subsample': trial.suggest_float('subsample', 0.5, 1.0),
#         'min_child_weight': trial.suggest_float('min_child_weight', 1, 10),
#         'gamma': trial.suggest_float('gamma', 0, 5),
#         'enable_categorical': True,
#         'verbose': -1,
#         'seed': 42
#         }

#     train_optuna_df = train_df.copy()
#     # Shuffle the DataFrame
#     train_optuna_df = train_optuna_df.sample(frac=1, random_state=42).reset_index(drop=True)
#     train_optuna_df[CAT_FEATURES] = train_optuna_df[CAT_FEATURES].astype('category')

#     # Use cross-validation for evaluation (5-folds)
#     model = xgb.XGBRegressor(**params)

#     # Perform 5-fold cross-validation with RMSE
#     cv_scores = cross_val_score(model,
#                                 X=train_optuna_df.drop(target_cols, axis=1),
#                                 y=train_optuna_df["km_label"],
#                                 scoring="neg_root_mean_squared_error", error_score='raise')

#     # Return the negative mean of RMSE scores from cross-validation
#     return -cv_scores.mean()

# # Optuna study setup
# study = optuna.create_study(direction='minimize') # Minimize RMSE
# study.optimize(objective, n_trials=30)

# print(f"Best trail: {study.best_trial}")
# print(f"Best parameters: {study.best_trial.params}")

