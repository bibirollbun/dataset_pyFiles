!pip install -q --no-index --find-links=/kaggle/input/pip-install-lifelines lifelines


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter
from scipy import stats
from warnings import filterwarnings
from xgboost import XGBRegressor, XGBClassifier
from  lightgbm import LGBMRegressor,LGBMClassifier,log_evaluation,early_stopping, callback
from catboost import CatBoostClassifier, Pool, CatBoostRegressor
from lifelines.utils import concordance_index
from lifelines import KaplanMeierFitter
import joblib
from colorama import Fore, Back, Style

filterwarnings('ignore')


# Load data
train = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/train.csv')
test = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/test.csv')
data_dict = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/data_dictionary.csv')

# Basic exploration
print("Dataset shape:", train.shape)
print("\nFirst 5 rows:")
display(train.head())

# Target variable analysis
print("\nTarget variable distribution:")
print(train['efs'].value_counts(normalize=True))



train.info()


train.nunique()



# Missing values analysis
plt.figure(figsize=(20, 8))
missing = train.isna().mean().sort_values(ascending=False)
missing[missing > 0].plot(kind='bar', title='Missing Values Distribution')
plt.ylabel('Proportion Missing')
plt.show()




# Survival analysis visualization
kmf = KaplanMeierFitter()
plt.figure(figsize=(10, 6))

for name, grouped_df in train.groupby('efs'):
    kmf.fit(grouped_df["efs_time"], 
            grouped_df["efs"], 
            label=f'Event = {name}')
    kmf.plot_survival_function()

plt.title('Kaplan-Meier Survival Curve')
plt.xlabel('Time (months)')
plt.ylabel('Survival Probability')
plt.show()




# Categorical variables analysis
cat_cols = data_dict[data_dict['type'] == 'Categorical']['variable'].tolist()
cat_cols = [c for c in cat_cols if c in train.columns and c not in ['efs', 'efs_time']]

plt.figure(figsize=(20, 30))
for i, col in enumerate(cat_cols[:15]):  # First 15 categorical features
    plt.subplot(5, 3, i+1)
    train[col].value_counts().plot(kind='bar')
    plt.title(f'{col} Distribution')
plt.tight_layout()
plt.show()



# Numerical variables analysis
num_cols = data_dict[data_dict['type'] == 'Numerical']['variable'].tolist()

# Calculate grid dimensions dynamically
n_cols = 4
n_rows = (len(num_cols) + n_cols - 1) // n_cols  # Round up division

plt.figure(figsize=(20, 5*n_rows))  # Adjust height based on rows
for i, col in enumerate(num_cols):
    plt.subplot(n_rows, n_cols, i+1)
    sns.histplot(train[col].dropna(), kde=True)  # Add dropna to handle missing values
    plt.title(f'{col} Distribution')
plt.tight_layout()
plt.show()


# Correlation analysis
plt.figure(figsize=(15, 10))
corr_matrix = train[num_cols + ['efs_time', 'efs']].corr()
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.show()




# Age distribution
plt.figure(figsize=(12, 6))
sns.histplot(np.log(train['age_at_hct']), bins=30, kde=True)
plt.title('Distribution of Age')
plt.xlabel('Year')
plt.show()




# Time-to-event distribution
plt.figure(figsize=(12, 6))
sns.histplot(train['efs_time'], bins=30, kde=True)
plt.title('Distribution of Time to Event')
plt.xlabel('Months')
plt.show()


# Race group analysis
if 'race_group' in train.columns:
    plt.figure(figsize=(12, 6))
    sns.countplot(data=train, x='race_group', hue='efs')
    plt.title('Event Distribution by Race Group')
    plt.xticks(rotation=45)
    plt.show()

# Age analysis
if 'age_at_hct' in train.columns:
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='efs', y='age_at_hct', data=train)
    plt.title('Age Distribution by Event Status')
    plt.show()

# Comorbidity analysis
if 'comorbidity_score' in train.columns:
    plt.figure(figsize=(12, 6))
    sns.violinplot(x='efs', y='comorbidity_score', data=train)
    plt.title('Comorbidity Score Distribution by Event Status')
    plt.show()

# Treatment modality analysis
treatment_cols = ['graft_type', 'conditioning_intensity', 'in_vivo_tcd']
plt.figure(figsize=(18, 6))
for i, col in enumerate(treatment_cols):
    plt.subplot(1, 3, i+1)
    sns.countplot(x=col, hue='efs', data=train)
    plt.title(f'Event Distribution by {col}')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Statistical tests for numerical features
print("\nStatistical significance between event groups:")
for col in num_cols:
    group1 = train[train['efs'] == 'Event'][col]
    group0 = train[train['efs'] == 'Censoring'][col]
    t_stat, p_val = stats.ttest_ind(group1.dropna(), group0.dropna())
    print(f"{col}: t-stat = {t_stat:.2f}, p-value = {p_val:.4f}")


sns.kdeplot(data=train, x="age_at_hct", hue="efs", common_norm=False, fill=True)
plt.xlabel("Age at HCT")
plt.ylabel("Density")
plt.title("KDE Plot of Age at HCT by EFS")
plt.show()


def dropcols(df):
    df_dropped = df.drop(columns = ['cyto_score' , 'mrd_hct', 'tce_match','tce_div_match', 'cyto_score_detail', 'tce_imm_match'])
    return df_dropped
    
train1 = dropcols(train)
test1 = dropcols(test)


train1.info()


# fill categorical with nan
categorical_cols = data_dict[data_dict['type'] == 'Categorical']['variable'].tolist()
for i in ['cyto_score' , 'mrd_hct', 'tce_match','tce_div_match', 'cyto_score_detail', 'tce_imm_match']:
    categorical_cols.remove(i)


train1[categorical_cols] = train1[categorical_cols].fillna('nan')


train1.info()


plt.figure(figsize=(15, 10))
corr_matrix = train1[num_cols + ['efs_time', 'efs']].corr()
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.show()


def encode_dataframe(df):
    # Define ordinal columns
    ordinal_cols = ['dri_score', 'year_hct', 'comorbidity_score', 'conditioning_intensity', 'karnofsky_score']
    
    df_encoded = df.copy()
    
    df_encoded.drop(columns = ['race_group'], inplace = True)
    # Process each object column
    for col in df_encoded.select_dtypes(include='object').columns:
        if col in ordinal_cols:
            # Label encode ordinal columns (preserving order if known)
            df_encoded[col] = pd.Categorical(df_encoded[col]).codes
        else:
            # One-hot encode all other nominal object columns
            dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=False)
            df_encoded = pd.concat([df_encoded.drop(col, axis=1), dummies], axis=1)
    df_encoded['race_group'] = df['race_group']        
    return df_encoded
train2 = encode_dataframe(train1)


test2 = encode_dataframe(test1)


train2.info()


train2['race_group']


'''
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

num_cols = [
    'hla_match_c_high', 'hla_high_res_8', 'hla_low_res_6', 'hla_high_res_6',
    'hla_high_res_10', 'hla_match_dqb1_high', 'hla_nmdp_6', 'hla_match_c_low',
    'hla_match_drb1_low', 'hla_match_dqb1_low', 'year_hct', 'hla_match_a_high',
    'donor_age', 'hla_match_b_low', 'age_at_hct', 'hla_match_a_low',
    'hla_match_b_high', 'comorbidity_score', 'karnofsky_score', 'hla_low_res_8',
    'hla_match_drb1_high', 'hla_low_res_10'
]

imp = SimpleImputer(missing_values=np.nan, strategy='median', add_indicator=True)
scaler = StandardScaler()

X_num_transformed = imp.fit_transform(train2[num_cols])
X_num_scaled = scaler.fit_transform(X_num_transformed)

train3 = train2.copy() 
train3[num_cols] = pd.DataFrame(X_num_scaled, columns=num_cols, index=train2.index) 
train3
'''


class ParticipantVisibleError(Exception):
    pass


def custom_score(solution, submission, row_id_column_name, prediction_label='prediction', print_info=True):
    
    del solution[row_id_column_name]
    del submission[row_id_column_name]
    
    event_label = 'efs'
    interval_label = 'efs_time'
    
    for col in submission.columns:
        if not pd.api.types.is_numeric_dtype(submission[col]):
            raise ParticipantVisibleError(f'Submission column {col} must be a number')
    # Merging solution and submission dfs on ID
    merged_df = pd.concat([solution, submission], axis=1)
    merged_df.reset_index(inplace=True)
    merged_df_race_dict = dict(merged_df.groupby(['race_group']).groups)
    metric_dict = {}
    for race in sorted(merged_df_race_dict.keys()):
        # Retrieving values from y_test based on index
        indices = sorted(merged_df_race_dict[race])
        merged_df_race = merged_df.iloc[indices]
        # Calculate the concordance index
        c_index_race = concordance_index(
                        merged_df_race[interval_label],
                        -merged_df_race[prediction_label],
                        merged_df_race[event_label])

        metric_dict[race] = c_index_race

    race_c_index = list(metric_dict.values())
    c_score = float(np.mean(race_c_index) - np.std(race_c_index))
    if print_info:
        print(f"{Fore.GREEN}{Style.BRIGHT}# c-index={c_score:.4f}, mean={np.mean(race_c_index):.4f} std={np.std(race_c_index):.4f}{Style.RESET_ALL}")
    
    return c_score, metric_dict


def display_overall(df):
    
    race_groups = [
        'American Indian or Alaska Native', 'Asian',
       'Black or African-American', 'More than one race',
       'Native Hawaiian or other Pacific Islander', 'White'
    ]
    df['mean'] = df[race_groups].mean(axis=1)
    df['std'] = np.std(df[race_groups], axis=1)
    df['score'] = df['mean'] - df['std']
    df = df.T
    df['Overall'] = df.mean(axis=1)
    temp = df.drop(index=['std']).values
    display(df
            .iloc[:len(race_groups)]
            .style
            .format(precision=4)
            .background_gradient(axis=None, vmin=temp.min(), vmax=temp.max(), cmap="cool")
            .concat(df.iloc[len(race_groups):].style.format(precision=3))
           )


def transform_survival_probability(df, time_col='efs_time', event_col='efs'):
    """
    Transform using survival probability estimates
    """
    kmf = KaplanMeierFitter()
    kmf.fit(df[time_col], df[event_col])
    y = kmf.survival_function_at_times(df[time_col]).values
    
    return y


train2["label"] = transform_survival_probability(train2, time_col='efs_time', event_col='efs')
train2.loc[train2['efs']==0, 'label'] -= 0.2

sns.histplot(data=train2, x='label', hue='efs', element='step', common_norm=False)
plt.legend(title='efs')
plt.title('Distribution of Target by EFS')
plt.xlabel('Target')
plt.ylabel('Density')
plt.show()


RMV = ["ID","efs","efs_time","label",'y','kfold']
FEATURES = [c for c in train2.columns if not c in RMV]
print(f"There are {len(FEATURES)} FEATURES: {FEATURES}")


CAT_FEATURES = []
for c in FEATURES:
    if train2[c].dtype=="object":
        CAT_FEATURES.append(c)
        train2[c] = train2[c].fillna("NAN")
        test2[c] = test2[c].fillna("NAN")


combined = pd.concat([train2, test2], axis=0, ignore_index=True)
print("The CATEGORICAL FEATURES: ",end="")
for c in FEATURES:
    if c in CAT_FEATURES:
        print(f"{c}, ", end="")
        combined[c] = combined[c].astype("category")
    else:
        if combined[c].dtype=="float64":
            combined[c] = combined[c].astype("float32")
        if combined[c].dtype=="int64":
            combined[c] = combined[c].astype("int32")

traintree = combined.iloc[:len(train2)].copy()
testtree = combined.iloc[len(train2):].reset_index(drop=True).copy()


from sklearn.model_selection import StratifiedKFold, train_test_split

folds = 5
traintree['kfold'] = -1  

skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
groups = traintree['efs'].astype(str)
for fold, (train_idx, val_idx) in enumerate(skf.split(X=traintree, y=groups)):
    traintree.loc[val_idx, 'kfold'] = fold


def CIndexMetric_XGB(y_true, y_pred):
    ds_pred["prediction"] = y_pred
    cindex_score, _ = custom_score(ds_true.copy(), ds_pred.copy(), "ID", print_info=False)
    return -cindex_score

def CIndexMetric_LGB(y_true, y_pred):
    ds_pred["prediction"] = y_pred
    cindex_score, _ = custom_score(ds_true.copy(), ds_pred.copy(), "ID", print_info=False)
    return ('C-Index', cindex_score, True)


traintree


traintree['race_group']


traintree[traintree.select_dtypes(include=['object']).columns] = traintree.select_dtypes(include=['object']).astype('category')



#no preprocessing to test metric
test = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv")
train = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/train.csv")
train["label"] = transform_survival_probability(train, time_col='efs_time', event_col='efs')
train.loc[train['efs']==0, 'label'] -= 0.2
RMV = ["ID","efs","efs_time","label",'y','kfold']
FEATURES = [c for c in train.columns if not c in RMV]
print(f"There are {len(FEATURES)} FEATURES: {FEATURES}")
CAT_FEATURES = []
for c in FEATURES:
    if train[c].dtype=="object":
        CAT_FEATURES.append(c)
        train[c] = train[c].fillna("NAN")
        test[c] = test[c].fillna("NAN")
combined = pd.concat([train, test], axis=0, ignore_index=True)
print("The CATEGORICAL FEATURES: ",end="")
for c in FEATURES:
    if c in CAT_FEATURES:
        print(f"{c}, ", end="")
        combined[c] = combined[c].astype("category")
    else:
        if combined[c].dtype=="float64":
            combined[c] = combined[c].astype("float32")
        if combined[c].dtype=="int64":
            combined[c] = combined[c].astype("int32")

train = combined.iloc[:len(train)].copy()
test = combined.iloc[len(train):].reset_index(drop=True).copy()
folds = 5
train['kfold'] = -1  

skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
groups = train['efs'].astype(str)
for fold, (train_idx, val_idx) in enumerate(skf.split(X=train, y=groups)):
    train.loc[val_idx, 'kfold'] = fold


%%time
    
oof_lgb = train[['kfold','ID','efs','efs_time','label','race_group']].copy()
oof_lgb['prediction'] = 0.0
feature_importances_lgb = pd.DataFrame()
feature_importances_lgb['feature'] = FEATURES
metric_df = []

for fold in range(skf.n_splits):
    
    x_train = train[train.kfold != fold].copy()
    x_valid = train[train.kfold == fold].copy()

    y_train = x_train['label']
    y_valid = x_valid['label']
    y_label = x_valid['efs']

    x_train = x_train[FEATURES]
    x_valid = x_valid[FEATURES]

    ds_true = oof_lgb.loc[oof_lgb.kfold==fold, ["ID","efs","efs_time","race_group"]].copy().reset_index(drop=True)
    ds_pred = oof_lgb.loc[oof_lgb.kfold==fold, ["ID"]].copy().reset_index(drop=True)

    lgb_params = {
        'max_depth': 6,
        'num_leaves': 40,
        'learning_rate': 0.03,
        'n_estimators': 10000,
        'objective': 'l2',
        'subsample': 0.8,
        'colsample_bytree': 0.5,
        'n_jobs': -1,
        'verbose': -1,
        'metric': 'None' # only show the custom metric
    }
    clf = LGBMRegressor(**lgb_params)
    clf.fit(
        x_train, y_train,
        eval_set=[(x_valid, y_valid)],
        categorical_feature=CAT_FEATURES,
        eval_metric=CIndexMetric_LGB, # the custom metric
        callbacks=[callback.log_evaluation(500), callback.early_stopping(50)]
    )
    feature_importances_lgb[f'fold_{fold + 1}'] = clf.feature_importances_

    preds_valid = clf.predict(x_valid)
    oof_lgb.loc[oof_lgb.kfold==fold, 'prediction'] = preds_valid

    joblib.dump(clf, f"lgb_model_{fold}.pkl")

    y_true = oof_lgb.loc[oof_lgb.kfold==fold, ["ID","efs","efs_time","race_group"]].copy().reset_index(drop=True)
    y_pred = oof_lgb.loc[oof_lgb.kfold==fold, ["ID","prediction"]].copy().reset_index(drop=True)
    m, metric_dict = custom_score(y_true, y_pred, "ID", print_info=True)
    metric_df.append(metric_dict)



y_true = oof_lgb[["ID","efs","efs_time","race_group"]].copy().reset_index(drop=True)
y_pred = oof_lgb[["ID","prediction"]].copy().reset_index(drop=True)
m, _ = custom_score(y_true, y_pred, "ID", print_info=True)
print(f"Overall official SCORE: {m:.5f}")

metric_df_ = pd.DataFrame(metric_df)
display_overall(metric_df_)


feature_importances_lgb['importance'] = feature_importances_lgb.drop('feature', axis=1).mean(axis=1)
feature_importances_lgb = feature_importances_lgb.sort_values('importance', ascending=False).reset_index(drop=True)
feature_importances_lgb.head(20)


sub = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv', index_col = False)
sub['prediction'] = y_pred
sub.to_csv('submission.csv', index = False)


sub




