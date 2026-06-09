import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns


data_dict = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/data_dictionary.csv')
test = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv")
train = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/train.csv")


data_dict


train.info()
test.info()


train.describe()


missing_train = train.isnull()
missing_test = test.isnull()

fig, axes = plt.subplots(1, 2, figsize = (18, 6))

sns.heatmap(missing_train, cmap='viridis', cbar=True, yticklabels=False, ax=axes[0])
axes[0].set_title('Missing Values Heatmap - Training Dataset', fontsize=14)
axes[0].set_xlabel('Features', fontsize=12)
axes[0].set_ylabel('Entries', fontsize=12)

sns.heatmap(missing_test, cmap='viridis', cbar=True, yticklabels=False, ax=axes[1])
axes[1].set_title('Missing Values Heatmap - Test Dataset', fontsize=14)
axes[1].set_xlabel('Features', fontsize=12)
axes[1].set_ylabel('Entries', fontsize=12)

plt.tight_layout()
plt.show()


def missing_values_table(df):
    missing_cnt = df.isnull().sum()
    missing_percent = 100 * missing_cnt / len(df)
    datatypes = df.dtypes
    return pd.DataFrame({
        'Missing Values': missing_cnt,
        'Missing Percent': missing_percent,
        'Data Type': datatypes
    })

msv_train = missing_values_table(train)
msv_test = missing_values_table(test)


msv_train[msv_train["Missing Values"]>0].sort_values(by ="Missing Values")


cols_train = [row for row in msv_train.index if msv_train.loc[row, "Missing Values"] > 0]
data_dict.query('@data_dict.variable in @cols_train')


msv_test[msv_test["Missing Values"]>0]


cols_test = [row for row in msv_test.index if msv_test.loc[row, "Missing Values"] > 0]
data_dict.query('@data_dict.variable in @cols_train')



train_data = train.copy()
test_data = test.copy()


numeric_columns = train_data.select_dtypes(include=['number']).columns
for col in numeric_columns:
    if col in test_data.columns:
        mean_value = train_data[col].mean()  
        train_data[col] = train_data[col].fillna(mean_value)
        test_data[col] = test_data[col].fillna(mean_value)


def fill_missing_values(data, column_name):
    value_counts = data[column_name].value_counts(normalize=True)
    most_frequent_category = value_counts.idxmax()
    most_frequent_category_percentage = value_counts.max()
    
    if most_frequent_category_percentage > 0.6:
        data[column_name] = data[column_name].fillna(most_frequent_category)  
    else:
        data[column_name] = data[column_name].fillna('Missing')  

for col in train_data.select_dtypes(include=['object']).columns:
        if col in test_data.columns:
            fill_missing_values(train_data, col)



for col in train_data.select_dtypes(include=['object']).columns:
        if col in test_data.columns:
            value_counts = train_data[col].value_counts(normalize=True)
            most_frequent_category = value_counts.idxmax()
            most_frequent_category_percentage = value_counts.max()
            
            if most_frequent_category_percentage > 0.6:
                test_data[col] = test_data[col].fillna(most_frequent_category)  
            else:
                test_data[col] = test_data[col].fillna('Missing')  


plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.histplot(train_data['efs_time'], color='skyblue', bins=30, stat='density')
plt.title('Distribution of efs_time', fontsize=14)
plt.xlabel('Feature1', fontsize=12)
plt.ylabel('Density', fontsize=12)

plt.subplot(1, 2, 2)
sns.boxplot(train_data['efs_time'], color='salmon')
plt.title('Distribution of efs_time', fontsize=14)
plt.xlabel('efs_time', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()


columns_to_analyze = train_data.select_dtypes(include=['number']).columns.drop(["efs_time","efs"])
fig, axes = plt.subplots(len(columns_to_analyze), 3, figsize=(25, len(columns_to_analyze) * 5))

for i, column in enumerate(columns_to_analyze):
    # Histogram for train_data
    sns.histplot(train_data[column], bins=30, kde=True, color= "skyblue", ax=axes[i, 0])
    axes[i, 0].set_title(f'Distribution of {column} (Train)', fontsize=14)
    axes[i, 0].set_xlabel(column, fontsize=10)
    axes[i, 0].set_ylabel('Frequency', fontsize=10)
    axes[i, 0].grid(visible=True, linestyle='--', alpha=0.6)

    # Boxplot for train_data
    sns.boxplot(x=train_data[column], color="skyblue", ax=axes[i, 1])
    axes[i, 1].set_title(f'Boxplot of {column} (Train)', fontsize=14)
    axes[i, 1].set_xlabel(column, fontsize=10)
    axes[i, 1].grid(visible=True, linestyle='--', alpha=0.6)

    # Boxplot for test_data
    sns.boxplot(x=test_data[column], color="skyblue", ax=axes[i, 2])
    axes[i, 2].set_title(f'Boxplot of {column} (Test)', fontsize=12)
    axes[i, 2].set_xlabel(column, fontsize=10)
    axes[i, 2].grid(visible=True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()



# Compute the correlation matrix
correlation_matrix = train_data.select_dtypes(include=['number']).corr()
plt.figure(figsize=(15, 12))

# Create the heatmap
sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap='viridis',
    cbar=True,
    square=True,
    mask=np.triu(np.ones_like(correlation_matrix, dtype=bool)),
    linewidths=0.5
)

plt.title('Correlation Heatmap of Numerical Features (Excluding Target)', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()


def plot_categorical_distribution(data, column_name):
    plt.figure(figsize=(18, 4))

    # Bar plot for categorical distribution
    plt.subplot(1, 2, 1)
    sns.countplot(y=column_name, data=data, palette='Set2')
    plt.title(f'Distribution of {column_name}', fontsize=12)
    plt.xlabel('Count', fontsize=10)
    plt.ylabel(column_name, fontsize=10)

    ax = plt.gca()
    for p in ax.patches:
        count = int(p.get_width())
        ax.annotate(f'{count}',
                    (p.get_width() + 0.1, p.get_y() + p.get_height() / 2),
                    ha='left', va='center', fontsize=10, color='black')

    sns.despine(left=True, bottom=True)

    # Pie chart for percentage distribution
    plt.subplot(1, 2, 2)
    data[column_name].value_counts().plot.pie(
        autopct='%1.1f%%',
        colors=sns.color_palette('Set2', data[column_name].nunique()),
        startangle=90,
        explode=[0.05] * data[column_name].nunique(),
        shadow=True
    )
    plt.title(f'Percentage Distribution of {column_name}', fontsize=12)
    plt.ylabel('')

    plt.tight_layout()
    plt.show()

categorical_columns = train_data.select_dtypes(include=['object'])

for column in categorical_columns:
    plot_categorical_distribution(train_data, column)


import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from category_encoders import TargetEncoder



CATEGORICAL_VARIABLES = [
    'dri_score', 'graft_type', 'prod_type', 'prim_disease_hct','psych_disturb', 
    'diabetes', 'arrhythmia', 'vent_hist', 'renal_issue', 'pulm_moderate',
    'pulm_severe', 'obesity', 'hepatic_mild', 'hepatic_severe', 'peptic_ulcer', 'rheum_issue',
    'cardiac', 'prior_tumor', 'mrd_hct', 'tbi_status', 'cyto_score', 'cyto_score_detail', 
    'ethnicity', 'race_group','sex_match', 'donor_related', 'cmv_status', 
    'tce_match', 'tce_div_match','melphalan_dose', 'rituximab', 'gvhd_proph', 'in_vivo_tcd', 
    'conditioning_intensity'
]



def label_encode(df, categorical_vars):
    label_encoder = LabelEncoder()
    for cat_var in categorical_vars:
        df[cat_var] = label_encoder.fit_transform(df[cat_var])
    return df

train_f = label_encode(train_data, CATEGORICAL_VARIABLES)
test_f = label_encode(test_data, CATEGORICAL_VARIABLES)

def target_encode(df_train, df_test, categorical_vars, target):
    target_encoder = TargetEncoder()
    df_train[categorical_vars] = target_encoder.fit_transform(df_train[categorical_vars], df_train[target])
    df_test[categorical_vars] = target_encoder.transform(df_test[categorical_vars])
    return df_train, df_test  

train_f, test_f = target_encode(train_f, test_f, ['tce_imm_match'], target='efs')


train_f = train_f.drop("ID", axis = 1)
test_f = test_f.drop("ID", axis = 1)


!pip install /kaggle/input/files-for-installation/autograd-1.7.0-py3-none-any.whl
!pip install /kaggle/input/files-for-installation/autograd-gamma-0.5.0.tar.gz
!pip install /kaggle/input/files-for-installation/interface_meta-1.3.0-py3-none-any.whl
!pip install /kaggle/input/files-for-installation/formulaic-1.1.1-py3-none-any.whl
!pip install /kaggle/input/files-for-installation/lifelines-0.30.0-py3-none-any.whl


import pandas as pd
import numpy as np
import lightgbm as lgb
from lifelines import CoxPHFitter
from sklearn.model_selection import train_test_split
from lifelines.utils import concordance_index


X = train_f.drop(['efs', 'efs_time'], axis=1).values  
y_event = train_f['efs'].values  
y_duration = train_f['efs_time'].values  

X_train, X_val, y_train_event, y_val_event, y_train_duration, y_val_duration = train_test_split(
    X, y_event, y_duration, test_size=0.2, random_state=42
)

# Training an LGBM model to classify an event
model_class = lgb.LGBMClassifier(random_state=42)
model_class.fit(X_train, y_train_event)
y_pred_class = model_class.predict_proba(X_val)[:, 1]  # Event Probabilities

# Train LGBM model for regression (time to event)
model_reg = lgb.LGBMRegressor(random_state=42)
model_reg.fit(X_train, y_train_duration)
y_pred_duration = model_reg.predict(X_val)


# Preparing data for CoxPH 
# Combine original features with predictions of LGBM models
X_val_combined = np.column_stack((X_val, y_pred_class, y_pred_duration))
X_train_combined = np.column_stack((X_train, model_class.predict_proba(X_train)[:, 1], model_reg.predict(X_train)))

# Create a DataFrame for lifelines CoxPH
columns = [f'feature_{i}' for i in range(X_train_combined.shape[1])]
df_train_cox = pd.DataFrame(X_train_combined, columns=columns)
df_train_cox['duration'] = y_train_duration
df_train_cox['event'] = y_train_event

df_val_cox = pd.DataFrame(X_val_combined, columns=columns)
df_val_cox['duration'] = y_val_duration
df_val_cox['event'] = y_val_event

# Training CoxPH model
cox_model = CoxPHFitter()
cox_model.fit(df_train_cox, duration_col='duration', event_col='event')
cox_model.print_summary()




cox_predictions = cox_model.predict_partial_hazard(df_val_cox)

c_index = concordance_index(
    df_val_cox['duration'],  
    -cox_predictions,        
    event_observed=df_val_cox['event']  
)

print(f"C-index on validation set (using LGBM predictions): {c_index:.4f}")




X_test = test_f.values


# Prediction for event probability (classification)
y_pred_class_test = model_class.predict_proba(X_test)[:, 1]  

# Prediction for event duration (regression)
y_pred_duration_test = model_reg.predict(X_test)

# Combine predictions 
X_test_combined = np.column_stack((X_test, y_pred_class_test, y_pred_duration_test))

# Cox model risk score prediction
cox_predictions_test = cox_model.predict_partial_hazard(X_test_combined)

# Prepare submission dataframe
submission = pd.DataFrame({
    'ID': test_data['ID'],  
    'prediction': cox_predictions_test 
})

submission.to_csv('/kaggle/working/submission.csv', index=False)


submission

