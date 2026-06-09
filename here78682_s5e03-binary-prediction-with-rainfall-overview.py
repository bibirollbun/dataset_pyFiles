# installing needed packages
!pip install --upgrade seaborn -q
!pip install autoviz --upgrade -q


import numpy as np
import pandas as pd
import gc
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from sklearn.model_selection import StratifiedKFold as SKF
from lightgbm import LGBMClassifier as LGBMC, log_evaluation, early_stopping
from catboost import CatBoostClassifier as CBC
from tqdm.notebook import tqdm
from sklearn.preprocessing import StandardScaler, FunctionTransformer, LabelEncoder, OneHotEncoder
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector, make_column_transformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
import category_encoders as ce
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv', index_col = 'id')
test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv', index_col = 'id')


train.info()


# Custom styling for displaying dataframe
def style_dataframe(df):
    return df.style.set_table_styles(
        [{
            'selector': 'thead th',
            'props': [
                ('background-color', '#F2F230'),  
                ('color', '#151515'),  
                ('font-weight', 'bold'),
                ('text-align', 'center'),
                ('border', '1px solid #006400') 
            ]
        }, {
            'selector': 'tbody td',
            'props': [
                ('background-color', '#E2F9B4'),  
                ('border', '1px solid #006400'),  
                ('font-weight', 'bold'),
                ('color', '#000000')  
            ]
        }]
    ).set_properties(**{'text-align': 'center'}).set_table_attributes('style="width:100%;"').hide(axis='index')


styled_subset_train = style_dataframe(train.head())
styled_subset_train


categorical_features = train.select_dtypes(include='object').columns.tolist()
numerical_features = train.select_dtypes(exclude='object').columns.tolist()

df_numerical = pd.DataFrame({'Numerical Variables': numerical_features})
df_categorical = pd.DataFrame({'Categorical Variables': categorical_features})

styled_numerical_df = style_dataframe(df_numerical)
styled_numerical_df


styled_numerical_df = style_dataframe(df_categorical)
styled_numerical_df


columns = []
unique_values = []
unique_counts = []
null_counts = []
MAX_LIST_LENGTH = 8

# Iterate through the columns and collect data
for col in train.columns:
    columns.append(col)
    unique_vals = train[col].unique()
    null_counts = train[col].isnull().sum()
    unique_values.append(list(unique_vals))  # Store unique values as list
    unique_counts.append(len(unique_vals))   # Count of unique values

def truncate_list(lst, max_length=MAX_LIST_LENGTH):
    if len(lst) > max_length:
        return lst[:max_length] + ['...']  # Append '...' to indicate truncation
    return lst
    
# Create a DataFrame from the collected data
unique_info_df = pd.DataFrame({
    'Column': columns,
    'Unique Values': unique_values,
    'Total Unique Values': unique_counts,
    'Total Null Values': null_counts
        
})
# Truncate for display
unique_info_df['Unique Values'] = unique_info_df['Unique Values'].apply(lambda x: truncate_list(x))
styled_unique_info_df = style_dataframe(unique_info_df)
styled_unique_info_df


# Set a general aesthetic style for the plots
sns.set(style="whitegrid")
sns.set_palette("cool")

# Visualize Categorical Variables
for column in categorical_features:
    if train[column].nunique() <= 20:  # Only plot pie charts for features with <= 20 unique values
        f, ax = plt.subplots(1, 2, figsize=(18, 5.5))
        train[column].value_counts().plot.pie(autopct='%1.1f%%', ax=ax[0])
        ax[0].set_ylabel('')
        ax[0].set_title(f'{column.title()} Distribution (Pie Chart)', fontsize=14, fontweight='bold')
        ax[0].set_facecolor('#F0F8FF')  # Set background color for better contrast
        sns.countplot(x=column, data=train, ax=ax[1])
        ax[1].set_title(f'{column.title()} Count (Bar Plot)', fontsize=14, fontweight='bold')
        ax[1].set_facecolor('#F0F8FF')
        plt.suptitle(f'{column.title()} Visualization', fontsize=18, fontweight='bold')
        plt.subplots_adjust(wspace=0.4)  # Adjust space between plots
        plt.show()
        
    elif 20 < train[column].nunique() <= 150:  # For columns with more unique values, use only bar plot
        plt.figure(figsize=(18, 6))
        sns.countplot(x=column, data=train, palette='cool', order=train[column].value_counts().index)
        plt.title(f'{column.title()} Count (Bar Plot)', fontsize=16, fontweight='bold')
        plt.xticks(rotation=90)
        plt.gca().set_facecolor('#F0F8FF')
        plt.show()   
    else:
        print(f"{column.title()} have too many unique values | No. of unique values : {train[column].nunique()}")
# Create DataFrames for numerical and categorical variables
df_categorical = pd.DataFrame({'Categorical Variables': categorical_features})


sns.set_palette("cool")

# Visualize Numerical Variables
for column in numerical_features:
    f, ax = plt.subplots(1, 2, figsize=(18, 5.5))
    sns.histplot(train[column], ax=ax[0], kde=True, edgecolor='black')
    sns.boxplot(x=train[column], ax=ax[1])
    ax[0].set_title(f'{column.title()} Distribution (Histogram)', fontsize=14, fontweight='bold')
    ax[1].set_title(f'{column.title()} Distribution (Box Plot)', fontsize=14, fontweight='bold')
    ax[0].set_facecolor('#F0F8FF')
    ax[1].set_facecolor('#F0F8FF')
    plt.suptitle(f'{column.title()} Visualization', fontsize=18, fontweight='bold')
    plt.subplots_adjust(wspace=0.3)
    plt.show()
    
df_numerical = pd.DataFrame({'Numerical Variables': numerical_features})


import autoviz              
from autoviz.AutoViz_Class import AutoViz_Class

filename = '/kaggle/input/playground-series-s5e3/train.csv'
depVar = 'rainfall'
sep=','

AV = AutoViz_Class()
_ = AV.AutoViz(filename)


# Modifying and Merging data
def merge(df1, df2):
    # merging
    df1['Source'] = 'Competition'
    df1['Source'] = 'Original'
    merged_df = pd.concat([df1, df2], ignore_index=True)
    return merged_df

def convert_to_categorical(df, columns):
    for column in columns:
        if column in df.columns:
            df[column] = df[column].astype('category')
        else:
            print(f"Column '{column}' does not exist in the DataFrame.")
    return df


model_label = "CB_Baseline"
target = 'rainfall'
cv         = SKF(n_splits= 5, shuffle= True, random_state = 42)
test_preds = 0
scores     = []
drop_cols  = [target]
ftre_imp   = 0

# merged_df   = merge(train,Original)
merged_df   = train
merged_df   = merged_df[~merged_df['rainfall'].isna()]
# merged_df   = date(merged_df)

categorical_features = merged_df.select_dtypes(include='object').columns.tolist()
numerical_features = merged_df.select_dtypes(exclude='object').columns.tolist()

merged_df   = convert_to_categorical(merged_df,categorical_features) 

y = merged_df[target]
X = merged_df.drop(target, axis=1)
# X = preprocessing.fit_transform(X, y)

sel_cols   = X.drop(columns = drop_cols, errors = "ignore").columns
OOF_Preds = pd.DataFrame(X, columns = [f"{model_label}"],dtype = np.float32,)

# test        = date(test)
test        = convert_to_categorical(test,categorical_features)
# test = preprocessing.transform(test)

print(f"X : {X.shape} | Y : {y.shape} | Test : {test.shape}")


for fold_nb, (train_idx, dev_idx) in tqdm(enumerate(cv.split(X, y))):

    Xtr  = X.iloc[train_idx][sel_cols]
    ytr  = y.iloc[train_idx]
    Xdev = X.iloc[dev_idx][sel_cols]
    ydev = y.loc[Xdev.index]
    
    print(f'### Fold {fold_nb+1} Training ###')  
    params = {   
             'iterations'        : 200,
             'learning_rate'     : 0.015,
             'l2_leaf_reg'       : 0.20,
             'colsample_bylevel' : 0.30,
             'max_depth'         : 5,
             'random_state'      : 42,
             'loss_function'     : "Logloss",
             'eval_metric'       : "AUC",
             'verbose'           : 0
             } 
    
    model = CBC(
                  **params,
#                   device        = "gpu",
                 )


    model.fit(Xtr, ytr,
              eval_set  = [(Xdev, ydev)],
              # callbacks = [early_stopping(200)],
              )

    ftre_imp  = ftre_imp + model.feature_importances_

    score   = model.get_best_score()['validation']['AUC']

    dev_preds = model.predict_proba(Xdev)[:, -1]
    
    print(f" OOF score [ROC-AUC] = {score} | Fold {fold_nb +1}")
    scores.append(score)

    test_preds = test_preds + (model.predict_proba(test)[:, -1]/5) 
    
    OOF_Preds.loc[Xdev.index, f"{model_label}"] = dev_preds;
    
    del Xtr, Xdev, ytr, ydev, score;
    gc.collect();
    
# print(f'\n\n OOF AUC-ROC score: {np.mean(scores) :.6f} +- {np.std(scores) :.6f}\n')


display(pd.DataFrame(ftre_imp, index = sel_cols, columns = ["FtreImp"]).sort_values(["FtreImp"], ascending = False).transpose().style.format(formatter = "{:,.2f}").set_caption(f"Feature Importances").set_properties(**{"text-align": "center"}).background_gradient(subset = sel_cols,cmap = "cool", axis=1))


# For threshold selection

# Extract predicted probabilities and actual labels
y_pred_proba = OOF_Preds.values  # predicted probabilities
y_true = y.values  # actual labels (0 or 1)

# Define a range of thresholds to test
thresholds = np.linspace(0, 1, 1001)

# Store the metrics for each threshold
metrics = {
    'threshold': [],
    'accuracy': [],
    'precision': [],
    'recall': [],
    'f1_score': [],
    'roc_auc': []
}

# Iterate over thresholds and compute metrics
for threshold in thresholds:
    # Binarize predictions based on the threshold
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Compute metrics
    metrics['threshold'].append(threshold)
    metrics['accuracy'].append(accuracy_score(y_true, y_pred))
    metrics['precision'].append(precision_score(y_true, y_pred))
    metrics['recall'].append(recall_score(y_true, y_pred))
    metrics['f1_score'].append(f1_score(y_true, y_pred))
    metrics['roc_auc'].append(roc_auc_score(y_true, y_pred))  # Optional for binary classification

# Convert metrics to a DataFrame for easier visualization
metrics_df = pd.DataFrame(metrics)

# Plot the metrics to visualize the optimal threshold
plt.figure(figsize=(12, 8))
plt.plot(metrics_df['threshold'], metrics_df['accuracy'], label='Accuracy')
plt.plot(metrics_df['threshold'], metrics_df['precision'], label='Precision')
plt.plot(metrics_df['threshold'], metrics_df['recall'], label='Recall')
plt.plot(metrics_df['threshold'], metrics_df['f1_score'], label='F1-Score')
plt.plot(metrics_df['threshold'], metrics_df['roc_auc'], label='ROC AUC', linestyle='--')

# Customize plot
plt.title('Metrics vs. Threshold')
plt.xlabel('Threshold')
plt.ylabel('Metric Value')
plt.legend()
plt.grid(True)
plt.show()

# Find the threshold that gives the best F1-score (or choose your desired metric)
best_threshold = metrics_df.iloc[metrics_df['roc_auc'].idxmax()]['threshold']
print(f"Optimal threshold based on accuracy: {best_threshold}")



OOF_Preds.index.name = "id"
OOF_Preds.sort_index().reset_index().to_parquet(f'OOF_Preds_{model_label}.parquet')

t = pd.DataFrame(test_preds)
t.to_parquet(f'Mdl_Preds_{model_label}.parquet')

# t = t.map(lambda x: 1 if x >= best_threshold else 0)
# t = t.astype(int)

sub_fl = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')
sub_fl[target] = t
sub_fl.index = sub_fl['id']
sub_fl = sub_fl.drop(columns = ['id']) 
sub_fl.to_csv(f'Submission_{model_label}.csv',index= True)

!ls
print("\n\n")
!head 'Submission_{model_label}.csv'




