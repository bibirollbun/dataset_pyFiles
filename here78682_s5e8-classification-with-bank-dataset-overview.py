# installing needed packages
!pip install --upgrade seaborn -q
!pip install autoviz --upgrade -q
!pip install xgboost --upgrade -q


import numpy as np
import pandas as pd
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold as SKF
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier as LGBMC, log_evaluation, early_stopping
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import xgboost as xgb
from tqdm.notebook import tqdm

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv', index_col = 'id')
test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv', index_col = 'id')
# additional data
Original = pd.read_csv('/kaggle/input/bank-marketing-dataset-full/bank-full.csv', sep = ";").drop_duplicates()
Original.index = range(len(Original))
Original.index.name = "id"    
Original = Original[train.columns]


# Custom styling for displaying dataframe
def style_dataframe(df):
    return df.style.set_table_styles(
        [{
            'selector': 'thead th',
            'props': [
                ('background-color', '#B0D0D3'),  
                ('color', '#151515'),  
                ('font-weight', 'bold'),
                ('text-align', 'center'),
                ('border', '1px solid #006400') 
            ]
        }, {
            'selector': 'tbody td',
            'props': [
                ('background-color', '#E5EFF0'),  
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


marital_y = pd.crosstab(train['marital'], train['y'])
marital_y.plot(kind='bar', stacked=True, figsize=(8,5), colormap='Set2')
plt.title('Marital Status vs. Subscription')
plt.xlabel('Marital Status')
plt.ylabel('Count')
plt.legend(['No Subscription', 'Subscribed'])
plt.show()


plt.figure(figsize=(10,5))
sns.countplot(x='month', hue='y', data=train, order=['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'], palette='tab10')
plt.title('Month vs. Subscription')
plt.xlabel('Month')
plt.ylabel('Count')
plt.legend(['No Subscription', 'Subscribed'])
plt.show()


plt.figure(figsize=(10,8))
corr = train.select_dtypes(include=['int64', 'float64']).corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()


import autoviz              
from autoviz.AutoViz_Class import AutoViz_Class

AV = AutoViz_Class()
dft = AV.AutoViz(
    "",
    dfte=train,
    depVar='y',
    verbose=1,
#     lowess=False,
#     chart_format="html",
    )


# Modifying and Merging data
def merge(df1, df2):
    # merging
    merged_df = pd.concat([df1, df2], ignore_index=True)
    return merged_df

def convert_to_categorical(df, columns):
    for column in columns:
        if column in df.columns:
            df[column] = df[column].astype('category')
        else:
            print(f"Column '{column}' does not exist in the DataFrame.")
    return df


model_label = "Baseline"
target = 'y'
cv         = SKF(n_splits= 5, shuffle= True, random_state = 42)
test_preds = 0
scores     = []
drop_cols  = [target]
ftre_imp   = 0
cutoff     = 0.5

cat_c      = ['job','marital','education','default','housing','loan','contact','month','poutcome']
num_c      = ['age','balance','day','duration','campaign','pdays','previous','y']
# OOF_Preds = pd.DataFrame(X.index, columns = [f"{model_label}"],dtype = np.float32,)

replace_map = {'no': 0, 'yes': 1}
Original['y'] = Original['y'].replace(replace_map)
Original['y'] = Original['y'].astype(int)

merged_df   = merge(Original,train)
merged_df   = convert_to_categorical(merged_df,cat_c) 
test        = convert_to_categorical(test,cat_c)

y = merged_df[target]
X = merged_df.drop(target, axis=1)
sel_cols   = X.drop(columns = drop_cols, errors = "ignore").columns
print(f"X : {X.shape} | Y : {y.shape} | Test : {test.shape}")


print("===== Starting LightGBM Training =====")

lgbm_scores = []
lgbm_oof_preds = np.zeros(len(X))
lgbm_test_preds = np.zeros(len(test))
lgbm_ftre_imp = np.zeros(len(sel_cols))

for fold_nb, (train_idx, dev_idx) in tqdm(enumerate(cv.split(X, y))):

    Xtr  = X.iloc[train_idx][sel_cols]
    ytr  = y.iloc[train_idx]
    Xdev = X.iloc[dev_idx][sel_cols]
    ydev = y.loc[Xdev.index]
    
    print(f'### Fold {fold_nb+1} Training ###')  
    params = {   
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'reg_lambda': 1.6,
        'reg_alpha': .1,
        'num_leaves':90,
        'n_estimators': 500, 
        'colsample_bytree': .5, 
        'min_child_samples': 10,
        'subsample_freq': 3,
        'subsample': .89,
        'importance_type': 'gain',
        'learning_rate': 0.05,
        'max_depth': 9,  
        'max_bin': 255,
        'extra_trees': True
             } 
    model = LGBMC(
                  **params,
                  # device        = "gpu",
                  verbosity     = -1,
                  random_state  = 42,
                 )

    model.fit(Xtr, ytr,
              eval_set  = [(Xdev, ydev)],
              eval_names = [("Dev")],
              eval_metric = "AUC",
              callbacks = [early_stopping(200)],
              categorical_feature = cat_c,
              )


    lgbm_ftre_imp += model.feature_importances_
    score = model.best_score_['Dev']['auc']

    # Store OOF predictions for this fold
    lgbm_oof_preds[dev_idx] = model.predict_proba(Xdev)[:, -1]
    
    print(f" OOF score [ROC-AUC] = {score} | Fold {fold_nb +1}")
    lgbm_scores.append(score)

    # Accumulate test predictions (we average them after the loop)
    lgbm_test_preds += model.predict_proba(test[sel_cols])[:, -1] / cv.get_n_splits()
    
    del Xtr, Xdev, ytr, ydev, score
    gc.collect()
    

print(f'\n\n OOF ROC-AUC score: {np.mean(lgbm_scores) :.6f} +- {np.std(lgbm_scores) :.6f}\n')


display(pd.DataFrame(lgbm_ftre_imp, index = sel_cols, columns = ["FtreImp"]).sort_values(["FtreImp"], ascending = False).transpose().style.format(formatter = "{:,.2f}").set_caption(f"Feature Importances (LGBM)").set_properties(**{"text-align": "center"}).background_gradient(subset = sel_cols,cmap = "cool", axis=1))


print("===== Starting CatBoost Training =====")
cat_scores = []
cat_ftre_imp = np.zeros(len(sel_cols))
# Initialize prediction arrays
cat_oof_preds = np.zeros(len(X))
cat_test_preds = np.zeros(len(test))


for fold_nb, (train_idx, dev_idx) in tqdm(enumerate(cv.split(X, y)), total=cv.get_n_splits()):
    Xtr, ytr = X.iloc[train_idx][sel_cols], y.iloc[train_idx]
    Xdev, ydev = X.iloc[dev_idx][sel_cols], y.iloc[dev_idx]
    
    print(f'### Fold {fold_nb+1} Training ###')
    
    params = {
        'iterations': 3000, 
        'learning_rate': 0.04,
        'depth': 8,
        'l2_leaf_reg': 1.5, 
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'random_seed': 42,
        'colsample_bylevel': 0.6, 
        'subsample': 0.9,
        'bootstrap_type': 'Bernoulli',
        'verbose': 100,
        # 'task_type' : 'GPU'
    }
    
    model = CatBoostClassifier(**params)

    model.fit(Xtr, ytr,
              eval_set=[(Xdev, ydev)],
              cat_features=cat_c,
              early_stopping_rounds=200
             )

    cat_ftre_imp += model.get_feature_importance()

    score = model.get_best_score()['validation']['AUC']
    
    dev_preds = model.predict_proba(Xdev)[:, -1]
    cat_oof_preds[dev_idx] = dev_preds

    print(f" OOF score [ROC-AUC] = {score} | Fold {fold_nb +1}")
    cat_scores.append(score)

    cat_test_preds += model.predict_proba(test[sel_cols])[:, -1] / cv.get_n_splits()
    
    del Xtr, Xdev, ytr, ydev, score
    gc.collect()

print(f'\n\nCatBoost OOF ROC-AUC score: {np.mean(cat_scores):.6f} +- {np.std(cat_scores):.6f}\n')


display(pd.DataFrame(cat_ftre_imp, index = sel_cols, columns = ["FtreImp"]).sort_values(["FtreImp"], ascending = False).transpose().style.format(formatter = "{:,.2f}").set_caption(f"Feature Importances (CB)").set_properties(**{"text-align": "center"}).background_gradient(subset = sel_cols,cmap = "cool", axis=1))


for col in cat_c:
    X[col] = X[col].astype('category')
    test[col] = test[col].astype('category')

print("===== Starting XGBoost Training =====")
xgb_scores = []
xgb_ftre_imp = np.zeros(len(sel_cols))
# Initialize prediction arrays
xgb_oof_preds = np.zeros(len(X))
xgb_test_preds = np.zeros(len(test))


for fold_nb, (train_idx, dev_idx) in tqdm(enumerate(cv.split(X, y)), total=cv.get_n_splits()):
    Xtr, ytr = X.iloc[train_idx][sel_cols], y.iloc[train_idx]
    Xdev, ydev = X.iloc[dev_idx][sel_cols], y.iloc[dev_idx]
    
    print(f'### Fold {fold_nb+1} Training ###')
    
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'booster': 'gbtree',
        'n_estimators': 3000,
        'eta': 0.04, 
        'max_depth': 8,
        'lambda': 1.5, 
        'alpha': 0.1,  
        'subsample': 0.9,
        'colsample_bytree': 0.6,
        'tree_method': 'hist', 
        'enable_categorical': True, 
        'seed': 42,
        'early_stopping_rounds' : 300,
        # 'device': 'gpu' # Uncomment if you have a GPU
    }
    
    model = XGBClassifier(**params)

    model.fit(Xtr, ytr,
              eval_set=[(Xdev, ydev)],
              verbose=100
             )

    xgb_ftre_imp += model.feature_importances_

    score = model.best_score
    
    # Store OOF predictions
    dev_preds = model.predict_proba(Xdev)[:, -1]
    xgb_oof_preds[dev_idx] = dev_preds
    
    print(f" OOF score [ROC-AUC] = {score} | Fold {fold_nb +1}")
    xgb_scores.append(score)

    xgb_test_preds += model.predict_proba(test[sel_cols])[:, -1] / cv.get_n_splits()
    
    del Xtr, Xdev, ytr, ydev, score
    gc.collect()

print(f'\n\nXGBoost OOF ROC-AUC score: {np.mean(xgb_scores):.6f} +- {np.std(xgb_scores):.6f}\n')


display(pd.DataFrame(xgb_ftre_imp, index = sel_cols, columns = ["FtreImp"]).sort_values(["FtreImp"], ascending = False).transpose().style.format(formatter = "{:,.2f}").set_caption(f"Feature Importances (XGB)").set_properties(**{"text-align": "center"}).background_gradient(subset = sel_cols,cmap = "cool", axis=1))


oof_df = pd.DataFrame({
    'lgbm': lgbm_oof_preds,
    'catboost': cat_oof_preds,
    'xgboost': xgb_oof_preds
}, index=X.index)

test_df = pd.DataFrame({
    'lgbm': lgbm_test_preds,
    'catboost': cat_test_preds,
    'xgboost': xgb_test_preds
}, index=test.index)

print("----- Individual Model OOF Scores -----")
print(f"LGBM OOF Score:     {roc_auc_score(y, oof_df['lgbm']):.6f}")
print(f"CatBoost OOF Score: {roc_auc_score(y, oof_df['catboost']):.6f}")
print(f"XGBoost OOF Score:  {roc_auc_score(y, oof_df['xgboost']):.6f}")

# Less correlated models often lead to better blends.
print("\n----- OOF Prediction Correlation -----")
print(oof_df.corr())


print("\n----- Blending Strategy 1: Simple Averaging -----")

oof_df['blend_avg'] = oof_df[['lgbm', 'catboost', 'xgboost']].mean(axis=1)
test_df['blend_avg'] = test_df[['lgbm', 'catboost', 'xgboost']].mean(axis=1)

# Evaluate the blend
avg_blend_score = roc_auc_score(y, oof_df['blend_avg'])
print(f"Simple Average Blend OOF Score: {avg_blend_score:.6f}")


print("\n----- Blending Strategy 2: Optimized Weighted Averaging -----")
# The optimizer can only minimize a function, so we minimize (1 - AUC)

def auc_optimizer(weights):
    """
    Function to be minimized.
    weights: a list of weights for lgbm, catboost, xgboost.
    """
    weighted_preds = (weights[0] * oof_df['lgbm'] +
                      weights[1] * oof_df['catboost'] +
                      weights[2] * oof_df['xgboost'])
    
    # Calculate the score and return the value to minimize (1 - score)
    auc = roc_auc_score(y, weighted_preds)
    return 1 - auc

# Initial guess for the weights (equal)
initial_weights = [1/3, 1/3, 1/3]

# Constraint: The sum of weights must be 1
constraints = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)})

# Bounds: Each weight must be between 0 and 1
bounds = [(0, 1)] * len(initial_weights)

# Run the optimization
result = minimize(
    fun=auc_optimizer,
    x0=initial_weights,
    bounds=bounds,
    constraints=constraints,
    method='SLSQP'
)

# Get the optimal weights
optimal_weights = result.x
print(f"Optimal Weights: \n- LGBM: {optimal_weights[0]:.4f}\n- CatBoost: {optimal_weights[1]:.4f}\n- XGBoost: {optimal_weights[2]:.4f}")

# Apply the optimal weights to the OOF and test predictions
oof_df['blend_weighted'] = (optimal_weights[0] * oof_df['lgbm'] +
                            optimal_weights[1] * oof_df['catboost'] +
                            optimal_weights[2] * oof_df['xgboost'])

test_df['blend_weighted'] = (optimal_weights[0] * test_df['lgbm'] +
                             optimal_weights[1] * test_df['catboost'] +
                             optimal_weights[2] * test_df['xgboost'])

weighted_blend_score = roc_auc_score(y, oof_df['blend_weighted'])
print(f"Optimized Weighted Blend OOF Score: {weighted_blend_score:.6f}")


print("\n----- Blending Strategy 3: Stacking with Meta-Model -----")

# The OOF predictions are our new training data
X_meta_train = oof_df[['lgbm', 'catboost', 'xgboost']]

# The test predictions are our new test data
X_meta_test = test_df[['lgbm', 'catboost', 'xgboost']]

# Create and train the meta-model
meta_model = LogisticRegression(random_state=42)
meta_model.fit(X_meta_train, y)

# Make predictions using the meta-model
oof_df['blend_stacked'] = meta_model.predict_proba(X_meta_train)[:, 1]
test_df['blend_stacked'] = meta_model.predict_proba(X_meta_test)[:, 1]

stacked_blend_score = roc_auc_score(y, oof_df['blend_stacked'])
print(f"Stacked Blend OOF Score: {stacked_blend_score:.6f}")


print("----- Final Blend Evaluation -----")
avg_blend_score = roc_auc_score(y, oof_df['blend_avg'])
weighted_blend_score = roc_auc_score(y, oof_df['blend_weighted'])
stacked_blend_score = roc_auc_score(y, oof_df['blend_stacked'])

print(f"Simple Average Blend OOF Score:     {avg_blend_score:.6f}")
print(f"Optimized Weighted Blend OOF Score: {weighted_blend_score:.6f}")
print(f"Stacked Blend OOF Score:            {stacked_blend_score:.6f}")

scores = {
    'Simple_Average_Blend': avg_blend_score,
    'Weighted_Blend': weighted_blend_score,
    'Stacked_Blend': stacked_blend_score
}

best_blend_name = max(scores, key=scores.get)

if best_blend_name == 'Simple_Average_Blend':
    final_test_preds = test_df['blend_avg']
elif best_blend_name == 'Weighted_Blend':
    final_test_preds = test_df['blend_weighted']
else:
    final_test_preds = test_df['blend_stacked']

# Update the model_label for filename
model_label = best_blend_name
print(f"\nSelected '{model_label}' for final submission.")

print("\n----- Creating Submission File -----")

# Load the sample submission file
sub_fl = pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')

# Assign our best blended predictions to the target column 'y'
sub_fl['y'] = final_test_preds.values

# Save the final submission file
submission_filename = f'Submission_{model_label}.csv'
sub_fl.to_csv(submission_filename, index=False)

# Saving OOFs for future use
oof_df.to_csv('oofs_baseline.csv')
test_df.to_csv('test_baseline.csv')

print(f"Submission file '{submission_filename}' created successfully.")

# Verify the Output 
!ls -lh {submission_filename}
print("\n\n")
!head {submission_filename}

