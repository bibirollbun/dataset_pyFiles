import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
from xgboost import XGBRegressor
import lightgbm as lgb
from lightgbm import LGBMRegressor, early_stopping
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import ElasticNetCV, BayesianRidge
from sklearn.linear_model import RidgeCV
import optuna

# Import packages for warnings
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)


train_df = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")

print(f"Train: {train_df.shape}")
print(f"Test: {test_df.shape}")


train_df.head()


test_df.head()    


# Check Missing Values
print('Missing values in Train:', test_df.isna().sum().sum())
print('Missing values in Test:', test_df.isna().sum().sum())


# Check duplicates without `id`
print('Duplicates without id in Train:', train_df.drop(columns=['id']).duplicated().sum())
print('Duplicates without id in Test:', test_df.drop(columns=['id']).duplicated().sum())


# Check duplicate rows in train set without `id`
train_df_dup = train_df.drop(columns=['id'])
train_df_dup = train_df_dup[train_df_dup.duplicated(keep=False)]
train_df_dup.sort_values(by=train_df_dup.columns.tolist()).head(6)     


# Check duplicate rows in test set without `id`
test_df_dup = test_df.drop(columns=['id'])
test_df_dup = test_df_dup[test_df_dup.duplicated(keep=False)]
test_df_dup.sort_values(by=test_df_dup.columns.tolist()).head(6)       


# Remove duplicate rows from train_df
train_df = train_df.drop(columns=['id'])
train_df = train_df.drop_duplicates(keep='first')

# Check
print('Duplicates in Train:', train_df.duplicated().sum()) 
print('Train:', train_df.shape)         


col = train_df.drop(columns=['Calories', 'Sex']).columns

# Setup subplots
fig, axes = plt.subplots(len(col), 2, figsize=(13, 5 * len(col)))

# Plot histogram for train_df and test_df
for i, var in enumerate(col):
    axes[i, 0].hist(train_df[var], alpha=0.5, label='Train')
    axes[i, 0].hist(test_df[var], alpha=0.5, label='Test')
    axes[i, 0].set_title(f'Histogram for {var}', weight='bold')
    axes[i, 0].legend()

    # Prepare data for boxplot
    combined = pd.concat([train_df[var].to_frame().assign(dataset='Train'),
                          test_df[var].to_frame().assign(dataset='Test')])    

    # Plot boxplot
    sns.boxplot(data=combined, x='dataset', y=var, ax=axes[i, 1], palette='Set2')
    axes[i, 1].set_title(f'Boxplot for {var}', weight='bold')

plt.tight_layout()
plt.show()   


train_df.describe()     


test_df.drop(columns=['id']).describe()


# Set subplots 
fig, axes = plt.subplots(1, 2, figsize=(10, 4)) 

# Plot Histogram 
sns.histplot(train_df['Calories'], ax=axes[0], kde=True) 
axes[0].set_title('Histogram of Calories',fontsize=10, weight='bold') 

# Plot Boxplot
sns.boxplot(x=train_df['Calories'], ax=axes[1]) 
axes[1].set_title('Boxplot of Calories', fontsize=10, weight='bold') 

# Adjust layout 
plt.tight_layout() 
plt.show()  


# Check skewness 
train_df['Calories'].skew() 


# Create donut chart to display gender distribution     
# Set subplots 
fig, axes = plt.subplots(1, 2, figsize=(10, 4)) 
plt.suptitle('Gender Distribution', weight='bold')

# Aggregate data for gender distribution
gender_count = train_df['Sex'].value_counts()
gender_count_test = test_df['Sex'].value_counts()

# Prepare labels and values
labels = ['female', 'male']
color = ['#FFB3BA', '#BAFFC9']
sizes = gender_count
seize_test = gender_count_test

# Train set
axes[0].pie(sizes, labels=labels, startangle=90, autopct='%1.1f%%',
        colors=color, textprops={'fontsize':10, 'fontweight':'bold'}, 
        wedgeprops={'width':0.4})
axes[0].set_title('Train', weight='bold') 

# Test set
axes[1].pie(seize_test, labels=labels, startangle=90, autopct='%1.1f%%',
        colors=color, textprops={'fontsize':10, 'fontweight':'bold'}, 
        wedgeprops={'width':0.4})
axes[1].set_title('Test', weight='bold') 

plt.show()         


# Heatmap for train_df
plt.figure(figsize=(8, 5))
sns.heatmap(train_df.corr(method='pearson', numeric_only=True), annot=True, cmap='viridis')
plt.title(f'Correlation Heatmap for Train Data', fontsize=14, weight='bold')
plt.show();

# Heatmap for test_df
plt.figure(figsize=(8, 5))
sns.heatmap(test_df.drop(columns=['id']).corr(method='pearson', numeric_only=True), annot=True, cmap='viridis')
plt.title(f'Correlation Heatmap for Test Data', fontsize=14, weight='bold')
plt.show();


# Change column names to lower_case
train_df.columns = train_df.columns.str.lower()
test_df.columns = test_df.columns.str.lower()


features = ['duration', 'heart_rate', 'body_temp', 'age', 'height', 'weight'] 

# Create aggregated dataframe
agg_data = {feature: train_df.groupby([feature, 'sex'])['calories'].median().reset_index() for feature in features}

# Create subplots 
fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)

# Loop through features and create line plots 
for i, feature in enumerate(features):
    row, col = divmod(i, 3)  
    sns.lineplot(ax=axes[row, col], data=agg_data[feature], x=feature, y='calories', hue='sex', linewidth=2)
    axes[row, col].set_title(f'Median Calories vs {feature.capitalize()} by Gender', weight='bold')
    axes[row, col].set_xlabel(feature.capitalize(), weight='bold')
    axes[row, col].grid(True)

# Adjust layout
plt.tight_layout()
plt.show()


def create_features(df):                
    df['BMI'] = df['weight'] / (df['height']/100) ** 2
    df['body_stress_index'] = df['heart_rate'] / df['body_temp']
    df['weight_by_duration'] = df['weight'] / df['duration']
    df['age_by_duration'] = df['age'] / df['duration']
    df['heart_rate_per_minute'] = df['heart_rate'] / df['duration']
    df['body_temp_per_minute'] = df['body_temp'] / df['duration']
    return df


# Create polynomial features - interactions and squared features 
poly_features = ['duration', 'heart_rate', 'body_temp']

# Create polynomial features
def create_polynomial_features(df):
    # Create interaction features 
    for i, col1 in enumerate(poly_features):
        for col2 in poly_features[i+1:]:
            df[f"{col1}_x_{col2}"] = df[col1] * df[col2]
    # Create squared features
    for col in poly_features:
        df[f"{col}_squared"] = df[col] ** 2
    return df

# Apply create_features
train_df = create_features(train_df)
test_df = create_features(test_df)  

# Apply create_polynomial_features
train_df = create_polynomial_features(train_df)
test_df = create_polynomial_features(test_df)  

# Encode `sex`
train_df['sex_enco'] = np.where(train_df['sex'] == 'female', 0, 1)
test_df['sex_enco'] = np.where(test_df['sex'] == 'female', 0, 1)        


# Set subplots 
fig, axes = plt.subplots(1, 2, figsize=(10, 4)) 

# Create Histogram for train_df
sns.histplot(data=train_df, x='BMI', hue='sex', kde=True, bins=30, ax=axes[0])
axes[0].set_title('BMI Distribution by Gender (Train)',fontsize=10, weight='bold')

# Create Histogram for test_df
sns.histplot(data=test_df, x='BMI', hue='sex', kde=True, bins=30, ax=axes[1])
axes[1].set_title('BMI Distribution by Gender (Test)',fontsize=10, weight='bold')

# Adjust layout 
plt.tight_layout() 
plt.show()


def add_bmi_category(df): 
    # Create BMI categories based on WHO guidelines
    # https://www.calculator.net/bmi-calculator.html
    conditions = [
        (df['BMI'] < 16),
        (df['BMI'] >= 16) & (df['BMI'] < 17),
        (df['BMI'] >= 17) & (df['BMI'] < 18.5),
        (df['BMI'] >= 18.5) & (df['BMI'] < 25),
        (df['BMI'] >= 25) & (df['BMI'] < 30),
        (df['BMI'] >= 30) & (df['BMI'] < 35),
        (df['BMI'] >= 35) & (df['BMI'] < 40),
        (df['BMI'] >= 40)]     
    
    values = [1, 2, 3, 4, 5, 6, 7, 8]
    
    df['BMI_cat'] = np.select(conditions, values, default=0)
    
    return df

# Apply `add_bmi_category`
train_df = add_bmi_category(train_df)
test_df = add_bmi_category(test_df)

# Create interactions between BMI_cat and important features
def add_bmi_cat_interactions(df):
  for feature in ['duration', 'heart_rate', 'body_temp']:
    df[f'BMI_cat_x_{feature}'] = df['BMI_cat'] * df[feature]
  return df 

# Apply `add_bmi_cat_interactions`
train_df = add_bmi_cat_interactions(train_df)
test_df = add_bmi_cat_interactions(test_df) 

# Create a feature that measures how far someone is from "normal" BMI
# Center point of normal range is ~21.75
train_df['BMI_deviation'] = abs(train_df['BMI'] - 21.75) 
test_df['BMI_deviation'] = abs(test_df['BMI'] - 21.75) 

train_df.head(2)    


# Function for binned target encoding
def add_binned_target_encoding(df, df_list, target, features, q=30): 
    for feature in features:
        df[f'{feature}_bin'] = pd.qcut(df[feature], q=q, duplicates='drop')

        aggs = ['mean', 'median', 'std']   
        stats = df.groupby(f'{feature}_bin')[target].agg(aggs).reset_index()
        stats.columns = [f'{feature}_bin'] + [f'{feature}_bin_{agg}' for agg in aggs]

        bin_edges = df[f'{feature}_bin'].cat.categories

        for i, temp_df in enumerate(df_list):
            temp_df[f'{feature}_bin'] = pd.cut(temp_df[feature], bins=bin_edges, include_lowest=True)
            temp_df = temp_df.merge(stats, on=f'{feature}_bin', how='left')
            temp_df.drop(columns=[f'{feature}_bin'], inplace=True)
            df_list[i] = temp_df  # update in place

    return df_list      


def add_target_encoding(df_train, df_valid, df_test, group_cols):  
     # Calculate group stats only on the training set
     for col in group_cols:
         agg_funcs = ['mean', 'median', 'std']
         agg_stats = df_train.groupby(col)['calories_log'].agg(agg_funcs).reset_index()
         agg_stats.columns = [col] + [f"{col}_{func}" for func in agg_funcs]
         
         # Merge into train, valid, test
         df_train = df_train.merge(agg_stats, on=[col], how='left')
         df_valid = df_valid.merge(agg_stats, on=[col], how='left')
         df_test = df_test.merge(agg_stats, on=[col], how='left')

     return df_train, df_valid, df_test     


def add_group_target_encoding(df_train, df_valid, df_test, group_cols):
    # Calculate group stats only for the training set
    for col in group_cols:   
        agg_funcs = ['mean', 'median', 'std'] 
        agg_stats = df_train.groupby([col, 'sex_enco'])['calories_log'].agg(agg_funcs).reset_index()
        agg_stats.columns = [col, 'sex_enco'] + [f"{col}_sex_{func}" for func in agg_funcs]

        # Merge into train, valid, test
        df_train = df_train.merge(agg_stats, on=[col, 'sex_enco'], how='left')
        df_valid = df_valid.merge(agg_stats, on=[col, 'sex_enco'], how='left')
        df_test = df_test.merge(agg_stats, on=[col, 'sex_enco'], how='left')

    return df_train, df_valid, df_test   


# Features & Target     
# Log-transform target variable (`calories`)
train_df['calories_log'] = np.log1p(train_df['calories'])
X = train_df.drop(columns=['calories', 'sex'])
y = train_df['calories_log']
X_test = test_df.drop(columns=['id', 'sex'])

# Custom RMSLE 
def rmsle(y_true, y_pred):
    y_true_original = np.expm1(y_true)
    y_pred_original = np.expm1(y_pred)
    y_pred_original = np.maximum(y_pred_original, 0)
    return np.sqrt(mean_squared_log_error(y_true_original, y_pred_original))  

# CatBoost hyper-parameter             
catboost_params = {
    'iterations': 10000,
    'learning_rate': 0.01,
    'depth': 8,    
    'l2_leaf_reg': 7.0,
    'random_seed': 42,
    'loss_function': 'RMSE',
    'eval_metric': 'RMSE',
    'task_type': 'GPU'  
}      

# LightGBM hyper-parameter
lgbm_params = {
    'n_estimators': 12000,
    'learning_rate': 0.01,
    'subsample': 0.75, 
    'colsample_bytree': 0.75,
    'reg_lambda': 0.3, 
    'reg_alpha': 0.3, 
    'num_leaves': 64, 
    'max_depth': 10, 
    'min_child_samples': 30, 
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'device': 'gpu',
    'random_state': 42
}

# XGBoost hyper-parameter
xgb_params = {
    'n_estimators': 2000,
    'learning_rate': 0.01,
    'max_depth': 10,
    'subsample': 0.75,
    'colsample_bytree': 0.75,
    'reg_lambda': 0.3,
    'reg_alpha': 0.3,
    'objective': 'reg:squarederror',
    'tree_method': 'gpu_hist',  
    'random_state': 42
}

X.shape, X_test.shape, y.shape       


# KFold Cross-Validation    

def train_model(X, y, X_test, model_name, params):
    print(f"\n##### Training {model_name} Model #####")

    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    rmsle_scores = [] 
    
    xgb_feature_importances = pd.DataFrame()  
    cat_feature_importances = pd.DataFrame()
    lgb_feature_importances = pd.DataFrame()    

    kf = KFold(n_splits=50, shuffle=True, random_state=42)

    for fold, (train_idx, valid_idx) in enumerate(kf.split(X)):
        print(f"\n##### Training Fold {fold + 1} #####")
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]
        X_test_fold = X_test.copy()    

        # Add target aggregated stats based on individual features
        for col in ['heart_rate', 'duration', 'BMI_cat']:
            X_train, X_valid, X_test_fold = add_target_encoding(
                df_train=X_train,  
                df_valid=X_valid,
                df_test=X_test_fold,
                group_cols=[col])      
        
        # Add group-wise target stats based on gender combinations
        for col in ['heart_rate', 'duration', 'age', 'BMI_cat']:
            X_train, X_valid, X_test_fold = add_group_target_encoding(
                df_train=X_train,  
                df_valid=X_valid,
                df_test=X_test_fold,
                group_cols=[col]) 

        # Add binned target stats for continuous features
        features = ['body_temp', 'BMI']
        X_train, X_valid, X_test_fold = add_binned_target_encoding(
            df=X_train,
            df_list=[X_train, X_valid, X_test_fold],
            target='calories_log',
            features=features)          
            
        # Drop Target variable 
        X_train.drop(columns=['calories_log'], inplace=True)
        X_valid.drop(columns=['calories_log'], inplace=True)

        # Impute missing values 
        imputer = SimpleImputer(strategy='median')
        X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
        X_valid = pd.DataFrame(imputer.transform(X_valid), columns=X_valid.columns)
        X_test_fold = pd.DataFrame(imputer.transform(X_test_fold), columns=X_test_fold.columns)   
        
        # Train Model
        if model_name == 'XGBoost':
            model = XGBRegressor(**params)
            model.fit(X_train, y_train,
                      eval_set=[(X_valid, y_valid)],
                      early_stopping_rounds=100,
                      verbose=500)
            
            # Feature importance for XGBoost
            importance_dict = model.get_booster().get_score(importance_type='gain')
            fold_importance = pd.DataFrame({
                "feature": list(importance_dict.keys()),
                "importance": list(importance_dict.values()),
                "fold": fold + 1})
            xgb_feature_importances = pd.concat([xgb_feature_importances, fold_importance], axis=0)

        elif model_name == 'LightGBM':
            model = LGBMRegressor(**params, verbose=-1)
            model.fit(X_train, y_train,
                      eval_set=[(X_valid, y_valid)], 
                      callbacks=[early_stopping(stopping_rounds=100, verbose=False)])
            
            # LightGBM Feature Importance
            fold_importance = pd.DataFrame({
                "feature": X_train.columns,
                "importance": model.feature_importances_,
                "fold": fold + 1})
            lgb_feature_importances = pd.concat([lgb_feature_importances, fold_importance], axis=0)

        elif model_name == 'CatBoost':
            model = CatBoostRegressor(**params)
            model.fit(X_train, y_train,
                      eval_set=(X_valid, y_valid),
                      use_best_model=True,
                      early_stopping_rounds=100,
                      verbose=0)
            
            # CatBoost Feature Importance
            importances = model.get_feature_importance(prettified=False)
            fold_importance = pd.DataFrame({
                "feature": X_train.columns,
                "importance": importances,
                "fold": fold + 1})
            cat_feature_importances = pd.concat([cat_feature_importances, fold_importance], axis=0)

        else:
            raise ValueError("Invalid model_name. Use 'XGBoost', 'LightGBM', or 'CatBoost'.")

        val_preds = model.predict(X_valid)
        test_fold_preds = model.predict(X_test_fold)      

        oof_preds[valid_idx] = val_preds
        test_preds += test_fold_preds / kf.n_splits

        score = rmsle(y_valid, val_preds)
        rmsle_scores.append(score)
        print(f"Fold {fold+1} RMSLE: {score:.6f}")

    # Mean RMSLE across folds
    mean_rmsle = np.mean(rmsle_scores)
    print(f"\nMean RMSLE across folds for {model_name}: {mean_rmsle:.6f}")
    
    # Compute final CV RMSLE
    final_score = rmsle(y, oof_preds)
    print(f"Overall Out-of-Fold (OOF) RMSLE for {model_name}: {final_score:.6f}\n")       
    
    if model_name == 'XGBoost':
        return oof_preds, test_preds, rmsle_scores, xgb_feature_importances
    elif model_name == 'LightGBM':
        return oof_preds, test_preds, rmsle_scores, lgb_feature_importances
    elif model_name == 'CatBoost':
        return oof_preds, test_preds, rmsle_scores, cat_feature_importances    


%%time                
# Run XGBoost
xgb_oof, xgb_test, xgb_scores, xgb_feature_importances = train_model(X, y, X_test, model_name='XGBoost', params=xgb_params)         


# Average importance from XGBoost         
avg_importance = xgb_feature_importances.groupby("feature")["importance"].mean().sort_values(ascending=False).reset_index()
xgb_15 = avg_importance.head(15)

# Plot XGBoost feature importance 
plt.figure(figsize=(10, 12))
sns.barplot(data=xgb_15, x='importance', y='feature')
plt.title("Top 15 XGBoost Feature Importances (Gain)", weight="bold")
plt.xlabel("Average Gain Importance");  


%%time        
# CatBoost  
cat_oof, cat_test, cat_scores, cat_feature_importances = train_model(X, y, X_test, model_name='CatBoost', params=catboost_params)


# Average importance from CatBoost 
cat_avg_importance = cat_feature_importances.groupby("feature")["importance"].mean().sort_values(ascending=False).reset_index()
cat_25 = cat_avg_importance.head(25)    

# Plot CatBoost feature importance 
plt.figure(figsize=(10, 15))
sns.barplot(data=cat_25, x='importance', y='feature')
plt.title("Top 25 CatBoost Feature Importances (Gain)", weight="bold")
plt.xlabel("Average Gain Importance");       


%%time 

# Combine base model predictions
X_stack = np.column_stack([xgb_oof, cat_oof])    
X_stack_test = np.column_stack([xgb_test, cat_test])

# Create additional meta-features
def create_stacking_features(base_preds):
    # Create additional meta-features from base model predictions
    xgb_pred, cat_pred = base_preds[:, 0], base_preds[:, 1]     
    
    # Statistical meta-features
    mean_pred = np.mean(base_preds, axis=1)
    median_pred = np.median(base_preds, axis=1) 
    std_pred = np.std(base_preds, axis=1)
    
    # Combine all features
    enhanced_features = np.column_stack([
        base_preds,  # Original predictions
        mean_pred, median_pred, std_pred,  # Statistical features
    ])
    
    return enhanced_features

# Stacking with multiple meta-models
kf = KFold(n_splits=50, shuffle=True, random_state=42)  

# Create enhanced features
X_stack_enhanced = create_stacking_features(X_stack)
X_stack_test_enhanced = create_stacking_features(X_stack_test)   

# Meta-models 
meta_models = {
    'Ridge': RidgeCV(alphas=np.logspace(-3, 3, 10), cv=5),
    'ElasticNet': ElasticNetCV(alphas=np.logspace(-3, 1, 10), cv=5, random_state=42),
    'BayesianRidge': BayesianRidge()
}

results = {}

for name, meta_model in meta_models.items():
    print(f"##### Training {name} Meta-Model #####")
    
    meta_oof = np.zeros(len(X_stack_enhanced))
    meta_test = np.zeros(len(X_stack_test_enhanced))
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_stack_enhanced)):
        X_train = X_stack_enhanced[train_idx]
        X_valid = X_stack_enhanced[val_idx]
        y_train = y.iloc[train_idx].values
        y_valid = y.iloc[val_idx].values
        
        meta_model.fit(X_train, y_train)
        val_preds = meta_model.predict(X_valid)
        meta_oof[val_idx] = val_preds
        meta_test += meta_model.predict(X_stack_test_enhanced) / kf.n_splits
        
        fold_score = rmsle(y_valid, val_preds)
        fold_scores.append(fold_score)
    
    overall_score = rmsle(y, meta_oof)
    results[name] = {
        'oof_score': overall_score,
        'fold_scores': fold_scores,
        'oof_preds': meta_oof, 
        'test_preds': meta_test}

    fold_rmsle = np.mean(fold_scores) 
    print(f"{name} Mean across fold: {fold_rmsle:.6f}")
    print(f"{name} OOF RMSLE: {overall_score:.6f}\n")      


%%time

optuna.logging.set_verbosity(optuna.logging.WARNING)  

# Actual OOF predictions from meta-models
ridge_oof = results['Ridge']['oof_preds']   
elastic_oof = results['ElasticNet']['oof_preds']
bayesian_oof = results['BayesianRidge']['oof_preds']

def objective(trial):
    w1 = trial.suggest_float("w_ridge", 0, 1)
    w2 = trial.suggest_float("w_elastic", 0, 1)
    w3 = trial.suggest_float("w_bayes", 0, 1)

    total = w1 + w2 + w3
    w1 /= total
    w2 /= total
    w3 /= total

    blended_preds = (
        w1 * ridge_oof +
        w2 * elastic_oof +
        w3 * bayesian_oof
    )

    score = rmsle(y, blended_preds)
    return score     

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)

print(f"Best RMSLE: {study.best_value:.6f}\n")   


# Apply to test predictions
best_weights = study.best_params
total = sum(best_weights.values())
w1 = best_weights['w_ridge'] / total
w2 = best_weights['w_elastic'] / total
w3 = best_weights['w_bayes'] / total

final_test_preds = (
    w1 * results['Ridge']['test_preds'] +
    w2 * results['ElasticNet']['test_preds'] +
    w3 * results['BayesianRidge']['test_preds'])  
 
print(f"Best Weights: w_ridge: {w1}, w_elastic: {w2}, w_bayes: {w3}")   

# Converting log predictions back to original values            
test_preds = np.expm1(final_test_preds) 

# Ensure no negative predictions
test_preds = np.maximum(test_preds, 0)

# Create the submission DataFrame      
submission = pd.DataFrame({
    'id': test_df['id'],
    'Calories': test_preds
})

# Save the submission file
submission.to_csv('submission.csv', index=False)
print("Final submission file created")  

submission.head()                                


submission['Calories'].describe()


# Test prediction           
plt.figure(figsize=(8, 4))
plt.hist(data=submission, x='Calories', bins=100)
plt.title('Data Distribution of Test Prediction', size=12, weight='bold');     

