import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt        

from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import lightgbm as lgb
from lightgbm import LGBMClassifier, early_stopping
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier 
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# Import packages for warnings
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')   

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)   


train_df = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
train_df.drop(columns=['id']).head()                                


test_df.head()


print("Train:", train_df.shape)
print("Test:", test_df.shape)


train_df['Personality'].value_counts()     


train_df['Personality'].value_counts(normalize=True) * 100    


train_df.drop(columns=['id']).describe()      


test_df.drop(columns=['id']).describe()      


print('Train:\n', train_df.isna().sum())
print()
print('Test:\n', test_df.isna().sum())


train_missing_perc = train_df.isna().sum().sum() / train_df.size * 100
test_missing_perc = test_df.isna().sum().sum() / test_df.size * 100

print(f"Train Dataset Missing Values: {train_missing_perc:.2f}%")
print(f"Test Dataset Missing Values: {test_missing_perc:.2f}%")     


missing_mean = lambda x: x.isna().mean() 
train_df.groupby('Personality')['Post_frequency'].apply(missing_mean) 


missing_counts = train_df[train_df['Post_frequency'].isna()].groupby('Personality')['Post_frequency'].size()   
missing_counts 


train_df.drop(columns=['id']).head()


train_df[(train_df['Personality'] == 'Introvert') & (train_df['Stage_fear'] == 'No')
         & (train_df['Post_frequency'] >= 5)].drop(columns=['id']).sample(5)      


train_df[(train_df['Personality'] == 'Extrovert') & (train_df['Stage_fear'] == 'Yes')
         & (train_df['Post_frequency'] < 5)].drop(columns=['id']).sample(5)      


np.isinf(train_df.select_dtypes(include='number')).sum().sum()


# Encoding `Stage_fear`         
train_df['Stage_fear_enco'] = np.where(train_df['Stage_fear'] == 'Yes', 1, 0)
test_df['Stage_fear_enco'] = np.where(test_df['Stage_fear'] == 'Yes', 1, 0)

# Encoding `Drained_after_socializing`
train_df['Drained_after_socializing_enco'] = np.where(train_df['Drained_after_socializing'] == 'Yes', 1, 0)
test_df['Drained_after_socializing_enco'] = np.where(test_df['Drained_after_socializing'] == 'Yes', 1, 0)

# Drop original features
train_df = train_df.drop(columns=['Stage_fear', 'Drained_after_socializing'])
test_df = test_df.drop(columns=['Stage_fear', 'Drained_after_socializing']) 

# Interaction & ratio features:
def interaction_ratio(df):
    df['going_out_x_drained'] = df['Going_outside'] * df['Drained_after_socializing_enco']
    df['time_spent_alone_x_stage_fear'] = df['Time_spent_Alone'] * df['Stage_fear_enco']
    df['social_vs_alone_ratio'] = df['Social_event_attendance'] / np.where(df['Time_spent_Alone'] == 0, 1, df['Time_spent_Alone'])
    df['friend_circle_vs_alone_ratio'] = df['Friends_circle_size'] / np.where(df['Time_spent_Alone'] == 0, 1, df['Time_spent_Alone'])
    df['posting_freq_vs_social_event_ratio'] = df['Post_frequency'] / np.where(df['Social_event_attendance'] == 0, 1, df['Social_event_attendance'])
    df['social_attendence_per_outing'] = df['Social_event_attendance'] / np.where(df['Going_outside'] == 0, 1, df['Going_outside'])
    df['alone_to_friends_ratio'] = df['Time_spent_Alone'] / np.where(df['Friends_circle_size'] == 0, 1, df['Friends_circle_size'])
    return df

# Apply
train_df = interaction_ratio(train_df)
test_df = interaction_ratio(test_df)  


train_df.head(1)


def probability_encoding(train_df, valid_df, test_df, group_cols, target_col):
    prefix = f"{'_'.join(group_cols)}_prob"
    ohe = pd.get_dummies(train_df[target_col], prefix=prefix, dtype=float)
    temp_df = pd.concat([train_df[group_cols].reset_index(drop=True), ohe], axis=1)
    group_prob = temp_df.groupby(group_cols).mean().reset_index()

    train_df = train_df.merge(group_prob, on=group_cols, how='left')
    valid_df = valid_df.merge(group_prob, on=group_cols, how='left')
    test_df = test_df.merge(group_prob, on=group_cols, how='left')
    
    return train_df, valid_df, test_df          


# Features & Target 
X = train_df.drop(columns=['id'])
y = np.where(train_df['Personality'] == 'Extrovert', 1, 0)
X_test = test_df.drop(columns=['id'])

print(X.shape, y.shape, X_test.shape)

# LightGBM hyper-parameter
lgbm_params = {  
    'n_estimators': 6000,
    'learning_rate': 0.01,
    'subsample': 0.75, 
    'colsample_bytree': 0.75,
    'reg_lambda': 0.3, 
    'reg_alpha': 0.3, 
    'num_leaves': 64, 
    'max_depth': 8, 
    'min_child_samples': 30, 
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'device': 'gpu',
    'random_state': 42,
    'verbosity': -1,
}

# XGBoost hyper-parameter
xgb_params = {
    'n_estimators': 10000,
    'learning_rate': 0.02,
    'max_depth': 8,
    'subsample': 0.75,
    'colsample_bytree': 0.75,
    'reg_lambda': 0.3,
    'reg_alpha': 0.3,
    'objective': 'binary:logistic',
    'tree_method': 'gpu_hist',  
    'random_state': 42,
    'eval_metric': 'auc'
}


def train_model(X, y, X_test, model_name, params):
    print(f"\n##### Training {model_name} #####")
    oof_preds = np.zeros(X.shape[0])
    test_preds = np.zeros(X_test.shape[0])
    accuracy_score_list = []
    feature_importances = pd.DataFrame()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)): 
        print(f"\n### Training Fold {fold +1} ###")

        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_valid, y_valid = X.iloc[valid_idx], y[valid_idx]
        X_test_fold = X_test.copy()
        
        # Add probability encoded (mean) features
        groupings = [
        ['Time_spent_Alone', 'Stage_fear_enco'],
        ['Friends_circle_size', 'Social_event_attendance'],
        ['Post_frequency', 'Going_outside']]  

        for group_cols in groupings:
            X_train, X_valid, X_test_fold = probability_encoding(
                train_df = X_train,
                valid_df = X_valid,
                test_df = X_test_fold,
                group_cols = group_cols,
                target_col = 'Personality')

        # Drop Target Variable
        X_train.drop(columns=['Personality'], inplace=True)
        X_valid.drop(columns=['Personality'], inplace=True)   

        # Train Model
        if model_name == 'LightGBM':   
                model = lgb.LGBMClassifier(**params)
                model.fit(X_train, y_train,
                         eval_set=[(X_valid, y_valid)],
                         callbacks=[early_stopping(stopping_rounds=50, verbose=False)])

                # Feature importance
                fold_importance = pd.DataFrame({
                    'feature': X_train.columns,
                    'importance': model.feature_importances_,
                    'fold': fold + 1
                })
                feature_importances = pd.concat([feature_importances, fold_importance], axis=0)
            
        elif model_name == 'XGBoost':
                  model = XGBClassifier(**params)
                  model.fit(X_train, y_train,
                           eval_set=[(X_valid, y_valid)],
                           early_stopping_rounds=200, verbose=100)    
                  
                  # Feature importance
                  importance_dict = model.get_booster().get_score(importance_type='gain')
                  fold_importance = pd.DataFrame({
                      'feature': list(importance_dict.keys()),
                      'importance': list(importance_dict.values()),
                      'fold': fold + 1
                  })
                  feature_importances = pd.concat([feature_importances, fold_importance], axis=0)
        else:
          raise ValueError("Invalid model_name. Choose 'LightGBM' or 'XGBoost'.")

        # Validation & test predictions
        valid_preds = model.predict_proba(X_valid)[:, 1]
        test_preds += model.predict_proba(X_test_fold)[:, 1] / skf.n_splits
    
        # OOF Predictions
        oof_preds[valid_idx] = valid_preds
    
        # Score
        fold_accuracy = accuracy_score(y_valid, (valid_preds > 0.5).astype(int))
        accuracy_score_list.append(fold_accuracy)
        print(f"Fold {fold+1} Accuracy: {fold_accuracy:.6f}")

    # Average Accuracy Score
    print(f"\nAverage Accuracy: {np.mean(accuracy_score_list):.5f}\n")
    # Final OOF Accuracy Score
    oof_accuracy = accuracy_score(y, (oof_preds > 0.5).astype(int))
    print(f"Overall CV (OOF) Accuracy: {oof_accuracy:.5f}")

    return oof_preds, test_preds, feature_importances, accuracy_score_list    


%%time                
# Run LightGBM
lgbm_oof, lgbm_test, lgbm_feature_importances, lgbm_accuracy_score = train_model(X, y, X_test, 
                                                                             model_name='LightGBM', params=lgbm_params)


for thresh in [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]:
    preds = (lgbm_oof > thresh).astype(int)
    acc = accuracy_score(y, preds)
    print(f"Threshold: {thresh:.2f} | Accuracy: {acc:.5f}")


lgbm_test


# Convert probabilities to binary predictions 
lgbm_oof_preds = (lgbm_oof > 0.5).astype(int)

# Compute confusion matrix
cm = confusion_matrix(y, lgbm_oof_preds)

# Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Introvert", "Extrovert"])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix', weight='bold')
plt.show()


tn, fp, fn, tp = cm.ravel()
print(f"True Negatives: {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Positives: {tp}")


# Generate Classificaiton Report (Precision, Recall, F1-score)      
print(classification_report(y, lgbm_oof_preds, target_names=['Introvert', 'Extrovert']))


# Convert probability predictions to binary class      
test_preds_binary = (lgbm_test > 0.5).astype(int)

# Map class integers to string label
label_map = {0: 'Introvert', 1: 'Extrovert'}
test_preds_label = pd.Series(test_preds_binary).map(label_map)

# Create Submission File
submission = pd.DataFrame({
    'id': test_df['id'],
    'Personality': test_preds_label
})

submission.to_csv("submission.csv", index=False)
print('Submission File Created')
submission.head()


# Average importance from LightGBM
lgb_avg_importance = lgbm_feature_importances.groupby('feature')['importance'].mean().sort_values(ascending=False).reset_index()

# Plot LightGBM feature importance 
plt.figure(figsize=(8, 5))
sns.barplot(data=lgb_avg_importance, x='importance', y='feature')
plt.title("LightGBM Feature Importances (Gain)", weight="bold")
plt.xlabel("Average Gain Importance");     


%%time                
# Run XGBoost  
xgb_oof, xgb_test, xgb_feature_importances, xgb_accuracy_score = train_model(X, y, X_test, 
                                                                             model_name='XGBoost', params=xgb_params)                                                                         


# Average importance from XGBoost
xgb_avg_importance = xgb_feature_importances.groupby('feature')['importance'].mean().sort_values(ascending=False).reset_index()

# Plot XGBoost feature importance 
plt.figure(figsize=(8, 5))
sns.barplot(data=xgb_avg_importance, x='importance', y='feature')
plt.title("XGBoost Feature Importances (Gain)", weight="bold")
plt.xlabel("Average Gain Importance");           

