import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import warnings
from xgboost import XGBClassifier, plot_importance
from lightgbm import LGBMClassifier, plot_importance
import lightgbm as lgb
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
import optuna
from scipy.stats import rankdata

pd.set_option('display.float_format', '{:.6f}'.format)


warnings.filterwarnings("ignore")


# load dataset
df_train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')


# view dataset summary
df_train.describe()


df_train.info()


df_train.isna().sum()


df_test.describe()


df_test.info()


df_test.isna().sum()

#1 missing value in winddirection in test set


# We drop 'id' and 'day' variables, since we are noobs and
# do not know how to deal with them yet T.T

df_train = df_train.drop(['id', 'day'], axis=1)
sub_test = df_test.drop(['id', 'day'], axis=1)


X = df_train.drop(['rainfall'], axis = 1)
y = df_train['rainfall']


y.value_counts()

# imbalance target variable


#make a df for classes
class_counts = y.value_counts().reset_index()
class_counts.columns = ['class', 'count']

sns.barplot(x='class', y='count', data=class_counts)
plt.xlabel('rainfall')
plt.ylabel('count')
plt.title('Distribution of Target Variable')
plt.show()


plt.figure(figsize=(12, 10))

for idx, col in enumerate(X.columns, 1):
    plt.subplot(4, 3, idx)

    sns.histplot(df_train, 
                 x=col, 
                 kde=True, 
                 hue='rainfall', 
                 alpha=0.5
                 )

    plt.title(col)

plt.tight_layout()
plt.show()
    


sns.pairplot(df_train, hue='rainfall')


# View correlations between independent variables
# some variables are highly correlated
sns.heatmap(X.corr(), annot=True, cmap='coolwarm', fmt='.2f')


# 1. Imbalance target variable
# 2. Independent variables are all numerical variables
# 3. Some independent variables are highly correlated

# solutions:
# 1. Oversampling SMOTE
# 3. Using non-linear models if want to keep all variables



# define a function for feature engineering without dropping the columns
def feature_engineering_no_drop(df):
    
    #df = df.drop(['id', 'day'], axis = 1)

    for col in df.columns:
        df[col].replace([np.inf, -np.inf], np.nan, inplace = True)
        df[col] = df[col].fillna(df[col].median())

    df['temp_diff'] = df['maxtemp'] - df['mintemp']

    df['relative_humidity'] = 100 * (np.exp((17.625 * df['dewpoint']) / (243.04 + df['dewpoint'])) / np.exp((17.625 * df['temparature']) / (243.04 + df['temparature'])))

    df['dewpoint_depression'] = df['temparature'] - df['dewpoint']

    df['cloud_sunshine_ratio'] = df['cloud'] / (df['sunshine']+ 1e-6)  # Avoid division by zero

    df['cloud_temp_ratio'] = df['cloud'] / df['temparature']

    #try humidity_temp_interaction
    df['humidity_temp_interaction'] = df['humidity'] * df['temparature']

    #try sunshine_temp_interaction
    df['sunshine_temp_interaction'] = df['sunshine'] * df['temparature']

    #try cloud_humidity_interaction
    #df['cloud_humidity_interaction'] = df['cloud'] * df['humidity']

###################################################################################################################################################################################
# The following features degrade my LB scores
# 'Cloud' and 'humidity' are most important features in my models,
# and i try to interact them with other variabes to increase my model complexity
# but some of them do not work well and degrade my LB scores
    
    #df['heat_index'] = 0.5 * (df['temparature'] + 61.0 + ((df['temparature'] - 68.0) * 1.2) + (df['humidity'] * 0.094))

    #df['temp_sunshine'] = df['temparature'] * df['sunshine']

    #df['dewpoint_humidity'] = df['dewpoint'] * df['humidity']

    #df['temp_humidity'] = df['temparature'] * df['humidity']

    #df['pressure_wind'] = df['pressure'] * df['windspeed']

    #df['cloud_sunshine'] = df['cloud'] * df['sunshine']

    #df['cloud_temp'] = df['cloud'] * df['temparature']

    #df['cloud_dewpoint'] = df['cloud'] * df['dewpoint']

    #df['cloud_pressure'] = df['cloud'] * df['pressure']

    #df['cloud_wind'] = df['cloud'] * df['windspeed']

    #for col in df.columns:
    #    df[col].replace([np.inf, -np.inf], np.nan, inplace = True)
    #    df[col] = df[col].fillna(df[col].median())

    return df


X_featured = feature_engineering_no_drop(X)
sub_test = feature_engineering_no_drop(sub_test)


plt.figure(figsize=(12,10))
sns.heatmap(X_featured.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.show()


X_train, X_test, y_train, y_test = train_test_split(X_featured, y, test_size=0.2, stratify=y, random_state=42)


stratified_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)


base_xgb = XGBClassifier(n_estimators=200, 
                         random_state=42, 
                         sampling_method='gradient_based', 
                         eta=0.1,
                         max_depth=0,
                         eval_metric='auc',
                         device='gpu'
                        )

base_lgbm = LGBMClassifier(n_estimators=200, 
                           random_state=42,
                           max_depth=-1,
                           device='gpu'
                          )

base_rf = RandomForestClassifier(n_estimators=200, 
                                random_state=42,
                                max_depth=None
                                )

base_gb = GradientBoostingClassifier(n_estimators=200, 
                                    random_state=42)

base_mlp = MLPClassifier(max_iter=1000, 
                       random_state=42, 
                       activation='tanh', 
                       learning_rate='adaptive', 
                       early_stopping=True)

base_models_list = {
    'XGBoost': base_xgb,
    'LightGBM': base_lgbm,
    'Random Forest': base_rf,
    'Gradient Boosting': base_gb,
    'MLP': base_mlp
}



def evaluate_models(models):

    final_scores = {}
    final_oof_pred = {}
    final_test_pred = {}


    for name, model in models.items():
        print(f"Training {name}...")
    
        model_pipeline = Pipeline(steps = [
            ('scaler', StandardScaler()),
            ('oversampling', SMOTE(sampling_strategy='minority', random_state=42)),
            ('model', model)
        ])
    
        scores = []
        oof_pred = np.zeros(X_train.shape[0])
        test_pred = np.zeros(X_test.shape[0])
    
        for fold_idx, (train_idx, val_idx) in enumerate(stratified_cv.split(X_train, y_train)):
            print(f" Fold {fold_idx + 1}")
            X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
            model_pipeline.fit(X_train_fold, y_train_fold)
            y_pred_val = model_pipeline.predict_proba(X_val_fold)[:, 1]
    
            oof_pred[val_idx] = y_pred_val
            test_pred += model_pipeline.predict_proba(X_test)[:, 1] / stratified_cv.n_splits
    
    
            score = roc_auc_score(y_val_fold, y_pred_val)
            scores.append(score)
            
    
        final_scores[name] = np.mean(scores)
        final_oof_pred[name] = oof_pred
        final_test_pred[name] = test_pred

    print('Done')

    final_scores_df = pd.DataFrame(final_scores, index = ['AUC']).T
    
    oof_auc_scores = {}
    for name, oof_pred in final_oof_pred.items():
        oof_auc_scores[name] = roc_auc_score(y_train, oof_pred)
    
    # Convert to DataFrame
    oof_auc_scores = pd.DataFrame(oof_auc_scores, index=['oof_AUC']).T
    
    test_auc_scores = {}
    for name, test_pred in final_test_pred.items():
        test_auc_scores[name] = roc_auc_score(y_test, test_pred)
    
    # Convert to DataFrame
    test_auc_scores = pd.DataFrame(test_auc_scores, index=['test_AUC']).T
    
    # Combine the scores into a single DataFrame
    combined_scores = pd.concat([final_scores_df, oof_auc_scores, test_auc_scores], axis=1).sort_values('test_AUC', ascending=False)
    combined_scores['gaps_between_test_oof'] = combined_scores['test_AUC'] - combined_scores['oof_AUC']
    
    return combined_scores



evaluate_models(base_models_list)


# 1. MLP and Random Forest have the highest test_AUC scores
# 2. Random Forest is the most stable tree model based on the gaps between oof and test scores
# 3. We will choose MLP and Random Forest as our best base model
# 4. We choose to ensemble MLP and Random Forest to improve our final scores, 
# since Random Forest, XGB, LGBM and GB are all tree based models


ensembled_models = [
    ('Random Forest', base_rf),
    ('MLP', base_mlp)
]

def final_voting(models):

    voting_clf = VotingClassifier(
        estimators = models,
        voting = 'soft',
        n_jobs = -1
    )

    model_pipeline = Pipeline(steps = [
        ('scaler', StandardScaler()),
        ('oversampling', SMOTE(sampling_strategy='minority', random_state=42)),
        ('model', voting_clf)
    ])

    model_pipeline.fit(X_train, y_train)
    y_pred = model_pipeline.predict_proba(sub_test)[:,1]

    final_sub = df_test.copy()
    final_sub['rainfall'] = y_pred
    final_submission = final_sub[['id', 'rainfall']]

    return final_submission

final_sub = final_voting(ensembled_models)
final_sub.head(10)


final_sub.to_csv('final_voting.csv', index=False) # got LB 0.85


# I saws many pros use this method for higher LB scores
# The basic theory behind this is to set weights of your current submission and previous submission,
# then, combine them together



my_prev_sub = pd.read_csv('/kaggle/input/rainfall-mlp-result/final_mlp.csv')


my_prev_sub.head(10)


ranked_sub = my_prev_sub.copy()
current_score = final_sub.rainfall.values
prev_score = my_prev_sub.rainfall.values


ranked_sub.rainfall = -0.5*rankdata(current_score) + 1.25*rankdata(prev_score)
ranked_sub.rainfall = rankdata(ranked_sub.rainfall) / len(ranked_sub)

ranked_sub.head(10)


ranked_sub.to_csv('final_ranked_submission.csv', index=False)

