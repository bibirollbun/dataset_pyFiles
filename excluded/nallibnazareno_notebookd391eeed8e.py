import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
import optuna
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from eli5.sklearn import PermutationImportance
import eli5
from sklearn.feature_selection import RFE


subm = True


file_path = '/kaggle/input/playground-series-s5e3/train.csv'
train = pd.read_csv(file_path)
test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')

def con(t=True):
    global df
    if t:
        df = pd.concat([train, test])
    else:
        df=train

con(subm)


print("### Dataset Info ###\n")
df.info()


# Missing values count & percentage
missing_values = df.isnull().sum()
missing_percentage = (missing_values / len(df)) * 100
print("\n### Missing Values ###\n")
print(pd.DataFrame({'Missing Values': missing_values, 'Percentage': missing_percentage}).sort_values(by='Percentage', ascending=False))

# Correlation matrix
corr_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Matrix")
plt.show()


df['temp_range'] = df['maxtemp'] - df['mintemp']
df['dew_vs_temp'] = df['dewpoint'] - df['temparature']
df['pressure_temp_interaction'] = df['pressure'] * df['temparature']
df['wind_temp_interaction'] = df['windspeed'] * df['temparature']
df['humidity_windspeed'] = df['humidity'] * df['windspeed']

df['temp_bin'] = pd.cut(df['temparature'], bins=[-10, 0, 15, 30, 45], labels=['Cold', 'Cool', 'Warm', 'Hot'])
df['windspeed_bin'] = pd.cut(df['windspeed'], bins=[0, 10, 20, 30, 50], labels=['Calm', 'Moderate', 'Strong', 'Very Strong'])

df['wind_direction_sin'] = np.sin(df['winddirection'] * (2. * np.pi / 360))
df['wind_direction_cos'] = np.cos(df['winddirection'] * (2. * np.pi / 360))

df['pressure_lag'] = df['pressure'] - df['pressure'].shift(1)
df['Humidity_lag'] = df['humidity'] - df['humidity'].shift(1)
df['prev3_temp_avg'] = df['temparature'].rolling(window=3).mean()

df.drop(['winddirection','windspeed','maxtemp'], axis=1,inplace=True)


df['cloud/sunshine'] = df['cloud'] / df['sunshine'].clip(lower=0.1)


#One-Hot Encoding for categorical columns
df = pd.get_dummies(df, columns=['temp_bin', 'windspeed_bin'], drop_first=True)

# get features and num cols
features =[col for col in df.columns if col not in ['id','rainfall']]
numerical_cols = df[features].select_dtypes(include=['float64', 'int64']).columns
# Impute missing values
numeric_imputer = SimpleImputer(strategy='median')
df[features] = numeric_imputer.fit_transform(df[features])

# Normalize numerical features
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])


def split(df, target,submission=True):
    if not target:
        return 'no target'

    global X_train,X_test, y_train, y_test
    if submission:
        global X_test_id
        df_train = df.iloc[:len(train)]
        df_test = df.iloc[len(train):]
        X_train = df_train[features]
        y_train = df_train[target]
        X_test = df_test[features]
        X_test_id = df_test['id']
    else:
        X_train,X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2)

split(df, 'rainfall', subm)

## apply SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)


xgb_params = {'max_depth': 6, 'learning_rate': 0.011736595060580784, 'n_estimators': 146, 'subsample': 0.9247623803149709, 'colsample_bytree': 0.5709669447548013, 'lambda': 0.0008119952013126283, 'alpha': 0.03521973235241369}

lgb_params = {'n_estimators': 38, 'learning_rate': 0.034947387651004636, 'num_leaves': 12, 'max_depth': 1, 'feature_fraction': 0.3028229805611067, 'bagging_fraction': 0.4335067293589081, 'bagging_freq': 3, 'min_data_in_leaf': 28}

cb_params ={'iterations': 140, 'learning_rate': 0.03656952154730269, 'depth': 4, 'l2_leaf_reg': 5.403429330021046, 'bagging_temperature': 0.1761186933020996, 'border_count': 127, 'colsample_bylevel': 0.9040956903001539, 'boosting_type': 'Ordered'}

xgb_model = XGBClassifier(**xgb_params)
lgb_model = lgb.LGBMClassifier(**lgb_params)
cb_model = CatBoostClassifier(**cb_params)


"""def objective(trial):
    # Hiperparámetros a optimizar
    param = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'lambda': trial.suggest_float('lambda', 1e-8, 10.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-8, 10.0, log=True)
    }

    # Entrenar modelo
    model = XGBClassifier(**param)
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)
    
    return auc


study = optuna.create_study(direction='maximize')  # Maximizar precisión
study.optimize(objective, n_trials=50)  # Ejecutar 50 iteraciones

# Mostrar mejores hiperparámetros
print("Mejores hiperparámetros:", study.best_params)
print("Mejor precisión:", study.best_value)


def objective2(trial):
    # Hiperparámetros a optimizar
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 10, 1000),
        'learning_rate': trial.suggest_float('learning_rate',0.01,0.3),
        'num_leaves': trial.suggest_int('num_leaves', 10, 50),
        'max_depth': trial.suggest_int('max_depth',1, 10),
        'feature_fraction':trial.suggest_float('feature_fraction', 0.1, 1),
        'bagging_fraction':trial.suggest_float('bagging_fraction', 0.1, 1),
        'bagging_freq':trial.suggest_int('bagging_freq', 1, 3),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 30)
    }

    # Entrenar modelo
    model = lgb.LGBMClassifier(**param)
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)
    
    return auc


study2 = optuna.create_study(direction='maximize')  # Maximizar precisión
study2.optimize(objective2, n_trials=50)  # Ejecutar 50 iteraciones
print("Mejores hiperparámetros:", study2.best_params)
print("Mejor precisión:", study2.best_value)


def objective3(trial):
    # Define hyperparameter search space
    param = {
        "iterations": trial.suggest_int("iterations", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
        "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
        "loss_function": "Logloss",
        "verbose": 0  # Suppress training logs
    }

    # Initialize and train the model
    model = CatBoostClassifier(**param)
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)
    
    return auc  # Maximizing accuracy


study3 = optuna.create_study(direction='maximize')  # Maximizar precisión
study3.optimize(objective3, n_trials=50)  # Ejecutar 50 iteraciones
print("Mejores hiperparámetros:", study3.best_params)
print("Mejor precisión:", study3.best_value)"""


def feature_selection(sub=subm):
    global X_train, X_test
    # Initialize RFE
    selector = RFE(xgb_model, n_features_to_select=7, step=1)
    
    # Fit RFE
    selector.fit(X_train, y_train)
    
    
    
    print(type(selector.support_))
    print(selector.support_.shape)
    print("Selected Features:", selector.support_)
    
    print("Feature Ranking:", selector.ranking_)
    
    X_train = X_train.loc[:, selector.support_]
    X_test = X_test.loc[:, selector.support_]

feature_selection()
print(X_train)



xgb_model.fit(X_train, y_train)
lgb_model.fit(X_train, y_train)
cb_model.fit(X_train, y_train)



def permutation(sub=True):
    if not sub:
        catboost_perm = PermutationImportance(cb_model, random_state=42).fit(X_test, y_test)
        
        
        # Calculate permutation importance for XGBoost
        
        xgb_perm = PermutationImportance(xgb_model, random_state=42).fit(X_test, y_test)
        
        
        # Calculate permutation importance for LightGBM
        
        lgbm_perm = PermutationImportance(lgb_model, random_state=42).fit(X_test, y_test)
        
        
        # Extract feature importances
        
        catboost_importances = catboost_perm.feature_importances_
        
        xgb_importances = xgb_perm.feature_importances_
        
        lgbm_importances = lgbm_perm.feature_importances_
        
        
        # Calculate mean importances
        
        mean_importances = np.mean([catboost_importances, xgb_importances, lgbm_importances], axis=0)
        
        
        # Get feature names
        
        feature_names = X_test.columns.tolist()  # Ensure X_test is a DataFrame
        
        
        # Create a DataFrame for better visualization
        
        importances_df = pd.DataFrame({
        
            'Feature': feature_names,
        
            'Mean Permutation Importance': mean_importances
        
        })
        
        
        # Sort the DataFrame by importance
        
        importances_df = importances_df.sort_values(by='Mean Permutation Importance', ascending=False)
        
        
        # Display the feature importances using eli5
        
        eli5.show_weights(catboost_perm, feature_names=feature_names)
        
        eli5.show_weights(xgb_perm, feature_names=feature_names)
        
        eli5.show_weights(lgbm_perm, feature_names=feature_names)

        print(importances_df)




corr_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=1)
plt.title("Feature Correlation Matrix")
plt.show()


def roc(sub=True):
    if not sub:
        xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
    
        lgb_probs = lgb_model.predict_proba(X_test)[:, 1]
    
        cb_probs = cb_model.predict_proba(X_test)[:, 1]
    
    
        # Compute FPR, TPR for each model
        fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_probs)
        fpr_lgb, tpr_lgb, _ = roc_curve(y_test, lgb_probs)
        fpr_cb, tpr_cb, _ = roc_curve(y_test, cb_probs)
    
    
        # Compute AUC scores
    
        auc_xgb = roc_auc_score(y_test, xgb_probs)
        auc_lgb = roc_auc_score(y_test, lgb_probs)
        auc_cb = roc_auc_score(y_test, cb_probs)
        
        
        # Plot ROC Curves
        plt.figure(figsize=(8, 6))
        plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {auc_xgb:.3f})', linestyle='dotted', color='red')
        plt.plot(fpr_lgb, tpr_lgb, label=f'LightGBM (AUC = {auc_lgb:.3f})', linestyle='solid', color='green')
        plt.plot(fpr_cb, tpr_cb, label=f'CatBoost (AUC = {auc_cb:.3f})', linestyle='dashed', color='blue')
        
        
        # Reference line for random model
        
        plt.plot([0, 1], [0, 1], linestyle='--', color='black', alpha=0.6)
        
        # Labels and Title
        
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title('ROC Curve Comparison')
        plt.legend(loc='lower right')
        plt.grid(True)
        
        
        # Show plot
        
        plt.show()




permutation(subm)

roc(subm)





print(X_test_id)
X_test_id = X_test_id.reset_index()['id']
print(X_test_id)


y_pred = pd.DataFrame({'1': xgb_model.predict(X_test), '2': cb_model.predict(X_test), '3': lgb_model.predict(X_test)})
y_pred = y_pred.mean(axis=1).round()
submission = pd.DataFrame({'id': X_test_id, 'pred': y_pred})
print(submission)
submission.to_csv('submission.csv', index=False)

submission2 = pd.DataFrame({'id':X_test_id,'pred':cb_model.predict(X_test)})
submission2.to_csv('submission2.csv', index=False)


