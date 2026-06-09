import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


import warnings
warnings.filterwarnings('ignore')



df = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")

print(f'Shape of Train Data: {df.shape}')
print(f'Shape of Test Data: {test_data.shape}')


df.head()


df.isnull().sum()


test_data.isnull().sum()


df.info()


df.Sex.value_counts()


sns.boxplot(x='Sex', y='Calories', data=df)
plt.title('Calories Distribution by Gender')
plt.show()


df.Sex.value_counts()


sns.pairplot(df[['Sex','Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Calories']], diag_kind='kde')
plt.show()


cols = df.columns
cat_cols = [col for col in cols if df[col].dtype == "object"]
num_cols = [col for col in df.columns if col not in cat_cols and col != "id"]



num_features = len(num_cols)
fig, axes = plt.subplots(num_features, 2, figsize=(12, 5 * num_features))

# Loop through numerical columns and create plots
for i, col in enumerate(num_cols):
    # Histogram
    sns.histplot(df[col], bins=30, kde=True, ax=axes[i, 0])
    axes[i, 0].set_title(f'Distribution of {col}')
    
    # Boxplot
    sns.boxplot(x=df[col], ax=axes[i, 1])
    axes[i, 1].set_title(f'Boxplot of {col}')

plt.tight_layout()
plt.show()


# Use only numeric columns
numeric_df = df.select_dtypes(include=['float64', 'int64'])

# Example: heatmap of correlations
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.show()


df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})
test_data['Sex'] = test_data['Sex'].map({'male':1, 'female': 0})


y = df['Calories']
Xx = df.drop(['Calories','id'],axis = 1)
test = test_data.drop(columns=['id'],axis=1)



from sklearn.preprocessing import StandardScaler

sc =  StandardScaler()
X = sc.fit_transform(Xx)
test = sc.transform(test)


from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, mean_squared_log_error
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, Ridge, HuberRegressor, ElasticNetCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor



# Define models
models = [
    ("LinearRegression", LinearRegression()),
    ("Ridge", Ridge()),
    ("Huber", HuberRegressor()),
    ("ElasticNetCV", ElasticNetCV()),
    ("DecisionTree", DecisionTreeRegressor()),
    ("RandomForest", RandomForestRegressor()),
    ("ExtraTrees", ExtraTreesRegressor()),
    ("GradientBoosting", GradientBoostingRegressor()),
    ("XGBoost", XGBRegressor(verbosity=0)),
    ("CatBoost", CatBoostRegressor(verbose=0)),
    ("LightGBM", LGBMRegressor())
]

# RMSLE scorer
def rmsle(y_true, y_pred):
    y_pred = np.maximum(0, y_pred)  # clip negatives
    return np.sqrt(mean_squared_log_error(y_true, y_pred))
    
rmsle_scorer = make_scorer(rmsle, greater_is_better=False)



# # Define 5-fold CV
# kf = KFold(n_splits=5, shuffle=True, random_state=42)

# # Store results
# results = []

# for name, model in models:
    
#     scores = cross_val_score(model, X, y, cv=kf, scoring=rmsle_scorer)
#     mean_score = -scores.mean()
#     results.append((name, model, mean_score))

# for name, _, score in results:
#     print(f"{name}: RMSLE = {score}")



import optuna


# RMSLE scorer
def rmsle(y_true, y_pred):
    y_pred = np.maximum(0, y_pred)  # clip negatives
    return np.sqrt(mean_squared_log_error(y_true, y_pred))
    
# rmsle_scorer = make_scorer(rmsle, greater_is_better=False)

def cat_objective(trial):
    
    params = {
        'iterations': trial.suggest_int('iterations', 250, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.009, 0.3),
        'depth': trial.suggest_int('depth', 3, 9),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0, 20),
        'random_strength': trial.suggest_float('random_strength', 0, 10),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 5),
        'random_state': 42,

    }
    
    # Define StratifiedKFold cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    rmsle_scores = []  # To store accuracy for each fold
    
    #cross validation
    for train_index , val_index in cv.split(X,y):
        
        X_train_fold, X_val_fold = X[train_index],X[val_index]
        y_train_fold, y_val_fold = y[train_index],y[val_index]
        
        
        model = CatBoostRegressor(**params,logging_level='Silent')
        model.fit(X_train_fold,y_train_fold)
        
        # Predict and calculate accuracy for this fold
        preds = model.predict(X_val_fold)
        rmsle_score = rmsle(y_val_fold, preds)
        rmsle_scores.append(rmsle_score)
    
    return sum(rmsle_scores) / len(rmsle_scores)


# best_params = []

# study = optuna.create_study(direction='minimize')
# study.optimize(cat_objective, n_trials=50)
    
# best_params.append(study.best_params)


best_cat_params = {
    'iterations': 893,
    'learning_rate': 0.0850161338239067,
    'depth': 9,
    'l2_leaf_reg': 1.0140499700812582,
    'random_strength': 0.037010325792570575,
    'bagging_temperature': 0.3126279893276654
}



model = CatBoostRegressor(**best_cat_params)
model.fit(X,y)

y_pred = model.predict(test)

# Clip Predictions to Avoid Negative Values
y_preds = y_pred.clip(0)


submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')
submission['Calories'] = y_preds
submission.to_csv('submission.csv', index=False)
print("Submission file saved as submission.csv!")





