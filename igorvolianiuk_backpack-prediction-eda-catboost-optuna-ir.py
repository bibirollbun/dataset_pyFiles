import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from catboost import CatBoostRegressor, Pool
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings
warnings.filterwarnings("ignore")


train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv").drop('id', axis=1)
train.head()


train.info()


test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv").drop('id', axis=1)
test.head()


test.info()


original = pd.read_csv('/kaggle/input/student-bag-price-prediction-dataset/Noisy_Student_Bag_Price_Prediction_Dataset.csv')
original.head()


sns.set(style="whitegrid")

fig, axes = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

sns.histplot(train['Price'], kde=True, bins=30, color='skyblue', ax=axes[0])
axes[0].set_title("Target Distribution - Train Data")
axes[0].set_ylabel("Count")
axes[0].grid(axis='y', linestyle='--', alpha=0.7)

sns.histplot(original['Price'], kde=True, bins=30, color='skyblue', ax=axes[1])
axes[1].set_title("Target Distribution - Original Data")
axes[1].set_xlabel("Price")
axes[1].set_ylabel("Count")
axes[1].grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()


cat_cols = ['Brand', 'Material', 'Size', 'Compartments', 'Laptop Compartment',
       'Waterproof', 'Style', 'Color']

fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(12, 16))

for ax, col in zip(axes.flatten(), cat_cols):
    train[col].value_counts().plot(
        kind='barh', 
        color='r', 
        title=f'Backpacks {col}',
        ax=ax 
    )

plt.tight_layout() 
plt.show()


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

X =  pd.concat([train, original], axis=0).drop('Price', axis=1)
y = [0] * len(train) + [1] * len(original)

cols_to_update = ['Brand', 'Material', 'Size', 'Compartments', 'Laptop Compartment',
       'Waterproof', 'Style', 'Color']

for col in cols_to_update:
    X[col].fillna('None')
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

X['Weight Capacity (kg)'] = X['Weight Capacity (kg)'].fillna(X['Weight Capacity (kg)'].median())
model = RandomForestClassifier(random_state=0)
cv_preds = cross_val_predict(model, X, y, cv=5, n_jobs=-1, method='predict_proba')

score = roc_auc_score(y_true=y, y_score=cv_preds[:,1])
print(f"roc-auc score: {score:0.3f}")


train[cols_to_update] = train[cols_to_update].fillna('None').astype('string').astype('category')
median_weight = train['Weight Capacity (kg)'].median()

# The "Weight Capacity (kg)" feature is crucial and is split into two: 
# one categorical and one numerical column
train['Weight Capacity (kg) categorical'] = train['Weight Capacity (kg)'].fillna(median_weight).astype('string')
train['Weight Capacity (kg)'] = train['Weight Capacity (kg)'].fillna(median_weight).astype('float64')

X = train.drop('Price', axis=1)
y = train.Price

# Same goes for test data
test[cols_to_update] = test[cols_to_update].fillna('None').astype('string').astype('category')
test['Weight Capacity (kg) categorical'] = test['Weight Capacity (kg)'].fillna(median_weight).astype('string')
test['Weight Capacity (kg)'] = test['Weight Capacity (kg)'].fillna(median_weight)


cat_cols = ['Brand', 'Material', 'Size', 'Compartments', 'Laptop Compartment',
       'Waterproof', 'Style', 'Color', 'Weight Capacity (kg) categorical']

# Optuna Tuning
def objective(trial):
    params = {
    'loss_function': 'RMSE',
    'eval_metric': 'RMSE',
    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
    'iterations': 2000,
    'depth': trial.suggest_int('depth', 3, 10),
    'random_strength': 0,
    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
    'task_type':'GPU',
    'random_seed':42,
    'verbose':False
    }
    
    
    cv = KFold(5, shuffle=True, random_state=0)
    cv_splits = cv.split(X, y)
    scores = list()
    for train_idx, val_idx in cv_splits:
        model = CatBoostRegressor(**params)
        X_train_fold, X_val_fold = X.loc[train_idx], X.loc[val_idx]
        y_train_fold, y_val_fold = y.loc[train_idx], y.loc[val_idx]
        X_train_pool = Pool(X_train_fold, y_train_fold, cat_features=cat_cols)
        X_valid_pool = Pool(X_val_fold, y_val_fold, cat_features=cat_cols)
        model.fit(X=X_train_pool, eval_set=X_valid_pool, verbose=False, early_stopping_rounds=200)
        val_pred = model.predict(X_valid_pool)
        score = np.sqrt(mean_squared_error(y_val_fold, val_pred))
        scores.append(score)  
    return np.mean(scores)

sqlite_db = "sqlite:///catboost.db"
study_name = "catboost"

optimize = False

if optimize:
    study = optuna.create_study(storage=sqlite_db, study_name=study_name, 
                                sampler=TPESampler(n_startup_trials=35, multivariate=True, seed=0),
                                direction="minimize", load_if_exists=True)

    study.optimize(objective, n_trials=100)
    print(f"best optimized RMSE: {study.best_value:0.5f}") 
    print(f"best hyperparameters: {study.best_params}") 
    catboost_params = study.best_params
else:
    catboost_params = {
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        'learning_rate': 0.05550266178302702,
        'iterations': 2000,
        'depth': 4,
        'random_strength': 0,
        'l2_leaf_reg': 5.189087598805998,
        'task_type':'CPU',
        'random_seed': 42,
        'verbose': False    
    }


def postprocess(y_true, y_pred, y_pred_test):
    IR = IsotonicRegression(out_of_bounds='clip')
    IR.fit(y_pred.reshape(-1, 1), y_true)
    return IR.predict(y_pred_test)

cv = KFold(5, shuffle=True, random_state=0)
cv_splits = cv.split(X, y)

scores = []
test_preds = []

X_test_pool = Pool(test, cat_features=cat_cols)

for train_idx, val_idx in cv_splits:
    model = CatBoostRegressor(**catboost_params)
    
    X_train_fold, X_val_fold = X.loc[train_idx], X.loc[val_idx]
    y_train_fold, y_val_fold = y.loc[train_idx], y.loc[val_idx]
    
    X_train_pool = Pool(X_train_fold, y_train_fold, cat_features=cat_cols)
    X_valid_pool = Pool(X_val_fold, y_val_fold, cat_features=cat_cols)
    
    model.fit(X=X_train_pool, eval_set=X_valid_pool, verbose=200, early_stopping_rounds=200)
    
    val_pred = model.predict(X_valid_pool)
    score = np.sqrt(mean_squared_error(y_val_fold, val_pred))
    scores.append(score)

    test_pred = model.predict(X_test_pool)
    test_pred = postprocess(y_val_fold, val_pred, test_pred)
    test_preds.append(test_pred)

print(f'Cross-validated RMSE score: {np.mean(scores):.3f} +/- {np.std(scores):.3f}')
print(f'Max RMSE score: {np.max(scores):.3f}')
print(f'Min RMSE score: {np.min(scores):.3f}')


explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train_pool)


shap.summary_plot(
    shap_values, 
    X, 
    plot_type="bar", 
    class_names=np.unique(y),
    color='purple',
    show=False
)
plt.xticks(fontsize=14)  
plt.yticks(fontsize=14)  
plt.xlabel('Mean Absolute SHAP Value', fontsize=14) 
plt.title('Feature Importance by SHAP Values', fontsize=16) 
plt.grid(visible=True, which='both', linestyle='--', linewidth=0.5) 
plt.show()


shap.summary_plot(shap_values, X.columns, plot_type="violin", show=False)
plt.xticks(fontsize=12) 
plt.yticks(fontsize=12)  
plt.title('Feature Importance Distribution by SHAP Values', fontsize=14) 
plt.grid(visible=True, which='both', linestyle='--', linewidth=0.5)  
plt.show()


sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')
sample_submission['Price'] = np.mean(test_preds, axis=0)
sample_submission.to_csv('submission.csv', index=False)
sample_submission.head(10)

