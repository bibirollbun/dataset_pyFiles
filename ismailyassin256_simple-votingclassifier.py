import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import optuna
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score, recall_score



# Loading the datasets
train_data = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")

# The shape of the data
print("#"*40)
print(f"The shape of train data is: {train_data.shape}")
print(f"The shape of test data is: {test_data.shape}")
print("#"*40)


# Function to load and wrangle the data
def wrangle_train(filepath):
    # Loading the data
    df = pd.read_csv(filepath)

    # Dropping null values
    df = df.dropna()

    # Dropping redundant columns
    df = df.drop('id', axis=1)
    
    return df


train = wrangle_train("/kaggle/input/playground-series-s5e3/train.csv")


train.head()


# Features
X = train.iloc[:,:-1]

# Target vector
y = train['rainfall']

print("#"*40)
print(f"The shape of X is: {X.shape}")
print(f"The shape of y is: {y.shape}")
print("#"*40)

# since the data is small, no splitting into X_train, X_test


# Instantiating the model
model = RandomForestClassifier()

# Fitting the model
model.fit(X,y)

importances = model.feature_importances_

importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
})

# Sort the DataFrame by importance
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot using seaborn
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
plt.title('Feature Importance')
plt.show()


# Create a dictionary of classifiers 
cls = {
    'RandomForest': RandomForestClassifier(class_weight='balanced', verbose=0),
    'GradientBoosting': GradientBoostingClassifier(), 
    'CatBoost': CatBoostClassifier(verbose=0, scale_pos_weight=1), 
    'LightGBM': LGBMClassifier(class_weight='balanced', verbose=0)
}


# Cross validating all the mdoels
results = {}
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for name, model in cls.items():
    scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy', n_jobs=-1)
    results[name] = scores.mean()

results


# RandomForest

def objective_RF(trial):
    # Hyperparameters for RandomForest
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 3, 30)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 30)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
                                         
    model = RandomForestClassifier(n_estimators=n_estimators, 
                                   max_depth=max_depth, 
                                   min_samples_split=min_samples_split, 
                                   class_weight='balanced',
                                   min_samples_leaf=min_samples_leaf,
                                   verbose=0)

    score = cross_val_score(model, X, y, cv=kf, scoring='accuracy', n_jobs=-1).mean()

    return score

study_RF = optuna.create_study(direction='maximize')
study_RF.optimize(objective_RF, n_trials=50)
best_params_RF =study_RF.best_params

# Print the best hyperparameters found by Optuna
print(f"Best hyperparameters: {best_params_RF}")
print(f"Best accuracy: {study_RF.best_value:.4f}")



# PLotting the optimization history
optuna.visualization.plot_optimization_history(study_RF)


# parameter importance
optuna.visualization.plot_param_importances(study_RF)


# GradientBoosting

def objective_GB(trial):
         # Suggest hyperparameters for GradientBoostingClassifier
        n_estimators = trial.suggest_int('n_estimators', 50, 300)  
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.9)  
        max_depth = trial.suggest_int('max_depth', 3, 30) 
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)  
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20) 
        subsample = trial.suggest_float('subsample', 0.5, 1.0)  
        max_features = trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2', None])  
                                         
        model = GradientBoostingClassifier(
                                            n_estimators=n_estimators,
                                            learning_rate=learning_rate,
                                            max_depth=max_depth,
                                            min_samples_split=min_samples_split,
                                            min_samples_leaf=min_samples_leaf,
                                            subsample=subsample,
                                            max_features=max_features,
                                            random_state=42
                                        )

        score = cross_val_score(model, X, y, cv=kf, scoring='accuracy', n_jobs=-1).mean()

        return score

study_GB = optuna.create_study(direction='maximize')
study_GB.optimize(objective_GB, n_trials=50)
best_params_GB =study_GB.best_params

# Print the best hyperparameters found by Optuna
print(f"Best hyperparameters: {best_params_GB}")
print(f"Best accuracy: {study_GB.best_value:.4f}")


# PLotting the optimization history
optuna.visualization.plot_optimization_history(study_GB)


# parameter importance
optuna.visualization.plot_param_importances(study_GB)


# CatBoostClassifier

def objective_CB(trial):
    # Suggest hyperparameters for CatBoostClassifier
    iterations = trial.suggest_int('iterations', 50, 300)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.9)
    depth = trial.suggest_int('depth', 3, 12)  
    l2_leaf_reg = trial.suggest_float('l2_leaf_reg', 1, 10)  
    subsample = trial.suggest_float('subsample', 0.6, 1.0)  
    colsample_bylevel = trial.suggest_float('colsample_bylevel', 0.6, 1.0) 
    class_weights = trial.suggest_categorical('class_weights', [None, 'balanced']) 
    random_strength = trial.suggest_float('random_strength', 1, 10) 
    bagging_temperature = trial.suggest_float('bagging_temperature', 0, 2) 
    one_hot_max_size = trial.suggest_int('one_hot_max_size', 2, 16)  
    grow_policy = trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide'])
    border_count = trial.suggest_int('border_count', 32, 255)  
    boosting_type = trial.suggest_categorical('boosting_type', ['Ordered', 'Plain']) 


    model = CatBoostClassifier(
                                iterations=iterations,
                                learning_rate=learning_rate,
                                depth=depth,
                                l2_leaf_reg=l2_leaf_reg,
                                subsample=subsample,
                                colsample_bylevel=colsample_bylevel,
                                class_weights=class_weights,
                                random_strength=random_strength,
                                bagging_temperature=bagging_temperature,
                                one_hot_max_size=one_hot_max_size,
                                grow_policy=grow_policy,
                                border_count=border_count,
                                boosting_type=boosting_type,
                                random_state=42,
                                verbose=0  
                            )
    score = cross_val_score(model, X, y, cv=kf, scoring='accuracy', n_jobs=-1).mean()

    return score

study_CB = optuna.create_study(direction='maximize')
study_CB.optimize(objective_GB, n_trials=50)
best_params_CB =study_CB.best_params

# Print the best hyperparameters found by Optuna
print(f"Best hyperparameters: {best_params_CB}")
print(f"Best accuracy: {study_CB.best_value:.4f}")


# PLotting the optimization history
optuna.visualization.plot_optimization_history(study_CB)


# parameter importance
optuna.visualization.plot_param_importances(study_CB)


# Lightgbm

def objective_LGBM(trial):
    # Define hyperparameter lightgbm
    n_estimators = trial.suggest_int("n_estimators", 50, 300)
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3)
    num_leaves = trial.suggest_int("num_leaves", 10, 150)
    max_depth = trial.suggest_int("max_depth", 3, 16)
    min_child_samples = trial.suggest_int("min_child_samples", 5, 30)
    subsample = trial.suggest_float("subsample", 0.5, 1.0)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)
    lambda_l1 = trial.suggest_float("lambda_l1", 0.0, 10.0)
    lambda_l2 = trial.suggest_float("lambda_l2", 0.0, 10.0)
    min_gain_to_split = trial.suggest_float("min_gain_to_split", 0.0, 5.0)
    class_weight = trial.suggest_categorical("class_weight", [None, "balanced"])

    model = LGBMClassifier(
                            n_estimators=n_estimators,
                            learning_rate=learning_rate,
                            num_leaves=num_leaves,
                            max_depth=max_depth,
                            min_child_samples=min_child_samples,
                            subsample=subsample,
                            colsample_bytree=colsample_bytree,
                            lambda_l1=lambda_l1,
                            lambda_l2=lambda_l2,
                            min_gain_to_split=min_gain_to_split,
                            class_weight=class_weight,
                            random_state=42
                        )

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    score = cross_val_score(model, X, y, cv=kf, scoring="accuracy", n_jobs=-1).mean()
    
    return score

# Create Optuna study and optimize
study_LGBM = optuna.create_study(direction="maximize")
study_LGBM.optimize(objective_LGBM, n_trials=50)
best_params_LGBM =study_LGBM.best_params

# Print best hyperparameters
print("Best hyperparameters:", best_params_LGBM)
print(f"Best accuracy: {study_LGBM.best_value:.4f}")


 optuna.visualization.plot_optimization_history(study_LGBM) 


optuna.visualization.plot_param_importances(study_LGBM) 



# Remove incompatible parameters for CatBoost
best_params_CB_cleaned = {k: v for k, v in best_params_CB.items() if k not in ['min_samples_split', 'min_samples_leaf', 'max_features']}



# Models
RandomForest = RandomForestClassifier(**best_params_RF)
GradientBoosting = GradientBoostingClassifier(verbose= False, **best_params_GB)
CatBoost = CatBoostClassifier(verbose=False, **best_params_CB_cleaned)
LightGBM = LGBMClassifier(verbose=0, **best_params_LGBM)

# The votingclassifier
best_model = VotingClassifier(
    estimators = [
        ('RandomForest', RandomForest),
        ('GradientBoosting', GradientBoosting),
        ('CatBoost', CatBoost),
        ('LightGBM', LightGBM)
        ],
    voting = 'soft'
).fit(X,y)


# Making prediction
y_pred = best_model.predict(X)
y_true = y

accuracy = accuracy_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)

print(f"Model Performance:\nAccuracy: {accuracy:.4f}\nRecall: {recall:.4f}")


# Function to load and wrangle the data
def wrangle_test(filepath):
    # Loading the data
    df = pd.read_csv(filepath)

    # Dropping null values
    df = df.fillna(df['winddirection'].mean())

    # Dropping redundant columns
    df = df.drop('id', axis=1)
    
    return df


test = wrangle_test('/kaggle/input/playground-series-s5e3/test.csv')


# making prediction

prediction = best_model.predict_proba(test)[:,1]



test_data['rainfall'] = prediction
submission = test_data[['id', 'rainfall']]


# Final submission file
submission.to_csv('submission.csv', index = False)

