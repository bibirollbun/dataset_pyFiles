import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import time
import optuna
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
# Load and preprocess data (same as before)
def load_and_preprocess_data():
    train = pd.read_csv('/kaggle/input/playground-series-s4e11/train.csv')
    test = pd.read_csv('/kaggle/input/playground-series-s4e11/test.csv')
    all_data = pd.concat([train, test], axis=0, ignore_index=True)

    def preprocess_data(df):
        df['Academic Pressure'].fillna(df['Academic Pressure'].mean(), inplace=True)
        df['Work Pressure'].fillna(df['Work Pressure'].mean(), inplace=True)
        df['CGPA'].fillna(df['CGPA'].mean(), inplace=True)
        df['Study Satisfaction'].fillna(df['Study Satisfaction'].mean(), inplace=True)
        df['Job Satisfaction'].fillna(df['Job Satisfaction'].mean(), inplace=True)
        
        le = LabelEncoder()
        categorical_cols = ['Gender', 'City', 'Working Professional or Student', 'Profession', 
                            'Sleep Duration', 'Dietary Habits', 'Degree']
        for col in categorical_cols:
            df[col] = le.fit_transform(df[col].astype(str))
        
        df['Have you ever had suicidal thoughts ?'] = df['Have you ever had suicidal thoughts ?'].map({'Yes': 1, 'No': 0})
        df['Family History of Mental Illness'] = df['Family History of Mental Illness'].map({'Yes': 1, 'No': 0})
        
        return df

    all_data = preprocess_data(all_data)
    train = all_data[:len(train)]
    test = all_data[len(train):]
    
    features = [col for col in train.columns if col not in ['id', 'Name', 'Depression']]
    X = train[features]
    y = train['Depression']
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_val, y_train, y_val, test[features]

X_train, X_val, y_train, y_val, X_test = load_and_preprocess_data()

# Define the objective function for Optuna (TPE)
def objective(trial):
    # TPE optimizer code with increased ranges in both directions
    params = {
        'max_depth': trial.suggest_int('max_depth', 1, 20),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 10.0),
        'n_estimators': trial.suggest_int('n_estimators', 10, 5000),
        'min_child_weight': trial.suggest_int('min_child_weight', 0, 100),
        'subsample': trial.suggest_uniform('subsample', 0.1, 1.0),  
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.1, 1.0),  
    }
    model = XGBClassifier(**params, use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    return accuracy

# Perform TPE optimization
start_time = time.time()
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
study.optimize(objective, n_trials=100, n_jobs=-1)
end_time = time.time()

print(f"Best parameters: {study.best_params}")
print(f"Best cross-validation score: {study.best_value:.4f}")
print(f"Time taken: {end_time - start_time:.2f} seconds")

# Train final model on entire training data
best_params = study.best_params
best_model = XGBClassifier(**best_params, use_label_encoder=False, eval_metric='logloss', random_state=42)
best_model.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))

# Make predictions on test set
test_predictions = best_model.predict(X_test)

# Create submission file
submission = pd.DataFrame({'id': pd.read_csv('/kaggle/input/playground-series-s4e11/test.csv')['id'], 'Depression': test_predictions})
submission.to_csv('submission.csv', index=False)
print("Submission file created.")

# Visualization functions
def plot_optimization_history(study, title):
    plt.figure(figsize=(10, 6))
    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_param_importances(study, title):
    plt.figure(figsize=(12, 8))
    optuna.visualization.matplotlib.plot_param_importances(study)
    plt.title(title, pad=20)
    plt.tight_layout()
    plt.show()

# Create visualizations
plot_optimization_history(study, "TPE Optimization History")
plot_param_importances(study, "TPE Parameter Importances")

plt.figure(figsize=(12, 8))
optuna.visualization.matplotlib.plot_optimization_history(study)
plt.title("TPE Optimization History")
plt.xlabel("Number of Trials")
plt.ylabel("Objective Value")
plt.tight_layout()
plt.show()
# Modify random search code to create similar visualizations
from sklearn.model_selection import RandomizedSearchCV
# Random search code with increased ranges in both directions
param_dist = {
    'n_estimators': [10, 50, 100, 500, 1000, 2500, 5000],
    'max_depth': [1, 3, 5, 8, 10, 15, 20],
    'learning_rate': [0.0001, 0.001, 0.01, 0.1, 1.0, 5.0, 10.0],
    'subsample': [0.1, 0.3, 0.6, 0.7, 0.8, 0.9, 1.0],  
    'colsample_bytree': [0.1, 0.3, 0.6, 0.7, 0.8, 0.9, 1.0],  
    'min_child_weight': [0, 1, 10, 25, 50, 75, 100]
}
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

random_search = RandomizedSearchCV(xgb, param_distributions=param_dist, n_iter=100, 
                                   cv=5, random_state=42, n_jobs=-1, verbose=1)

start_time = time.time()
random_search.fit(X_train, y_train)
end_time = time.time()

print(f"Best parameters: {random_search.best_params_}")
print(f"Best cross-validation score: {random_search.best_score_:.4f}")
print(f"Time taken: {end_time - start_time:.2f} seconds")

# Create visualizations for random search
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(random_search.cv_results_['mean_test_score']) + 1), random_search.cv_results_['mean_test_score'])
plt.title("Random Search Optimization History")
plt.xlabel("Iteration")
plt.ylabel("Mean CV Score")
plt.tight_layout()
plt.show()

# Convert the list of dictionaries to a DataFrame
params_df = pd.DataFrame(random_search.cv_results_['params'])

# Calculate feature importances
importances = mutual_info_regression(params_df, 
                                     random_search.cv_results_['mean_test_score'])

# Create a dataframe of importances
importance_df = pd.DataFrame({'parameter': list(param_dist.keys()),
                              'importance': importances})
importance_df = importance_df.sort_values('importance', ascending=False)
plt.figure(figsize=(12, 8))
plt.bar(importance_df['parameter'], importance_df['importance'])
plt.title("Random Search Parameter Importances")
plt.xlabel("Parameter")
plt.ylabel("Importance Score")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Train final model and create submission for random search
best_random_model = XGBClassifier(**random_search.best_params_, use_label_encoder=False, eval_metric='logloss', random_state=42)
best_random_model.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))

random_test_predictions = best_random_model.predict(X_test)

random_submission = pd.DataFrame({'id': pd.read_csv('/kaggle/input/playground-series-s4e11/test.csv')['id'], 'Depression': random_test_predictions})
random_submission.to_csv('submission_random.csv', index=False)
print("Random search submission file created.")


