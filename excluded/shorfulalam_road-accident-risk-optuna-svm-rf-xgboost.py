import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import optuna
import warnings
warnings.filterwarnings('ignore')


# import data
train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv', index_col='id')
test=pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv',index_col='id')


train.head()


train.info()


test.info()


#categorical_columns = train.select_dtypes(include=['object', 'bool']).columns
#categorical_columns


categorical_features = [
    'road_type', 'lighting', 'weather', 'road_signs_present',
    'public_road', 'time_of_day', 'holiday', 'school_season'
]

def show_unique_values(df, columns):
    for c in columns:
        print(f"\n{c}:")
        print(df[c].unique())

show_unique_values(train, categorical_features)


for col in categorical_features:
    train[col].value_counts().plot(kind='pie', autopct='%1.1f%%', figsize=(5,5), title=col)
    plt.ylabel('')
    plt.show()


# Risk encodings
weather = {'clear': 0, 'rainy': 1, 'foggy': 2}
lighting = {'daylight': 0, 'dim': 1, 'night': 2}
time = {'morning': 0, 'afternoon': 1, 'evening': 2, 'night': 3}
road = {'rural': 0, 'urban': 1, 'highway': 2}

train['weather'] = train['weather'].map(weather)
train['lighting'] = train['lighting'].map(lighting)
train['time_of_day'] = train['time_of_day'].map(time)                                                                  # Road type encoding 
train['road_type'] = train['road_type'].map(road)

test['weather'] = test['weather'].map(weather)
test['lighting'] = test['lighting'].map(lighting)
test['time_of_day'] = test['time_of_day'].map(time)                                                                  # Road type encoding 
test['road_type'] = test['road_type'].map(road)
train.head()


def create_features(df):
    # Feature 1: speed * curvature
    df['speed_curvature'] = df['curvature'] * df['speed_limit']
    df['curvature_speed2'] = df['curvature'] * (df['speed_limit'] ** 2)
    
    # Feature 2: road_type (ordinal) * speed_limit
    df['road_speed'] = df['road_type'] * df['speed_limit']
    
    # Feature 3: num_lanes * speed_limit
    df['lanes_speed'] = df['num_lanes'] * df['speed_limit']
    
    # Feature 4: lighting + weather
    df['visibility_risk'] = df['lighting'] + df['weather']
    
    # Feature 5: peak_hour = morning or evening AND not holiday
    df['peak_hour'] = ((df['time_of_day'].isin([0, 2])) & (~df['holiday'])).astype(int)

    # Night risk (lighting = night) & weather bad (rainy/foggy)
    df['night_bad_weather'] = ((df['lighting'] >= 1) & (df['weather'] >= 1)).astype(int)

    
    return df




create_features(train)


create_features(test)


# Data Normalization Using StandardScaler
columns=test.columns
scaler = StandardScaler()
train[columns] = scaler.fit_transform(train[columns])
test[columns] = scaler.transform(test[columns])


train


# Prepare input data
X_train = train.drop(['accident_risk'], axis=1)
y_train = train['accident_risk']

# Split the dataset into training and validation data
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
print(X_train.shape)
print(X_val.shape)


xgb_model = XGBRegressor(
    n_estimators=900,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    early_stopping_rounds=50,
    eval_metric='rmse',
)

xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)


# Objective function for Optuna
def objective(trial):
    # Suggest hyperparameters for XGBoost
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 3000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'random_state': 42,
        'n_jobs': -1
    }

    # Initialize model
    model = XGBRegressor(**params)

    # Evaluate using cross-validation
    score = cross_val_score(model, X_train, y_train, cv=3, scoring='neg_root_mean_squared_error').mean()

    # Return RMSE (positive value for minimization)
    rmse = -score
    return rmse


# Create and run Optuna study
study = optuna.create_study( direction='minimize', sampler=optuna.samplers.TPESampler(seed=42)) #Bayesian optimization–based sampler.
study.optimize(objective, n_trials=100, show_progress_bar=True)


# Best parameters and score
best_trial = study.best_trial
print("Best trial parameters:", best_trial.params)
print("Best trial accuracy:", best_trial.value)


# Train a RandomForestClassifier using the best hyperparameters from Optuna
best_model = XGBRegressor(**study.best_trial.params)

# Fit the model to the training data
best_model.fit(X_train, y_train)


from xgboost import plot_importance

plt.figure(figsize=(10, 6))
plot_importance(best_model, importance_type='gain', max_num_features=15)
plt.title('XGBoost Feature Importance (Top 15)')
plt.show()



# Make predictions on the test set
y_pred = best_model.predict(test)


submission = pd.DataFrame({
    'id': test.index,
    'accident_risk': y_pred
})
submission['accident_risk'] = submission['accident_risk'].clip(0, 1)
submission.to_csv('submission.csv', index=False)
# Display the submission DataFrame
display(submission.head())

