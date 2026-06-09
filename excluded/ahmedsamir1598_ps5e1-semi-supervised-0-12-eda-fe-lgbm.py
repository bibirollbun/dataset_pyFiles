import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import re
import string
from datetime import datetime, timedelta
from sklearn.preprocessing import OneHotEncoder
import optuna
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import GroupKFold
from lightgbm import LGBMRegressor
!mkdir /kaggle/working/checkpoint


data_path = '/kaggle/input/playground-series-s5e1'
checkpoint_path = '/kaggle/working/checkpoint'

train = pd.read_csv(f'{data_path}/train.csv')
test = pd.read_csv(f'{data_path}/test.csv')
train.drop(columns=['id'],inplace=True)
test.drop(columns=['id'],inplace=True)

train['date'] = pd.to_datetime(train['date'], errors='coerce')
test['date'] = pd.to_datetime(test['date'], errors='coerce')

train.head()


train.info()


train.describe().T


train.nunique()


test.info()


fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

# Plot histograms
for i, country in enumerate(train['country'].unique()):
    ax = axes[i]
    ax.hist(train[train['country'] == country]['num_sold'], bins=300)
    ax.set_title(f'Country: {country}')
    ax.set_xlabel('Number Sold')
    ax.set_ylabel('Frequency')

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


# Function to plot mean numbers sold by days in the week for each country
def mean_num_sold(df:pd.DataFrame, country:str, month_list:list) -> None:

    # Filter the DataFrame for select country
    prague_df = df[df['country'] == country]

    prague_df = prague_df[prague_df['date'].dt.month.isin(month_list)]

    # Extract year and day of the week
    prague_df['year'] = prague_df['date'].dt.year
    prague_df['day_of_week'] = prague_df['date'].dt.day_name()

    # Calculate mean numbers sold for each day of the week for each year
    mean_num_sold_by_day_year = prague_df.groupby(['year', 'day_of_week'])['num_sold'].mean().unstack()

    teamp_days_num_sold = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    days_order = []
    for day in teamp_days_num_sold:
        if day in mean_num_sold_by_day_year.columns:
            days_order.append(day)
    mean_num_sold_by_day_year = mean_num_sold_by_day_year[days_order]

    plt.figure(figsize=(14, 6))
    for year in mean_num_sold_by_day_year.index:
        plt.plot(mean_num_sold_by_day_year.columns, mean_num_sold_by_day_year.loc[year], marker='o', linestyle='-', label=f'Year {year}')

    plt.title(f'Mean Numbers Sold by Day of the Week for {country}')
    plt.xlabel('Day of the Week')
    plt.ylabel('Mean Numbers Sold')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.show()


train['country'].unique()


countries = ['Canada', 'Finland', 'Italy', 'Kenya', 'Norway', 'Singapore']

for country in countries:
    mean_num_sold(train, country, month_list=[1, 2, 3])


train_c = train.copy()
test_c = test.copy()
#train_c['num_sold'] = train['num_sold'].fillna(train['num_sold'].mean()) ##### Rookie mistake: don't fill with mean without 
#checking the distribution
#train_c.dropna(inplace=True)
train_c.info()


def preprocess(df, onehot_encoder=None, fit_mode=True):
    # Extract temporal features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['month_name'] = df['date'].dt.month_name()
    
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['date'].dt.dayofweek >= 5
    df['is_sunday'] = df['date'].dt.dayofweek == 6  # Add Sunday feature
    df['quarter'] = df['date'].dt.quarter
    
    # Cyclical encoding for periodic features
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['cos_year'] = np.cos(df['year'] * (2 * np.pi) / 100)
    df['sin_year'] = np.sin(df['year'] * (2 * np.pi) / 100)
    # Interaction features
    df['country_store'] = df['country'] + '_' + df['store']
    df['store_product'] = df['store'] + '_' + df['product']
    
    # One-hot encode categorical features
    CATEGORICAL_COLS = ['country', 'store', 'product', 'country_store', 'store_product','month_name','day_of_week']
    if onehot_encoder is None:
        onehot_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    
    if fit_mode:
        encoded = onehot_encoder.fit_transform(df[CATEGORICAL_COLS])
    else:
        encoded = onehot_encoder.transform(df[CATEGORICAL_COLS])
    
    encoded_df = pd.DataFrame(encoded, columns=onehot_encoder.get_feature_names_out(CATEGORICAL_COLS), index=df.index)
    df = pd.concat([df, encoded_df], axis=1)
    
    # Drop original categorical columns
    df.drop(columns=CATEGORICAL_COLS, inplace=True)
    
    return df, onehot_encoder

# Apply preprocessing to train and test datasets
onehot_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

# Preprocess training data
train_c, onehot_encoder = preprocess(train_c, onehot_encoder=onehot_encoder, fit_mode=True)

# Preprocess test data (use the fitted encoder)
test_c, _ = preprocess(test_c, onehot_encoder=onehot_encoder, fit_mode=False)


train_c.head()


train_c.info()


train_c[train_c['num_sold'].isnull()]


train_c_train = train_c[train_c['num_sold'].notnull()]
train_c_test = train_c[train_c['num_sold'].isnull()]
# Prepare features and target
X_train = train_c_train.drop(columns=['num_sold', 'date'])
y_train = train_c_train['num_sold']
group_col = train_c_train['year']

X_test = train_c_test.drop(columns=['num_sold', 'date'])



params = {'reg_alpha': 0.0193448774446158, 'reg_lambda': 0.001145334071923676, 'learning_rate': 0.0891306023864947, 'colsample_bytree': 0.8853552116695744, 'min_child_weight': 0.9396092576991832, 'num_leaves': 49, 'min_child_samples': 6}
## 0.079
def optunatune(tune=False):
    global best_model
    if tune:
        def objective(trial):
            # Define parameter search space
            trial_params = {
                'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-5, 1.0),
                'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-5, 1.0),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.5),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'min_child_weight': trial.suggest_loguniform('min_child_weight', 1e-5, 1.0),
                'num_leaves': trial.suggest_int('num_leaves', 10, 50),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 30),
            }

            # Use GPU for training
            model = LGBMRegressor(objective='regression', 
                                  verbose=-1, n_estimators=1000, **trial_params)

            # Cross-validation for MAPE
            fold_mape = []
            gkf = GroupKFold(n_splits=5)
            for train_index, val_index in gkf.split(X_train, y_train, groups=group_col):
                X_fold_train, X_fold_val = X_train.iloc[train_index], X_train.iloc[val_index]
                y_fold_train, y_fold_val = y_train.iloc[train_index], y_train.iloc[val_index]

                model.fit(X_fold_train, y_fold_train)
                y_pred = model.predict(X_fold_val)

                # Calculate MAPE
                fold_mape.append(mean_absolute_percentage_error(y_fold_val, y_pred))

            return np.mean(fold_mape)

        # Run Optuna optimization
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=500, show_progress_bar=True)

        # Use best parameters
        best_params = study.best_params
        print(f"Best Parameters: {best_params}")

        # Train final model on entire training data with GPU
        best_model = LGBMRegressor(objective='regression',
                                  verbose=-1, n_estimators=1000, **best_params)
        best_model.fit(X_train, y_train)
    else:
        best_model = LGBMRegressor(objective='regression',
                                  verbose=-1, n_estimators=1000, **params)
        best_model.fit(X_train, y_train)



# Train model
optunatune(tune=True)

# Predict missing num_sold for test set
train_c_test['num_sold'] = best_model.predict(X_test)



final_data = pd.concat([train_c_train, train_c_test], axis=0)
final_data


final_data.info()


params = {'reg_alpha': 0.01022851857188049, 'reg_lambda': 0.03097737055880182, 'learning_rate': 0.09716505312120992, 'colsample_bytree': 0.9344556185684774, 'min_child_weight': 0.00021110769740997144, 'num_leaves': 48, 'min_child_samples': 18}
## 0.10
def optunatune(tune=False):
    if tune:
        def objective(trial):
            # Define parameter search space
            trial_params = {
                'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-5, 1.0),
                'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-5, 1.0),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.5),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'min_child_weight': trial.suggest_loguniform('min_child_weight', 1e-5, 1.0),
                'num_leaves': trial.suggest_int('num_leaves', 10, 50),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 30)
            }
            
            # Create LGBMRegressor with trial parameters
            model = LGBMRegressor(objective='regression', verbose=-1, n_estimators=1000, **trial_params)
            X = final_data.drop(columns=['num_sold', 'date'])
            y = final_data['num_sold']
            group_col = final_data['year']
            
            # Cross-validation for MAPE
            fold_mape = []
            gkf = GroupKFold(n_splits=5)
            for train_index, val_index in gkf.split(X, y, groups=group_col):
                X_train, X_val = X.iloc[train_index], X.iloc[val_index]
                y_train, y_val = y.iloc[train_index], y.iloc[val_index]
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                
                # Calculate MAPE
                fold_mape.append(mean_absolute_percentage_error(y_val, y_pred))
            
            return np.mean(fold_mape)  # Return average MAPE
        
        # Run Optuna optimization
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=500, show_progress_bar=True)
        
        # Update params with best found values
        params.update(study.best_params)
        print(f"Best Parameters: {study.best_params}")

optunatune(tune=True)



params


X = final_data.drop(columns=['num_sold', 'date'])
y = final_data['num_sold']
model = LGBMRegressor(objective='regression_l1', verbose=-1, n_estimators=1000, **params)
group_col = final_data['year']
# Cross-validation for MAPE
fold_mape = []
gkf = GroupKFold(n_splits=5)
for train_index, val_index in gkf.split(X, y, groups=group_col):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index] 
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    # Calculate MAPE
    fold_mape.append(mean_absolute_percentage_error(y_val, y_pred))
print(f"Prediction MAPE Mean: {np.mean(fold_mape)}")



all_mapes = []
gkf = GroupKFold(n_splits=5)

group_col = final_data['year']
X = final_data.drop(columns=['num_sold', 'date', 'year'])
y = final_data['num_sold']

for fold, (train_index, val_index) in enumerate(gkf.split(X, y, groups=group_col)):
    print(f"\nFold {fold + 1}:")
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    
    model = LGBMRegressor(objective='regression_l1', verbose=-1, n_estimators=1000, **params)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_val)
    
    mape = mean_absolute_percentage_error(y_val, y_pred)
    all_mapes.append(mape)
    
    print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}")
    print(f"Fold {fold + 1} MAPE: {mape:.4f}")

valid_mapes = [m for m in all_mapes if not np.isnan(m)]
if valid_mapes:
    print(f"\nAverage MAPE across all folds: {np.mean(valid_mapes):.4f}")
else:
    print("\nNo valid MAPE scores calculated")



X = final_data.drop(columns=['num_sold', 'date'])
y = final_data['num_sold']
model = LGBMRegressor(objective='regression', verbose=-1, n_estimators=1000, **params)
model.fit(X,y)
y_preds = model.predict(test_c.drop(columns=['date']))


sub = pd.read_csv(f'{data_path}/sample_submission.csv')
sub['num_sold'] = y_preds
sub.to_csv("submission.csv", index=False)


sub


fig, axs = plt.subplots(1, 1, figsize=(10, 20))

palette = sns.color_palette("RdYlGn_r", len(final_data.drop(columns=['num_sold', 'date']).columns))
lgbm_importances = pd.Series(model.feature_importances_, index=final_data.drop(columns=['num_sold', 'date']).columns).sort_values(ascending=False)
sns.barplot(y=lgbm_importances.index, x=lgbm_importances.values, orient='h', palette=palette)

plt.tight_layout()
plt.show()




