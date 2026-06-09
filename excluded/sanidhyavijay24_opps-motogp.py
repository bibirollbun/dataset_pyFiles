# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


train_df=pd.read_csv('/kaggle/input/burnout-datathon-ieeecsmuj/train.csv')
train_df


test_df=pd.read_csv('/kaggle/input/burnout-datathon-ieeecsmuj/test.csv')
test_df


val_df=pd.read_csv('/kaggle/input/burnout-datathon-ieeecsmuj/val.csv')
val_df


train_df.isnull().sum()


print(train_df.info())


train_df['Penalty']


with pd.option_context('display.max_columns', None):
    print(train_df.head())


print(train_df['category_x'].value_counts())


team_means1 = train_df.groupby('circuit_name')['Lap_Time_Seconds'].mean()
team_means2 = train_df.groupby('shortname')['Lap_Time_Seconds'].mean()
team_means3 = train_df.groupby('team_name')['Lap_Time_Seconds'].mean()
team_means4 = train_df.groupby('rider_name')['Lap_Time_Seconds'].mean()
team_means5 = train_df.groupby('bike_name')['Lap_Time_Seconds'].mean()


def feature_engineering(df):
    df = pd.get_dummies(df, columns=['category_x'], prefix='moto', dtype=int)
    df = pd.get_dummies(df, columns=['Track_Condition'], prefix='Track_Condition', dtype=int,drop_first=True)
    df = pd.get_dummies(df, columns=['Tire_Compound_Rear'], prefix='rear_tire', dtype=int, drop_first=True)
    df = pd.get_dummies(df, columns=['Tire_Compound_Front'], prefix='front_tire', dtype=int, drop_first=True)
    df = pd.get_dummies(df, columns=['Penalty'], prefix='Penalty', dtype=int, drop_first=True)
    df = pd.get_dummies(df, columns=['Session'], prefix='Session', dtype=int, drop_first=True)
    df = pd.get_dummies(df, columns=['weather'], prefix='weather', dtype=int, drop_first=True)
    df = pd.get_dummies(df, columns=['track'], prefix='track', dtype=int, drop_first=True)
    team_means1 = df.groupby('circuit_name')['Lap_Time_Seconds'].mean()
    df['Circuit_name_encoded'] = df['circuit_name'].map(team_means1)
    team_means2 = df.groupby('shortname')['Lap_Time_Seconds'].mean()
    df['Short_name_encoded'] = df['shortname'].map(team_means2)
    team_means3 = df.groupby('team_name')['Lap_Time_Seconds'].mean()
    df['team_encoded'] = df['team_name'].map(team_means3)
    team_means4 = df.groupby('rider_name')['Lap_Time_Seconds'].mean()
    df['rider_name_encoded'] = df['rider_name'].map(team_means4)
    team_means5 = df.groupby('bike_name')['Lap_Time_Seconds'].mean()
    df['bike_name_encoded'] = df['bike_name'].map(team_means5)
    df=df.drop(["circuit_name","shortname","team_name","rider_name","bike_name",
                            "Unique ID","Rider_ID"],axis=1)
    return df
    
train_df = feature_engineering(train_df)
val_df = feature_engineering(val_df)


def feature_engineering(df):
    df = pd.get_dummies(df, columns=['category_x'], prefix='moto', dtype=int)
    df = pd.get_dummies(df, columns=['Track_Condition'], prefix='Track_Condition', dtype=int,drop_first=True)
    df = pd.get_dummies(df, columns=['Tire_Compound_Rear'], prefix='rear_tire', dtype=int, drop_first=True)
    df = pd.get_dummies(df, columns=['Tire_Compound_Front'], prefix='front_tire', dtype=int, drop_first=True)
    df = pd.get_dummies(df, columns=['Penalty'], prefix='Penalty', dtype=int, drop_first=True)
    df = pd.get_dummies(df, columns=['Session'], prefix='Session', dtype=int, drop_first=True)
    df = pd.get_dummies(df, columns=['weather'], prefix='weather', dtype=int, drop_first=True)
    df = pd.get_dummies(df, columns=['track'], prefix='track', dtype=int, drop_first=True)
    df['Circuit_name_encoded'] = df['circuit_name'].map(team_means1)
    df['Short_name_encoded'] = df['shortname'].map(team_means2)
    df['team_encoded'] = df['team_name'].map(team_means3)
    df['rider_name_encoded'] = df['rider_name'].map(team_means4)
    df['bike_name_encoded'] = df['bike_name'].map(team_means5)
    df=df.drop(["circuit_name","shortname","team_name","rider_name","bike_name",
                            "Unique ID","Rider_ID"],axis=1)
    return df

test_df = feature_engineering(test_df)


from sklearn.metrics import mean_squared_error
import numpy as np

X_train = train_df.drop(['Lap_Time_Seconds'], axis=1)  # Remove target and original categorical columns
y_train = train_df['Lap_Time_Seconds']


X_val = val_df.drop(['Lap_Time_Seconds'], axis=1)
y_val = val_df['Lap_Time_Seconds']


from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import AdamW
from tensorflow import keras
from tensorflow.keras import layers
model1 = keras.Sequential([
    layers.Input(shape=(57,)),
    layers.Dense(units=512,kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(1e-4),activation=None),
    layers.BatchNormalization(),
    #layers.Activation('relu'),
    layers.LeakyReLU(alpha=0.1),
    layers.Dropout(0.1),
    layers.Dense(units=512,kernel_regularizer=regularizers.l2(1e-4),activation=None),
    layers.BatchNormalization(),
    #layers.Activation('relu'),
    layers.LeakyReLU(alpha=0.1),
    layers.Dropout(0.1),
    layers.Dense(units=128,activation=None),
    layers.BatchNormalization(),
    #layers.Activation('relu'),
    layers.LeakyReLU(alpha=0.1),
    layers.Dropout(0.1),
    layers.Dense(units=128,activation=None),
    layers.BatchNormalization(),
    #layers.Activation('relu'),
    layers.LeakyReLU(alpha=0.1),
    layers.Dropout(0.1),
    layers.Dense(units=16,activation=None),
    layers.BatchNormalization(),
    #layers.Activation('relu'),
    layers.LeakyReLU(alpha=0.1),
   # layers.Dropout(0.1),
    layers.Dense(units=1, activation=None)
])
model1.compile(
    optimizer=AdamW(learning_rate=1e-3, weight_decay=1e-4),  # Lower learning rate
    loss='mse',
    metrics=['mae']
)

# More patient early stopping
early_stopping = keras.callbacks.EarlyStopping(
    patience=25,           # Increased from 15
    min_delta=0.00001,     # Smaller threshold (was 0.0001)
    restore_best_weights=True,
    monitor='val_loss'
)

# More patient learning rate scheduler
lr_scheduler = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.8,            # Less aggressive reduction (was 0.7)
    patience=15,           # Increased from 10
    min_lr=1e-7,           # Lower minimum (was 1e-6)
    verbose=1
)
history1=model1.fit(
    X_train,y_train,
    validation_data=(X_val,y_val),
    batch_size=1024,
    epochs=500,
    callbacks=[early_stopping,lr_scheduler],
    verbose=1
)

from sklearn.metrics import mean_squared_error

train_pred = model1.predict(X_train)
val_pred = model1.predict(X_val)

train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))

print(f"Train RMSE: {train_rmse:.4f}")
print(f"Validation RMSE: {val_rmse:.4f}")


sample_submission = pd.read_csv('/kaggle/input/burnout-datathon-ieeecsmuj/sample_submission.csv')

# Check the structure of sample submission
print("Sample submission structure:")
print(f"Shape: {sample_submission.shape}")
print(f"Columns: {sample_submission.columns.tolist()}")
print("\nFirst few rows:")
print(sample_submission.head())


id_column = sample_submission['Unique ID']

print(f"\nID column extracted:")
print(f"Column name: 'Unique ID'")
print(f"Number of IDs: {len(id_column)}")
print(f"First few IDs: {id_column.head().tolist()}")

# Generate predictions on your test data
# Make sure test_df has the same 57 features in the same order as training
test_predictions = model1.predict(test_df)

# Create submission using the exact same format as sample
submission = pd.DataFrame({
    'Unique ID': id_column,
    'Lap_Time_Seconds': test_predictions.flatten()
})

# Save submission
submission.to_csv('submission.csv', index=False)

print(f"\nSubmission created with columns: {submission.columns.tolist()}")
print("Submission saved as 'submission.csv'")


'''
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. Standardize your data (important!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)  # X is your feature matrix

# 2. Apply PCA
pca = PCA(n_components=0.95)  # Keep 95% of variance
X_pca = pca.fit_transform(X_scaled)

X_val_scaled = scaler.transform(X_val)  # Note: transform, not fit_transform
X_val_pca = pca.transform(X_val_scaled)

# 3. Check results
print(f"Original features: {X_train.shape[1]}")
print(f"PCA features: {X_pca.shape[1]}")
print(f"Variance explained: {pca.explained_variance_ratio_.sum():.3f}")
'''


'''
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
train_data = lgb.Dataset(X_pca, label=y_train)
val_data = lgb.Dataset(X_val_pca, label=y_val, reference=train_data)

# 6. Set LightGBM parameters
params = {
    'objective': 'regression',
    'metric': ['rmse', 'mae'],
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0,
    'random_state': 42
}

# 7. Train the model
print("Training LightGBM model...")
model = lgb.train(
    params,
    train_data,
    valid_sets=[train_data, val_data],
    valid_names=['train', 'val'],
    num_boost_round=1000,
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=100)
    ]
)

# 8. Make predictions
train_pred = model.predict(X_pca, num_iteration=model.best_iteration)
val_pred = model.predict(X_val_pca, num_iteration=model.best_iteration)

# 9. Calculate metrics
train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
train_mae = mean_absolute_error(y_train, train_pred)
val_mae = mean_absolute_error(y_val, val_pred)

print(f"\n=== Model Performance ===")
print(f"Training RMSE: {train_rmse:.4f}")
print(f"Validation RMSE: {val_rmse:.4f}")
print(f"Training MAE: {train_mae:.4f}")
print(f"Validation MAE: {val_mae:.4f}")

# 10. Feature importance (top 10 PCA components)
feature_importance = model.feature_importance(importance_type='gain')
print(f"\n=== Top 10 PCA Components by Importance ===")
for i in range(min(10, len(feature_importance))):
    idx = np.argsort(feature_importance)[::-1][i]
    print(f"PCA Component {idx}: {feature_importance[idx]:.0f}")

# Optional: Plot feature importance
import matplotlib.pyplot as plt
lgb.plot_importance(model, max_num_features=15, importance_type='gain')
plt.title('Top 15 PCA Components Feature Importance')
plt.tight_layout()
plt.show()
'''


'''
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from itertools import product

print("=" * 60)
print("LIGHTGBM STAGED GRID SEARCH HYPERPARAMETER TUNING")
print("=" * 60)

def train_and_evaluate(params, X_train, y_train, X_val, y_val):
    """Train model and return validation RMSE"""
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        valid_names=['val'],
        num_boost_round=1000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0)
        ]
    )
    
    val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    return val_rmse, model

# Stage 1: Find best num_leaves and learning_rate
print("Stage 1: Tuning num_leaves and learning_rate...")
stage1_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'random_state': 42
}

stage1_grid = {
    'num_leaves': [10, 15, 20, 25, 31],
    'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.15]
}

best_stage1_rmse = float('inf')
best_stage1_params = None

results_stage1 = []
total_combinations = len(stage1_grid['num_leaves']) * len(stage1_grid['learning_rate'])
current_combo = 0

for num_leaves, lr in product(stage1_grid['num_leaves'], stage1_grid['learning_rate']):
    current_combo += 1
    params = stage1_params.copy()
    params.update({'num_leaves': num_leaves, 'learning_rate': lr})
    
    rmse, _ = train_and_evaluate(params, X_pca, y_train, X_val_pca, y_val)
    results_stage1.append({
        'num_leaves': num_leaves,
        'learning_rate': lr,
        'rmse': rmse
    })
    
    print(f"  [{current_combo:2d}/{total_combinations}] num_leaves: {num_leaves:2d}, lr: {lr:.3f} -> RMSE: {rmse:.4f}")
    
    if rmse < best_stage1_rmse:
        best_stage1_rmse = rmse
        best_stage1_params = {'num_leaves': num_leaves, 'learning_rate': lr}
        print(f"    ✓ New best!")

print(f"\nBest Stage 1 RMSE: {best_stage1_rmse:.4f}")
print(f"Best num_leaves: {best_stage1_params['num_leaves']}")
print(f"Best learning_rate: {best_stage1_params['learning_rate']}")

# Stage 2: Tune regularization with best num_leaves and learning_rate
print("\n" + "-" * 60)
print("Stage 2: Tuning regularization...")
stage2_params = stage1_params.copy()
stage2_params.update(best_stage1_params)

stage2_grid = {
    'lambda_l1': [0, 0.1, 0.3, 0.5, 1.0],
    'lambda_l2': [0, 0.1, 0.3, 0.5, 1.0]
}

best_stage2_rmse = float('inf')
best_stage2_params = None

results_stage2 = []
total_combinations = len(stage2_grid['lambda_l1']) * len(stage2_grid['lambda_l2'])
current_combo = 0

for l1, l2 in product(stage2_grid['lambda_l1'], stage2_grid['lambda_l2']):
    current_combo += 1
    params = stage2_params.copy()
    params.update({'lambda_l1': l1, 'lambda_l2': l2})
    
    rmse, _ = train_and_evaluate(params, X_pca, y_train, X_val_pca, y_val)
    results_stage2.append({
        'lambda_l1': l1,
        'lambda_l2': l2,
        'rmse': rmse
    })
    
    print(f"  [{current_combo:2d}/{total_combinations}] L1: {l1:.1f}, L2: {l2:.1f} -> RMSE: {rmse:.4f}")
    
    if rmse < best_stage2_rmse:
        best_stage2_rmse = rmse
        best_stage2_params = {'lambda_l1': l1, 'lambda_l2': l2}
        print(f"    ✓ New best!")

print(f"\nBest Stage 2 RMSE: {best_stage2_rmse:.4f}")
print(f"Best lambda_l1: {best_stage2_params['lambda_l1']}")
print(f"Best lambda_l2: {best_stage2_params['lambda_l2']}")

# Stage 3: Fine-tune sampling parameters
print("\n" + "-" * 60)
print("Stage 3: Tuning sampling parameters...")
stage3_params = stage2_params.copy()
stage3_params.update(best_stage2_params)

stage3_grid = {
    'feature_fraction': [0.6, 0.7, 0.8, 0.9],
    'bagging_fraction': [0.6, 0.7, 0.8, 0.9],
    'min_data_in_leaf': [10, 20, 30, 50]
}

best_stage3_rmse = float('inf')
best_stage3_params = None
best_model = None

results_stage3 = []
total_combinations = len(stage3_grid['feature_fraction']) * len(stage3_grid['bagging_fraction']) * len(stage3_grid['min_data_in_leaf'])
current_combo = 0

for ff, bf, mdil in product(stage3_grid['feature_fraction'], 
                           stage3_grid['bagging_fraction'],
                           stage3_grid['min_data_in_leaf']):
    current_combo += 1
    params = stage3_params.copy()
    params.update({
        'feature_fraction': ff,
        'bagging_fraction': bf,
        'min_data_in_leaf': mdil
    })
    
    rmse, model = train_and_evaluate(params, X_pca, y_train, X_val_pca, y_val)
    results_stage3.append({
        'feature_fraction': ff,
        'bagging_fraction': bf,
        'min_data_in_leaf': mdil,
        'rmse': rmse
    })
    
    print(f"  [{current_combo:2d}/{total_combinations}] FF: {ff:.1f}, BF: {bf:.1f}, MDL: {mdil:2d} -> RMSE: {rmse:.4f}")
    
    if rmse < best_stage3_rmse:
        best_stage3_rmse = rmse
        best_stage3_params = {
            'feature_fraction': ff,
            'bagging_fraction': bf,
            'min_data_in_leaf': mdil
        }
        best_model = model
        print(f"    ✓ New best!")

print(f"\nBest Stage 3 RMSE: {best_stage3_rmse:.4f}")
print(f"Best feature_fraction: {best_stage3_params['feature_fraction']}")
print(f"Best bagging_fraction: {best_stage3_params['bagging_fraction']}")
print(f"Best min_data_in_leaf: {best_stage3_params['min_data_in_leaf']}")

# Final best parameters
final_best_params = stage3_params.copy()
final_best_params.update(best_stage1_params)
final_best_params.update(best_stage2_params)
final_best_params.update(best_stage3_params)

print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)
print(f"Original RMSE: 10.3841")
print(f"Best Grid Search RMSE: {best_stage3_rmse:.4f}")
print(f"Improvement: {10.3841 - best_stage3_rmse:.4f} ({((10.3841 - best_stage3_rmse)/10.3841)*100:.2f}%)")

print("\nFinal Best Parameters:")
for param, value in final_best_params.items():
    if param not in ['objective', 'metric', 'boosting_type', 'verbose', 'random_state']:
        print(f"  {param}: {value}")

# Train final model and get detailed metrics
print("\nTraining final model with best parameters...")
final_train_pred = best_model.predict(X_pca, num_iteration=best_model.best_iteration)
final_val_pred = best_model.predict(X_val_pca, num_iteration=best_model.best_iteration)

final_train_rmse = np.sqrt(mean_squared_error(y_train, final_train_pred))
final_val_rmse = np.sqrt(mean_squared_error(y_val, final_val_pred))
final_train_mae = np.mean(np.abs(y_train - final_train_pred))
final_val_mae = np.mean(np.abs(y_val - final_val_pred))

print(f"\nFinal Model Performance:")
print(f"Training RMSE: {final_train_rmse:.4f}")
print(f"Validation RMSE: {final_val_rmse:.4f}")
print(f"Training MAE: {final_train_mae:.4f}")
print(f"Validation MAE: {final_val_mae:.4f}")
print(f"Overfitting Gap: {final_val_rmse - final_train_rmse:.4f}")

# Display top 5 parameter combinations from each stage
print("\n" + "=" * 60)
print("TOP 5 COMBINATIONS FROM EACH STAGE")
print("=" * 60)

print("\nStage 1 - Top 5 (num_leaves, learning_rate):")
df_stage1 = pd.DataFrame(results_stage1).sort_values('rmse')
for i, row in df_stage1.head().iterrows():
    print(f"  {row['num_leaves']:2d}, {row['learning_rate']:.3f} -> RMSE: {row['rmse']:.4f}")

print("\nStage 2 - Top 5 (lambda_l1, lambda_l2):")
df_stage2 = pd.DataFrame(results_stage2).sort_values('rmse')
for i, row in df_stage2.head().iterrows():
    print(f"  {row['lambda_l1']:.1f}, {row['lambda_l2']:.1f} -> RMSE: {row['rmse']:.4f}")

print("\nStage 3 - Top 5 (feature_frac, bagging_frac, min_data_leaf):")
df_stage3 = pd.DataFrame(results_stage3).sort_values('rmse')
for i, row in df_stage3.head().iterrows():
    print(f"  {row['feature_fraction']:.1f}, {row['bagging_fraction']:.1f}, {row['min_data_in_leaf']:2d} -> RMSE: {row['rmse']:.4f}")

print(f"\n" + "=" * 60)
print("COPY-PASTE BEST PARAMETERS:")
print("=" * 60)
print("params = {")
for param, value in final_best_params.items():
    if isinstance(value, str):
        print(f"    '{param}': '{value}',")
    else:
        print(f"    '{param}': {value},")
print("}")
'''




