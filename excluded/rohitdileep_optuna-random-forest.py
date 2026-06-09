import numpy as np
import pandas as pd


train_labels = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/train_labels.csv')
train_sequences = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/train_sequences.csv')
submission = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/sample_submission.csv')
test_sequences = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/test_sequences.csv')


## left join ##
train_labels['ID'] = train_labels['ID'].str.rsplit('_', n=1).str[0]
train_df  = train_labels.merge(how = 'left' , left_on = 'ID' , right_on = 'target_id' , right = train_sequences )


print('No of rows and cols ' , train_df.shape)
print()
print('Missing values' , train_df.isna().sum())
print()


##datetime conversion ###
train_df['temporal_cutoff'] = pd.to_datetime(train_df['temporal_cutoff']).astype('int64') // 10**9
test_sequences['temporal_cutoff'] = pd.to_datetime(test_sequences['temporal_cutoff']).astype('int64') // 10**9
## using len of sequence 
train_df['seq_length'] = train_df['sequence'].str.len()
test_sequences['seq_length'] =  test_sequences['sequence'].str.len()


submission['ID']  = submission['ID'].str.rsplit('_' ,n =1  ).str[0]


test  = test_sequences.merge(how = 'left' , left_on = 'target_id' , right_on = 'ID' , right = submission)


# Define columns
categorical_cols = ['resname', 'target_id', 'description', 'all_sequences']
numerical_cols = ['temporal_cutoff', 'resid', 'seq_length']
target_cols = ['x_1', 'y_1', 'z_1']


# Encode categoricals
from sklearn.preprocessing import LabelEncoder

# Encode categoricals
def encode_categoricals(df, categorical_cols):
    df_encoded = df.copy()
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        encoders[col] = le
    return df_encoded, encoders


## dropna values 
train_df.dropna(inplace = True)


# Prepare data
df_encoded, encoders = encode_categoricals(train_df, categorical_cols)
X = df_encoded[categorical_cols + numerical_cols]
y = df_encoded[target_cols]


from sklearn.model_selection import train_test_split
trainx , testx , trainy , testy = train_test_split(X , y , random_state = 0 , test_size = 0.25)


# # # # Objective function for Optuna

# import optuna
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error
# import warnings

# warnings.filterwarnings("ignore", category=UserWarning)
# warnings.filterwarnings("ignore", category=FutureWarning)

# # Prepare training and validation data
# X_train = trainx[categorical_cols + numerical_cols]
# y_train = trainy[target_cols]
# X_valid = testx[categorical_cols + numerical_cols]
# y_valid = testy[target_cols]

# def objective(trial):
#     params = {
#         "n_estimators": trial.suggest_int("n_estimators", 25, 1000),
#         "max_depth": trial.suggest_int("max_depth", 5, 35),
#         "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
#         "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
#         "max_features": trial.suggest_categorical("max_features", ["auto", "sqrt", "log2"]),
#         "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
#         "random_state": 42,
#         "n_jobs": -1 ,
#         "verbose" : 0  
#     }

#     # Train one model per target column
#     losses = []
#     for target in target_cols:
#         model = RandomForestRegressor(**params)
#         model.fit(X_train, y_train[target])
#         preds = model.predict(X_valid)
#         mse = mean_squared_error(y_valid[target], preds)
#         losses.append(mse)

#     return sum(losses) / len(losses)

# # Run Optuna optimization
# study = optuna.create_study(direction="minimize")
# study.optimize(objective, n_trials=100)

# print("Best params:", study.best_params)


## training models on hyperparameters obtained from optuna 

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error



# Prepare training and validation data
X_train = trainx[categorical_cols + numerical_cols]
y_train = trainy[target_cols]
X_valid = testx[categorical_cols + numerical_cols]
y_valid = testy[target_cols]


params = {'n_estimators': 520, 'max_depth': 34, 'min_samples_split': 8, 'min_samples_leaf': 1, 'max_features': 'auto', 
              'bootstrap': True , 'n_jobs' : -1 }

models = {}
for target in target_cols:
    lgb_reg = RandomForestRegressor(**params)
    lgb_reg.fit(X_train, y_train[target])
    models[target] = lgb_reg


## Feature importance 
import pandas as pd
import matplotlib.pyplot as plt

# Collect feature importances for each target
for target in target_cols:
    model = models[target]
    importances = model.feature_importances_
    
    # Create a DataFrame for better visualization
    importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': importances
    }).sort_values(by='importance', ascending=False)
    
    print(f"\nTop Features for Target: {target}")
    print(importance_df.head(10))

    # Optional: Plot
    plt.figure(figsize=(10, 5))
    plt.title(f'Feature Importance for {target}')
    plt.bar(importance_df['feature'], importance_df['importance'])
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()



x_test = test[categorical_cols + numerical_cols]
x_test.head()


##  Data Preprocessing for inference
from sklearn.preprocessing import LabelEncoder



for col in categorical_cols:
    # Handle potential NaN values before encoding
    test[col] = test[col].astype(str).fillna('missing')
    
    # Create and fit label encoder
    le = LabelEncoder()
    test[col] = le.fit_transform(test[col])
    encoders[col] = le



x_test = test[categorical_cols + numerical_cols].copy()
x_test.head()


## Prediction on test data ##

# Predict on test set (or new data)

predictions = pd.DataFrame()

for target in target_cols:
    predictions[target] = models[target].predict(x_test)


submission[['x_1', 'y_1', 'z_1']] = predictions
submission[['ID', 'resname', 'resid', 'x_1', 'y_1', 'z_1']].head()


# Create x_2-x_5, y_2-y_5, z_2-z_5 columns with same values as x_1/y_1/z_1
for i in range(2, 6):
    submission[f'x_{i}'] = submission['x_1']
    submission[f'y_{i}'] = submission['y_1']
    submission[f'z_{i}'] = submission['z_1']


submission['ID'] = submission['ID'].astype(str) + '_' + submission['resid'].astype(str)


submission.to_csv('submission.csv' , index = False)


submission.head()




