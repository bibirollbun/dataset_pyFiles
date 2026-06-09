import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
pd.options.mode.copy_on_write = True
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
from cuml.preprocessing import TargetEncoder
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm
from itertools import combinations
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter('ignore')


def process_combinations_fast(df, columns_to_encode, pair_size, max_batch_size=2000):
    # Precompute string versions of all columns once
    str_df = df[columns_to_encode]
    le = LabelEncoder()
    str_df = str_df.astype(str)
    total_new_cols = 0
    
    for r in pair_size:
        print(f"Processing {r}-combinations")
        
        # Count total combinations for this r-value
        n_combinations = np.math.comb(len(columns_to_encode), r)
        print(f"Total {r}-combinations to process: {n_combinations}")
        
        # Process combinations in batches to manage memory
        combos_iter = combinations(columns_to_encode, r)
        batch_cols = []
        batch_names = []
        
        with tqdm(total=n_combinations) as pbar:
            while True:
                # Collect a batch of combinations
                batch_cols.clear()
                batch_names.clear()
                
                # Fill the current batch
                for _ in range(max_batch_size):
                    try:
                        cols = next(combos_iter)
                        batch_cols.append(list(cols))
                        batch_names.append('+'.join(cols))
                    except StopIteration:
                        break
                
                if not batch_cols:  # No more combinations
                    break
                
                # Process this batch vectorized
                for i, (cols, new_name) in enumerate(zip(batch_cols, batch_names)):
                    # Fast vectorized concatenation
                    result = str_df[cols[0]].copy()
                    for col in cols[1:]:
                        result += '' + str_df[col]
                    
                    df[new_name] = le.fit_transform(result) + 1
                    pbar.update(1)
                
                total_new_cols += len(batch_cols)
                if len(batch_cols) == max_batch_size:  # Only print on full batches
                    print(f"Progress: {total_new_cols}/{n_combinations} combinations processed")
        
        print(f"Completed all {r}-combinations. Total columns now: {len(df.columns)}")
    
    return df


# Load data
df_train = pd.read_csv("/kaggle/input/training-data-for-podcast/podcast_listening_training_data.csv")
df_test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')

df = pd.concat([df_train, df_test], axis=0, ignore_index=True)

df.drop(columns=['id'], inplace=True)
df = df.drop_duplicates()

# outlier removal
df['Episode_Length_minutes'] = np.maximum(0, np.minimum(120, df['Episode_Length_minutes']))
df['Host_Popularity_percentage'] = np.maximum(20, np.minimum(100, df['Host_Popularity_percentage']))
df['Guest_Popularity_percentage'] = np.maximum(0, np.minimum(100, df['Guest_Popularity_percentage']))
df['Host_Popularity_bin'] = pd.cut(df['Host_Popularity_percentage'], bins=[20,40,60,80,100], labels=[1,2,3,4])
df.loc[df['Number_of_Ads'] > 3, 'Number_of_Ads'] = 0

# Encode categorical features
day_mapping = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7}
df['Publication_Day'] = df['Publication_Day'].map(day_mapping)

time_mapping = {'Morning': 1, 'Afternoon': 2, 'Evening': 3, 'Night': 4}
df['Publication_Time'] = df['Publication_Time'].map(time_mapping)

sentiment_map = {'Negative': 1, 'Neutral': 2, 'Positive': 3}
df['Episode_Sentiment'] = df['Episode_Sentiment'].map(sentiment_map)

df['Episode_Title'] = df['Episode_Title'].str.replace('Episode ', '', regex=True)
df['Episode_Title'] = df['Episode_Title'].astype('int')
df['Title_Episode_Length'] = df['Episode_Title'] / (df['Episode_Length_minutes'] + 1)
le = LabelEncoder()
for col in df.select_dtypes('object').columns:
    df[col] = le.fit_transform(df[col]) + 1

# Some Feature engineering
for col in ['Episode_Length_minutes']:
    df[[col + '_sqrt', col + '_squared']] = np.column_stack([
    np.sqrt(df[col]),
    df[col] ** 2
    ])

for col in tqdm(['Episode_Sentiment', 'Genre', 'Publication_Day', 'Podcast_Name', 'Episode_Title',
                 'Guest_Popularity_percentage', 'Host_Popularity_percentage', 'Number_of_Ads']):
    df[col + '_EP'] = df.groupby(col)['Episode_Length_minutes'].transform('mean')

df = process_combinations_fast(df, ['Episode_Length_minutes', 'Episode_Title', 'Publication_Time', 'Host_Popularity_percentage', 'Number_of_Ads', 'Episode_Sentiment', 
                     'Publication_Day', 'Podcast_Name','Genre','Guest_Popularity_percentage'], [2,3,5,7], 1000)

df = df.astype('float32')

df_train = df.iloc[:-len(df_test)]
df_test = df.iloc[-len(df_test):].reset_index(drop=True)

df_train = df_train[df_train['Listening_Time_minutes'].notnull()]

target = df_train.pop('Listening_Time_minutes')
df_test.pop('Listening_Time_minutes')

df_train.shape, df_test.shape


class EarlyStopping(xgb.callback.TrainingCallback):
    def __init__(self, patience=30, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.wait = 0
        self.stopped_epoch = 0
        self.best_iteration = 0

    def after_iteration(self, model, epoch, evals_log):
        score = evals_log['validation']['rmse'][-1]
        
        if self.best_score is None or score < self.best_score - self.min_delta:
            self.best_score = score
            self.wait = 0
            self.best_iteration = epoch
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print(f"Early stopping at epoch {epoch}, best epoch was {self.best_iteration} with RMSE {self.best_score:.5f}")
                return True 
        return False


class HistoryLogger(xgb.callback.TrainingCallback):
    def __init__(self):
        self.history = {'epoch': [], 'lr': [], 'train_rmse': [], 'valid_rmse': []}
    
    def after_iteration(self, model, epoch, evals_log):
        lr = lr_decay(epoch)
        
        self.history['epoch'].append(epoch)
        self.history['lr'].append(lr)  
        self.history['train_rmse'].append(evals_log['train']['rmse'][-1])
        self.history['valid_rmse'].append(evals_log['validation']['rmse'][-1])
        
        return False

def plot_training_history(history):
    epochs = history['epoch']
    train_rmse = history['train_rmse']
    valid_rmse = history['valid_rmse']
    
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axs[0].plot(epochs, train_rmse, label='Train RMSE')
    axs[0].plot(epochs, valid_rmse, label='Validation RMSE')
    axs[0].set_ylabel('RMSE')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(epochs, history['lr'], label='Learning Rate', color='orange')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Learning Rate')
    axs[1].legend()
    axs[1].grid(True)

    plt.show()


seed1 = 42
cv = KFold(7, random_state=seed1, shuffle=True)
pred_test = np.zeros((250000,))


def lr_decay(epoch):
    lr_start = 0.02
    lr_end = 0.005
    decay_speed = 0.01  

    lr = lr_end + (lr_start - lr_end) * np.exp(-decay_speed * epoch)
    return lr
    
callbacks = xgb.callback.LearningRateScheduler(lr_decay)
# XGBoost parameters
params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'seed': seed1,
    'max_depth': 19,
    'learning_rate': 0.03,
    'min_child_weight': 50,
    'reg_alpha': 5,
    'reg_lambda': 1,
    'subsample': 0.85,
    'colsample_bytree': 0.6,
    'colsample_bynode': 0.5,
    'device': "cuda"
}
all_histories = []
for idx_train, idx_valid in cv.split(df_train):
    
    X_train, y_train = df_train.iloc[idx_train], target.iloc[idx_train]
    X_valid, y_valid = df_train.iloc[idx_valid], target.iloc[idx_valid]
    X_test = df_test[X_train.columns].copy()

    features = df_train.columns
    
    encoder1 = TargetEncoder(n_folds=5, seed=seed1, stat="mean")

    for col in tqdm(features[:20]):
        X_train[col+'_te1'] = encoder1.fit_transform(X_train[[col]], y_train)
        X_valid[col+'_te1'] = encoder1.transform(X_valid[[col]])
        X_test[col+'_te1'] = encoder1.transform(X_test[[col]])

    for col in tqdm(features[20:]):
        X_train[col] = encoder1.fit_transform(X_train[[col]], y_train)
        X_valid[col] = encoder1.transform(X_valid[[col]])
        X_test[col] = encoder1.transform(X_test[[col]])

    # Create DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_valid, label=y_valid)
    dtest = xgb.DMatrix(X_test)
    
    early_stopping = EarlyStopping(patience=30, min_delta=0.0005)
    history_logger = HistoryLogger()
    model = xgb.train(
        params, 
        dtrain, 
        num_boost_round=1000000, 
        evals=[(dtrain, 'train'), (dval, 'validation')], 
        early_stopping_rounds=30, 
        verbose_eval=500,
        callbacks=[early_stopping, callbacks, history_logger]
    )
    all_histories.append(history_logger.history)
    plot_training_history(history_logger.history)
    predictions = model.predict(dval)

    # Generate predictions for test set and save submission
    pred_test += np.maximum(0, np.minimum(120, model.predict(dtest)))
    print("----------------------------------------------------------------")

pred_test /= 7


df_sub = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv")
df_sub.Listening_Time_minutes = pred_test
df_sub.to_csv('submission.csv', index=False)

