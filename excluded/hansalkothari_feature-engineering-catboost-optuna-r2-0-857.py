import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
pd.set_option("display.max_column", 999)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



path = '/kaggle/input/playground-series-s5e4/'
test = pd.read_csv(path+'/test.csv')
train = pd.read_csv(path+'/train.csv')

test2 = test.copy()
train2 = train.copy()
train.info()


# Our target variable is Listening_Time_minutes

test.drop(['id'], axis=1, inplace = True)
train.drop(['id'], axis=1, inplace = True)
train2.drop(['id'], axis=1, inplace = True)
test2.drop(['id'], axis=1, inplace = True)


#finding null values in both train and test
print("description of null values in train dataset ")
print(train.isnull().sum())

print("-"*90)

print("description of null values in test dataset ")
print(test.isnull().sum())



train.describe().style.background_gradient(cmap="coolwarm")


train['Episode_Number'] = train['Episode_Title'].str.extract('(\d+)').astype(int)
train.drop(['Episode_Title'], axis=1, inplace=True)

test['Episode_Number'] = test['Episode_Title'].str.extract('(\d+)').astype(int)
test.drop(['Episode_Title'], axis=1, inplace=True)

train2['Episode_Number'] = train2['Episode_Title'].str.extract('(\d+)').astype(int)
train2.drop(['Episode_Title'], axis=1, inplace=True)

test2['Episode_Number'] = test2['Episode_Title'].str.extract('(\d+)').astype(int)
test2.drop(['Episode_Title'], axis=1, inplace=True)



# separate numerical and categorical variables
numerical_variables = test.select_dtypes(include=['int64', 'float64']).columns
categorical_variables = [col for col in test.columns if col not in numerical_variables]


# drop na values with .

train.dropna(subset=['Episode_Length_minutes','Guest_Popularity_percentage' ], inplace=True)
train2.dropna(subset=['Episode_Length_minutes','Guest_Popularity_percentage' ], inplace=True)
test.dropna(subset=['Episode_Length_minutes','Guest_Popularity_percentage' ], inplace=True)
test2 = test2.fillna(0)


# univariate analysis of numerical featues.
custom_palette = ['#3498db', '#e74c3c']

# Add 'Dataset' column to distinguish between train and test data
train['Dataset'] = 'Train'
test['Dataset'] = 'Test'

# Function to create and display a row of plots for a single variable
def create_variable_plots(variable):
    sns.set_style('whitegrid')
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Box plot
    plt.subplot(1, 3, 1)
    sns.boxplot(data=pd.concat([train,test]), x=variable, y="Dataset", palette=custom_palette)
    plt.xlabel(variable)
    plt.title(f"Box Plot for {variable}")

    # Separate Histograms
    plt.subplot(1, 3, 2)
    sns.histplot(data=train, x=variable, color=custom_palette[0], kde=True, bins=30, label="Train")
    # sns.histplot(data=test, x=variable, color=custom_palette[1], kde=True, bins=30, label="Test")
    plt.subplot(1, 3, 3)
    sns.histplot(data=test, x=variable, color=custom_palette[1], kde=True, bins=30, label="Test")
    

    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.title(f"Histogram for {variable} [TRAIN, TEST] ]")
    plt.legend()

    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plots
    plt.show()
    
# Perform univariate analysis for each variable
for variable in numerical_variables:
    create_variable_plots(variable)

# Drop the 'Dataset' column after analysis
train.drop('Dataset', axis=1, inplace=True)
test.drop('Dataset', axis=1, inplace=True)




categorical_variables
for col in categorical_variables:
    plt.figure(figsize=(10, 4))
    sns.countplot(data=train, x=col, palette="coolwarm", order=train[col].value_counts().index)
    plt.xticks(rotation=90)
    plt.title(f'Distribution of {col}')
    plt.show()


for feature in categorical_variables:
    if feature not in ['Episode_Title','Listening_Time_minutes']:
        plt.figure(figsize=(10, 6))
        
        # Violin plot with KDE
        sns.violinplot(
            data=train, 
            x=feature, 
            y="Listening_Time_minutes", 
            palette="coolwarm", 
            inner="box"  # Keeps a boxplot inside for median & quartiles
        )
        
        plt.xlabel(feature, fontsize=12)
        plt.ylabel("Listening Time (Minutes)", fontsize=12)
        plt.title(f"Violin Plot for {feature} vs Listening Time", fontsize=14)
        plt.xticks(rotation=80)
        plt.tight_layout()
        
        # plt.show()


from sklearn.preprocessing import LabelEncoder

label_encoders = {}

for col in categorical_variables:
    le = LabelEncoder()
    
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col]) 

    train2[col] = le.fit_transform(train2[col])
    test2[col] = le.transform(test2[col]) 
    
    label_encoders[col] = le  



from sklearn.preprocessing import OrdinalEncoder
encoder = OrdinalEncoder()


def feature_engineering(df_train, df_test):

    df_train['Is_Weekend'] = df_train['Publication_Day'].apply(lambda x: 1 if x in [6, 7] else 0)
    df_test['Is_Weekend'] = df_test['Publication_Day'].apply(lambda x: 1 if x in [6, 7] else 0)
    
    # Guest-Host interaction ratio
    df_train['Total_Popularity'] = df_train['Host_Popularity_percentage'] + df_train['Guest_Popularity_percentage'] 
    df_test['Total_Popularity'] = df_test['Host_Popularity_percentage'] + df_test['Guest_Popularity_percentage'] 

    
    df_train['Popularity_Diff'] = df_train['Host_Popularity_percentage'] - df_train['Guest_Popularity_percentage'] 
    df_test['Popularity_Diff'] = df_test['Host_Popularity_percentage'] - df_test['Guest_Popularity_percentage'] 

    
    df_train['Host_Guest_Popularity_Ratio'] = df_train['Host_Popularity_percentage'] / (df_train['Guest_Popularity_percentage'] + 1e-5)
    df_test['Host_Guest_Popularity_Ratio'] = df_test['Host_Popularity_percentage'] / (df_test['Guest_Popularity_percentage'] + 1e-5)

    
    df_train['Popularity_Interaction'] = (df_train['Host_Popularity_percentage'] + 1) * (df_train['Guest_Popularity_percentage'] + 1)
    df_test['Popularity_Interaction'] = (df_test['Host_Popularity_percentage'] + 1) * (df_test['Guest_Popularity_percentage'] + 1)

    # Ad density
    df_train['Ads_per_Minute'] = df_train['Number_of_Ads'] / (df_train['Episode_Length_minutes'] + 1e-5)
    df_test['Ads_per_Minute'] = df_test['Number_of_Ads'] / (df_test['Episode_Length_minutes'] + 1e-5)

    #popularity score
    df_train['Popularity_Score'] = (df_train['Host_Popularity_percentage'] + df_train['Guest_Popularity_percentage']) / 2 
    df_test['Popularity_Score'] = (df_test['Host_Popularity_percentage'] + df_test['Guest_Popularity_percentage']) / 2 

    #long episode 
    df_train['Long_Episode'] = (df_train['Episode_Length_minutes'] > 75).astype(int)
    df_test['Long_Episode'] = (df_test['Episode_Length_minutes'] > 75).astype(int)

    # Highly popular guest
    df_train['Highly_Popular_Guest'] = (df_train['Guest_Popularity_percentage'] > 75).astype(int)
    df_test['Highly_Popular_Guest'] = (df_test['Guest_Popularity_percentage'] > 75).astype(int)

    # Highly popular host
    df_train['Highly_Popular_Host'] = (df_train['Host_Popularity_percentage'] > 75).astype(int)
    df_test['Highly_Popular_Host'] = (df_test['Host_Popularity_percentage'] > 75).astype(int)
    
    
    df_train['Ad_Sentiment_Burden'] = df_train['Episode_Sentiment'] * df_train['Number_of_Ads']
    df_test['Ad_Sentiment_Burden'] = df_test['Episode_Sentiment'] * df_test['Number_of_Ads']
    
    df_train['Length_Sentiment_Burden'] = df_train['Episode_Sentiment'] * df_train['Episode_Length_minutes']
    df_test['Length_Sentiment_Burden'] = df_test['Episode_Sentiment'] * df_test['Episode_Length_minutes']

    df_train['Ad_Impact'] = df_train['Number_of_Ads'] * df_train['Episode_Length_minutes']
    df_test['Ad_Impact'] = df_test['Number_of_Ads'] * df_test['Episode_Length_minutes']
    
     

# Apply inplace modifications
feature_engineering(train, test)
feature_engineering(train2, test2)



train


# separate numerical and categorical variables
numerical_variables = test.select_dtypes(include=['int64', 'float64']).columns
categorical_variables = [col for col in test.columns if col not in numerical_variables]



df = train[list(numerical_variables) + ['Listening_Time_minutes']]
corr = df.corr()

plt.figure(figsize=(15, 6))
ax = sns.heatmap(corr, annot=True, fmt=".2f", cmap="magma", cbar=True, linewidths=0.5)
ax.xaxis.tick_top()

plt.title("Correlation Heatmap of Numerical Variables", fontsize=14)
plt.show()


features = list(train.columns)

remove_cols = [
    'Listening_Time_minutes', 'Episode_Number', 'Guest_Popularity_percentage',
]

for col in remove_cols:
    if col in features:  
        features.remove(col)

print(features)


#Lets use catboost 

import optuna
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error

#RUN ONLY TO FIND BEST_PARAMS. ELSE THE BEST_PARAMS FOUND USING THIS CODE IS ALREADY MENTIONED IN BELOW CELL

# def kfold(df, model, df_test_=None, plot=False, verbose=0):

#     eps = df['Episode_Number'].unique()
#     cvs = []
#     pred_tests = []
#     target = "Listening_Time_minutes" 
    
#     for i in range(5, len(eps),6):

#         train_episodes = eps[i-5:i]  # Take last 5 episodes
#         val_episode = eps[i]
        
#         if verbose:
#             print(f'\nTraining on episodes {train_episodes}, validating on episode {val_episode}')
        
#         df_train = df[df['Episode_Number'].isin(train_episodes)].reset_index(drop=True).copy()
#         df_val = df[df['Episode_Number'] == val_episode].reset_index(drop=True).copy()
#         df_test = df_test_.copy()
        
#         # df_train, df_val, df_test = rescale(features, df_train, df_val, df_test)
        
#         model.fit(df_train[features], df_train[target])
#         pred = model.predict(df_val[features])

        
#         if df_test is not None:
#             pred_test = model.predict(df_test[features])
#             # pred_test = (pred_test - pred_test.min()) / (pred_test.max() - pred_test.min())
                
#             pred_tests.append(pred_test)
        
#         # pred = (pred - pred.min()) / (pred.max() - pred.min())
#         # pred = np.clip(pred, 0, 1)

#         # score = ((df_val['Listening_Time_minutes'].values - pred) ** 2).mean()
#         score = mean_squared_error(df_val['Listening_Time_minutes'].values, pred)
#         cvs.append(score)

#         if verbose:
#             print(f'Scored {score:.3f}')
        
#     print(f'\n Local CV is {np.mean(cvs):.3f}')
    
#     return pred, df_val['Listening_Time_minutes'].values

# def objective(trial):
#     cat_params = dict(
#         iterations=trial.suggest_int("iterations", 100, 400),
#         learning_rate=trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
#         depth=trial.suggest_int("depth", 3, 10),
#         l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1e-2, 10.0, log=True),
#         bagging_temperature=trial.suggest_float('bagging_temperature', 0, 1.5),
#         random_strength=trial.suggest_float("random_strength", 1e-3, 5.0, log=True),
#         task_type='GPU',
#         early_stopping_rounds=100,
#         verbose=False
#     )
    
#     model = CatBoostRegressor(**cat_params)
    
#     y_pred , y_val = kfold(train.sort_values('Episode_Number'), model, test, plot=False, verbose=1)
#     score = mean_squared_error(y_val, y_pred)
    
#     return score



# study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials=7)



# best_params = study.best_params
# best_params.update({
#     "task_type": "GPU",  # or "CPU" depending on your setup
#     "early_stopping_rounds": 200,
#     "verbose": False
# })
# best_params

# BEST_PARAMS ACCORDING TO ABOVE TRAILS, GIVING  MSE=113.034

# best_params = {'iterations': 804,
#     'learning_rate': 0.00787995274245964,
#     'depth': 9,
#     'l2_leaf_reg': 0.355577629174872,
#     'bagging_temperature': 0.9597044362376916,
#     'random_strength': 0.010006740830816919,
#     'task_type': 'GPU',
#     'early_stopping_rounds': 200,
#     'verbose': False
# }

# best_params = {'iterations': 368,
#  'learning_rate': 0.025007449165913655,
#  'depth': 8,
#  'l2_leaf_reg': 1.4274202665840763,
#  'bagging_temperature': 0.8174956060640513,
#  'random_strength': 0.004017108472901758,
#  'task_type': 'GPU',
#  'early_stopping_rounds': 200,
#  'verbose': False
# }


# from sklearn.metrics import mean_squared_error, r2_score
# def kfold(df, model, df_test_=None, plot=False, verbose=0):

#     eps = df['Episode_Number'].unique()
#     cvs = []
#     pred_tests = []
#     target = "Listening_Time_minutes" 
    
#     for i in range(5, len(eps),6):

#         train_episodes = eps[i-5:i]  # Take last 5 episodes
#         val_episode = eps[i]
        
#         if verbose:
#             print(f'\nTraining on episodes {train_episodes}, validating on episode {val_episode}')
        
#         df_train = df[df['Episode_Number'].isin(train_episodes)].reset_index(drop=True).copy()
#         df_val = df[df['Episode_Number'] == val_episode].reset_index(drop=True).copy()
#         df_test = df_test_.copy()
        
#         model.fit(df_train[features], df_train[target])
#         pred = model.predict(df_val[features])

        
#         if df_test is not None:
#             pred_test = model.predict(df_test[features])
#             pred_tests.append(pred_test)
        
        
#         score = mean_squared_error(df_val['Listening_Time_minutes'].values, pred)
#         r2 = r2_score(df_val['Listening_Time_minutes'].values, pred)
#         cvs.append(score)

#         if verbose:
#             print(f'Scored {score:.3f} and R2_score {r2:.3f}')
        
#     print(f'\n Local CV is {np.mean(cvs):.3f}')
    
#     return pred_test

    
# final_model = CatBoostRegressor(**best_params)
# y_pred = kfold(train.sort_values('Episode_Number'), final_model, test2, plot=False, verbose=1)




y = train['Listening_Time_minutes']
train = train.drop(['Listening_Time_minutes'],axis=1)

X = train
X_test = test2


import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import KFold
import xgboost as xgb
from tqdm import tqdm 

def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))

# Define XGBoost parameters
xgb_params = {
    'n_estimators': 560,
    'max_depth': 14,
    'learning_rate': 0.04221,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'tree_method': 'hist',
    'n_jobs': -1,
    'eval_metric': 'rmse'
}

# Set up K-Fold cross-validation
n_splits = 10
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

scores = []
test_preds = np.zeros(len(X_test))
models = []  # Optional: store models per fold

# Cross-validation loop
for fold, (train_idx, val_idx) in enumerate(tqdm(kf.split(X, y), total=n_splits)):
    print(f"\nğŸ“¦ Training fold {fold + 1}/{n_splits}...")
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = xgb.XGBRegressor(**xgb_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=False
    )

    val_pred = model.predict(X_val)
    score = rmse(y_val, val_pred)
    scores.append(score)
    test_preds += model.predict(X_test) / n_splits
    models.append(model)  # Optional

    print(f"âœ… Fold {fold + 1} RMSE: {score:.4f} | Best iteration: {model.best_iteration}")

# Summary
print("\nğŸ“Š Cross-Validation Results:")
print(f"Average RMSE: {np.mean(scores):.4f}")
print(f"Std Dev RMSE: {np.std(scores):.4f}")
print(f"Max RMSE:    {np.max(scores):.4f}")
print(f"Min RMSE:    {np.min(scores):.4f}")




# create submission df
test_preds.shape



# make the sumbission

sample = pd.read_csv(path+'/sample_submission.csv')
sample['Listening_Time_minutes'] = test_preds

sample.to_csv('submission.csv', index=False)
# submission.to_csv('/kaggle/working/submission.csv', index=False)
# submission


# !rm /kaggle/working/submission.csv


# !rm -rf /kaggle/working/


sample




