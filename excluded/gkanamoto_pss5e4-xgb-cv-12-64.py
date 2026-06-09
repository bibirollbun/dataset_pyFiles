import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor, Pool
from gensim.models import Word2Vec

# Ignore all warnings
import warnings
warnings.simplefilter("ignore")


train_df = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")


train_df.head()


test_df.head()


# basic information
print("="*20, "Training data", "="*20)
print(train_df.info())
print()
print("="*20, "Test data", "="*20)
print(test_df.info())


print("="*20, "Training data", "="*20)
print(train_df.isnull().sum())
print()
print("="*20, "Test data", "="*20)
print(test_df.isnull().sum())


train_df.describe()


test_df.describe()


fig, axes = plt.subplots(1, 2, figsize=(10, 7))

# Training data
sns.heatmap(train_df.isnull(), cmap="viridis", cbar=False, ax=axes[0])
axes[0].set_title("Training Data: Missing Values")

# Test data
sns.heatmap(test_df.isnull(), cmap="viridis", cbar=False, ax=axes[1])
axes[1].set_title("Test Data: Missing Values")

plt.tight_layout()
plt.show()


# Check training data
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# Episode_Length_minutes
sns.histplot(train_df["Episode_Length_minutes"], bins=30, kde=True, color="blue", ax=axes[0])
axes[0].set_title(f"Distribution of Episode_Length_minutes")

# Guest_Popularity_percentage
sns.histplot(train_df["Guest_Popularity_percentage"], bins=30, kde=True, color="blue", ax=axes[1])
axes[1].set_title(f"Distribution of Guest_Popularity_percentage")

# Number_of_Ads
sns.histplot(train_df["Number_of_Ads"], bins=30, kde=True, color="blue", ax=axes[2])
axes[2].set_title(f"Distribution of Number_of_Ads")

plt.tight_layout()
plt.show()


# Check test data
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Episode_Length_minutes
sns.histplot(test_df["Episode_Length_minutes"], bins=30, kde=True, color="blue", ax=axes[0])
axes[0].set_title(f"Distribution of Episode_Length_minutes")

# Guest_Popularity_percentage
sns.histplot(test_df["Guest_Popularity_percentage"], bins=30, kde=True, color="blue", ax=axes[1])
axes[1].set_title(f"Distribution of Guest_Popularity_percentage")

plt.tight_layout()
plt.show()


# train_df['Episode_Length_minutes'].fillna(train_df["Episode_Length_minutes"].median(), inplace=True)
# train_df['Guest_Popularity_percentage'].fillna(train_df["Guest_Popularity_percentage"].median(), inplace=True)
# train_df['Number_of_Ads'].fillna(train_df["Number_of_Ads"].median(), inplace=True)

# test_df['Episode_Length_minutes'].fillna(test_df["Episode_Length_minutes"].median(), inplace=True)
# test_df['Guest_Popularity_percentage'].fillna(test_df["Guest_Popularity_percentage"].median(), inplace=True)


categorical_features =train_df.select_dtypes(exclude=['number']).columns.tolist()
print("Categorical Features:", categorical_features)


## One-Hot encoding
cat_cols = ['Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']
# cat_cols = ['Genre']

all_df = pd.concat([train_df, test_df], sort=False).reset_index(drop=True)
all_df = pd.get_dummies(all_df, columns=cat_cols)


## Mapping -Publication_Day, Publication_Time, and Episode_Sentiment
# Publication_Day
# level_mapping = {
#     'Sunday': 1,
#     'Thursday': 2,
#     'Friday': 3,
#     'Saturday': 4,
#     'Wednesday': 5,
#     'Monday': 6,
#     'Tuesday': 7
# }

# all_df['Publication_Day'] = all_df['Publication_Day'].map(level_mapping)

# # Publication_Time
# level_mapping = {
#     'Evening': 1,
#     'Morning': 2,
#     'Afternoon': 3,
#     'Night': 4,
# }

# all_df['Publication_Time'] = all_df['Publication_Time'].map(level_mapping)

# # Episode_Sentiment
# level_mapping = {
#     'Negative': 1,
#     'Neutral': 2,
#     'Positive': 3
# }

# all_df['Episode_Sentiment'] = all_df['Episode_Sentiment'].map(level_mapping)


## Handling Episode_Title
all_df["Episode_Number"] = all_df["Episode_Title"].str.replace("Episode ", "", regex=True).astype(int)
all_df = all_df.drop(columns=["Episode_Title"])


## Target encoding
agg_stats = train_df.groupby('Podcast_Name')['Listening_Time_minutes'].agg(
    # Podcast_Mean_ListenTime='mean',
    Podcast_Median_ListenTime='median',
    Podcast_Q25_ListenTime=lambda x: x.quantile(0.25),
    Podcast_Q75_ListenTime=lambda x: x.quantile(0.75),
).reset_index()

all_df = all_df.merge(agg_stats, on='Podcast_Name', how='left')

# podcast_mean = train_df.groupby('Podcast_Name')['Listening_Time_minutes'].mean()
# all_df['Podcast_Mean_ListenTime'] = all_df['Podcast_Name'].map(podcast_mean)

all_df = all_df.drop(columns=['Podcast_Name'])


## Handling Podcast_Name Episode_Title
# from sklearn.decomposition import PCA

# Trining the Word2Vec model
# categorical_features = ['Podcast_Name']
# all_df['target_cat'] = all_df[categorical_features].astype(str).agg(' '.join, axis=1)

# sentences = [text.split() for text in all_df['target_cat']]
# word2vec_model = Word2Vec(sentences, vector_size=50, window=10, sg=1, min_count=1, workers=1)

# def get_avg_vector(sentence, model):
#     vectors = [model.wv[word] for word in sentence if word in model.wv]
#     if vectors:
#         return np.mean(vectors, axis=0)
#     else:
#         return np.zeros(model.vector_size)

# word2vec_vectors = [get_avg_vector(sentence, word2vec_model) for sentence in sentences]

# # apply PCA
# pca = PCA()
# word2vec_pca = pca.fit_transform(word2vec_vectors)
# cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)
# n_components = np.argmax(cumulative_explained_variance >= 0.95) + 1
# pca = PCA(n_components=n_components)
# word2vec_pca_reduced = pca.fit_transform(word2vec_vectors)

# word2vec_df = pd.DataFrame(word2vec_pca_reduced, columns=[f'word2vec_pca_{i}' for i in range(word2vec_pca_reduced.shape[1])])
# # word2vec_df = pd.DataFrame(word2vec_vectors, columns=[f'word2vec_{i}' for i in range(word2vec_model.vector_size)])

# # Combine Word2Vec vectors with the original dataframe
# all_df = pd.concat([all_df.reset_index(drop=True), word2vec_df], axis=1)

# # Drop the original categorical features and combined column
# all_df = all_df.drop(columns=['target_cat', 'Podcast_Name'])
# all_df = all_df.drop(columns=['Podcast_Name'])


# NEW FEATURES

# Average Host & Guest Popularity
all_df['Host_Guest_Avg_Popularity'] = (all_df['Host_Popularity_percentage'] + all_df['Guest_Popularity_percentage']) / 2

# Binary feature for Ads (1 if ads exist, 0 otherwise)
all_df['Has_Ads'] = all_df['Number_of_Ads'].apply(lambda x: 1 if x > 0 else 0)

all_df['Is_High_Host_Popularity'] = (all_df['Host_Popularity_percentage'] > 70).astype(int)
all_df['Is_High_Guest_Popularity'] = (all_df['Guest_Popularity_percentage'] > 70).astype(int)
all_df['Host_Guest_Popularity_Gap'] = all_df['Host_Popularity_percentage'] - all_df['Guest_Popularity_percentage']
all_df['Ad_Density'] = all_df['Number_of_Ads'] / all_df['Episode_Length_minutes']
all_df['Ad_Density'].replace([np.inf, -np.inf], np.nan, inplace=True)
all_df['Is_Long_Episode'] = (all_df['Episode_Length_minutes'] > 60).astype(int)


all_df.head()


train_df = all_df[~all_df["Listening_Time_minutes"].isnull()]
test_df = all_df[all_df["Listening_Time_minutes"].isnull()]


X = train_df.drop(columns=["id", "Listening_Time_minutes"])
y = train_df["Listening_Time_minutes"]


SEED = 42
N_SPLITS = 5


# ## Optuna
# import optuna
# from sklearn.model_selection import train_test_split

# X_train, X_val, y_train, y_val = train_test_split(
#     X, y, test_size=0.2, random_state=SEED
# )

# def objective(trial):
#     param = {
#         "objective": "reg:squarederror",
#         "eval_metric": "rmse",
#         "tree_method": "gpu_hist",
#         "predictor": "gpu_predictor",
#         "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.3, step=0.01),
#         "max_depth": trial.suggest_int("max_depth", 5, 30),
#         "subsample": trial.suggest_float("subsample", 0.6, 1.0, step=0.05),
#         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0, step=0.05),
#         "max_bin": trial.suggest_int("max_bin", 256, 1024, step=128),
#         "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
#         "gamma": trial.suggest_float("gamma", 0, 5),
#         "lambda": trial.suggest_float("lambda", 1e-3, 10.0, log=True),
#         "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),
#         "random_state": SEED
#     }

#     dtrain = xgb.DMatrix(X_train, label=y_train)
#     dval = xgb.DMatrix(X_val, label=y_val)

#     model = xgb.train(
#         params=param,
#         dtrain=dtrain,
#         num_boost_round=100,
#         evals=[(dtrain, "train"), (dval, "valid")],
#         early_stopping_rounds=50,
#         verbose_eval=0
#     )

#     y_pred = model.predict(dval, iteration_range=(0, model.best_iteration))
#     rmse = np.sqrt(mean_squared_error(y_val, y_pred))
#     return rmse

# study = optuna.create_study(direction="minimize")
# study.optimize(objective, n_trials=100)

# print("Best hyperparameters:", study.best_params)
# print("Best RMSE score:", study.best_value)


best_params =  {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "gpu_hist",
    'learning_rate': 0.05, 
    'max_depth': 20, 
    'subsample': 0.75, 
    'colsample_bytree': 0.7, 
    'max_bin': 768, 
    'min_child_weight': 3, 
    'gamma': 3.3471817494642293, 
    'lambda': 1.7785513842743859, 
    'alpha': 0.13471663687947796,
    "n_jobs": 2,
    "random_state": SEED
}


# kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

# scores = []
# models = []

# for fold, (train_index, valid_index) in enumerate(kf.split(X)):
#     X_train, X_val = X.iloc[train_index], X.iloc[valid_index]
#     y_train, y_val = y.iloc[train_index], y.iloc[valid_index]

#     train_pool = Pool(X_train, label=y_train)
#     val_pool = Pool(X_val, label=y_val)

#     model = CatBoostRegressor(
#         iterations=1500,
#         learning_rate=0.08777255350163136,
#         depth=10,
#         l2_leaf_reg=0.1259643500248322,
#         bootstrap_type='Bayesian',
#         random_strength=4.276181166674371e-08,
#         bagging_temperature=0.35995482350907326,
#         od_type='Iter',
#         od_wait=39,
#         verbose=200,
#         allow_writing_files=False,
#         random_seed=SEED
#     )

#     model.fit(train_pool, eval_set=val_pool)

#     y_pred_val = model.predict(X_val)
#     score = np.sqrt(mean_squared_error(y_val, y_pred_val))
#     print(f"Fold: {fold + 1} RMSE score: {score:.5f}")

#     scores.append(score)
#     models.append(model)


kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

scores = []
models = []

for fold, (train_index, valid_index) in enumerate(kf.split(X)):
    X_train, X_val = X.iloc[train_index], X.iloc[valid_index]
    y_train, y_val = y.iloc[train_index], y.iloc[valid_index]
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    # Modeling
    model = xgb.train(
        params=best_params,
        dtrain=dtrain,
        num_boost_round=1500,
        evals=[(dtrain, "train"), (dval, "valid")],
        early_stopping_rounds=100,
        verbose_eval=0
    )

    # Prediction
    y_pred_val = model.predict(dval, iteration_range=(0, model.best_iteration))

    score = np.sqrt(mean_squared_error(y_val, y_pred_val))
    print(f'Fold: {fold+1} RMSE score: {np.mean(score):.5f}') 

    scores.append(score)
    models.append(model)


# kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

# scores = []
# models = []

# for fold, (train_index, valid_index) in enumerate(kf.split(X)):
#     X_train, X_val = X.iloc[train_index], X.iloc[valid_index]
#     y_train, y_val = y.iloc[train_index], y.iloc[valid_index]
#     df_train = lgb.Dataset(X_train, y_train)
#     df_val = lgb.Dataset(X_val, y_val, reference=df_train)
    
#     parameter = {
#         'objective': 'regression',
#         'metric': 'rmse',
#         'boosting_type': 'gbdt',
#         'learning_rate': 0.05,
#         'num_leaves': 64,
#         'max_depth': 40,
#         'feature_fraction': 0.8,
#         'bagging_fraction': 0.8,
#         'bagging_freq': 1,
#         'lambda_l1': 0.1,
#         'lambda_l2': 0.1,
#         'min_data_in_leaf': 50,
#         'min_sum_hessian_in_leaf': 1e-2,
#         'verbosity': -1,
#         'random_state': SEED
#     }
    
#     # Modeling
#     model = lgb.train(parameter,
#                       train_set=df_train,
#                       valid_sets=[df_train, df_val],
#                       num_boost_round=100,
#                       callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=True), lgb.log_evaluation(0)]
#                      )
    
#     # Predict
#     y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)
    
#     score = np.sqrt(mean_squared_error(y_val, y_pred_val))
    
#     scores.append(score)
#     models.append(model)


print(f'Cross-validated RMSE score: {np.mean(scores):.5f} +/- {np.std(scores):.5f}') 





test_id = test_df["id"]
test = test_df.drop(columns=["id", "Listening_Time_minutes"])
submit_score = []

dtest = xgb.DMatrix(test)
for fold_, model in enumerate(models):
    # predict test data
    pred_ = model.predict(dtest, iteration_range=(0, model.best_iteration)) # XGBoost
    # pred_ = model.predict(test, num_iteration=model.best_iteration) # LightGBM
    # pred_ = model.predict(X_val)
    submit_score.append(pred_)

# predict test data
pred = np.mean(submit_score, axis=0)


submission = pd.DataFrame({
    'id': test_id,
    'Listening_Time_minutes': pred
})

# Save
submission.to_csv('submission.csv', index=False)


submission




