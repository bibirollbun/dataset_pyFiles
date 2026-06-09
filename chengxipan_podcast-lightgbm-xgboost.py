import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor, Pool
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy import stats

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer

SEED = 42
n_splits = 8
n_estimators=5000
early_stopping_rounds = 100


train_data = pd.read_csv(r"/kaggle/input/playground-series-s5e4/train.csv")
test_data = pd.read_csv(r"/kaggle/input/playground-series-s5e4/test.csv")
data = pd.read_csv(r"/kaggle/input/playground-series-s5e4/sample_submission.csv")

print("train_data shape :",train_data.shape)
print("test_data shape :",test_data.shape)
print("data shape :",data.shape)


def tfidf_vectorization(train_data, test_data, column_name):
    # Vectorize the 'Podcast_Name' column using TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_train = vectorizer.fit_transform(train_data[column_name])
    feature_names = vectorizer.get_feature_names_out()
    tfidf_train_df = pd.DataFrame(tfidf_train.toarray(), columns=feature_names)
    
    # Transform the test data using the same vectorizer
    tfidf_test = vectorizer.transform(test_data[column_name])
    tfidf_test_df = pd.DataFrame(tfidf_test.toarray(), columns=feature_names)
    
    return tfidf_train_df, tfidf_test_df


tfidf_train_podcast, tfidf_test_podcast = tfidf_vectorization(train_data, test_data, 'Podcast_Name')
tfidf_train_publication, tfidf_test_publication = tfidf_vectorization(train_data, test_data, 'Publication_Time')


# Mapeo para Publication_Day basado en tiempo promedio de escucha
day_mapping = {
    'Monday': 0,      # 45.97 min
    'Tuesday': 1,     # 46.13 min (más escuchado)
    'Wednesday': 2,   # 45.81 min
    'Thursday': 3,    # 44.87 min
    'Friday': 4,      # 45.21 min
    'Saturday': 5,    # 45.33 min
    'Sunday': 6       # 44.82 min (menos escuchado)
}

# Mapeo para Publication_Time basado en tiempo promedio de escucha
time_mapping = {
    'Night': 0,      # 46.46 min (más escuchado)
    'Afternoon': 1,  # 45.53 min
    'Morning': 2,    # 44.96 min
    'Evening': 3     # 44.76 min (menos escuchado)
}




def data_process(df):
    df['Episode_Title_num'] = df['Episode_Title'].astype(str).str.replace('Episode ', '').astype(int)
    df['Guest_Popularity_percentage'].fillna(df['Guest_Popularity_percentage'].median(), inplace=True)
    df['Episode_Length_minutes'].fillna(df['Episode_Length_minutes'].median(), inplace=True)
    df['Number_of_Ads'].fillna(df['Number_of_Ads'].median(), inplace=True)
    df['Guest_Popularity_percentage'] = df['Guest_Popularity_percentage'].clip(upper=3)
    df['Episode_Sentiment'] = df['Episode_Sentiment'].replace({'Neutral': 0, 'Positive': 1, 'Negative': -1})

    df['Ad_Density'] = df['Number_of_Ads'] / (df['Episode_Length_minutes'] + 1e-3)
    df['Popularity_Diff'] = df['Host_Popularity_percentage'] - df['Guest_Popularity_percentage']
    df['Popularity_Interaction'] = df['Host_Popularity_percentage'] * df['Guest_Popularity_percentage']
    df['Host_Popularity_squared'] = df['Host_Popularity_percentage'] ** 2
    df['Popularity_Average'] = (df['Host_Popularity_percentage'] + df['Guest_Popularity_percentage'])/2
    
    # target encoding Genre
    genre_target_mean = train_data.groupby('Genre')['Listening_Time_minutes'].mean()
    df['Genre_Target_Encoding'] = df['Genre'].map(genre_target_mean)
    # frequency encoding Genre
    genre_frequency = train_data['Genre'].value_counts(normalize=True)
    df['Genre_Frequency_Encoding'] = df['Genre'].map(genre_frequency)
    
    # 是否为周末
    df['Is_Weekend'] = df['Publication_Day'].apply(lambda x: 1 if x in ['Saturday', 'Sunday'] else 0)
    
    # df['Genre_Num'] = LabelEncoder().fit_transform(df['Genre'])
    df['Publication_Day_Num'] = df['Publication_Day'].map(day_mapping)
    df['Publication_Time_Num'] = df['Publication_Time'].map(time_mapping)

    return df

train_data=data_process(train_data)
test_data=data_process(test_data)



correlation_matrix=train_data[train_data.select_dtypes(include=['number']).columns.tolist()].corr() # correlation matrix, pearson
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='RdBu', linewidths=0.5, vmin=-1, vmax=1)
plt.title('Feature Correlation Heatmap')    
plt.tight_layout()
plt.show()


cat_cols = train_data.select_dtypes(exclude=['number']).columns.tolist()
train_data[cat_cols] = train_data[cat_cols].fillna("Missing")
test_data[cat_cols] = test_data[cat_cols].fillna("Missing")
print(cat_cols) # ['Podcast_Name', 'Episode_Title', 'Genre', 'Publication_Day', 'Publication_Time']
for col in cat_cols:
    le = LabelEncoder()
    train_data[col] = le.fit_transform(train_data[col])
    test_data[col] = le.transform(test_data[col]) 


train_data = pd.concat([train_data, tfidf_train_podcast, tfidf_train_publication], axis=1)
test_data = pd.concat([test_data, tfidf_test_podcast, tfidf_test_publication], axis=1)


X = train_data.drop(['id','Listening_Time_minutes'],axis=1)
y = train_data['Listening_Time_minutes']
test = test_data.drop(['id'],axis=1)

print('true y min:',y.min())
print('true y max:',y.max())
print('true y mean:',y.mean())
print('true y median:',y.median())


from tqdm import tqdm
from sklearn.metrics import mean_squared_error as rmse


kf = KFold(n_splits, shuffle=True, random_state=42)
kf_splits = kf.split(X)
scores1 = []
test_preds1 = []

lgbm_params1 = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    'n_estimators': n_estimators,
    'learning_rate': 0.08,
    'max_depth': 15,
    'num_leaves': 64, 
    'reg_alpha' : 1,
    'reg_lambda' : 8,
    'colsample_bytree' : 0.7,
    'subsample' : 1.0,
    'subsample_freq' : 6,
    'seed': SEED,
    'verbose': -1,
    'device' : 'cpu' 
}

for i, (train_idx, val_idx) in tqdm(enumerate(kf_splits)):
    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
    
    callbacks = [lgb.early_stopping(stopping_rounds=early_stopping_rounds),lgb.log_evaluation(period=1000)]
    model = lgb.LGBMRegressor(**lgbm_params1)
    model.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)],eval_metric='rmse', categorical_feature=cat_cols, callbacks=callbacks)
    
    val_pred = model.predict(X_val_fold, num_iteration=model.best_iteration_)
    score = rmse(y_val_fold, val_pred)
    scores1.append(score)
    
    test_pred = np.maximum(model.predict(test, num_iteration=model.best_iteration_),0)
    test_preds1.append(test_pred)

    
    print(f'LightGBM Fold {i + 1} rmse: {score}')
print(f'LightGBM rmse: {np.mean(scores1):.5f};');



y_preds_lgb = np.mean(test_preds1, axis=0)
y_preds_lgb = y_preds_lgb
print('predict mean :',y_preds_lgb.mean())
print('predict median :',np.median(y_preds_lgb))
y_preds_lgb = np.clip(y_preds_lgb,0,119.97)
print('predict mean final:',y_preds_lgb.mean())
print('predict median final:',np.median(y_preds_lgb))


kf = KFold(n_splits, shuffle=True, random_state=43)
kf_splits = kf.split(X)
scores1 = []
test_preds1 = []

xgb_params1 = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'n_estimators': n_estimators,
    'learning_rate': 0.08,
    'max_depth': 15,
    'subsample': 1.0,
    'colsample_bytree': 0.7,
    'reg_alpha': 0.8,
    'reg_lambda': 4,
    'seed': SEED,
    'tree_method': 'hist',  # 使用 hist 方法加速训练
    'device': 'gpu',
    'early_stopping_rounds': early_stopping_rounds
}

for i, (train_idx, val_idx) in enumerate(kf_splits):
    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

    model = xgb.XGBRegressor(**xgb_params1)
    model.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)], verbose=1000)

    val_pred = model.predict(X_val_fold)
    score = rmse(y_val_fold, val_pred)
    scores1.append(score)

    test_pred = np.maximum(model.predict(test), 0)
    test_preds1.append(test_pred)

    print(f'XGBoost Fold {i + 1} rmse: {score}')
print(f'XGBoost rmse: {np.mean(scores1):.5f};')


y_preds_xgb = np.mean(test_preds1, axis=0)
y_preds_xgb = y_preds_xgb
print('predict mean :',y_preds_xgb.mean())
print('predict median :',np.median(y_preds_xgb))
y_preds_xgb = np.clip(y_preds_xgb,0,119.97)
print('predict mean final:',y_preds_xgb.mean())
print('predict median final:',np.median(y_preds_xgb))


kf = KFold(n_splits, shuffle=True, random_state=45)
kf_splits = kf.split(X)
scores1 = []
test_preds1 = []

random_forest_params = {
    'n_estimators': 100,  # 树的数量
    'max_depth': 15,  # 树的最大深度
    'min_samples_split': 2,  # 内部节点再划分所需最小样本数
    'min_samples_leaf': 1,  # 叶节点最少样本数
    'max_features': 128,  # 寻找最佳分割时要考虑的特征数量
    'bootstrap': True,
    'random_state': SEED,
    'n_jobs': -1,  # 使用所有可用的处理器
}

for i, (train_idx, val_idx) in enumerate(kf_splits):
    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

    model = RandomForestRegressor(**random_forest_params)
    model.fit(X_train_fold, y_train_fold)

    val_pred = model.predict(X_val_fold)
    score = rmse(y_val_fold, val_pred)
    scores1.append(score)

    test_pred = np.maximum(model.predict(test), 0)
    test_preds1.append(test_pred)

    print(f'Random Forest Fold {i + 1} rmse: {score}')

print(f'Random Forest rmse: {np.mean(scores1):.5f};')


y_preds_rf = np.mean(test_preds1, axis=0)
y_preds_rf = y_preds_rf
print('predict mean :',y_preds_rf.mean())
print('predict median :',np.median(y_preds_rf))
y_preds_rf = np.clip(y_preds_rf,0,119.97)
print('predict mean final:',y_preds_rf.mean())
print('predict median final:',np.median(y_preds_rf))


y_preds = y_preds_lgb*0.2 + y_preds_xgb*0.7 + y_preds_rf*0.1
# Save predictions for submission
submission = pd.DataFrame({'id': test_data['id'], 'Listening_Time_minutes': y_preds})
submission.to_csv('submission.csv', index=False)
print(submission.head())

