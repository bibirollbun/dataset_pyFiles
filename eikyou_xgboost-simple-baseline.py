import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')

train.head()


combined = pd.concat([train, test])

mean_episode_length = combined.groupby('Podcast_Name')['Episode_Length_minutes'].mean()
mean_guest_popularity = combined.groupby('Podcast_Name')['Guest_Popularity_percentage'].mean()

train['Episode_Length_minutes'] = train['Episode_Length_minutes'].fillna(train['Podcast_Name'].map(mean_episode_length))
train['Guest_Popularity_percentage'] = train['Guest_Popularity_percentage'].fillna(train['Podcast_Name'].map(mean_guest_popularity))

test['Episode_Length_minutes'] = test['Episode_Length_minutes'].fillna(test['Podcast_Name'].map(mean_episode_length))
test['Guest_Popularity_percentage'] = test['Guest_Popularity_percentage'].fillna(test['Podcast_Name'].map(mean_guest_popularity))


train.loc[~train['Number_of_Ads'].isin([0, 1, 2, 3]), 'Number_of_Ads'] = 0
test.loc[~test['Number_of_Ads'].isin([0, 1, 2, 3]), 'Number_of_Ads'] = 0


train['Episode_Title'] = train['Episode_Title'].apply(lambda x: x.split()[1])

test['Episode_Title'] = test['Episode_Title'].apply(lambda x: x.split()[1])


train['Publication'] = train['Publication_Day'].astype(str) + "_" + train['Publication_Time'].astype(str)

test['Publication'] = test['Publication_Day'].astype(str) + "_" + test['Publication_Time'].astype(str)


CATS = ['Podcast_Name', 'Genre', 'Publication_Day', 'Publication_Time', 'Publication', 'Number_of_Ads', 'Episode_Sentiment']
NUMS = ['Episode_Title', 'Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage']
TARGET = 'Listening_Time_minutes'
FEATURES = CATS + NUMS

for col in CATS:
    train[col] = train[col].astype('category')
    test[col] = test[col].astype('category')

for col in NUMS:
    train[col] = train[col].astype('float32')
    test[col] = test[col].astype('float32')
    mean_val = train[col].mean()  
    std_val = train[col].std()    
    train[col] = (train[col] - mean_val) / std_val
    test[col] = (test[col] - mean_val) / std_val  


from xgboost import XGBRegressor
import xgboost as xgb
print("Using XGBoost version",xgb.__version__)


from sklearn.model_selection import KFold
FOLDS = 5
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)


%%time
    
oof_xgb = np.zeros(len(train))
pred_xgb = np.zeros(len(test))

for i, (train_index, test_index) in enumerate(kf.split(train)):

    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)
    
    X_train = train.loc[train_index,FEATURES].copy()
    y_train = train.loc[train_index,TARGET].copy()
    X_val = train.loc[test_index,FEATURES].copy()
    y_val = train.loc[test_index,TARGET].copy()
    X_test = test[FEATURES].copy()
    
    model_xgb = XGBRegressor(n_estimators=10000,
                             learning_rate=0.02,
                             enable_categorical=True,
                             device='cuda', 
                             random_state=42,
                            )
    model_xgb.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],  
        verbose=200,
        early_stopping_rounds=500,
    )

    # INFER OOF
    oof_xgb[test_index] = model_xgb.predict(X_val)
    # INFER TEST
    pred_xgb += model_xgb.predict(X_test)

# COMPUTE AVERAGE TEST PREDS
pred_xgb /= FOLDS


from sklearn.metrics import mean_squared_error

print('CV RMSE for XGBoost', np.sqrt(mean_squared_error(oof_xgb, train.loc[:, TARGET])))


feature_importance = model_xgb.feature_importances_
importance_df = pd.DataFrame({
    "Feature": FEATURES,  # Replace FEATURES with your list of feature names
    "Importance": feature_importance
}).sort_values(by="Importance", ascending=False)
plt.figure(figsize=(10, 15))
plt.barh(importance_df["Feature"], importance_df["Importance"])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("XGBoost")
plt.gca().invert_yaxis()  # Flip features for better readability
plt.show()


sub = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv")
sub.Listening_Time_minutes = pred_xgb
sub.to_csv("submission.csv",index=False)
print("Sub shape:",sub.shape)
sub.head()

