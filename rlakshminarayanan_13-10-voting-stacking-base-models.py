import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


original_df = pd.read_csv('/kaggle/input/podcast-listening-time-prediction-dataset/podcast_dataset.csv')
train_df = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
sample_submission_df = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')


print(f"The original data has {original_df.shape[0]} rows and {original_df.shape[1]} columns")
print(f"The training data has {train_df.shape[0]} rows and {train_df.shape[1]} columns")
print(f"The test data has {test_df.shape[0]} rows and {test_df.shape[1]} columns")
print(f"The submission file should have {sample_submission_df.shape[0]} rows and {sample_submission_df.shape[1]} columns")


train_df1 = train_df.drop(columns = ['id'])
original_df1 = original_df.dropna()
merged_df = pd.concat([original_df1,train_df1], axis=0)
merged_df = merged_df.drop_duplicates()
merged_df.shape


merged_df.isna().sum()


def feat_eng(dataframe):
    df = dataframe.copy()
    df['Episode_Number'] = df['Episode_Title'].str.split(" ", expand = True)[1].astype(np.uint16)
    df['Episode_Freq'] = df['Episode_Title'].map(df['Episode_Title'].value_counts(normalize=True))
    df = df.drop(columns = 'Episode_Title')
    
    df['Publication_Day'] = df['Publication_Day'].map({'Monday':0, 'Tuesday':1, 'Wednesday':2, 'Thursday':3, 'Friday': 4, 'Saturday': 5, 'Sunday':6})
    df['Publication_Day_sin'] = np.sin(2 * np.pi * df['Publication_Day']/7)
    df['Publication_Day_cos'] = np.cos(2 * np.pi * df['Publication_Day']/7)
    
    df['Publication_Time'] = df['Publication_Time'].map({'Morning':0, 'Afternoon':1, 'Evening':2, 'Night':3})
    df['Publication_Time_sin'] = np.sin(2 * np.pi * df['Publication_Time']/4)
    df['Publication_Time_cos'] = np.cos(2 * np.pi * df['Publication_Time']/4)
    return df


merged_df1 = feat_eng(merged_df)
merged_df1.shape


merged_df1.columns


from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

X_merged = merged_df1.drop(columns = ['Listening_Time_minutes'])
y_merged = merged_df1.Listening_Time_minutes

num_cols = ['Episode_Length_minutes','Host_Popularity_percentage','Publication_Day',
           'Publication_Time','Guest_Popularity_percentage','Number_of_Ads',
            'Episode_Number', 'Episode_Freq', 'Publication_Day_sin', 'Publication_Day_cos',
           'Publication_Time_sin', 'Publication_Time_cos']
cat_cols = ['Podcast_Name','Genre','Episode_Sentiment']

num_processor = Pipeline([
    ('imputer',SimpleImputer(strategy = 'mean')),
    ('scaler',StandardScaler())
])

cat_processor = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
    ('ohe',OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_processor, num_cols),
        ('cat', cat_processor, cat_cols)
    ],
    remainder='drop'
)

X_merged_trn = preprocessor.fit_transform(X_merged)
X_merged_trn.shape


y_merged.shape


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

X_train, X_test, y_train, y_test = train_test_split(X_merged_trn, y_merged, test_size=0.2, random_state=42)
lr = LinearRegression()
lr.fit(X_train,y_train)
y_pred_lr = lr.predict(X_test)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
rmse_lr


%load_ext cuml.accel


from xgboost import XGBRegressor
xgbr = XGBRegressor()
xgbr.fit(X_train,y_train)
y_pred_xgbr = xgbr.predict(X_test)
rmse_xgbr = np.sqrt(mean_squared_error(y_test, y_pred_xgbr))
rmse_xgbr


from lightgbm import LGBMRegressor
lgbm = LGBMRegressor()
lgbm.fit(X_train,y_train)
y_pred_lgbm = lgbm.predict(X_test)
rmse_lgbm = np.sqrt(mean_squared_error(y_test, y_pred_lgbm))
rmse_lgbm


from catboost import CatBoostRegressor
cbr = CatBoostRegressor(verbose=0)
cbr.fit(X_train,y_train)
y_pred_cbr = cbr.predict(X_test)
rmse_cbr = np.sqrt(mean_squared_error(y_test, y_pred_cbr))
rmse_cbr


from sklearn.ensemble import VotingRegressor
vr = VotingRegressor(estimators = [('xgbr', xgbr),('cbr', cbr)])
vr.fit(X_train,y_train)
y_pred_vr = vr.predict(X_test)
rmse_vr = np.sqrt(mean_squared_error(y_test, y_pred_vr))
rmse_vr


test_df.head()


vr_final = VotingRegressor(estimators = [('xgbr', xgbr),('cbr', cbr)])
vr_final.fit(X_merged_trn,y_merged)
X_test_df = test_df.drop(columns = ['id'])
X_test_feat = feat_eng(X_test_df)
X_trn = preprocessor.transform(X_test_feat)
y_pred_test_vr = vr_final.predict(X_trn)
y_pred_test_vr


sample_submission_df.shape


sample_submission_df_vr = sample_submission_df.copy()
sample_submission_df_vr['Listening_Time_minutes'] = y_pred_test_vr
sample_submission_df_vr.to_csv('submission.csv',index=False)


# define the base models
level0 = list()
level0.append(('xgbr', XGBRegressor()))
level0.append(('lgbm', LGBMRegressor(verbose=-1)))
level0.append(('cbr', CatBoostRegressor()))
# define meta learner model
level1 = LinearRegression()
# define the stacking ensemble
model = StackingRegressor(estimators=level0, final_estimator=level1, cv=5)
model.fit(X_train,y_train)
y_pred_stack = model.predict(X_test)
rmse_stack = np.sqrt(mean_squared_error(y_test, y_pred_stack))
rmse_stack


model = StackingRegressor(estimators=level0, final_estimator=level1, cv=5)
model.fit(X_merged_trn,y_merged)
y_pred_stack = model.predict(X_trn)


y_pred_stack


sample_submission_df_vr = sample_submission_df.copy()
sample_submission_df_vr['Listening_Time_minutes'] = y_pred_stack
sample_submission_df_vr.to_csv('submission.csv',index=False)

