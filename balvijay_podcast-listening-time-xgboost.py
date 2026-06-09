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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

import warnings
warnings.filterwarnings('ignore')

# -----------------------
# ğŸ”¹ 1. Load Data
# -----------------------
train_df = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
subm_df = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv', index_col='id')

# -----------------------
# ğŸ”¹ 2. Handle Missing Values
# -----------------------
train_df.fillna(train_df.mean(numeric_only=True), inplace=True)
test_df.fillna(test_df.mean(numeric_only=True), inplace=True)

# -----------------------
# ğŸ”¹ 3. Encode Categorical Columns
# -----------------------
genr_dict = {v: i for i, v in enumerate(train_df['Genre'].unique())}
podc_dict = {v: i for i, v in enumerate(train_df['Podcast_Name'].unique())}
week_dict = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
             'Friday': 4, 'Saturday': 5, 'Sunday': 6}
time_dict = {'Morning': 0, 'Afternoon': 1, 'Evening': 2, 'Night': 3}
sent_dict = {'Positive': 1, 'Neutral': 0, 'Negative': -1}

for df in [train_df, test_df]:
    df['Genre'] = df['Genre'].replace(genr_dict).astype('category')
    df['Podcast_Name'] = df['Podcast_Name'].replace(podc_dict).astype('category')
    df['Publication_Day'] = df['Publication_Day'].replace(week_dict).astype('category')
    df['Publication_Time'] = df['Publication_Time'].replace(time_dict).astype('category')
    df['Episode_Sentiment'] = df['Episode_Sentiment'].replace(sent_dict).astype('category')
    df['Ad_Density'] = df['Number_of_Ads'] / df['Episode_Length_minutes']
    df['Ad_Density'] = df['Ad_Density'].replace([np.inf, -np.inf], 0)

# -----------------------
# ğŸ”¹ 4. Feature Engineering for Interaction
# -----------------------
def interaction_features(df):
    df['Avg_Listen_Rate'] = df['Listening_Time_minutes'] / df['Episode_Length_minutes'] if 'Listening_Time_minutes' in df else 0
    df['Ads_Effect'] = df['Number_of_Ads'] * df['Episode_Length_minutes']
    df['Sentiment_Effect'] = df['Episode_Sentiment'].astype(int) * df['Episode_Length_minutes']
    df['Genre_Effect'] = df['Genre'].astype(int) * df['Episode_Length_minutes']
    return df

train_df = interaction_features(train_df)
test_df = interaction_features(test_df)

# -----------------------
# ğŸ”¹ 5. Define Features
# -----------------------
features = [
    'Genre', 'Podcast_Name', 'Episode_Length_minutes', 'Publication_Day',
    'Publication_Time', 'Episode_Sentiment', 'Number_of_Ads', 'Ad_Density',
    'Ads_Effect', 'Sentiment_Effect', 'Genre_Effect'
]
target = 'Listening_Time_minutes'

X = train_df[features]
y = train_df[target]
X_test = test_df[features]

# -----------------------
# ğŸ”¹ 6. Normalize Features
# -----------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# -----------------------
# ğŸ”¹ 7. Train-Validation Split
# -----------------------
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# -----------------------
# ğŸ”¹ 8. Train XGBoost Model
# -----------------------
xgb_model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42)
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=20, verbose=False)

# -----------------------
# ğŸ”¹ 9. Evaluate Model
# -----------------------
val_preds = xgb_model.predict(X_val)
r2 = r2_score(y_val, val_preds)
print(f"âœ… RÂ² Score on Validation: {r2:.4f}")

# -----------------------
# ğŸ”¹ 10. Generate Submission
# -----------------------
test_preds = xgb_model.predict(X_test_scaled)
subm_df["Listening_Time_minutes"] = test_preds
subm_df.to_csv("/kaggle/working/submission_xgboost.csv")
print("ğŸ“� Submission saved: submission_xgboost.csv")



subm_df.head()

