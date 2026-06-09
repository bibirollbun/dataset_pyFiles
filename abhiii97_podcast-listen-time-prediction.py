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


# Basic Libraries
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Model and Evaluation
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error





# Load the data
train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')


## view the dataset
train.head()


test.head()


train.info


train.columns


train.isnull().sum()


# Drop columns that are not useful for modeling
train.drop(['id', 'Podcast_Name', 'Episode_Title'], axis=1, inplace=True)
test.drop(['id', 'Podcast_Name', 'Episode_Title'], axis=1, inplace=True)


# Convert string columns with numbers to float
train['Number_of_Ads'] = pd.to_numeric(train['Number_of_Ads'], errors='coerce')
test['Number_of_Ads'] = pd.to_numeric(test['Number_of_Ads'], errors='coerce')


# Fill missing values with median
train['Episode_Length_minutes'].fillna(train['Episode_Length_minutes'].median(), inplace=True)
train['Guest_Popularity_percentage'].fillna(train['Guest_Popularity_percentage'].median(), inplace=True)

test['Episode_Length_minutes'].fillna(test['Episode_Length_minutes'].median(), inplace=True)
test['Guest_Popularity_percentage'].fillna(test['Guest_Popularity_percentage'].median(), inplace=True)


# One-hot encode categorical features
train = pd.get_dummies(train, columns=['Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment'], drop_first=True)
test = pd.get_dummies(test, columns=['Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment'], drop_first=True)


# Align train and test columns
X = train.drop('Listening_Time_minutes', axis=1)
y = train['Listening_Time_minutes']
X, test = X.align(test, join='left', axis=1, fill_value=0)


from sklearn.impute import SimpleImputer

# Impute any remaining missing values with mean
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
test = pd.DataFrame(imputer.transform(test), columns=test.columns)



# Feature Engineering
X['ad_density'] = X['Number_of_Ads'] / (X['Episode_Length_minutes'] + 1)
X['length_x_popularity'] = X['Episode_Length_minutes'] * X['Guest_Popularity_percentage']
X['log_length'] = np.log1p(X['Episode_Length_minutes'])
X['popularity_per_min'] = X['Guest_Popularity_percentage'] / (X['Episode_Length_minutes'] + 1)

# Do same for test set
test['ad_density'] = test['Number_of_Ads'] / (test['Episode_Length_minutes'] + 1)
test['length_x_popularity'] = test['Episode_Length_minutes'] * test['Guest_Popularity_percentage']
test['log_length'] = np.log1p(test['Episode_Length_minutes'])
test['popularity_per_min'] = test['Guest_Popularity_percentage'] / (test['Episode_Length_minutes'] + 1)

# Correlation Heatmap

# Combine features and target for correlation
X_with_target = X.copy()
X_with_target['Listening_Time_minutes'] = y

# Compute correlations with target
correlation_with_target = X_with_target.corr()['Listening_Time_minutes'].sort_values(ascending=False)

# Display as heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_with_target.to_frame(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation of Features with Listening Time")
plt.show()


# Model with improved parameters

from sklearn.model_selection import cross_val_score
model = XGBRegressor(
    n_estimators=2000,          # More trees, allows smaller learning rate
    learning_rate=0.04,        # Slower, more stable learning
    max_depth=9,                # Slightly shallower trees (avoids overfitting)
    subsample=0.9,
    colsample_bytree=0.9,
    gamma=0.2,
    reg_alpha=0.5,              # More regularization
    reg_lambda=1.2,
    min_child_weight=5,
    random_state=42,
    tree_method='gpu_hist',
    predictor='gpu_predictor'
)

# 5-fold Cross-Validation RMSE
scores = cross_val_score(model, X, y, scoring='neg_root_mean_squared_error', cv=5)
print("Average RMSE from 5-fold CV:", -np.mean(scores))



# Fit on full data now
model.fit(X, y)


# predict on test set
test_pred = model.predict(test)


print(test_pred[:10])


submission = sample_submission.copy()
submission['Listening_Time_minutes'] = test_pred
submission.to_csv('submission_xgb_tuned.csv', index=False)



submission.head()




