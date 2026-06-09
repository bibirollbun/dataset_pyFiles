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


# Reading Data
train = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')


# Checking for missing values
print(train.isna().sum())
print()
print(test.isna().sum())


# Statistical description 
# For train
train.describe()


# For test
test.describe()


# Importing necessary modules
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import lightgbm as lgb
from sklearn.metrics import mean_squared_error


# Feature Engineering
def fe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    df['Rhythm_Energy'] = df['RhythmScore'] * df['Energy']
    df['Rhythm_Loudness'] = df['RhythmScore'] * df['AudioLoudness']
    df['Duration_Minutes'] = df['TrackDurationMs'] / 60000  
    df['Duration_Energy_Ratio'] = df['TrackDurationMs'] / (df['Energy'] * 10000 + 1)  
    df['RhythmScore_Squared'] = df['RhythmScore'] ** 2
    df['Energy_Squared'] = df['Energy'] ** 2
    df['Log_Duration'] = np.log1p(df['TrackDurationMs']) 
    df['Acoustic_Instrumental_Ratio'] = df['AcousticQuality'] / (df['InstrumentalScore'] + 0.01) 
    df['Vocal_Energy'] = df['VocalContent'] * df['Energy']
    df['Live_Energy'] = df['LivePerformanceLikelihood'] * df['Energy']
    df['Mood_Rhythm'] = df['MoodScore'] * df['RhythmScore']
    df['Audio_Intensity'] = (df['Energy'] * np.abs(df['AudioLoudness'])) / 10  
    df['Performance_Character'] = (df['LivePerformanceLikelihood'] + df['MoodScore']) / 2
    df['Energy_Loudness_Ratio'] = df['Energy'] / (np.abs(df['AudioLoudness']) + 0.01)
    df['Rhythm_Duration_Density'] = df['RhythmScore'] / df['Duration_Minutes']

    return df

train_fe = fe(train)
test_fe = fe(test)


# Partitioning into feature matrix X and target vector y 
X = train_fe.drop(['id', 'BeatsPerMinute'], axis=1)
y = train_fe['BeatsPerMinute']

# Dropping id from test dataset
test_fe.drop(['id'], axis=1, inplace=True)


# Applying columntransformer to apply standard scaler to numerical values 
data_processor = ColumnTransformer(
    transformers=[
        ('numerical', StandardScaler(), X.select_dtypes(include=['float64', 'int64']).columns),
    ]
)
data_processor


# Splitting the dataset for training and validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# Fit data processor separately
processor_fitted = data_processor.fit(X_train)
# Transform train and val for LightGBM
X_train_proc = processor_fitted.transform(X_train)
X_val_proc = processor_fitted.transform(X_val)

model = lgb.LGBMRegressor(
    n_estimators=200,
    learning_rate=0.01,
    max_depth=15,
    num_leaves=31,
    random_state=42,
    n_jobs=-1,
    verbosity=-1,
    subsample=0.6, 
    reg_lambda=0.5, 
    reg_alpha=0.1,
    min_child_samples=50,  
    colsample_bytree=1.0
)
model


# Creating the pipeline to apply the transformations
pipeline = Pipeline(
    steps=[
        ('data_processor', data_processor),
        ('lgbm_classifier', model)
    ]
)
pipeline


%%time
# Fitting the pipeline which already contains the proecssing steps and model
pipeline.named_steps['lgbm_classifier'].fit(
    X_train_proc, y_train,
    eval_set=[(X_val_proc, y_val)],
    callbacks=[
        lgb.early_stopping(200),
        lgb.log_evaluation(50)
    ]
)


# Calculate the rmse
y_pred = pipeline.predict(X_val)
print(f"RMSE: {np.sqrt(mean_squared_error(y_val, y_pred)):.4f}")


# Submission
sub = pd.read_csv('/kaggle/input/playground-series-s5e9/sample_submission.csv')
test_preds = pipeline.predict(test_fe)
submission = pd.DataFrame({
    'id': sub['id'],
    'BeatsPerMinute': test_preds
})
submission.to_csv('submission.csv', index=False)

