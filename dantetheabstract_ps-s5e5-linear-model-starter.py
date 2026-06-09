import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()

import warnings; warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv', index_col='id')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv', index_col='id')
sub = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')

train.info(); train.head()


target = 'Calories'
features = ['Sex', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']

train['Sex'] = train['Sex'].map({'male': 1, 'female': 0})
test['Sex'] = test['Sex'].map({'male': 1, 'female': 0})


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))
sns.histplot(train[target], kde=True, ax=ax1)
ax1.set_title('Original Histogram')

sns.histplot(np.log1p(train[target]), kde=True, ax=ax2)
ax2.set_title('Log-Transformed Histogram')

plt.tight_layout()
plt.show()


train.skew()


train.corr()[[target]].T


def feature_engineering(df):
    df['bmi'] = df['Weight'] / ((df['Height'] / 100) ** 2)    
    df['exercise_intensity'] = df['Heart_Rate'] / df['Duration']
    df['heart_rate_duration'] = df['Heart_Rate'] * df['Duration']
    df['temp_duration'] = df['Body_Temp'] * df['Duration']
    df['hr_to_temp'] = df['Heart_Rate'] / df['Body_Temp']
    df['hr_to_age'] = df['Heart_Rate'] / df['Age']
    df['age_bmi'] = df['Age'] * df['bmi']
    df['max_heart_rate'] = 220 - df['Age']
    df['heart_rate_intensity'] = df['Heart_Rate'] / df['max_heart_rate']

    # add log-transform of skewed features
    skewed_feats = ['Age', 'Weight', 'Body_Temp', 'Height', 'Duration', 'Heart_Rate']
    for feat in skewed_feats:
        df[f'log_{feat}'] = np.log1p(df[feat])
    return df


from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error

from sklearn.linear_model import LinearRegression
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline


X = train[features]
y = train[target]
test = test[features]

X = feature_engineering(X)
test = feature_engineering(test)
print('number of features:', X.shape[1])


def training(X, y, test, model, cv):
    scores = []
    predictions = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        print('Fold', fold + 1, '...', end=' ')
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        pred_test = model.predict(X_test).clip(0)
        test_rmsle = np.sqrt(mean_squared_log_error(y_test, pred_test))
        scores.append(test_rmsle)
        print('test rmsle:', test_rmsle)

        y_pred = model.predict(test).clip(0)
        predictions.append(y_pred)

    predictions = np.mean(predictions, axis=0)
    rmsle = np.mean(scores)
    print('Average validated rmsle:', rmsle)
    return predictions, rmsle, model


SEED = 208
FOLDS = 5
cv = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

regressor = TransformedTargetRegressor(
    regressor=LinearRegression(),
    func=np.log1p,
    inverse_func=np.expm1
)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(include_bias=False)),
    ('regressor', regressor)
])

predictions, test_rmsle, model = training(X, y, test, pipeline, cv)


sub[target] = predictions
sub.to_csv('submission.csv', index=False)
print('Submission file saved.')

