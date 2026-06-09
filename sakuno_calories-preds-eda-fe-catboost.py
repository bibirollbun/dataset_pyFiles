import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from catboost import CatBoostRegressor

from sklearn.metrics import mean_squared_error, mean_squared_log_error, mean_absolute_error

import warnings
warnings.filterwarnings('ignore')


trainpath = '/kaggle/input/playground-series-s5e5/train.csv'
testpath = '/kaggle/input/playground-series-s5e5/test.csv'


df_train = pd.read_csv(trainpath)
print(df_train.shape)


df_test = pd.read_csv(testpath)
print(df_test.shape)


df_train.info()


df_train.head()


df_test.head()


df_train.isnull().sum()


df_test.isnull().sum()


print(f"There are {df_train.duplicated().sum()} duplicates in train set.")


print(f"There are {df_test.duplicated().sum()} duplicates in test set.")


df_train['BMI'] = df_train['Weight'] / ((df_train['Height'] / 100) ** 2)


df_test['BMI'] = df_test['Weight'] / ((df_test['Height'] / 100) ** 2)


sns.histplot(data=df_train, x=df_train.Calories)


# Define numerical features
numerical_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'BMI']


for column in numerical_features:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    sns.histplot(data=df_train, x=column, ax=axes[0])
    mean_value = df_train[column].mean()
    median_value = df_train[column].median()
    axes[0].axvline(mean_value, color='orange', linestyle='--', linewidth=2, label=f'Mean: {mean_value:.2f}')
    axes[0].axvline(median_value, color='yellow', linestyle='-.', linewidth=2, label=f'Median: {median_value:.2f}')
    axes[0].set_title(f'Histogram of {column}')
    axes[0].legend()

    sns.boxplot(data=df_train, x=column, ax=axes[1])
    axes[1].set_title(f'Boxplot of {column}')

    plt.tight_layout()
plt.show()


# Define categorical features
categorical_features = ['Sex']


for column in categorical_features:
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df_train, x=column)
    plt.title(f'Distribution of {column}')
plt.show()


sns.boxplot(df_train.iloc[:,1:])
plt.yscale("log")
plt.xticks(rotation=45)
plt.show()


df_train = df_train.drop("id", axis=1)


plt.figure(figsize=(10, 6))
sns.heatmap(df_train.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Matrix")
plt.show()


X = df_train.drop(columns=['Calories'])
y = df_train['Calories']


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)


class CustomFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, resting_hr=60):
        self.resting_hr = resting_hr

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df['Height_m'] = df['Height'] / 100
        df['Max_HR'] = 220 - df['Age']
        df['HR_reserve_frac'] = (df['Heart_Rate'] - self.resting_hr) / (df['Max_HR'] - self.resting_hr)

        # Add polynomial features
        poly_cols = ['Duration', 'Heart_Rate', 'Body_Temp', 'BMI']
        poly = PolynomialFeatures(degree=2, include_bias=False)
        poly_array = poly.fit_transform(df[poly_cols])
        poly_feature_names = poly.get_feature_names_out(poly_cols)
        df_poly = pd.DataFrame(poly_array, columns=poly_feature_names, index=df.index)

        df = pd.concat([df, df_poly.loc[:, ~df_poly.columns.isin(df.columns)]], axis=1)
        return df


categorical_cols = ['Sex']

sample = CustomFeatureEngineer().fit_transform(X_train)
numeric_cols = sample.columns.difference(['Sex']).tolist()


preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols),
    ('num', StandardScaler(), numeric_cols)
])

pipeline1 = Pipeline([
    ('feature_eng', CustomFeatureEngineer()),
    ('preprocess', preprocessor),
    ('model', CatBoostRegressor(verbose=0))
])


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='mean_squared_error',
              metrics=['mean_squared_error'])


pipeline2 = Pipeline([
    ('feature_eng', CustomFeatureEngineer()),
    ('preprocess', preprocessor)
])

X_train_processed = pipeline2.fit_transform(X_train)
X_val_processed = pipeline2.transform(X_val)


pipeline1.fit(X_train, y_train)
y_pred_catboot = pipeline1.predict(X_val)


history = model.fit(X_train_processed, y_train,
                    validation_data=(X_val_processed, y_val),
                    epochs=100,
                    batch_size=32,
                    verbose=1)


y_pred_clipped = np.maximum(y_pred, 0)

rmsle = np.sqrt(mean_squared_log_error(y_val, y_pred_clipped))
print("✅ RMSLE:", rmsle)


test_ids = df_test["id"]
X_test = df_test.drop(columns=["id"])

test_preds = pipeline.predict(X_test)


test_preds_clipped = np.maximum(test_preds, 0)


# Create the submission DataFrame
submission = pd.DataFrame({
    'id': test_ids,
    'Calories': test_preds_clipped
})

submission.to_csv("submission.csv", index=False)

