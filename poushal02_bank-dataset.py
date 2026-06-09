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


train_df=pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
train_df.head()


train_df.drop('id', axis=1, inplace=True)


train_df.shape


train_df.columns


train_df.isnull().sum()


train_df.head()


train_df.age.unique()


# binning age 
bins = [15, 18, 40, 60, 95]
labels = ['Too Young', 'Young Adult', 'midage','Senior Citizen']
train_df['age'] = pd.cut(train_df['age'], bins=bins, labels=labels)


train_df.head()


train_df.job.unique()


train_df['job'][train_df['job'] == 'unknown'].count()


train_df['job'].mode()[0]


mode_Job = train_df['job'].mode()[0]
train_df['job'] = train_df['job'].replace("unknown", mode_Job)
train_df.info()


train_df.marital.unique()


train_df.education.unique()


train_df['education'].mode()[0]


train_df['education'] = train_df['education'].replace('unknown', 'secondary')
train_df.info()


train_df.default.unique()


train_df['balance'][train_df['balance'] < 0].unique()


train_df.columns


train_df['housing'].unique()


train_df['loan'].unique()


train_df.contact.unique()


train_df['day'].unique()


train_df['month'].unique()


train_df['duration'].unique()


train_df['campaign'].unique()


train_df['pdays'].unique()


train_df['previous'].unique()


train_df['poutcome'].unique()


train_df['y'].unique()


def seperate_cols(df: pd.DataFrame) -> tuple:
    NUMS = []
    CATS = []
    ORDS = []
    for each in df.columns:
        if df[each].dtype == 'object':
            CATS.append(each)
        elif df[each].dtype == 'category':
            ORDS.append(each)
        else:
            NUMS.append(each)
    return NUMS, CATS, ORDS


nums, cats, ords = seperate_cols(train_df)
print(f'Numerical columns {len(nums)}', end=',')
print(f" Categorical columns {len(cats)}", end=',')
print(f" Ordinal Columns {len(ords)}")


nums


cats


ords


train_df.info()


nums.pop()


nums


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

preprocessor = ColumnTransformer(
    transformers=[
        ('ord', OrdinalEncoder(), ords), # for age
        ('onehot', OneHotEncoder(handle_unknown='ignore'), cats), # for categorical columns, ohe helps in deep learning
        ('num', StandardScaler(), nums) # scaling for numerical columns
    ],
    remainder='passthrough'
)
X_processed = preprocessor.fit_transform(train_df)


ohe_feature_names = preprocessor.named_transformers_['onehot'].get_feature_names_out(cats)
# Combine ordinal and onehot feature names
all_feature_names = (
    list(ords) + 
    list(ohe_feature_names) + 
    list(nums) +
    [col for col in train_df.columns if col not in cats + ords + nums]
)

new_df = pd.DataFrame(X_processed, columns=all_feature_names)


new_df.sample()


new_df.columns


new_df.sample()


import warnings
warnings.filterwarnings('ignore')


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

# Define the model
model = Sequential([
    Dense(256, activation='relu', input_shape=(49,)),
    BatchNormalization(),
    Dropout(0.4),

    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),

    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),

    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'],
             )

# Early stopping callback (optional)
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

# Summary
model.summary()


X = new_df.drop('y', axis=1)
y = new_df['y']


test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
ids = test['id']
test.drop('id', axis=1, inplace=True)
test.sample()


# binning age 
bins = [15, 18, 40, 60, 95]
labels = ['Too Young', 'Young Adult', 'midage','Senior Citizen']
test['age'] = pd.cut(test['age'], bins=bins, labels=labels)


test['y'] = np.zeros(250000, dtype='int')


mode_Job_test = test['job'].mode()[0]
test['job'] = test['job'].replace("unknown", mode_Job_test)
test['education'] = test['education'].replace('unknown', 'secondary')


X_test_processed = preprocessor.fit_transform(test)

# For OneHotEncoder, get feature names
ohe_feature_names = preprocessor.named_transformers_['onehot'].get_feature_names_out(cats)
# Combine ordinal and onehot feature names
all_feature_names = (
    list(ords) + 
    list(ohe_feature_names) + 
    list(nums) +  # Scaled numeric columns keep their original names
    [col for col in train_df.columns if col not in cats + ords + nums]
)
# Create DataFrame from the transformed array
new_df_test = pd.DataFrame(X_test_processed, columns=all_feature_names)
new_df_test


y_test=new_df_test['y']
X_test=new_df_test.drop('y', axis=1)


history = model.fit(X, y,
                    epochs=100,
                    batch_size=32,
                    validation_data=(X_test, y_test),
                    callbacks=[early_stop])


y_pred = model.predict(X_test)


y_pred


y_pred = np.array([each[0] for each in y_pred])


y_pred_class = (y_pred >= 0.5).astype(int)


from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred_class)
print("Accuracy:", accuracy)


submission = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")
submission.sample()


submission


submission_df = pd.DataFrame({
    'id': ids,
    'y': y_pred
})


submission_df.sample()


submission_df.to_csv('submission.csv', index=False)


subs_by_me = pd.read_csv('/kaggle/working/submission.csv')
subs_by_me.sample()




