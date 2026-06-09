import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)

from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler


import os



for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))




df1 = pd.read_csv("/kaggle/input/playground-series-s4e12/train.csv")
df1.head()


df2 = pd.read_csv("/kaggle/input/playground-series-s4e12/test.csv")
df2.head()


df1.shape, df2.shape


df = pd.concat([df1,df2])


df.sample(4)


df.info()


df['Policy Start Date'] = pd.to_datetime(df['Policy Start Date'])

df['Policy_Year'] = df['Policy Start Date'].dt.year

df['Policy_Month'] = df['Policy Start Date'].dt.month

df['Policy_Day'] = df['Policy Start Date'].dt.day

df['Policy_Weekday'] = df['Policy Start Date'].dt.weekday

df['Policy_Is_Weekend'] = df['Policy_Weekday'].isin([5, 6]).astype(int)

def get_season(month):
    if month in [12, 1, 2]:
        return "Winter"  
    elif month in [3, 4, 5]:
        return "Spring"  
    elif month in [6, 7, 8]:
        return "Summer"  
    else:
        return "Autumn"

df['Policy_Season'] = df['Policy_Month'].apply(get_season)

df['Policy_Age_Days'] = (pd.to_datetime("today") - df['Policy Start Date']).dt.days


df.drop("Policy Start Date", axis=1, inplace=True)


df.describe().round(3).T


categorical_columns = df.select_dtypes(include=['object']).columns
numerical_columns = df.select_dtypes(exclude=['object']).columns

print("\nCategorical Columns:", categorical_columns.tolist())
print("\nNumerical Columns:", numerical_columns.tolist())


for column in categorical_columns:
    num_unique = df[column].nunique()
    print(f"'{column}' has {num_unique} unique categories.")


df.sample(5)


df.isnull().sum()


missing_cols = df.columns[df.isnull().sum() > 0]
missing_cols = missing_cols.drop('Premium Amount')

num_cols = df[missing_cols].select_dtypes(include='number').columns
cat_cols = df[missing_cols].select_dtypes(include='object').columns

df[num_cols] = SimpleImputer(strategy='median').fit_transform(df[num_cols])
df[cat_cols] = SimpleImputer(strategy='constant', fill_value='Unknown').fit_transform(df[cat_cols])


education_order = {
    "High School": 0,
    "Bachelor's": 1,
    "Master's": 2,
    "PhD": 3
}
df["Education Level"] = df["Education Level"].map(education_order)

exercise_order = {
    "Rarely": 0,
    "Monthly": 1,
    "Weekly": 2,
    "Daily": 3
}
df["Exercise Frequency"] = df["Exercise Frequency"].map(exercise_order)

policy_type_order = {
    "Basic": 0,
    "Comprehensive": 1,
    "Premium": 2
}
df["Policy Type"] = df["Policy Type"].map(policy_type_order)


df = pd.get_dummies(df, drop_first=True)


train=df[:1200000]
test=df[1200000:]


x = train.drop(["id","Premium Amount"], axis=1)
y = np.log1p(train["Premium Amount"])
test = test.drop(["id","Premium Amount"], axis=1)


scaler = StandardScaler()
x = scaler.fit_transform(x)
test = scaler.transform(test)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state=8)


model = Sequential([
    Dense(512, activation='relu', input_shape=(x_train.shape[1],)),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1)
])


model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mean_absolute_error',
    metrics=['mean_absolute_error']
)



early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    x_train, y_train,
    validation_split=0.1,
    epochs=20,
    batch_size=512,  
    callbacks=[early_stop],
    verbose=1
)


early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
model.fit(x, y, epochs=20, batch_size=512, callbacks=[early_stop], verbose=1)


prediction_log = model.predict(test)
prediction = np.expm1(prediction_log)


prediction


submission = pd.DataFrame({
    "id": df2["id"].values,
    "Premium Amount": prediction.flatten()
})


submission.head()


submission.to_csv("submission.csv", index=False)




