import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_squared_log_error
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os


os.listdir("/kaggle/input")


df_train = pd.read_csv("../input/competition-data/data/Predict Calorie Expenditure/train.csv")
df_train


df_test = pd.read_csv("../input/competition-data/data/Predict Calorie Expenditure/test.csv")
df_test


display (df_train.describe())


cols = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Calories']

plt.figure(figsize=(16, 12))
for i, col in enumerate(cols, 1):
    plt.subplot(3, 3, i)
    sns.histplot(data=df_train, x=col, hue='Sex', kde=True, bins=30, palette='Set2')
    plt.title(f'Distribution of {col}')
    plt.tight_layout()

plt.show()





def add_features(df):
    df = df.copy()
    df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)
    df['Duration_per_kg'] = df['Duration'] / df['Weight']
    df['Heart_Rate_per_min'] = df['Heart_Rate'] / df['Duration']
    df['Temp_Deviation'] = df['Body_Temp'] - 37
    df['Intensity_Index'] = (df['Heart_Rate'] * df['Body_Temp']) / df['Duration']
    df['Age_BMI'] = df['Age'] * df['BMI']
    df['Sex_encoded'] = LabelEncoder().fit_transform(df['Sex'])
    return df



# add feartures
df_train_feat = add_features(df_train)
df_test_feat = add_features(df_test)

features = [
    'Sex_encoded', 'Age', 'Height', 'Weight', 'Duration',
    'Heart_Rate', 'Body_Temp',
    'BMI', 'Duration_per_kg', 'Heart_Rate_per_min',
    'Temp_Deviation', 'Intensity_Index', 'Age_BMI'
]

X = df_train_feat[features]
y = df_train_feat['Calories']
X_test = df_test_feat[features]



X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestRegressor(
    n_estimators=400,
    max_depth=30,
    max_features='sqrt',
    min_samples_leaf=5,
    min_samples_split=10,
    n_jobs=-1,
    random_state=42
)


model.fit(X_train, y_train)

y_val_pred = model.predict(X_val)
rmsle_val = np.sqrt(mean_squared_log_error(y_val, y_val_pred))
print(f'Validation RMSLE: {rmsle_val:.4f}')


y_test_pred = model.predict(X_test)


submission = pd.DataFrame({
    'id': df_test_feat['id'], 
    'Calories': y_test_pred.clip(0)
})

submission.to_csv('submission.csv', index=False)




