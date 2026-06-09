import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train_df = pd.read_csv('/kaggle/input/playground-series-s4e4/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s4e4/test.csv')


train_df.shape


train_df.head()


print(train_df.isnull().sum())


train_df.info()


plt.figure(figsize=(10, 6))
sns.histplot(train_df['Rings'], kde=True)
plt.title('Distribution of Rings')
plt.show()


# Explore relationship between Sex and Rings
plt.figure(figsize=(10, 6))
sns.boxplot(x='Sex', y='Rings', data=train_df)
plt.title('Rings by Sex')
plt.show()

sex_rings_stats = train_df.groupby('Sex')['Rings'].agg(['mean', 'median', 'std', 'count'])
print(sex_rings_stats)


## Encoding
le = LabelEncoder()
train_df["Sex"] = le.fit_transform(train_df["Sex"])

test_df["Sex"] = test_df["Sex"].apply(lambda x: x if x in le.classes_ else le.classes_[0])
test_df["Sex"] = le.transform(test_df["Sex"])


features = [col for col in train_df.columns if col != "Rings"]
X = train_df[features]
y = train_df["Rings"]


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


y_pred_val = model.predict(X_val)

val_mse = mean_squared_error(y_val, y_pred_val)
val_mae = mean_absolute_error(y_val, y_pred_val)
val_r2 = r2_score(y_val, y_pred_val)

print(f"\nValidation MSE: {val_mse:.2f}")
print(f"Validation MAE: {val_mae:.2f}")
print(f"Validation R-squared: {val_r2:.2f}")



X_test = test_df[features]
y_pred_test = model.predict(X_test)

y_pred_test = np.round(y_pred_test).astype(int)


results_df = pd.DataFrame({'id': test_df['id'], 'Rings': y_pred_test})
results_df.head()




