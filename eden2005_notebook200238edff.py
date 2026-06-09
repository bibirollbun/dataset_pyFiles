# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# ğŸ“¦ ×™×™×‘×•×� ×¡×¤×¨×™×•×ª
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error

# ğŸ�¨ ×¢×™×¦×•×‘ ×’×¨×¤×™×�
sns.set_theme(style="whitegrid")

# ğŸ“¥ ×˜×¢×™× ×ª × ×ª×•× ×™×� (×œ-Kaggle)
train = pd.read_csv("/kaggle/input/bike-sharing-demand/train.csv")

# ğŸ§  ×”× ×“×¡×ª ×ª×›×•× ×•×ª
train['datetime'] = pd.to_datetime(train['datetime'])
train['hour'] = train['datetime'].dt.hour
train['day'] = train['datetime'].dt.day
train['weekday'] = train['datetime'].dt.weekday
train['month'] = train['datetime'].dt.month
train['year'] = train['datetime'].dt.year

# ğŸ�¯ ×”×’×“×¨×ª ×ª×›×•× ×•×ª ×œ×�×•×“×œ
features = ['temp', 'humidity', 'windspeed', 'hour', 'weekday', 'month']
X = train[features]
y = train['count']

# ğŸ§ª ×¤×™×¦×•×œ ×œ-Train/Test
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ğŸš€ ×�×™×�×•×Ÿ ×�×•×“×œ
model = RandomForestRegressor(n_estimators=20, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# ğŸ§® ×—×™×–×•×™ ×•×—×™×©×•×‘ RMSLE
val_preds = model.predict(X_val)
val_preds = np.maximum(0, val_preds)  # ×‘×™×˜×•×œ ×¢×¨×›×™×� ×©×œ×™×œ×™×™×�
rmsle = np.sqrt(mean_squared_log_error(y_val, val_preds))

# âœ… ×”×“×¤×¡×ª ×”×ª×•×¦×�×”
print("ğŸ“Š RMSLE ×¢×œ ×¡×˜ ×�×™×�×•×ª:", round(rmsle, 4))

# ğŸ“Š ×’×¨×£ ×�×�×•×¦×¢ ×”×©×›×¨×•×ª ×œ×¤×™ ×©×¢×”
hourly = train.groupby('hour')['count'].mean().reset_index()
plt.figure(figsize=(12, 6))

# ×’×¨×£ ×¤×©×•×˜ ×‘×œ×™ hue
barplot = sns.barplot(data=hourly, x='hour', y='count', palette="viridis")

# ×¢×¨×›×™×� ×¢×œ ×›×œ ×¢×�×•×“×”
for index, row in hourly.iterrows():
    barplot.text(row.name, row['count'] + 5, f"{int(row['count'])}", color='black', ha="center", fontsize=8)

# ×›×•×ª×¨×•×ª ×�×¡×•×“×¨×•×ª
plt.title("×�×�×•×¦×¢ ×”×©×›×¨×•×ª ×œ×¤×™ ×©×¢×”"[::-1], fontsize=16)
plt.xlabel("×©×¢×” ×‘×™×•×�"[::-1], fontsize=12)
plt.ylabel("×›×�×•×ª ×�×�×•×¦×¢×ª ×©×œ ×”×©×›×¨×•×ª"[::-1], fontsize=12)
plt.xticks(range(0, 24))
plt.tight_layout()
plt.show()
# ×™×¦×™×¨×ª ×§×•×‘×¥ ×”×’×©×” ×œ×¤×™ ×”×¤×•×¨×�×˜ ×©×œ Kaggle
test = pd.read_csv("/kaggle/input/bike-sharing-demand/test.csv")

# × ×‘×¦×¢ ×”× ×“×¡×ª ×ª×›×•× ×•×ª ×›×�×• ×‘Ö¾train
test['datetime'] = pd.to_datetime(test['datetime'])
test['hour'] = test['datetime'].dt.hour
test['day'] = test['datetime'].dt.day
test['weekday'] = test['datetime'].dt.weekday
test['month'] = test['datetime'].dt.month
test['year'] = test['datetime'].dt.year

# ×™×¦×™×¨×ª ×ª×—×–×™×ª
X_test = test[features]
preds = model.predict(X_test)
preds = np.maximum(0, preds)  # ×‘×™×˜×•×œ ×¢×¨×›×™×� ×©×œ×™×œ×™×™×�

# ×™×¦×™×¨×ª ×§×•×‘×¥ submission
submission = pd.DataFrame({
    "datetime": test["datetime"],
    "count": preds
})

# ×©×�×™×¨×ª ×”×§×•×‘×¥ ×œ×ª×™×§×™×™×ª Kaggle working
submission.to_csv("/kaggle/working/submission.csv", index=False)


