import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


train=pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')


train.head(3)


train.info()


train.columns


train.describe().T


train.isna().sum()


test.head(3)


test.info()


test.describe().T


test.isna().sum()


columns = ['RhythmScore', 'AudioLoudness', 'VocalContent', 'AcousticQuality',
           'InstrumentalScore', 'LivePerformanceLikelihood', 'MoodScore',
           'TrackDurationMs', 'Energy', 'BeatsPerMinute']
plt.figure(figsize=(20, 15))
for i, col in enumerate(columns):
    plt.subplot(4, 3, i+1)
    sns.histplot(train[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()


plt.figure(figsize=(20, 15))
for i, col in enumerate(columns):
    plt.subplot(4, 3, i+1)
    sns.boxplot(y=train[col])
    plt.title(f'Box Plot of {col}')
plt.tight_layout()
plt.show()


print("\nCorrelation Matrix:\n")
corr_matrix = train.corr()
print(corr_matrix)


X=train.drop(columns=['id','BeatsPerMinute'])
y=train['BeatsPerMinute']
test_id=test['id']
test=test.drop(columns='id',axis=1)


from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


model=CatBoostRegressor(random_state=42)


model.fit(X_train,y_train)


# Training RMSE
train_preds = model.predict(X_train)
train_rmse = mean_squared_error(y_train, train_preds, squared=False)

val_preds = model.predict(X_test)
val_rmse = mean_squared_error(y_test, val_preds, squared=False)

print(f"CatBoost train RMSE: {train_rmse:.4f}")
print(f"CatBoost validation RMSE: {val_rmse:.4}")


test_predictions = model.predict(test)


submission = pd.DataFrame({
    "id": test_id,
    "BeatsPerMinute": test_predictions
})
submission.to_csv("submission.csv", index=False)
print("Saved submission.csv")




