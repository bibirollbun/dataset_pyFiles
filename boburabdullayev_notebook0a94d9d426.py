import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
print("Kutubxonalar yuklandi!")


train_df = pd.read_csv('/kaggle/input/yovvoyi-ko-k-maymunjon-hosildorligini-aniqlash/train.csv')
test_df = pd.read_csv('/kaggle/input/yovvoyi-ko-k-maymunjon-hosildorligini-aniqlash/test.csv')
sample_submission = pd.read_csv('/kaggle/input/yovvoyi-ko-k-maymunjon-hosildorligini-aniqlash/sample_submission.csv')

print("Train data shape:", train_df.shape)
print("Test data shape:", test_df.shape)
print(train_df.head())


# Xususiyatlar va nishonni ajratish
X = train_df.drop(['id', 'yield'], axis=1)
y = train_df['yield']
X_test = test_df.drop('id', axis=1)

# Yo‘q qiymatlarni to‘ldirish
imputer = SimpleImputer(strategy='median')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# Normalizatsiya
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

print("Ma'lumotlar tayyor!")


# Train va validation qismlarga ajratish
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelni yaratish va o‘qitish
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Validationda baholash
y_val_pred = model.predict(X_val)
mae = mean_absolute_error(y_val, y_val_pred)
print(f"Validation MAE: {mae}")

# Test uchun bashorat
y_test_pred = model.predict(X_test)


submission = pd.DataFrame({
    'id': test_df['id'],
    'yield': y_test_pred
})
submission.to_csv('submission.csv', index=False)
print("Submission fayli 'submission.csv' sifatida saqlandi!")
print(submission.head())







