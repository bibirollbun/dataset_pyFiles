import pandas as pd

# Baca dataset
df = pd.read_csv("/kaggle/input/allstate-claims-severity/train.csv")

# Ukuran data
print("Jumlah baris dan kolom:", df.shape)

# Info struktur data
print(df.info())


print(df.head())


missing = df.isnull().sum()
print(missing[missing > 0])


print("Duplikat:", df.duplicated().sum())


import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(df['loss'], kde=True)
plt.title("Distribusi 'loss'")
plt.show()


from sklearn.preprocessing import LabelEncoder

categorical_cols = [col for col in df.columns if 'cat' in col]
le = LabelEncoder()

for col in categorical_cols:
    df[col] = le.fit_transform(df[col])


df = pd.get_dummies(df, columns=categorical_cols)


from sklearn.preprocessing import StandardScaler

numeric_cols = [col for col in df.columns if 'cont' in col]
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.figure(figsize=(12,5))

# Distribusi asli
plt.subplot(1,2,1)
sns.histplot(df['loss'], bins=50, kde=True)
plt.title("Distribusi Original 'loss'")

# Distribusi setelah log transform
plt.subplot(1,2,2)
sns.histplot(np.log1p(df['loss']), bins=50, kde=True)
plt.title("Distribusi Log-Transformed 'loss'")

plt.tight_layout()
plt.show()


# Korelasi dengan target
corr = df[[f'cont{i}' for i in range(1,15)] + ['loss']].corr()

plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Korelasi antar fitur kontinyu dan 'loss'")
plt.show()


plt.figure(figsize=(12,6))
sns.boxplot(x='cont1', y='loss', data=df)
plt.title("Distribusi 'loss' per kategori 'cont1'")
plt.xticks(rotation=45)
plt.show()


from sklearn.model_selection import train_test_split

# Gunakan log(loss) sebagai target agar distribusi lebih stabil
df['log_loss'] = np.log1p(df['loss'])

# Buang kolom yang tidak digunakan
X = df.drop(columns=['id', 'loss', 'log_loss'])
y = df['log_loss'] 


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

# Model
model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)

# Latih model
model.fit(X_train, y_train)

# Prediksi
y_pred_log = model.predict(X_test)

# Balik transformasi log
y_test_orig = np.expm1(y_test)
y_pred_orig = np.expm1(y_pred_log)


rmse = mean_squared_error(y_test_orig, y_pred_orig, squared=False)
mae = mean_absolute_error(y_test_orig, y_pred_orig)
r2 = r2_score(y_test_orig, y_pred_orig)

print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R2 Score: {r2:.3f}")


import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
plt.scatter(y_test_orig, y_pred_orig, alpha=0.3)
plt.plot([y_test_orig.min(), y_test_orig.max()], [y_test_orig.min(), y_test_orig.max()], 'r--')
plt.xlabel("Actual Loss")
plt.ylabel("Predicted Loss")
plt.title("Prediksi vs Nilai Sebenarnya")
plt.grid(True)
plt.show()


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Target
df['log_loss'] = np.log1p(df['loss'])
X = df.drop(columns=['id', 'loss', 'log_loss'])
y = df['log_loss']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Fungsi evaluasi
def evaluate_model(model, name):
    model.fit(X_train, y_train)
    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_test_orig = np.expm1(y_test)
    
    rmse = mean_squared_error(y_test_orig, y_pred, squared=False)
    mae = mean_absolute_error(y_test_orig, y_pred)
    r2 = r2_score(y_test_orig, y_pred)
    
    print(f"{name}")
    print(f"  RMSE : {rmse:.2f}")
    print(f"  MAE  : {mae:.2f}")
    print(f"  R2   : {r2:.3f}")
    print("-" * 30)


from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100, random_state=42)
evaluate_model(rf, "Random Forest")


from catboost import CatBoostRegressor

cb = CatBoostRegressor(verbose=0, random_state=42)
evaluate_model(cb, "CatBoost Regressor")

