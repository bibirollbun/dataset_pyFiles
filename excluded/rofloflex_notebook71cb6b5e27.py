import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Загрузка обучающих данных
train = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv')
X = train[['acc_x', 'acc_y', 'acc_z', 'sequence_counter']]
y = train['behavior']

# Кодирование целевой переменной
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Обучение модели
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X, y_encoded)

# ОБЯЗАТЕЛЬНАЯ функция
def predict(test_df):
    X_test = test_df[['acc_x', 'acc_y', 'acc_z', 'sequence_counter']]
    preds = model.predict(X_test)
    labels = le.inverse_transform(preds)
    return pd.Series(labels)

