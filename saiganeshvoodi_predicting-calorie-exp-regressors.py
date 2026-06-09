import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
import warnings
warnings.filterwarnings("ignore")


def rmsle(y_true, y_pred):
    y_pred = np.maximum(0, y_pred)
    return np.sqrt(mean_squared_log_error(y_true, y_pred))


train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')


le = LabelEncoder()
train['Sex'] = le.fit_transform(train['Sex'])
test['Sex'] = le.transform(test['Sex'])


plt.figure(figsize=(10, 4))
sns.histplot(train['Calories'], kde=True, bins=50)
plt.title("Calories Distribution")
plt.show()


X = train.drop(columns=['id', 'Calories'])
y = train['Calories']
X_test = test.drop(columns=['id'])


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)


hgb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1, random_state=42)
hgb.fit(X_train, y_train)
hgb_preds = hgb.predict(X_val)
hgb_score = rmsle(y_val, hgb_preds)



etr = ExtraTreesRegressor(n_estimators=200, random_state=42)
etr.fit(X_train, y_train)
etr_preds = etr.predict(X_val)
etr_score = rmsle(y_val, etr_preds)


mlp = MLPRegressor(hidden_layer_sizes=(128, 64), activation='relu',
                   alpha=0.001, learning_rate='adaptive', max_iter=1000, random_state=42)
mlp.fit(X_train_scaled, y_train)
mlp_preds = mlp.predict(X_val_scaled)
mlp_score = rmsle(y_val, mlp_preds)


scores = {
    'HistGradientBoosting': hgb_score,
    'ExtraTrees': etr_score,
    'MLPRegressor': mlp_score
}
for name, score in scores.items():
    print(f"{name} RMSLE: {score:.4f}")


plt.figure(figsize=(8, 5))
sns.barplot(x=list(scores.keys()), y=list(scores.values()))
plt.title("Model Comparison (RMSLE)")
plt.ylabel("RMSLE (lower is better)")
plt.show()



best_model = min(scores, key=scores.get)
print(f"Best model: {best_model}")



if best_model == "HistGradientBoosting":
    final_model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1, random_state=42)
    final_model.fit(X, y)
    preds = final_model.predict(X_test)
elif best_model == "ExtraTrees":
    final_model = ExtraTreesRegressor(n_estimators=200, random_state=42)
    final_model.fit(X, y)
    preds = final_model.predict(X_test)
else:  # MLP
    final_model = MLPRegressor(hidden_layer_sizes=(128, 64), activation='relu',
                               alpha=0.001, learning_rate='adaptive', max_iter=1000, random_state=42)
    final_model.fit(X_scaled, y)
    preds = final_model.predict(X_test_scaled)


final_preds = np.maximum(0, preds)


submission = pd.DataFrame({
    'id': test['id'],
    'Calories': final_preds
})
submission.to_csv('submission.csv', index=False)
print("submission.csv saved")




