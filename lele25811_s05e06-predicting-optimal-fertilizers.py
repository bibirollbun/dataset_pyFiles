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


train_df = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')


train_df


test_df


train_df.isnull().sum()


train_df['Fertilizer Name'].unique()


y = train_df['Fertilizer Name']
X = train_df[["id", "Temparature", "Humidity", "Moisture", "Soil Type", "Crop Type", "Nitrogen", "Potassium", "Phosphorous"]]


from sklearn.preprocessing import LabelEncoder

cols = ["Soil Type", "Crop Type"]

le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)

for col in cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    test_df[col] = le.transform(test_df[col])


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2)


# Definizione MAP@k
def mapk(y_true, y_prob, k=3):
    topk = np.argsort(-y_prob, axis=1)[:, :k]
    score = 0.0
    for i in range(len(y_true)):
        if y_true[i] in topk[i]:
            rank = np.where(topk[i] == y_true[i])[0][0]
            score += 1.0 / (rank + 1)
    return score / len(y_true)



from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

xgb_model = XGBClassifier(device='cuda',
                            tree_method='hist',
                            predictor='gpu_predictor',
                            use_label_encoder=False,
                            eval_metric='logloss',
                            verbosity=0)

param = {
    'n_estimators': [1500, 2000, 3000],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [4, 6]
}

grid_search = GridSearchCV(estimator=xgb_model,
                           param_grid=param,
                           cv=3,
                           scoring='accuracy',
                           verbose=1)

print("Avvio della Grid Search...")
grid_search.fit(X_train, y_train)
print("Grid Search completata.")


print("Migliori parametri:", grid_search.best_params_)


val_probs = grid_search.predict_proba(X_test)
map3 = mapk(y_test, val_probs, k=3)
print(f"MAP@3 on validation set: {map3:.4f}")


# Predici probabilità
probs = grid_search.predict_proba(test_df)

# Prendi i top-3 per ogni riga
top_3 = np.argsort(-probs, axis=1)[:, :3]

# Converti back da int (classi) a nomi fertilizzanti
top_3_labels = [le_target.inverse_transform(row) for row in top_3]

# Crea la stringa spazio-delimitata
predictions = [" ".join(row) for row in top_3_labels]

# Crea il dataframe di submission
submission = pd.DataFrame({
    "id": test_df["id"],
    "Fertilizer Name": predictions
})

# Salva il file CSV
submission.to_csv("submission.csv", index=False)
print("File 'submission.csv' salvato correttamente.")


submission.head()

