import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=FutureWarning)


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV


# Fine tune the hypyerparmater
param_dist = {
    'n_estimators': [10000],
    'max_depth': [8],
    'learning_rate': [0.001],
    'subsample': [0.6],
    'colsample_bytree': [1],
    'gamma': [1],
    'reg_alpha': [0.5],
    'reg_lambda': [2]
}


train_dataset = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test_dataset = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')


train_dataset.head()


train_dataset.info()


# show statitical derscription for numerical features
train_dataset.describe()


train_dataset.nunique()


fig, ax = plt.subplots(1, 3, figsize=(18, 5))
sns.countplot(data=train_dataset, x='Soil Type', ax=ax[0])
sns.countplot(data=train_dataset, x='Crop Type', ax=ax[1])
sns.countplot(data=train_dataset, x='Fertilizer Name', ax=ax[2])
plt.tight_layout()



num_cols = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
train_dataset[num_cols].hist(bins=20, figsize=(12, 8))
plt.tight_layout()


plt.figure(figsize=(15, 6))
sns.boxplot(data=train_dataset[num_cols], orient='h')



num_cols = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']

# Create a 2x3 grid of boxplots
fig, axes = plt.subplots(2, 3, figsize=(10, 8))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    sns.boxplot(
        x='Fertilizer Name', 
        y=col, 
        data=train_dataset, 
        ax=axes[i],
        showmeans=True,  # Show mean markers
        meanprops={"marker":"x", "markerfacecolor":"white", "markeredgecolor":"black"}
    )
    axes[i].set_title(f'{col} by Fertilizer', fontsize=12)
    axes[i].tick_params(axis='x', rotation=45)  # Rotate labels

plt.tight_layout()
plt.show()



train_dataset = train_dataset.drop('id', axis=1)
y = train_dataset['Fertilizer Name']
X = train_dataset.drop(['Fertilizer Name'], axis=1)


ids = test_dataset["id"]


X_test = test_dataset.drop('id' , axis = 1)


numerical_cols = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
# Step 2: Initialize the scaler
scaler = StandardScaler()
# Step 3: Fit on training data and transform both train and test
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])


X.head()


X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])


X_test.head()


X_encoded = pd.get_dummies(X, columns=['Soil Type', 'Crop Type'])
X_encoded = X_encoded.astype(int)
X_encoded = X_encoded.to_numpy()


label_encoder = LabelEncoder()
Y_encoded = label_encoder.fit_transform(y)


X_test_encoded =  pd.get_dummies(X_test, columns=['Soil Type', 'Crop Type'])
X_test_encoded = X_test_encoded.astype(int)
X_test_encoded = X_test_encoded.to_numpy()


X_encoded.shape


X_test_encoded.shape


Y_encoded.shape


xgb_model = XGBClassifier(
   objective='multi:softprob',
    num_class=7,
    eval_metric='mlogloss',
    verbosity=1,
    tree_method='gpu_hist',      # Required for GPU
    predictor='gpu_predictor',   # Optional but recommended
    gpu_id=0 
)


import numpy as np
from sklearn.metrics import make_scorer

def map3_score(y_true, y_pred_proba):
    try:
        top_3 = np.argsort(y_pred_proba, axis=1)[:, -3:][:, ::-1]  # top 3 predictions per sample
        score = 0.0
        for i, true_label in enumerate(y_true):
            if true_label in top_3[i]:
                rank = np.where(top_3[i] == true_label)[0][0] + 1
                score += 1.0 / rank
        return score / len(y_true)
    except Exception as e:
        print("MAP@3 scoring failed:", e)
        return np.nan  # graceful fallback

map3_scorer = make_scorer(map3_score, needs_proba=True)




from sklearn.metrics import top_k_accuracy_score, make_scorer
top3_scorer = make_scorer(top_k_accuracy_score, needs_proba=True, k=5)



import joblib
import traceback

try:
    random_search = RandomizedSearchCV(
        xgb_model,
        param_distributions=param_dist,
        n_iter=1,
        scoring=top3_scorer,  # ← Use the wrapped function here
        cv=3,
        verbose=3,
        n_jobs=-1,
        random_state=42
    )
    random_search.fit(X_encoded, Y_encoded , eval_set=[(X_encoded, Y_encoded)], verbose=True)

except KeyboardInterrupt:
    print("Training interrupted by user.")
    joblib.dump(random_search, 'random_search_partial.pkl')
except Exception as e:
    print("An error occurred:", e)
    traceback.print_exc()
    joblib.dump(random_search, 'random_search_partial.pkl')




random_search.


import joblib

# Save
joblib.dump(random_search, 'random_search_model.pkl')
best_model = random_search.best_estimator_
joblib.dump(best_model, "xgb_best_model.pkl")

# Load later
loaded_search = joblib.load('random_search_model.pkl')



best_model_xgb = loaded_search.best_estimator_
y_xgb_pred = best_model_xgb.predict_proba(X_test_encoded)



y_xgb_pred


top3_indices = np.argsort(y_xgb_pred, axis=1)[:, -3:][:, ::-1]  # descending order
top3_labels = label_encoder.inverse_transform(top3_indices.ravel())
top3_labels = top3_labels.reshape(top3_indices.shape) 
pred_strings = [' '.join(row) for row in top3_labels]
submission_df = pd.DataFrame({
    "id": ids,
    "Fertilizer Name": pred_strings
})
submission_df.to_csv("sample_submission.csv", index=False)




