!pip install pytorch_tabnet


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from pytorch_tabnet.tab_model import TabNetClassifier
import torch


train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')


print("Train columns:", train.columns.tolist())
print("Test columns:", test.columns.tolist())


print(train.info())
print(train.describe())
print(train.head())


target_col = 'Fertilizer Name'

target_encoder = LabelEncoder()
train[target_col] = target_encoder.fit_transform(train[target_col])


def add_features(df):
    df = df.copy()
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    soil_encoder = LabelEncoder()
    crop_encoder = LabelEncoder()
    df['soil_enc'] = soil_encoder.fit_transform(df['Soil Type'])
    df['crop_enc'] = crop_encoder.fit_transform(df['Crop Type'])
    df['soil_crop'] = df['soil_enc'] * 100 + df['crop_enc']
    df['temp_hum'] = df['Temparature'] * df['Humidity']
    df['temp_hum_moisture_ratio'] = df['temp_hum'] / (df['Moisture'] + 1e-5)
    df['n_to_p'] = df['Nitrogen'] / (df['Phosphorous'] + 1e-5)
    df['n_to_k'] = df['Nitrogen'] / (df['Potassium'] + 1e-5)
    df['p_to_k'] = df['Phosphorous'] / (df['Potassium'] + 1e-5)
    df = df.select_dtypes(include=[np.number])
    scaler = StandardScaler()
    df[df.columns.difference(['id'])] = scaler.fit_transform(df[df.columns.difference(['id'])])
    return df


X = add_features(train)
y = train[target_col].values

X = X.drop(columns=['id',target_col])

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


tabnet_model = TabNetClassifier(
    optimizer_params=dict(lr=1e-3),
    scheduler_params={"step_size": 10, "gamma": 0.9},
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    verbose=1,
    seed=28,
    device_name='cuda' 
)

tabnet_model.fit(
    X_train=X_train.values, y_train=y_train,
    eval_set=[(X_valid.values, y_valid)],
    eval_name=["val"],
    eval_metric=["accuracy"],
    max_epochs=100,
    patience=5,
    batch_size=1024,
    virtual_batch_size=128
)


test_features = add_features(test).drop(columns=['id'])
preds = tabnet_model.predict(test_features.values)

predicted_labels = target_encoder.inverse_transform(preds)

submission = pd.DataFrame({
    "id": test["id"],
    "Fertilizer Name": predicted_labels
})
submission.to_csv("submission.csv", index=False)



submission

