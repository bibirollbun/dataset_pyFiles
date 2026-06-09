import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping


df=pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')


df.shape


df.info()


df2=pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')


df2.shape


df2.isnull().sum()


#combining the train and training extra dataset
df=pd.concat([df,df2])


df.head()


df.info()


df.shape


df.isnull().sum()


#price distribution
plt.figure(figsize=(10, 6))
sns.histplot(df["Price"], kde=True, color="teal", bins=30)
plt.title("Price Distribution of Bags", fontsize=16)
plt.xlabel("Price", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.tight_layout()
plt.show()


#price vs brand
plt.figure(figsize=(10, 6))
brand_price = df.groupby("Brand")["Price"].mean().sort_values(ascending=False)
sns.barplot(x=brand_price.index, y=brand_price.values, palette="muted")
plt.title("Average Price by Brand", fontsize=16)
plt.xlabel("Brand", fontsize=12)
plt.ylabel("Average Price", fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


#price vs size
plt.figure(figsize=(10, 6))
sns.boxplot(x="Size", y="Price", data=df, palette="pastel")
plt.title("Price Distribution by Size", fontsize=16)
plt.xlabel("Bag Size", fontsize=12)
plt.ylabel("Price", fontsize=12)
plt.tight_layout()
plt.show()


#style popularity
plt.figure(figsize=(10, 6))
style_counts = df["Style"].value_counts()
sns.barplot(y=style_counts.index, x=style_counts.values, palette="crest")
plt.title("Style Popularity", fontsize=16)
plt.xlabel("Number of Bags", fontsize=12)
plt.ylabel("Style", fontsize=12)
plt.tight_layout()
plt.show()


#price vs material
plt.figure(figsize=(10, 6))
sns.boxplot(x="Material", y="Price", data=df, palette="pastel")
plt.title("Price Distribution by Material", fontsize=16)
plt.xlabel("Material", fontsize=12)
plt.ylabel("Price", fontsize=12)
plt.tight_layout()
plt.show()


#dropping the 'id' column
df = df.drop(columns=["id"])


#filling the empty numeric columns with mean
numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())


#filling the empty categorical columns with mode
categorical_cols = df.select_dtypes(include=["object"]).columns
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])


#one-hot encoding categorical features
df = pd.get_dummies(df, drop_first=True)

x = df.drop(columns=["Price"])
y = df["Price"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#scaling
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


#building the model
model = Sequential([
    Dense(512, activation="relu", input_shape=(x_train_scaled.shape[1],)),
    Dropout(0.3),
    Dense(256, activation="relu"),
    Dropout(0.2),
    Dense(128, activation="relu"),
    Dropout(0.2),
    Dense(64, activation="relu"),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dense(1)
])

#compiling the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss="mse", metrics=["mae"])


#early stopping
early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

#training the model
history = model.fit(x_train_scaled, y_train, validation_data=(x_test_scaled, y_test), 
                    epochs=30, batch_size=1024, callbacks=[early_stop], verbose=0)


#evaluate the model
predictions = model.predict(x_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print(f"RMSE: {rmse}")


model.summary()


#RMSE over epochs
rmse_per_epoch = [val_loss ** 0.5 for val_loss in history.history['val_loss']]
plt.figure(figsize=(10, 6))
plt.plot(rmse_per_epoch, label='Validation RMSE', color='blue', lw=2)
plt.title("RMSE Over Epochs", fontsize=16, fontweight='bold')
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("RMSE", fontsize=12)
plt.grid(alpha=0.3)
plt.legend(fontsize=10, shadow=True)
plt.tight_layout()
plt.show()


#RMSE distribution of predictions
plt.figure(figsize=(10, 6))
errors = np.sqrt((y_test - predictions.flatten())**2)
sns.histplot(errors, kde=True, color="teal", bins=30)
plt.title("RMSE Distribution of Predictions", fontsize=16)
plt.xlabel("RMSE", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()


#calculating RMSE for each feature
feature_errors = []

for i in range(x_test_scaled.shape[1]):
    modified_test = np.zeros_like(x_test_scaled)
    modified_test[:, i] = x_test_scaled[:, i]
    
    predictions = model.predict(modified_test, verbose=0)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    feature_errors.append(rmse)

#RMSE by features
plt.figure(figsize=(12, 6))
sns.barplot(x=x.columns, y=feature_errors, palette="crest")
plt.title("RMSE Contribution by Features", fontsize=16)
plt.xlabel("Features", fontsize=12)
plt.ylabel("RMSE", fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


test=pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')

#preprocessing test data
submission = pd.DataFrame({'id': test['id']})
test = test.drop(columns=['id'])

#one-hot encoding the test data and align columns
test = pd.get_dummies(test, drop_first=True)
test = test.reindex(columns=x.columns, fill_value=0)

#scaling
test_scaled = scaler.transform(test)


#making predictions on test data
predictions = model.predict(test_scaled)

#adding predicted values to df and create submission file
submission['Price'] = predictions.flatten()
submission.to_csv('submission.csv', index=False)

