import numpy as np
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
df


df.shape


df.dtypes


from sklearn.preprocessing import LabelEncoder

# Create a LabelEncoder instance
le_sex = LabelEncoder()

# Fit and transform the 'Sex' column
df['Sex_Encoded'] = le_sex.fit_transform(df['Sex'])

# Optional: check mapping
print(dict(zip(le_sex.classes_, le_sex.transform(le_sex.classes_))))


df.isnull().sum()


# Correlation matrix
correlation_matrix = df.corr(numeric_only=True)

# Correlation with target 'Calories'
calorie_corr = correlation_matrix['Calories'].sort_values(ascending=False)

print("Correlation of features with Calories:\n")
print(calorie_corr)


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='Height', y='Calories', alpha=0.3)
plt.title('Height vs Calories')
plt.grid(True)
plt.show()


df.drop(columns=['id','Sex'], inplace=True)


from sklearn.model_selection import train_test_split

# Define features and target
X = df.drop(columns=['Calories'])  # All columns except target
y = df['Calories']                 # Target column

# Split into training and test sets (90% train, 10% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)

# Optional: Confirm sizes
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout



# There is a lot of noisy behaviour, Lets scale our inputs


from sklearn.preprocessing import StandardScaler

# Initialize scaler
scaler = StandardScaler()

# Fit only on training data
X_train_scaled = scaler.fit_transform(X_train)

# Transform test data using the same scaler
X_test_scaled = scaler.transform(X_test)


# It is clearly the case of overfitting, lets reduce the complexity of our neural network


# Still overfitting


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import regularizers

model = Sequential([
    Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001), input_shape=(X_train.shape[1],)),
    Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    Dense(1)
])



import tensorflow.keras.backend as K

def rmsle(y_true, y_pred):
    return K.sqrt(K.mean(K.square(K.log(y_pred + 1.0) - K.log(y_true + 1.0))))


model.compile(
    optimizer='adam',
    loss=rmsle,         # You can use this as your loss
    metrics=['mae', rmsle]  # Track both
)


from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_test_scaled, y_test),
    epochs=10,                # Train more epochs; early stopping will handle early termination
    batch_size=128,
    callbacks=[early_stop],
    verbose=1
)


import pandas as pd

df2 = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
df2.head()


print(df2.columns)



df2['Sex_Encoded'] = le_sex.transform(df2['Sex'])


# Drop original 'Sex' column
df2_cleaned = df2.drop(columns=['Sex'])


X_df2 = df2_cleaned.drop(columns=['id'])  # Drop ID for prediction


X_df2_scaled = scaler.transform(X_df2)  # Use previously fitted StandardScaler


calorie_preds = model.predict(X_df2_scaled).flatten()  # Flatten to 1D


submission = pd.DataFrame({
    'id': df2['id'],
    'Calories': calorie_preds
})


submission.to_csv('submission.csv', index=False)




