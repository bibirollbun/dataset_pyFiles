import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
import tensorflow.keras.backend as K


train_df = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
train_df.head()


train_df.shape


# ============ STEP 1: Feature Engineering Function ============
def feature_engineering(df):
    df = df.copy()
    
    # Drop non-informative 'id' column
    if 'id' in df.columns:
        df.drop(columns=['id'], inplace=True)
    
    # Compute BMI
    df['BMI'] = df['Weight'] / (df['Height'] / 100) ** 2

    # Interaction features
    df['HR_x_Duration'] = df['Heart_Rate'] * df['Duration']
    df['Temp_x_Duration'] = df['Body_Temp'] * df['Duration']
    df['HR_per_Duration'] = df['Heart_Rate'] / df['Duration'].replace(0, np.nan)
    df['Temp_Deviation'] = df['Body_Temp'] - 36.5

    # Fill inf/nan
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    return df


# ============ STEP 2: Full Preprocessing Pipeline ============
def create_preprocessing_pipeline():
    # Define column categories
    categorical_features = ['Sex']
    numerical_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp',
                          'BMI', 'HR_x_Duration', 'Temp_x_Duration', 'HR_per_Duration', 'Temp_Deviation']

    # Transformers
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='first'))
    ])

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Combine transformers
    preprocessor = ColumnTransformer(transformers=[
        ('cat', categorical_transformer, categorical_features),
        ('num', numerical_transformer, numerical_features)
    ])

    return preprocessor


# ============ STEP 3: Prepare Data ============

def prepare_data(df, target_column='Calories'):
    # Step 1: Feature Engineering
    df = feature_engineering(df)
    
    # Separate target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Step 2: Build and apply preprocessing pipeline
    preprocessor = create_preprocessing_pipeline()
    X_processed = preprocessor.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, preprocessor


# Preprocess
X_train, X_test, y_train, y_test, preprocessor = prepare_data(train_df)


def rmsle(y_true, y_pred):
    return K.sqrt(K.mean(K.square(K.log(y_pred + 1) - K.log(y_true + 1))))


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='relu')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss=rmsle, metrics=['mae'])

history = model.fit(X_train, y_train, callbacks=[early_stop, lr_schedule], epochs=100, validation_split=0.2)


plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('RMSLE')
plt.ylim([0, 0.1])
plt.legend(loc='upper right')


# Load test data
test_df = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
test_id = test_df['id']
test_df = feature_engineering(test_df)
X_test_final = preprocessor.transform(test_df)


# Predict
predictions = model.predict(X_test_final)

# Format submission
submission = pd.DataFrame({'id': test_id.values, 'Calories': predictions.flatten()})
submission.to_csv('predictions.csv', index=False)
print("Prediction results saved to predictions.csv")


submission.head()

