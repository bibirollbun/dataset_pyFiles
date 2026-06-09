import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, concatenate, BatchNormalization, Dropout, Add, Lambda
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import holidays

# Load datasets
train = pd.read_csv('/kaggle/input/train-pssse1/train_PSSSE1.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')


# Set seed for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


# Handle missing values
train = train.dropna()


# Handle 'id' Column
ids = test['id']
train.drop(columns=['id'], inplace=True) 
test.drop(columns=['id'], inplace=True)


# Adding Advanced Features: Public Holidays
def add_holidays(df):
    countries_holidays = {
        'Finland': holidays.Finland(),
        'Canada': holidays.Canada(),
        'Italy': holidays.Italy(),
        'Kenya': holidays.Kenya(),
        'Singapore': holidays.Singapore(),
        'Norway': holidays.Norway()
    }
    
    df['is_holiday'] = df.apply(
        lambda row: int(row['date'] in countries_holidays[row['country']]), axis=1
    )
    return df


# Transforming Date Features
def transform_date(df):
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['weekday'] = df['date'].dt.weekday
    df['day_of_year'] = df['date'].dt.dayofyear
    df['week_of_year'] = df['date'].dt.isocalendar().week
    df['is_weekend'] = (df['weekday'] >= 5).astype(int)
    df['season'] = ((df['month'] % 12 + 3) // 3)  
    df['elapsed_days'] = (df['date'] - df['date'].min()).dt.days
    df['is_end_of_month'] = df['date'].dt.is_month_end.astype(int)
    df['is_end_of_quarter'] = df['date'].dt.is_quarter_end.astype(int)
    
    # Sinusoidal transformations for cyclic features
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    
    return add_holidays(df)


# Transform train and test data
train = transform_date(train)
test = transform_date(test)

# Log-transform the target variable
train['num_sold'] = np.log1p(train['num_sold'])


# Split train data into features and target
X = train.drop(columns=['num_sold', 'date'])
y = train['num_sold']


# Encoding Categorical Features:
categorical_cols = ['country', 'store', 'product', 'is_weekend', 'season', 'is_end_of_month', 'is_end_of_quarter', 'is_holiday']
numerical_cols = [col for col in X.columns if col not in categorical_cols]

encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    test[col] = le.transform(test[col])
    encoders[col] = le


# Scaling Numerical Features:
scaler = MinMaxScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
test[numerical_cols] = scaler.transform(test[numerical_cols])


def attention_block(x):
    attention = Dense(x.shape[-1], activation='softmax')(x)
    return Lambda(lambda inputs: inputs[0] * inputs[1])([x, attention])

def build_advanced_model():
    # Inputs for categorical features
    inputs = []
    embeddings = []
    for col in categorical_cols:
        unique_values = X[col].nunique()
        input_cat = Input(shape=(1,))
        embedding = Embedding(input_dim=unique_values, output_dim=min(50, (unique_values + 1) // 2))(input_cat)
        flatten = Flatten()(embedding)
        inputs.append(input_cat)
        embeddings.append(flatten)

    # Input for numerical features
    input_num = Input(shape=(len(numerical_cols),))
    inputs.append(input_num)
    
    # Combine all inputs
    x = concatenate(embeddings + [input_num])

    # Add dense layers with residual connections
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    residual = x

    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    residual = Dense(256)(residual)
    x = Add()([x, residual])

    x = attention_block(x)  # Attention mechanism

    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Dense(1)(x)  # Linear activation for regression output
    
    model = Model(inputs=inputs, outputs=x)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model


# Prepare inputs for the model
def prepare_inputs(X):
    inputs = [X[col] for col in categorical_cols]
    inputs.append(X[numerical_cols].values)
    return inputs

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=42)

model = build_advanced_model()

history = model.fit(
    prepare_inputs(X_train),
    y_train,
    validation_data=(prepare_inputs(X_valid), y_valid),
    epochs=200,
    batch_size=1024,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, min_delta=0.001),
        tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, min_lr=1e-6)
    ]
)


test_predictions = np.expm1(model.predict(prepare_inputs(test)))


submission = pd.DataFrame({
    'id': ids,
    'num_sold': test_predictions.flatten()
})

submission.to_csv('submission.csv', index=False)
print("submission.csv saved.")

