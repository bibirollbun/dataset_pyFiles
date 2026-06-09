import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error


def preprocess_data(df):
    """
    Simple preprocessing function with NaN handling
    """
    df = df.copy()
    
    # Drop rows with NaN in num_sold if it exists in the dataframe
    if 'num_sold' in df.columns:
        df = df.dropna(subset=['num_sold'])
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Basic date features
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['date'].dt.dayofweek.isin([5, 6]).astype(int)
    
    # Handle missing values in categorical columns
    categorical_cols = ['store', 'country', 'product']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna('missing')
            df[col] = pd.Categorical(df[col]).codes
    
    return df


def train_model(train_df):
    """
    Train a simple Random Forest model
    """
    df = preprocess_data(train_df)
    features = ['month', 'day_of_week', 'is_weekend', 'store', 'country', 'product']
    
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    
    model.fit(df[features], df['num_sold'])
    return model, features


def make_predictions(model, features, test_df):
    """
    Make predictions on test data
    """
    df = preprocess_data(test_df)
    predictions = model.predict(df[features])
    predictions = np.maximum(0, np.round(predictions))
    
    return pd.DataFrame({
        'id': test_df['id'],
        'num_sold': predictions
    })


def evaluate_performance(model, features, validation_df):
    """
    Calculate MAPE for model evaluation
    """
    df = preprocess_data(validation_df)
    predictions = model.predict(df[features])
    return mean_absolute_percentage_error(df['num_sold'], predictions) * 100


# Load your data
train_df = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')


print("Training model...")
model, features = train_model(train_df)


print("Evaluating model...")
mape = evaluate_performance(model, features, train_df)
print(f"Training MAPE: {mape:.2f}%")


print("Making predictions...")
submissions = make_predictions(model, features, test_df)
submissions.to_csv('predictions.csv', index=False)
print("Predictions saved to 'predictions.csv'")




