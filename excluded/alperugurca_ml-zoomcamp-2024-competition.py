# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from datetime import datetime

# Define data paths
DATA_PATH = '/kaggle/input/ml-zoomcamp-2024-competition'

def load_and_check_data():
    """Load data and print basic information"""
    # Load training data
    sales_df = pd.read_csv(f'{DATA_PATH}/sales.csv')
    
    # Load and process test data
    test_df = pd.read_csv(f'{DATA_PATH}/test.csv', sep=';')  # Use separator
    
    print("Sales DataFrame Info:")
    print(sales_df.info())
    print("\nSales DataFrame Head:")
    print(sales_df.head())
    
    print("\nTest DataFrame Info:")
    print(test_df.info())
    print("\nTest DataFrame Head:")
    print(test_df.head())
    
    return sales_df, test_df

def prepare_features(df, is_train=True):
    """Create features from the available data"""
    if is_train:
        # For training data
        df['date'] = pd.to_datetime(df['date'])
    else:
        # For test data
        df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y')
    
    # Create date-based features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    if is_train:
        # Create lag features for training data
        df['lag_1'] = df.groupby(['item_id', 'store_id'])['quantity'].shift(1)
        df['lag_7'] = df.groupby(['item_id', 'store_id'])['quantity'].shift(7)
        
        # Create rolling mean features
        df['rolling_mean_7'] = df.groupby(['item_id', 'store_id'])['quantity'].transform(
            lambda x: x.rolling(window=7, min_periods=1).mean())
        df['rolling_mean_30'] = df.groupby(['item_id', 'store_id'])['quantity'].transform(
            lambda x: x.rolling(window=30, min_periods=1).mean())
    else:
        # For test data, initialize these columns with 0
        df['lag_1'] = 0
        df['lag_7'] = 0
        df['rolling_mean_7'] = 0
        df['rolling_mean_30'] = 0
    
    # Fill NaN values
    df = df.fillna(0)
    
    return df

def prepare_training_data(df):
    """Prepare data for model training"""
    # Select features for training
    feature_columns = [
        'year', 'month', 'day', 'day_of_week', 'is_weekend',
        'lag_1', 'lag_7', 'rolling_mean_7', 'rolling_mean_30',
        'store_id'  # Adding store_id as a feature
    ]
    
    X = df[feature_columns]
    y = df['quantity']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test, feature_columns

def train_model(X_train, y_train):
    """Train XGBoost model"""
    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=7,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    return model

def evaluate_model(y_true, y_pred, model_name):
    """Evaluate model performance using multiple metrics"""
    metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred)
    }
    
    print(f"\nModel: {model_name}")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    return metrics

def generate_submission(model, test_df, feature_columns):
    """Generate submission file"""
    # Make predictions
    test_features = test_df[feature_columns]
    predictions = model.predict(test_features)
    
    # Create submission dataframe
    submission = pd.DataFrame({
        'row_id': test_df['row_id'],
        'quantity': predictions
    })
    
    # Save submission file
    submission.to_csv('submission.csv', index=False)
    print("Submission file created successfully!")

def main():
    # Load and check data
    print("Loading and checking data...")
    train_df, test_df = load_and_check_data()
    
    # Prepare features
    print("Preparing features...")
    train_df = prepare_features(train_df, is_train=True)
    test_df = prepare_features(test_df, is_train=False)
    
    # Prepare training data
    print("Preparing training data...")
    X_train, X_test, y_train, y_test, feature_columns = prepare_training_data(train_df)
    
    # Train model
    print("Training model...")
    model = train_model(X_train, y_train)
    
    # Make predictions and evaluate
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    metrics = evaluate_model(y_test, y_pred, "XGBoost")
    
    # Generate submission
    print("Generating submission...")
    generate_submission(model, test_df, feature_columns)
    
    return model, metrics

if __name__ == "__main__":
    main()


# Import necessary libraries
import pandas as pd
import numpy as np
from datetime import datetime

# Define data paths
DATA_PATH = '/kaggle/input/ml-zoomcamp-2024-competition'

# Load the data first
print("Loading data...")
train_df = pd.read_csv(f'{DATA_PATH}/sales.csv')

# Now perform EDA
def perform_eda(sales_df):
    """Perform Exploratory Data Analysis"""
    print("=== Basic Statistics ===")
    print("\nDescriptive Statistics for Numerical Columns:")
    print(sales_df.describe())
    
    print("\n=== Missing Values Analysis ===")
    missing_values = sales_df.isnull().sum()
    print(missing_values[missing_values > 0])
    
    print("\n=== Target Variable Analysis ===")
    print("\nQuantity Statistics:")
    print(f"Mean quantity: {sales_df['quantity'].mean():.2f}")
    print(f"Median quantity: {sales_df['quantity'].median():.2f}")
    print(f"Min quantity: {sales_df['quantity'].min():.2f}")
    print(f"Max quantity: {sales_df['quantity'].max():.2f}")
    
    # Time-based analysis
    sales_df['date'] = pd.to_datetime(sales_df['date'])
    print("\n=== Temporal Analysis ===")
    monthly_sales = sales_df.groupby(sales_df['date'].dt.to_period('M'))['quantity'].sum()
    print("\nMonthly Sales Trends:")
    print(monthly_sales)
    
    # Store and Item Analysis
    print("\n=== Store Analysis ===")
    store_stats = sales_df.groupby('store_id')['quantity'].agg(['mean', 'count'])
    print("\nStore-wise Statistics:")
    print(store_stats)
    
    print("\n=== Item Analysis ===")
    item_stats = sales_df.groupby('item_id')['quantity'].agg(['mean', 'count'])
    print("\nTop 5 Items by Average Quantity:")
    print(item_stats.sort_values('mean', ascending=False).head())
    
    # Feature Correlations
    numeric_cols = sales_df.select_dtypes(include=[np.number]).columns
    correlations = sales_df[numeric_cols].corr()
    print("\n=== Feature Correlations ===")
    print(correlations['quantity'].sort_values(ascending=False))

    return {
        'monthly_sales': monthly_sales,
        'store_stats': store_stats,
        'item_stats': item_stats,
        'correlations': correlations
    }

# Perform EDA
print("Performing EDA...")
eda_results = perform_eda(train_df)

