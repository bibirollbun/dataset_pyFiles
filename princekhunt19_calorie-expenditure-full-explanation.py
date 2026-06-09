import numpy as np
import pandas as pd
import os
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def load_csv(file_path):
    """Loads a CSV file into a DataFrame"""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Error: File '{file_path}' not found!")

        df = pd.read_csv(file_path)

        print(f"âœ… CSV file! Shape: {df.shape}")

        return df
    except FileNotFoundError as fnf_error:
        print(fnf_error)
    except pd.errors.EmptyDataError:
        print("Error: The CSV file is empty!")
    except pd.errors.ParserError:
        print("Error: CSV parsing issue. Check the file format.")
    except Exception as e:
        print(f"Unexpected error: {e}")

    return None  


train_df = load_csv("/kaggle/input/playground-series-s5e5/train.csv")
test_df = load_csv("/kaggle/input/playground-series-s5e5/test.csv")


def get_basic_info(df):
    """Get basic information about the dataset"""
    print(f"ğŸ”¹ Number of Rows: {df.shape[0]}")
    print(f"ğŸ”¹ Number of Columns: {df.shape[1]}")
    print(f"ğŸ”¹ Columns: {df.columns.tolist()}")
    print(f"ğŸ”¹ Column Data Types: {df.dtypes}")
    print(f"ğŸ”¹ Missing Values: {df.isnull().sum().sum()}")
    print(f"ğŸ”¹ Missing Values (Per Column) : {df.isnull().sum()}")
    print(f"ğŸ”¹ Missing Values (Per Column Percentage) : {(df.isnull().mean() * 100)}")
    print(f"ğŸ”¹ Unique Values: {df.nunique()}")


get_basic_info(train_df)


def plot_missing_bar(df):
    """Plot a bar chart showing the count of missing values per column."""
    missing_values = df.isnull().sum()
    missing_values = missing_values[missing_values > 0].sort_values(ascending=False)

    if missing_values.empty:
        print("âœ… No missing values to plot.")
        return

    plt.figure(figsize=(12, 6))
    sns.barplot(x=missing_values.index, y=missing_values.values, palette="magma")
    plt.xticks(rotation=90)
    plt.title("Missing Values Count per Column")
    plt.ylabel("Count")
    plt.show()


plot_missing_bar(train_df)


train_df.isnull().sum()


def plot_histogram(df, numerical_cols):
    """Plot histograms for numerical features to visualize their distributions."""
    num_plots = len(numerical_cols)
    num_rows = (num_plots // 3) + (1 if num_plots % 3 != 0 else 0)  
    
    fig, axes = plt.subplots(num_rows, 3, figsize=(15, num_rows * 5))
    axes = axes.flatten()
    
    for i, col in enumerate(numerical_cols):
        sns.histplot(df[col], kde=False, bins=30, color='skyblue', edgecolor='black', ax=axes[i])
        axes[i].set_title(f"Histogram of {col}")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Frequency")
    
    for i in range(num_plots, len(axes)):
        axes[i].axis('off') 
    
    plt.tight_layout()
    plt.show()


def plot_boxplot(df, numerical_cols):
    """Plot boxplots for numerical features to visualize the distribution and detect outliers."""
    num_plots = len(numerical_cols)
    num_rows = (num_plots // 3) + (1 if num_plots % 3 != 0 else 0)
    
    fig, axes = plt.subplots(num_rows, 3, figsize=(15, num_rows * 5))
    axes = axes.flatten() 
    
    for i, col in enumerate(numerical_cols):
        sns.boxplot(x=df[col], color='lightgreen', ax=axes[i])
        axes[i].set_title(f"Boxplot of {col}")
        axes[i].set_xlabel(col)
    
    for i in range(num_plots, len(axes)):
        axes[i].axis('off') 
    
    plt.tight_layout()
    plt.show()


def plot_kde(df, numerical_cols):
    """Plot KDE (Kernel Density Estimate) for numerical features to visualize smooth distributions."""
    num_plots = len(numerical_cols)
    num_rows = (num_plots // 3) + (1 if num_plots % 3 != 0 else 0)
    
    fig, axes = plt.subplots(num_rows, 3, figsize=(15, num_rows * 5))
    axes = axes.flatten()
    
    for i, col in enumerate(numerical_cols):
        sns.kdeplot(df[col], shade=True, color='blue', alpha=0.5, ax=axes[i])
        axes[i].set_title(f"KDE of {col}")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Density")
    
    for i in range(num_plots, len(axes)):
        axes[i].axis('off') 
    
    plt.tight_layout()
    plt.show()


def classify_columns(df):
    """
    Classify columns into binary, numerical, and categorical types.
    """
    binary_cols = []
    numerical_cols = []
    categorical_cols = []
    
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']: 
            if df[col].nunique() == 2:
                binary_cols.append(col)
            else:
                numerical_cols.append(col)
        elif df[col].dtype == 'object' or df[col].dtype.name == 'category':  
            if df[col].nunique() == 2: 
                binary_cols.append(col)
            else:
                categorical_cols.append(col)
        elif df[col].dtype == 'bool':  
            binary_cols.append(col) 

    return {
        'binary': binary_cols,
        'numerical': numerical_cols,
        'categorical': categorical_cols
    }


classify_columns(train_df)


classified_columns_object = {'binary': ['Sex'],
 'numerical': ['id',
  'Age',
  'Height',
  'Weight',
  'Duration',
  'Heart_Rate',
  'Body_Temp',
  'Calories'],
 'categorical': []}


plot_histogram(train_df, classified_columns_object["numerical"])


plot_boxplot(train_df, classified_columns_object["numerical"])


plot_kde(train_df, classified_columns_object["numerical"])


def plot_correlation_heatmap(df, numerical_cols):
    """Plot a heatmap to visualize the correlation between numerical features."""
    plt.figure(figsize=(12, 6))
    correlation_matrix = df[numerical_cols].corr() 
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("ğŸ”¹ Correlation Heatmap")
    plt.show()


def plot_pairplot(df, numerical_cols):
    """Plot pairplot for numerical columns to observe feature relationships."""
    sns.pairplot(df[numerical_cols], diag_kind="kde", corner=True)  
    plt.show()


plot_correlation_heatmap(train_df, classified_columns_object["numerical"])


# Drop unnecessary columns 

train_df = train_df.drop(['id', 'Sex', 'Age', 'Height', 'Weight'], axis=1)


new_classified_columns = classify_columns(train_df)
new_classified_columns


scale_dict = {
    'standard': ['Duration', 'Heart_Rate', 'Body_Temp']
}


def scale_features(df, scale_dict):
    """
    StandardScaler, MinMaxScaler, or RobustScaler based on the provided scale_dict.
    """
    df_scaled = df.copy()
    
    if 'standard' in scale_dict:
        scaler = StandardScaler()
        df_scaled[scale_dict['standard']] = scaler.fit_transform(df[scale_dict['standard']])
    
    if 'minmax' in scale_dict:
        scaler = MinMaxScaler()
        df_scaled[scale_dict['minmax']] = scaler.fit_transform(df[scale_dict['minmax']])
    
    if 'robust' in scale_dict:
        scaler = RobustScaler()
        df_scaled[scale_dict['robust']] = scaler.fit_transform(df[scale_dict['robust']])
    
    return df_scaled


train_scaled_df = scale_features(train_df, scale_dict)
train_scaled_df.head()


train_df.head()


def train_test_split_custom(df, target_col, test_size=0.2, stratify=False, random_state=42):
    """Function to split data into training and testing sets."""
    X = df.drop(columns=[target_col]) 
    y = df[target_col] 

    if stratify:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = train_test_split_custom(train_df, "Calories", test_size=0.1)
X_scaled_train, X_scaled_test, y_scaled_train, y_scaled_test = train_test_split_custom(train_scaled_df, "Calories", test_size=0.1)


X_train.head()


X_test.head()


X_scaled_train.head()


def train_models_regression(X_train, X_test, y_train, y_test):
    """Train and compare multiple regression models"""
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(),
        "Lasso Regression": Lasso(),
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest": RandomForestRegressor(),
        "Gradient Boosting": GradientBoostingRegressor(),
        "XGBoost": xgb.XGBRegressor()
    }
    
    print("ğŸ”¹ğŸ”¹ğŸ”¹ğŸ”¹ğŸ”¹ğŸ”¹ğŸ”¹ğŸ”¹ğŸ”¹ğŸ”¹ğŸ”¹ğŸ”¹ğŸ”¹ğŸ”¹ğŸ”¹ğŸ”¹ğŸ”¹ğŸ”¹ğŸ”¹ğŸ”¹ğŸ”¹ğŸ”¹")
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate RÂ², Mean Squared Error and Mean Absolute Error
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        print("ğŸ”¹ğŸ”¹ğŸ”¹ğŸ”¹ğŸ”¹ğŸ”¹ğŸ”¹ğŸ”¹ğŸ”¹ğŸ”¹ğŸ”¹ğŸ”¹ğŸ”¹ğŸ”¹ğŸ”¹ğŸ”¹ğŸ”¹ğŸ”¹ğŸ”¹ğŸ”¹ğŸ”¹ğŸ”¹")
        print(f"{name}\nRÂ²: {r2:.4f}\nMSE: {mse:.4f}\nMAE: {mae:.4f}")
        print("ğŸ”¹ğŸ”¹ğŸ”¹ğŸ”¹ğŸ”¹ğŸ”¹ğŸ”¹ğŸ”¹ğŸ”¹ğŸ”¹ğŸ”¹ğŸ”¹ğŸ”¹ğŸ”¹ğŸ”¹ğŸ”¹ğŸ”¹ğŸ”¹ğŸ”¹ğŸ”¹ğŸ”¹ğŸ”¹")
    
    print("ğŸ”¹ğŸ”¹ğŸ”¹ğŸ”¹ğŸ”¹ğŸ”¹ğŸ”¹ğŸ”¹ğŸ”¹ğŸ”¹ğŸ”¹ğŸ”¹ğŸ”¹ğŸ”¹ğŸ”¹ğŸ”¹ğŸ”¹ğŸ”¹ğŸ”¹ğŸ”¹ğŸ”¹ğŸ”¹")


# Train on without scaled data 
train_models_regression(X_train, X_test, y_train, y_test)


# Train on scaled data

train_models_regression(X_scaled_train, X_scaled_test, y_scaled_train, y_scaled_test)


X__, _, y__, _ = train_test_split_custom(train_df, "Calories", test_size=0.0000001)


xgb_model = xgb.XGBRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)


xgb_model = xgb_model.fit(X__, y__)


test_df


X_test = test_df[["Duration", "Heart_Rate", "Body_Temp"]]


X_test


preds = xgb_model.predict(X_test)


submission_df = test_df[["id"]].copy()
submission_df["Calories"] = preds


submission_df


submission_df["Calories"] = submission_df["Calories"].round(3)


submission_df.to_csv("submission.csv", index=False)


def plot_feature_importance(model, X, feature_names=None):
    """Plot the feature importance for tree-based models."""
    if hasattr(model, 'feature_importances_'): 
        importance = model.feature_importances_

        if feature_names is None:
            if hasattr(X, 'columns'):
                feature_names = X.columns
            else:
                feature_names = [f"Feature {i}" for i in range(X.shape[1])]
        
        indices = np.argsort(importance)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(X.shape[1]), importance[indices], align="center")
        plt.yticks(range(X.shape[1]), [feature_names[i] for i in indices])
        plt.title("Feature Importance for Random Forest")
        plt.xlabel('Importance')
        plt.show()


def feature_importance_rf(X_train, y_train, feature_names=None, random_state=42):
    """Train Random Forest model and plot feature importance."""
    rf_model = RandomForestRegressor(random_state=random_state)
    rf_model.fit(X_train, y_train)
    plot_feature_importance(rf_model, X_train, feature_names)


feature_importance_rf(X_train, y_train, train_df.columns)

