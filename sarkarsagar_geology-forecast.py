import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import warnings
import random
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.feature_selection import mutual_info_regression
from sklearn.svm import SVR
from scipy import stats
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.impute import KNNImputer
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
warnings.filterwarnings('ignore')



train=pd.read_csv('/kaggle/input/geology-forecast-challenge-open/data/train.csv')
test=pd.read_csv('/kaggle/input/geology-forecast-challenge-open/data/test.csv')
sample_submission=pd.read_csv('/kaggle/input/geology-forecast-challenge-open/data/sample_submission.csv')



list1=set(train.columns.to_list()) 
list2=set(test.columns.to_list())

input_cols=list(list2-{'geology_id'})
target_cols=list(list1-list2)
target_cols_with_id=target_cols.copy()
target_cols_with_id.append('geology_id')
main_target_cols=list(set(map(str, range(1, 322))) & set(target_cols))


def knn_null_dealing(data,cols):
    data=data[cols]
    # Initialize KNNImputer (k=2)
    imputer = KNNImputer(n_neighbors=2)
    
    # Impute missing values
    imputed_data = imputer.fit_transform(data)
    
    # Convert back to DataFrame
    imputed_df = pd.DataFrame(imputed_data, columns=data.columns)
    return imputed_df


def null_dealing(data,input_cols):
    df=data[input_cols].copy()
    # Assuming df is your DataFrame with 30 columns
    # First, identify the two columns with no null values
    complete_cols = [col for col in df.columns if df[col].notnull().all()]

    if len(complete_cols) < 2:
        raise ValueError("There must be at least two complete columns to use this approach")

    # For each column with missing values (excluding the two complete columns)
    for column in [col for col in df.columns if col not in complete_cols and df[col].isnull().any()]:
        # Create temporary dataset without missing values in the target column
        temp_df = df.dropna(subset=[column])
        
        # Prepare features (the two complete columns) and target (current column)
        X = temp_df[complete_cols]
        y = temp_df[column]
        
        # Train linear regression model
        model =LinearRegression()
        model.fit(X, y)
        
        # Identify rows where the current column is null
        null_mask = df[column].isnull()
        
        # Predict missing values using the two complete columns
        if null_mask.any():
            X_pred = df.loc[null_mask, complete_cols]
            df.loc[null_mask, column] = model.predict(X_pred)
    return df


train_with_no_null=null_dealing(train,input_cols)


train_with_no_null.isna().sum().sum()


train_with_no_null=pd.concat([train_with_no_null,train[target_cols_with_id]],axis=1)[train.columns]


def winsorize_columns(df, limits=(0.05, 0.05)):
    """
    Winsorize all numeric columns in a DataFrame.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        limits (tuple): Lower and upper percentile bounds (e.g., (0.05, 0.05) caps 5% on each tail).
    
    Returns:
        pd.DataFrame: Winsorized DataFrame.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_winsorized = df.copy()
    
    for col in numeric_cols:
        df_winsorized[col] = winsorize(df[col], limits=limits)
    
    return df_winsorized
def remove_outliers_zscore_columnwise(df, threshold=3):
    """
    Remove outliers column-wise using Z-score method.
    For each numeric column, values beyond threshold standard deviations are set to NaN.
    Returns a new DataFrame with outliers removed column-wise.
    """
    df_clean = df.copy()
    
    # Process only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        z_scores = np.abs(stats.zscore(df[col]))
        df_clean[col] = df[col].where(z_scores < threshold)
    
    return df_clean
def remove_outliers(df, threshold=3):
    """
    Remove outliers using z-score method column-wise
    threshold: number of standard deviations to consider as outlier (default 3)
    """
    z_scores = np.abs((df - df.mean()) / df.std())
    return df[(z_scores < threshold).all(axis=0)]

# Alternative using IQR method



train_with_no_outliers=winsorize_columns(train_with_no_null)


train_with_no_outliers['0']=0


train_with_no_outliers.isna().sum().sum()


'''for col in target_cols+input_cols:
    train_with_no_outliers[col]=train_with_no_outliers[col].fillna(train_with_no_outliers[col].mean())'''


train_with_no_outliers.isna().sum()[train_with_no_outliers.isna().sum()>0]


def feature_extraction(data):
    data['Varience']=data[input_cols].dropna().std(axis=1)
    data['Sum']=data[input_cols].dropna().sum(axis=1)
    return data



train_with_no_outliers[main_target_cols]


X=train_with_no_outliers[input_cols]
y=train_with_no_outliers[target_cols]
discrete_features=X.dtypes==float




def make_mi_scores(X, y, discrete_features):
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores





def plot_mi_scores(scores, top_n=100):
    # Sort scores and select top N features
    scores = scores.sort_values(ascending=False).head(top_n)
    scores = scores.sort_values(ascending=True)  # Re-sort for horizontal bar plot
    
    # Plotting
    width = np.arange(len(scores))
    ticks = list(scores.index)
    
    plt.barh(width, scores, color='skyblue')
    plt.yticks(width, ticks)
    plt.title(f"Top {top_n} Mutual Information Scores")



mi_scores = make_mi_scores(X, y['277'], discrete_features)
# Example usage
plt.figure(dpi=100, figsize=(8, 5))
plot_mi_scores(mi_scores, top_n=20)  # Replace `mi_scores` with your actual scores
plt.show()



sns.boxplot(data=train_with_no_outliers['2'])


sns.kdeplot(data=train_with_no_outliers['-2'],fill=True)


sns.lmplot(x="32", y='258', data=train_with_no_outliers);


feature_extraction(train_with_no_outliers[input_cols])


input_scaler=StandardScaler()
output_scaler=StandardScaler()
input_scaler.fit(feature_extraction(train_with_no_outliers[input_cols]))
output_scaler.fit(train_with_no_outliers[target_cols])

train_scaled_X=input_scaler.fit_transform(feature_extraction(train_with_no_outliers[input_cols]))
train_scaled_y=output_scaler.fit_transform(train_with_no_outliers[target_cols])

test_scaled_X=input_scaler.fit_transform(feature_extraction(null_dealing(test,input_cols)))




X_train,X_test,y_train,y_test=train_test_split(train_scaled_X,train_scaled_y,shuffle=False)


dt =DecisionTreeRegressor()
dt.fit(X_train, y_train) 


model =MultiOutputRegressor(SVR(kernel="rbf"))
model.fit(X_train, y_train) 


'''

# Define the model architecture
def create_model(input_dim=300, output_dim=3000):
    model = models.Sequential([
        # Input layer
        layers.InputLayer(input_shape=(input_dim,)),
        
        # First hidden layer with 1024 neurons and ReLU activation
        layers.Dense(1024, activation='relu'),
        
        # Optional: Add batch normalization
        #layers.BatchNormalization(),
        
        # Second hidden layer with 2048 neurons
        layers.Dense(2048, activation='relu'),
        layers.Dense(512,activation='relu'),
        layers.Dense(1024,activation ='relu'),
        
        # Optional: Add dropout for regularization
        layers.Dropout(0.3),
        
        # Output layer with 3000 neurons (linear activation for regression tasks)
        layers.Dense(output_dim, activation='linear')  # Use 'softmax' for classification
    ])
    
    return model

# Create the model
model = create_model(input_dim=300, output_dim=3000)

# Compile the model (adjust optimizer and loss based on your task)
model.compile(optimizer='adam',
              loss='mean_squared_error',  # For regression
              # loss='categorical_crossentropy'  # For classification
              metrics=['accuracy'])

# Print model summary
model.summary()


# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)'''




# Get feature importances
importances = dt.feature_importances_
feature_names = X_train.columns if hasattr(X_train, 'columns') else [f'Feature {i}' for i in range(X_train.shape[1])]


# Sort features by importance (descending) and select top 20
sorted_idx = np.argsort(importances)[::-1][:5]
top_features = [feature_names[i] for i in sorted_idx]
top_importances = importances[sorted_idx]

# Create plot
plt.figure(figsize=(12, 8))
plt.barh(top_features, top_importances, color='skyblue')
plt.gca().invert_yaxis()  # Most important at top
plt.xlabel('Feature Importance Score')
plt.title('Top 20 Feature Importances')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


scaled_y_pred=model.predict(X_test)
y_pred=output_scaler.inverse_transform(scaled_y_pred)


mean_absolute_error(y_pred,y_test)


def categorical_negative_log_likelihood(actual, predicted_probs):
    """
    Calculate negative log-likelihood for multi-class classification
    
    Parameters:
    actual: array of true class indices (one-hot encoded)
    predicted_probs: array of predicted probabilities for each class
    
    Returns:
    Negative log-likelihood value
    """
    eps = 1e-15
    predicted_probs = np.clip(predicted_probs, eps, 1 - eps)
    nll = -np.sum(actual * np.log(predicted_probs))
    return nll


categorical_negative_log_likelihood(y_pred,y_test)


scaled_pred=model.predict(test_scaled_X)
pred=output_scaler.inverse_transform(scaled_pred)


submission_df=pd.DataFrame(pred,columns=target_cols)


submission_df['geology_id']=sample_submission.geology_id.tolist()


submission_df=submission_df[sample_submission.columns.to_list()]
submission_df.head()


submission_df.to_csv('submission.csv',index=False)

