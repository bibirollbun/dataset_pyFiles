# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = 'iframe'



df = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e1/sample_submission.csv")


df.sample(5)


df_test.sample(5)


#Check the shape of data
print(f'The Training Dataset has {df.shape[0]} rows and {df.shape[1]} columns.')
print(f'The Training Dataset has {df_test.shape[0]} rows and {df_test.shape[1]} columns.')


df.info()


df.describe()


df.describe(include="object")


def calculate_missing_percentage(data):
    """
    Calculate the percentage of missing values for each column in the dataset.
    """
    missing_percentage = (data.isnull().sum() / len(data)) * 100
    return missing_percentage.round(2)

def plot_missing_percentage(missing_percentage, title="Percentage of Missing Values by Column"):
    """
    Plot the percentage of missing values for each column as a horizontal bar chart.
    """
    missing_percentage.plot(kind='barh', figsize=(10, 6), color='skyblue', edgecolor="black", linewidth=1.0)
    plt.title(title)
    plt.xlabel('Percentage')
    plt.ylabel('Columns')
    plt.show()

def plot_missing_heatmap(data, title="Heatmap of Missing Values"):
    """
    Create a heatmap visualization of missing values in the dataset.
    """
    plt.figure(figsize=(12, 8))
    sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
    plt.title(title)
    plt.show()


print("Missing Percentage in Training Data")
print("------------------------------------")
train_missing_percentage = calculate_missing_percentage(df)
print(train_missing_percentage)


print("\nMissing Percentage in Testing Data")
print("------------------------------------")
test_missing_percentage = calculate_missing_percentage(df_test)
print(test_missing_percentage)


# Plot for training data
plot_missing_percentage(train_missing_percentage, title="Percentage of Missing Values in Training Data")
plot_missing_heatmap(df, title="Heatmap of Missing Values in Training Data")


# Plot for test data
plot_missing_percentage(test_missing_percentage, title="Percentage of Missing Values in Test Data")
plot_missing_heatmap(df_test, title="Heatmap of Missing Values in Test Data")



# Check for duplicate rows
duplicates = df.duplicated()
# Print the number of duplicate rows
print(f"Number of duplicate rows: {duplicates.sum()}")

test_duplicates = df_test.duplicated()

print(f"Number of duplicate rows: {test_duplicates.sum()}")


# Group-wise median imputation
df['num_sold'] = df.groupby(['country', 'store', 'product'])['num_sold'].transform(
    lambda x: x.fillna(x.median())
)

# Fallback: Overall median
overall_median = df['num_sold'].median()
df['num_sold'].fillna(overall_median, inplace=True)



print("Missing Percentage in Training Data")
print("------------------------------------")
train_missing_percentage = calculate_missing_percentage(df)
print(train_missing_percentage)


# Step 5: Convert `date` to Datetime and Extract Time Features
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['weekday'] = df['date'].dt.day_name()


import plotly.express as px

# Histogram for `num_sold`
fig = px.histogram(df, x='num_sold', nbins=30, title="Distribution of num_sold", color_discrete_sequence=['blue'])
fig.update_layout(bargap=0.2)
fig.show()



for col in ['country', 'store', 'product']:
    counts = df[col].value_counts().reset_index()
    counts.columns = [col, 'count']  
    fig = px.bar(
        counts, 
        x=col, y='count', 
        title=f"Distribution of {col}",
        labels={col: col, 'count': 'Count'}, 
        color_discrete_sequence=px.colors.qualitative.Vivid
    )
    fig.update_layout(xaxis_tickangle=45)
    fig.show()



fig = px.line(
    df.groupby('date')['num_sold'].sum().reset_index(), 
    x='date', y='num_sold', 
    title="Overall Sales Over Time", 
    labels={'num_sold': 'Number of Products Sold'}
)
fig.show()



# Monthly Sales
monthly_sales = df.groupby('month')['num_sold'].sum().reset_index()
fig = px.bar(monthly_sales, x='month', y='num_sold', title="Monthly Sales", color='num_sold', color_continuous_scale='greens')
fig.show()

# Weekday Sales
weekday_sales = df.groupby('weekday')['num_sold'].sum().reset_index()
fig = px.bar(weekday_sales, x='weekday', y='num_sold', title="Sales by Weekday", color='num_sold', color_continuous_scale='purples')
fig.show()



for col in ['country', 'store', 'product']:
    fig = px.box(df, x=col, y='num_sold', title=f"num_sold by {col}", color=col)
    fig.update_layout(xaxis_tickangle=45)
    fig.show()



import plotly.express as px

def plot_sales_trend_by_country(df, title="Sales Trend by Country"):
    """
    Plot the sales trend over time for each country using Plotly with enhanced visuals.
    """
    # Create the line plot by country
    fig = px.line(
        df, 
        x='date', 
        y='num_sold', 
        color='country', 
        title=title,
        labels={'num_sold': 'Number of Products Sold', 'date': 'Date'},
        line_shape='linear',  # Spline can be used for smoother lines
        markers=True  # Adds markers on the line
    )
    
    # Update layout for better readability
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Number of Products Sold",
        plot_bgcolor='white',  # Set background to white for clarity
        xaxis=dict(
            tickmode='array',  # Define x-axis ticks
            tickangle=45,  # Rotate x-axis labels for better readability
        ),
        yaxis=dict(
            showgrid=True,  # Show grid lines for better visualization
            gridcolor='lightgray'  # Lighter grid lines for better contrast
        ),
        legend=dict(
            title="Country",  # Add title to the legend
            orientation="h",  # Make the legend horizontal
            yanchor="bottom",  # Position the legend at the bottom
            y=1.05,  # Adjust the position slightly above the plot
            xanchor="center",
            x=0.5
        ),
        font=dict(family="Arial, sans-serif", size=12)  # Set font for clarity
    )

    # Show the plot
    fig.show()



def plot_sales_trend_by_product(df, title="Sales Trend by Product"):
    """
    Plot the sales trend over time for each product using Plotly with enhanced visuals.
    """
    # Create the line plot by product
    fig = px.line(
        df, 
        x='date', 
        y='num_sold', 
        color='product', 
        title=title,
        labels={'num_sold': 'Number of Products Sold', 'date': 'Date'},
        line_shape='linear',  # Spline can be used for smoother lines
        markers=True  # Adds markers on the line
    )
    
    # Update layout for better readability
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Number of Products Sold",
        plot_bgcolor='white',  # Set background to white for clarity
        xaxis=dict(
            tickmode='array',  # Define x-axis ticks
            tickangle=45,  # Rotate x-axis labels for better readability
        ),
        yaxis=dict(
            showgrid=True,  # Show grid lines for better visualization
            gridcolor='lightgray'  # Lighter grid lines for better contrast
        ),
        legend=dict(
            title="Product",  # Add title to the legend
            orientation="h",  # Make the legend horizontal
            yanchor="bottom",  # Position the legend at the bottom
            y=1.05,  # Adjust the position slightly above the plot
            xanchor="center",
            x=0.5
        ),
        font=dict(family="Arial, sans-serif", size=12)  # Set font for clarity
    )

    # Show the plot
    fig.show()



plot_sales_trend_by_country(df)


plot_sales_trend_by_product(df)


fig = px.box(df, y='num_sold', title="Boxplot of num_sold", color_discrete_sequence=['blue'])
fig.show()


# Handling Outliers
q1 = df['num_sold'].quantile(0.25)
q3 = df['num_sold'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr



# Cap outliers
df['num_sold'] = np.clip(df['num_sold'], lower_bound, upper_bound)



fig = px.box(df, y='num_sold', title="Boxplot of num_sold", color_discrete_sequence=['blue'])
fig.show()



# Compute Correlation Matrix
correlation = df[['num_sold', 'year', 'month', 'day']].corr()

# Correct colorscale
fig = px.imshow(
    correlation, 
    text_auto=True, 
    color_continuous_scale='Viridis',  # Replace 'coolwarm' with 'Viridis'
    title="Correlation Heatmap"
)
fig.show()



import plotly.graph_objects as go

def plot_pivot_heatmap_with_numbers(df, index='product', columns='country', values='num_sold', aggfunc='sum', title="Sales by Product and Country"):
    """
    Plot an interactive heatmap with numerical annotations inside the cells using Plotly.
    """
    # Create the pivot table
    pivot = df.pivot_table(index=index, columns=columns, values=values, aggfunc=aggfunc)
    
    # Generate the heatmap with numbers
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale='Blues',
            colorbar=dict(title="Sales"),
            hoverongaps=False,
            zmin=0,  # Ensures a consistent color scale starting at 0
            text=pivot.values,  # Display numbers in the cells
            texttemplate="%{text:.1f}",  # Format the numbers to one decimal place
            showscale=True
        )
    )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis=dict(title="Country"),
        yaxis=dict(title="Product"),
        template="plotly_white"
    )
    
    # Show the figure
    fig.show()



# Assuming df is your DataFrame
plot_pivot_heatmap_with_numbers(df)



df.head()


numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns

categorical_cols = df.select_dtypes(include=['object']).columns



df.drop(columns=['date'], inplace=True)  # Drop the original date column


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

encoder = OneHotEncoder(sparse=False, drop='first')
encoded_cats = pd.DataFrame(encoder.fit_transform(df[categorical_cols]), columns=encoder.get_feature_names_out())
df = pd.concat([df.drop(columns=categorical_cols), encoded_cats], axis=1)


df.head()





X = df.drop(columns=["num_sold"])  # Features
y = df['num_sold']  # Target

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
numerical_cols = numerical_cols.drop('num_sold', errors='ignore')  # Exclude 'num_sold'
scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_val[numerical_cols] = scaler.transform(X_val[numerical_cols])

# Final dataset shapes
print("Training Features Shape:", X_train.shape)
print("Validation Features Shape:", X_val.shape)
print("Training Target Shape:", y_train.shape)
print("Validation Target Shape:", y_val.shape)


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Train a baseline model
baseline_model = LinearRegression()
baseline_model.fit(X_train, y_train)

# Make predictions
y_train_pred = baseline_model.predict(X_train)
y_val_pred = baseline_model.predict(X_val)

# Evaluate performance
print("Baseline Model Performance:")
print("Training R^2:", r2_score(y_train, y_train_pred))
print("Validation R^2:", r2_score(y_val, y_val_pred))
print("Validation RMSE:", mean_squared_error(y_val, y_val_pred, squared=False))
print("Validation MAE:", mean_absolute_error(y_val, y_val_pred))



from sklearn.ensemble import RandomForestRegressor

# Train Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_val_pred_rf = rf_model.predict(X_val)

# Evaluate performance
print("Random Forest Performance:")
print("Validation R^2:", r2_score(y_val, y_val_pred_rf))
print("Validation RMSE:", mean_squared_error(y_val, y_val_pred_rf, squared=False))



from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Grid Search
grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3, scoring='neg_mean_squared_error', verbose=2)
grid_search.fit(X_train, y_train)

# Best parameters and evaluation
best_rf = grid_search.best_estimator_
y_val_pred_best_rf = best_rf.predict(X_val)

print("Tuned Random Forest Performance:")
print("Validation R^2:", r2_score(y_val, y_val_pred_best_rf))
print("Validation RMSE:", mean_squared_error(y_val, y_val_pred_best_rf, squared=False))



import matplotlib.pyplot as plt
import pandas as pd

# Feature importance
importances = rf_model.feature_importances_
feature_names = X_train.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel('Feature Importance')
plt.title('Feature Importance (Random Forest)')
plt.gca().invert_yaxis()
plt.show()



from xgboost import XGBRegressor
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)
y_val_pred_xgb = xgb_model.predict(X_val)
print("XGBoost Performance:", r2_score(y_val, y_val_pred_xgb))



from lightgbm import LGBMRegressor
lgb_model = LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
lgb_model.fit(X_train, y_train)
y_val_pred_lgb = lgb_model.predict(X_val)
print("LightGBM Performance:", r2_score(y_val, y_val_pred_lgb))



import matplotlib.pyplot as plt

models = ['Linear Regression', 'Random Forest', 'XGBoost', 'LightGBM']
r2_scores = [r2_score(y_val, y_val_pred),
             r2_score(y_val, y_val_pred_rf),
             r2_score(y_val, y_val_pred_xgb),
             r2_score(y_val, y_val_pred_lgb)]

plt.figure(figsize=(10, 6))
plt.bar(models, r2_scores, color='lightcoral')
plt.title('Model Comparison (RÂ² Scores)')
plt.ylabel('RÂ² Score')
plt.show()



df.head()


# Transform date to features
df_test['date'] = pd.to_datetime(df_test['date'])
df_test['year'] = df_test['date'].dt.year
df_test['month'] = df_test['date'].dt.month
df_test['day'] = df_test['date'].dt.day
df_test['weekday'] = df_test['date'].dt.day_name()


numerical_cols = df_test.select_dtypes(include=['float64', 'int64']).columns

categorical_cols = df_test.select_dtypes(include=['object']).columns
df_test.drop(columns=['date'], inplace=True)  # Drop the original date column


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

encoder = OneHotEncoder(sparse=False, drop='first')
encoded_cats = pd.DataFrame(encoder.fit_transform(df_test[categorical_cols]), columns=encoder.get_feature_names_out())
df_test = pd.concat([df_test.drop(columns=categorical_cols), encoded_cats], axis=1)


df_test.head()


# Predict on test data
test_predictions = rf_model.predict(df_test)

# Create the submission file
submission['num_sold'] = test_predictions
submission.to_csv('submission.csv', index=False)

print("Submission file saved as 'submission.csv'.")


