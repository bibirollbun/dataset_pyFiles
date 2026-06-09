import pandas as pd  # Data manipulation and analysis
import numpy as np  # Numerical operations
import seaborn as sns  # Statistical data visualization
import plotly.express as px
import matplotlib.pyplot as plt  # Plotting and visualization
from IPython.display import display, HTML  # Displaying HTML output in Jupyter notebooks
import warnings  # Suppress warnings
from colorama import Fore, Style  # Colored terminal text

# Import classification models from sklearn
from sklearn.tree import DecisionTreeClassifier  # Decision Tree Classifier
from sklearn.ensemble import RandomForestClassifier  # Random Forest Classifier
from sklearn.svm import SVC  # Support Vector Classifier
from sklearn.neighbors import KNeighborsClassifier  # K-Nearest Neighbors Classifier

# Import LightGBM classifier
from lightgbm import LGBMClassifier  # LightGBM Classifier

from sklearn.model_selection import GridSearchCV, train_test_split  # Model selection and hyperparameter tuning
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer  # Data scaling and transformation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  # Model evaluation metrics

warnings.filterwarnings("ignore", category=FutureWarning)  # Suppress FutureWarning messages
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
warnings.filterwarnings('ignore')


df_tr = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
df_ts = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")


def styled_heading(text, background_color='#14adc6', text_color='white'):
    return f"""
    <div style="
        text-align: center;
        background: {background_color};
        font-family: 'Montserrat', sans-serif;
        color: {text_color};
        padding: 15px;
        font-size: 30px;
        font-weight: bold;
        line-height: 1;
        border-radius: 20px 20px 0 0;
        margin-bottom: 20px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.2);
        border: 3px dashed {background_color};
    ">
        {text}
    </div>
    """

def style_table(df):
    styled_df = df.style.set_table_styles([
        {"selector": "th", "props": [("color", "white"), ("background-color", "#14adc6")]}
    ]).set_properties(**{"text-align": "center"}).hide(axis="index")
    return styled_df.to_html()

def print_dataset_analysis(train_dataset, n_top=5, heading_color='#14adc6', text_color='white'):
    train_heading = styled_heading(f"ğŸ”� Top {n_top} rows of Dataset", heading_color, text_color)
    display(HTML(train_heading))
    display(HTML(style_table(train_dataset.head(n_top))))

    summary_heading = styled_heading("ğŸ“Š Summary of Dataset", heading_color, text_color)
    display(HTML(summary_heading))
    display(HTML(style_table(train_dataset.describe())))

    null_heading = styled_heading("â�Œ Null Values in Dataset", heading_color, text_color)
    train_null_count = train_dataset.isnull().sum()
    display(HTML(null_heading))
    if train_null_count.sum() == 0:
        display(HTML("<p>No null values in the training dataset.</p>"))
    else:
        display(HTML("<h3>Training Dataset:</h3>"))
        display(HTML(style_table(train_null_count[train_null_count > 0].to_frame())))
        display(HTML("<p>These are the null values.</p>"))

    duplicate_heading = styled_heading("â™»ï¸� Duplicate Values in Dataset", heading_color, text_color)
    train_duplicates = train_dataset.duplicated().sum()
    display(HTML(duplicate_heading))
    display(HTML("<h3>Training Dataset:</h3>"))
    display(HTML(f"<p>{train_duplicates} duplicate rows found.</p>"))

    shape_heading = styled_heading("ğŸ“� Number of Rows and Columns", heading_color, text_color)
    display(HTML(shape_heading))
    display(HTML("<h3>Training Dataset:</h3>"))
    display(HTML(f"<p>Rows: {train_dataset.shape[0]}, Columns: {train_dataset.shape[1]}</p>"))

def print_unique_values(train_dataset, heading_color='#14adc6', text_color='white'):
    unique_values_heading = styled_heading("ğŸ”¢ Unique Values in Dataset", heading_color, text_color)
    display(HTML(unique_values_heading))
    unique_values_table = pd.DataFrame({
        'Column Name': train_dataset.columns,
        'Data Type': [train_dataset[col].dtype for col in train_dataset.columns],
        'Unique Values': [', '.join(map(str, train_dataset[col].unique()[:7])) for col in train_dataset.columns]
    })
    display(HTML(style_table(unique_values_table)))

# Example usage with `df_tr`
print_dataset_analysis(df_tr, n_top=5, heading_color='#14adc6', text_color='white')
print_unique_values(df_tr, heading_color='#14adc6', text_color='white')



background_color = '#5fa1bc'
sns.set_theme(style="whitegrid", rc={"axes.facecolor": background_color})
plt.subplots(figsize=(10, 5))
p = sns.countplot(y="Color", data=df_tr, palette='magma', edgecolor='white', linewidth=2, width=0.7)
for container in p.containers:
    plt.bar_label(container, label_type="center", color="black", fontsize=17, weight='bold', padding=6, position=(0.5, 0.5),
                  bbox={"boxstyle": "round", "pad": 0.2, "facecolor": "white", "edgecolor": "black", "linewidth": 2, "alpha": 1})
plt.title("Color Distribution")
plt.xlabel("Count")
plt.ylabel("Color")
plt.show()


background_color = '#5fa1bc'
sns.set_theme(style="whitegrid", rc={"axes.facecolor": background_color})
plt.subplots(figsize=(10, 5))
p = sns.countplot(y="Compartments", data=df_tr, palette='magma', edgecolor='white', linewidth=2, width=0.7)
for container in p.containers:
    plt.bar_label(container, label_type="center", color="black", fontsize=17, weight='bold', padding=6, position=(0.5, 0.5),
                  bbox={"boxstyle": "round", "pad": 0.2, "facecolor": "white", "edgecolor": "black", "linewidth": 2, "alpha": 1})
plt.title("Compartments Distribution")
plt.xlabel("Count")
plt.ylabel("Compartments")
plt.show()


background_color = '#5fa1bc'
sns.set_theme(style="whitegrid", rc={"axes.facecolor": background_color})
plt.subplots(figsize=(10, 5))
p = sns.countplot(y="Style", data=df_tr, palette='magma', edgecolor='white', linewidth=2, width=0.7)
for container in p.containers:
    plt.bar_label(container, label_type="center", color="black", fontsize=17, weight='bold', padding=6, position=(0.5, 0.5),
                  bbox={"boxstyle": "round", "pad": 0.2, "facecolor": "white", "edgecolor": "black", "linewidth": 2, "alpha": 1})
plt.title("Style Distribution")
plt.xlabel("Count")
plt.ylabel("Style")
plt.show()


background_color = '#5fa1bc'
sns.set_theme(style="whitegrid", rc={"axes.facecolor": background_color})
plt.subplots(figsize=(10, 5))
p = sns.countplot(y="Waterproof", data=df_tr, palette='magma', edgecolor='white', linewidth=2, width=0.7)
for container in p.containers:
    plt.bar_label(container, label_type="center", color="black", fontsize=17, weight='bold', padding=6, position=(0.5, 0.5),
                  bbox={"boxstyle": "round", "pad": 0.2, "facecolor": "white", "edgecolor": "black", "linewidth": 2, "alpha": 1})
plt.title("Waterproof Distribution")
plt.xlabel("Count")
plt.ylabel("Brand")
plt.show()


background_color = '#5fa1bc'
sns.set_theme(style="whitegrid", rc={"axes.facecolor": background_color})
plt.subplots(figsize=(10, 5))
p = sns.countplot(y="Brand", data=df_tr, palette='magma', edgecolor='white', linewidth=2, width=0.7)
for container in p.containers:
    plt.bar_label(container, label_type="center", color="black", fontsize=17, weight='bold', padding=6, position=(0.5, 0.5),
                  bbox={"boxstyle": "round", "pad": 0.2, "facecolor": "white", "edgecolor": "black", "linewidth": 2, "alpha": 1})
plt.title("Brands Distribution")
plt.xlabel("Count")
plt.ylabel("Brand")
plt.show()


background_color = '#5fa1bc'
sns.set_theme(style="whitegrid", rc={"axes.facecolor": background_color})
plt.subplots(figsize=(10, 5))
p = sns.countplot(y="Material", data=df_tr, palette='magma', edgecolor='white', linewidth=2, width=0.7)
for container in p.containers:
    plt.bar_label(container, label_type="center", color="black", fontsize=17, weight='bold', padding=6, position=(0.5, 0.5),
                  bbox={"boxstyle": "round", "pad": 0.2, "facecolor": "white", "edgecolor": "black", "linewidth": 2, "alpha": 1})
plt.title("Material in dataset")
plt.xlabel("Count")
plt.ylabel("Material")
plt.show()


background_color = '#5fa1bc'
sns.set_theme(style="whitegrid", rc={"axes.facecolor": background_color})
plt.subplots(figsize=(10, 5))
p = sns.countplot(y="Size", data=df_tr, palette='magma', edgecolor='white', linewidth=2, width=0.7)
for container in p.containers:
    plt.bar_label(container, label_type="center", color="black", fontsize=17, weight='bold', padding=6, position=(0.5, 0.5),
                  bbox={"boxstyle": "round", "pad": 0.2, "facecolor": "white", "edgecolor": "black", "linewidth": 2, "alpha": 1})
plt.title("Size in the Dataset")
plt.xlabel("Count")
plt.ylabel("Size")
plt.show()


background_color = '#5fa1bc'
sns.set_theme(style="whitegrid", rc={"axes.facecolor": background_color})
plt.subplots(figsize=(10, 5))
p = sns.countplot(y="Laptop Compartment", data=df_tr, palette='magma', edgecolor='white', linewidth=2, width=0.7)
for container in p.containers:
    plt.bar_label(container, label_type="center", color="black", fontsize=17, weight='bold', padding=6, position=(0.5, 0.5),
                  bbox={"boxstyle": "round", "pad": 0.2, "facecolor": "white", "edgecolor": "black", "linewidth": 2, "alpha": 1})
plt.title("Laptop Compartment in the Dataset")
plt.xlabel("Count")
plt.ylabel("Laptop Compartment")
plt.show()


sns.set_theme(style="whitegrid", rc={"axes.facecolor": "#5fa1bc"})
cmap = sns.color_palette("magma", as_cmap=True)
plt.figure(figsize=(10, 6))
histplot = sns.histplot(data=df_tr, x="Weight Capacity (kg)", bins=20, palette=cmap, edgecolor='white', kde=True)
histplot.get_lines()[0].set_color("#4cc9f0")
mean_value = df_tr["Weight Capacity (kg)"].mean()
median_value = df_tr["Weight Capacity (kg)"].median()
plt.axvline(mean_value, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_value:.2f}')
plt.axvline(median_value, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_value:.2f}')
plt.title("Distribution of Age in dataset with Mean and Median")
plt.xlabel("Weight Capacity (kg)")
plt.ylabel("Count")
plt.legend()
plt.show()


sns.set_theme(style="whitegrid", rc={"axes.facecolor": "#5fa1bc"})
cmap = sns.color_palette("magma", as_cmap=True)
plt.figure(figsize=(10, 6))
histplot = sns.histplot(data=df_tr, x="Price", bins=20, palette=cmap, edgecolor='white', kde=True)
histplot.get_lines()[0].set_color("#4cc9f0")
mean_value = df_tr["Price"].mean()
median_value = df_tr["Price"].median()
plt.axvline(mean_value, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_value:.2f}')
plt.axvline(median_value, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_value:.2f}')
plt.title("Price in dataset with Mean and Median")
plt.xlabel("Price ")
plt.ylabel("Count")
plt.legend()
plt.show()


df_tr.info()


convert_object_to_category = lambda df: df.apply(lambda col: col.astype('category') if col.dtype == 'object' else col)
df_tr = convert_object_to_category(df_tr)
df_ts = convert_object_to_category(df_ts)


import pandas as pd

def target_encode(data, target_column, categorical_columns):
    """
    Perform target encoding for the specified categorical columns.
    
    Parameters:
    - data (pd.DataFrame): The input DataFrame containing the data.
    - target_column (str): The name of the target column.
    - categorical_columns (list of str): The list of categorical columns to encode.
    
    Returns:
    - pd.DataFrame: The DataFrame with target encoded categorical columns.
    """
    
    # Copy the original DataFrame to avoid modifying it directly
    data_encoded = data.copy()
    
    for col in categorical_columns:
        # Compute the mean of the target variable for each category
        means = data.groupby(col)[target_column].mean()
        
        # Map the means to the corresponding categories in the original DataFrame
        data_encoded[col] = data[col].map(means)
        
        # Optionally, you can add a suffix to the encoded columns to identify them
        data_encoded.rename(columns={col: f"{col}_target_encoded"}, inplace=True)
    
    return data_encoded

# Example usage
target_column = 'Price'
categorical_columns = ['Brand', 'Material', 'Size','Laptop Compartment','Waterproof','Style','Color']
data_encoded = target_encode(df_tr, target_column, categorical_columns)
print(data_encoded.head())


df_tr.info()


df_tr.isnull().sum()


data_encoded.info()


import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

def train_lightgbm_model_cv(data, target_column, n_splits=10):
    # Convert object columns to category
    data = data.apply(lambda col: col.astype('category') if col.dtype == 'object' else col)
    
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Tuned hyperparameters
    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': -1,
        'min_data_in_leaf': 20,
        'bagging_fraction': 0.8,
        'feature_fraction': 0.8,
        'bagging_freq': 5,
        'verbosity': -1
    }
    
    # Initialize KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    rmses = []
    
    for fold, (train_index, test_index) in enumerate(kf.split(X), 1):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Create LightGBM dataset
        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        # Train the model
        model = lgb.train(params, train_data, valid_sets=[test_data])
        
        # Predict and evaluate
        y_pred = model.predict(X_test, num_iteration=model.best_iteration)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        rmses.append(rmse)
        print(f"Fold {fold}: RMSE = {rmse:.4f}")
    
    avg_rmse = np.mean(rmses)
    print(f"Average RMSE: {avg_rmse:.4f}")
    
    return model, avg_rmse

# Example usage
# data = pd.read_csv('your_dataset.csv')
model, avg_rmse = train_lightgbm_model_cv(data_encoded, 'Price')



df_ts.info()


# Load the test data
X_test = df_ts

# Load the submission IDs file
submission_ids = pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')  

# Predict on the test set
y_pred = model.predict(X_test)

# Create a DataFrame for submission
submission_df = pd.DataFrame({'id': submission_ids['id'], 'Price': y_pred})

# Save the submission DataFrame to a CSV file
submission_df.to_csv('submission.csv', index=False)

