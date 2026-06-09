# import libaraies for binary classification
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.express as px
from IPython.display import display, HTML
import warnings
from colorama import Fore, Style
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, roc_auc_score
from scipy.stats import randint
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings('ignore')

import plotly.io as pio
import plotly.graph_objects as go
import plotly.express as px

# Set the default renderer for both Plotly Express and Graph Objects
pio.renderers.default = 'iframe_connected'
import joblib


df_tr = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
df_ts = pd.read_csv ("/kaggle/input/playground-series-s5e8/test.csv")
sample= pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")


import pandas as pd
from IPython.display import display, HTML

# Load dataset
df_tr = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")

# Styled heading
def styled_heading(text, background_color='#800080', text_color='white'):
    return f"""
    <p style="
        background-color: {background_color};
        font-family: Roboto, sans-serif;
        font-size: 150%;
        color: {text_color};
        text-align: center;
        border-radius: 10px;
        padding: 10px;
        font-weight: bold;
        box-shadow: 0px 2px 4px rgba(0, 0, 0, 0.2);
        width: fit-content;
        margin: 20px auto;
    ">
        {text}
    </p>
    """

# Style the table
def style_table(df):
    styled_df = df.style.set_table_styles([
        {"selector": "th", "props": [("color", "white"), ("background-color", "#800080"), ("font-family", "Roboto, sans-serif")]}
    ]).set_properties(**{
        "text-align": "center",
        "font-family": "Roboto, sans-serif"
    }).hide(axis="index")
    return styled_df.to_html()

# Dataset overview
def print_dataset_analysis(train_dataset, n_top=5, heading_color='#800080', text_color='white'):
    display(HTML(styled_heading("ğŸ“Š Basic Overview of Data", heading_color, text_color)))
    display(HTML(style_table(train_dataset.head(n_top))))

    display(HTML(styled_heading("ğŸ”� Data Summary", heading_color, text_color)))
    display(HTML(style_table(train_dataset.describe())))

    display(HTML(styled_heading("ğŸš« Null Values in Data", heading_color, text_color)))
    train_null_count = train_dataset.isnull().sum()
    if train_null_count.sum() == 0:
        display(HTML("<p style='font-family: Roboto, sans-serif; text-align:center;'>âœ… No null values in the dataset.</p>"))
    else:
        display(HTML(style_table(train_null_count[train_null_count > 0].to_frame(name='Null Count'))))

    display(HTML(styled_heading("â™»ï¸� Duplicate Values in Data", heading_color, text_color)))
    train_duplicates = train_dataset.duplicated().sum()
    display(HTML(f"<p style='font-family: Roboto, sans-serif; text-align:center;'>ğŸ”� {train_duplicates} duplicate rows found.</p>"))

    display(HTML(styled_heading("ğŸ“� Data Shape", heading_color, text_color)))
    display(HTML(f"<p style='font-family: Roboto, sans-serif; text-align:center;'>ğŸ“� Rows: {train_dataset.shape[0]}, Columns: {train_dataset.shape[1]}</p>"))

# Unique value summary
def print_unique_values(train_dataset, heading_color='#800080', text_color='white'):
    display(HTML(styled_heading("ğŸ”¢ Unique Values in Data", heading_color, text_color)))
    unique_values_table = pd.DataFrame({
        'Column Name': train_dataset.columns,
        'Data Type': [train_dataset[col].dtype for col in train_dataset.columns],
        'Unique Sample Values': [', '.join(map(str, train_dataset[col].unique()[:5])) + (' ...' if train_dataset[col].nunique() > 5 else '') for col in train_dataset.columns]
    })
    display(HTML(style_table(unique_values_table)))

# Run analysis
print_dataset_analysis(df_tr, n_top=5)
print_unique_values(df_tr)



def summary_statistics(df):
    """
    Returns summary statistics (mean, median, min, max) for all numeric columns in the dataframe.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: A DataFrame containing mean, median, min, and max for numeric columns only.
    """
    # Select numeric columns only
    numeric_df = df.select_dtypes(include='number')
    
    # Compute summary statistics
    stats = numeric_df.agg(['mean', 'median', 'min', 'max']).transpose()
    
    # Round for better readability
    return stats.round(2)

# Example usage:
summary_stats = summary_statistics(df_tr)
display(summary_stats)  # better than print in notebooks



# ğŸ“Š Compute Descriptive Statistics on Numeric Columns Only
numeric_df = df_tr.select_dtypes(include='number')  # select numeric columns

stats = numeric_df.describe().T.copy()
stats['std_dev'] = numeric_df.std()
stats['skewness'] = numeric_df.skew()
stats['kurtosis'] = numeric_df.kurtosis()

# ğŸ�¯ Filter and round selected stats
selected_stats = stats[['mean', 'std_dev', 'skewness', 'kurtosis']].round(3)

# ğŸ“Œ Display with purple gradient and custom styling
from IPython.display import display

display(selected_stats.style
        .set_caption("ğŸ“Œ Summary Statistics: Mean, Std Dev, Skewness, Kurtosis (Numeric Only)")
        .background_gradient(cmap='Purples')  # keep purple gradient
        .set_properties(**{
            'border': '1px solid #ccc',
            'padding': '8px',
            'border-radius': '5px',
            'text-align': 'center',
            'font-family': 'Verdana'
        })
        .set_table_styles([{
            'selector': 'caption',
            'props': [
                ('color', '#4B0082'),
                ('font-size', '16px'),
                ('font-weight', 'bold'),
                ('text-align', 'center'),
                ('margin-bottom', '10px')
            ]
        }])
)



# Correlation heatmap for numerical features
plt.figure(figsize=(12, 6))
sns.heatmap(df_tr.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap of Numerical Features")
plt.show()


# Distribution of target variable
fig = px.histogram(df_tr, x='y', color='y', title='Distribution of Target Variable (y)')
fig.show()



# Age distribution
fig = px.histogram(df_tr, x='age', nbins=30, title='Age Distribution')
fig.show()


job_counts = df_tr['job'].value_counts().reset_index()
job_counts.columns = ['job', 'count']

fig = px.bar(job_counts, x='job', y='count',
             labels={'job': 'Job', 'count': 'Count'}, title='Job Distribution')
fig.show()



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df_tr = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")

# Define the numerical columns for visualization
numeric_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

# Set up the color map
coolwarm_cmap = plt.cm.coolwarm

# Create histograms with KDE for each numerical column
plt.figure(figsize=(15, 12))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(3, 3, i)
    color = coolwarm_cmap(i / len(numeric_cols))
    
    sns.histplot(df_tr[col], kde=True, color=color)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Density")

plt.tight_layout()
plt.show()


# Define numerical columns for boxplots
numeric_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

# Set the plotting style
sns.set(style="whitegrid")
plt.figure(figsize=(18, 20))

# Create boxplots of each numeric feature grouped by the target column 'y'
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(4, 2, i)
    sns.boxplot(x='y', y=col, data=df_tr, palette="Set2")
    plt.title(f"{col} vs Target (y)", fontsize=13)
    plt.xlabel("Target (y)")
    plt.ylabel(col)

plt.tight_layout()
plt.show()


from catboost import CatBoostClassifier, Pool
# âœ… Define target and features
target = 'y'
X = df_tr.drop(columns=['id', target])
y = df_tr[target]

# âœ… Identify categorical columns (CatBoost handles these natively)
cat_features = X.select_dtypes(include='object').columns.tolist()

# âœ… Split data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# âœ… Create Pool for CatBoost
train_pool = Pool(X_train, y_train, cat_features=cat_features)
valid_pool = Pool(X_valid, y_valid, cat_features=cat_features)

# âœ… Initialize and train CatBoost
model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.1,
    depth=6,
    eval_metric='Accuracy',
    early_stopping_rounds=50,
    verbose=100,
    random_state=42
)

model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

# âœ… Predict and evaluate
y_pred = model.predict(X_valid)
print("\nğŸ“Š Classification Report:")
print(classification_report(y_valid, y_pred))


# ============================
# ğŸŒŸ Predict on Test Data
# ============================

# âœ… Create test pool (if not already created)
X_test = df_ts.drop(columns=['id'])
test_pool = Pool(X_test, cat_features=cat_features)

# âœ… Predict labels (0 or 1)
test_preds = model.predict(test_pool)

# âœ… Fill submission file
submission_df = pd.DataFrame({
    "id": df_ts["id"],
    "y": test_preds.astype(int)  # Ensure binary output
})

# âœ… Save to CSV
submission_filename = "bank_binary_classification_submission.csv"
submission_df.to_csv(submission_filename, index=False)

# âœ… Show confirmation
print(f"\nâœ… Submission file saved as: {submission_filename}")
display(submission_df.head())

