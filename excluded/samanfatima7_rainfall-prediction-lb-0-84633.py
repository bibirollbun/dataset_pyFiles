import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.express as px
from IPython.display import display, HTML
import warnings
from colorama import Fore, Style

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



df_tr = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
df_ts = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
sample = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')


from IPython.display import display, HTML

def styled_heading(text, background_color='#14adc6', text_color='white'):
    return f"""
    <p style="
        background-color: {background_color};
        font-family: Pacifico, cursive;
        font-size: 150%;
        color: {text_color};
        text-align: center;
        border-radius: 10px;
        padding: 10px;
        font-weight: normal;
        box-shadow: 0px 2px 4px rgba(0, 0, 0, 0.2);
        width: fit-content;
        margin: 0 auto;
    ">
        {text}
    </p>
    """

def style_table(df):
    styled_df = df.style.set_table_styles([
        {"selector": "th", "props": [("color", "white"), ("background-color", "#14adc6")]}
    ]).set_properties(**{"text-align": "center"}).hide(axis="index")
    return styled_df.to_html()

def print_dataset_analysis(train_dataset, n_top=5, heading_color='#14adc6', text_color='white'):
    train_heading = styled_heading(f"📊 Basic Overview of Data", heading_color, text_color)
    display(HTML(train_heading))
    display(HTML(style_table(train_dataset.head(n_top))))

    summary_heading = styled_heading("🔍 Data Summary", heading_color, text_color)
    display(HTML(summary_heading))
    display(HTML(style_table(train_dataset.describe())))

    null_heading = styled_heading("🚫 Null Values in Data", heading_color, text_color)
    train_null_count = train_dataset.isnull().sum()
    display(HTML(null_heading))
    if train_null_count.sum() == 0:
        display(HTML("<p>No null values in the dataset.</p>"))
    else:
        display(HTML("<h3>Null Values:</h3>"))
        display(HTML(style_table(train_null_count[train_null_count > 0].to_frame())))
        display(HTML("<p>These are the null values.</p>"))

    duplicate_heading = styled_heading("♻️ Duplicate Values in Data", heading_color, text_color)
    train_duplicates = train_dataset.duplicated().sum()
    display(HTML(duplicate_heading))
    display(HTML("<h3>Duplicates:</h3>"))
    display(HTML(f"<p>{train_duplicates} duplicate rows found.</p>"))

    shape_heading = styled_heading("📏 Data Shape", heading_color, text_color)
    display(HTML(shape_heading))
    display(HTML("<h3>Shape:</h3>"))
    display(HTML(f"<p>Rows: {train_dataset.shape[0]}, Columns: {train_dataset.shape[1]}</p>"))

def print_unique_values(train_dataset, heading_color='#14adc6', text_color='white'):
    unique_values_heading = styled_heading("🔢 Unique Values in Data", heading_color, text_color)
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



def summary_statistics(df):
    """Returns mean, median, min, and max of all numeric columns."""
    return df.agg(['mean', 'median', 'min', 'max'])

# Example usage:
summary_stats = summary_statistics(df_tr)
print(summary_stats)


# Compute Standard Deviation, Skewness, Kurtosis
stats = df_tr.describe().T
stats['std_dev'] = df_tr.std()
stats['skewness'] = df_tr.skew()
stats['kurtosis'] = df_tr.kurtosis()

# Display statistics
print(stats[['mean', 'std_dev', 'skewness', 'kurtosis']])



# Split data by Rainfall (0 vs. 1)
plt.figure(figsize=(12, 6))
sns.boxplot(x=df_tr["rainfall"], y=df_tr["humidity"], palette="rainbow")
plt.title("Humidity Levels on Rainy vs. Non-Rainy Days 🌧️☀️")
plt.xlabel("Rainfall (0 = No Rain, 1 = Rain)")
plt.ylabel("Humidity (%)")
plt.show()

# Repeat for Temperature
plt.figure(figsize=(12, 6))
sns.boxplot(x=df_tr["rainfall"], y=df_tr["temparature"], palette="rainbow")
plt.title("Temperature on Rainy vs. Non-Rainy Days 🌧️☀️")
plt.xlabel("Rainfall (0 = No Rain, 1 = Rain)")
plt.ylabel("Temperature (°C)")
plt.show()



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define a rainbow color palette
rainbow_palette = sns.color_palette("rainbow", as_cmap=True)

# Select relevant features
columns_to_check = ['pressure', 'maxtemp', 'temparature', 'mintemp',
       'dewpoint', 'humidity', 'cloud', 'sunshine', 'winddirection',
       'windspeed', 'rainfall']

# Create histograms with KDE
plt.figure(figsize=(15, 12))
for i, col in enumerate(columns_to_check, 1):
    plt.subplot(4, 3, i)
    sns.histplot(df_tr[col], kde=True, color=rainbow_palette(i / len(columns_to_check)))
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Density")
plt.tight_layout()
plt.show()



plt.figure(figsize=(12, 8))
sns.heatmap(df_tr[columns_to_check].corr(), annot=True, cmap="rainbow", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap 🌧️")
plt.show()


plt.figure(figsize=(12, 6))

# Boxplot for Temperature
plt.subplot(1, 2, 1)
sns.boxplot(x=df_tr["rainfall"], y=df_tr["temparature"], palette="rainbow")
plt.title("Effect of Rainfall on Temperature")

# Boxplot for Humidity
plt.subplot(1, 2, 2)
sns.boxplot(x=df_tr["rainfall"], y=df_tr["humidity"], palette="rainbow")
plt.title("Effect of Rainfall on Humidity")

plt.tight_layout()
plt.show()



# ============================
# 🌟 Step 1: Prepare Features & Target
# ============================
X = df_tr.drop(columns=["rainfall"])  # Features
y = df_tr["rainfall"]  # Target variable

# Split into train-test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ============================
# 🌟 Step 2: Define Stable CatBoost Hyperparameters
# ============================
stable_params = {
    "iterations": 500,
    "depth": 6,
    "learning_rate": 0.05,
    "l2_leaf_reg": 3,
    "border_count": 128,
    "bagging_temperature": 0.8,
    "random_strength": 1.5,
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "verbose": 100
}

# ============================
# 🌟 Step 3: Train Model Using Stratified K-Fold Cross-Validation
# ============================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize CatBoostClassifier
model = CatBoostClassifier(**stable_params)

# Perform Cross-Validation
cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="roc_auc", n_jobs=-1)

# Train Final Model on Full Training Set
model.fit(X_train, y_train)

# ============================
# 🌟 Step 4: Predict Probabilities
# ============================
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Get probability scores for class 1

# Compute AUC-ROC Score
auc = roc_auc_score(y_test, y_pred_proba)

# Print Results
print(f"\n🔥 Cross-Validation AUC-ROC: {cv_scores.mean():.4f}")
print(f"🔥 Final Model AUC-ROC on Test Data: {auc:.4f}")
print("\n🔍 Classification Report:")
print(classification_report(y_test, model.predict(X_test)))



test_preds = model.predict_proba(df_ts)[:, 1] # ✅ This gives probabilities
sample["rainfall"] = test_preds
sample.to_csv("submission_prob_3.csv", index=False)
print(sample.shape)
sample.head()

