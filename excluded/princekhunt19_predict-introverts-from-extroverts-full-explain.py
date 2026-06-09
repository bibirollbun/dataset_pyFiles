import numpy as np
import pandas as pd
import os
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb


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


train_df = load_csv("/kaggle/input/playground-series-s5e7/train.csv")
test_df = load_csv("/kaggle/input/playground-series-s5e7/test.csv")
sample_submission_df = load_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")


def get_basic_info(df):
    """Get basic information about the dataset"""
    print(f"ðŸ”¹ Number of Rows: {df.shape[0]}")
    print(f"ðŸ”¹ Number of Columns: {df.shape[1]}")
    print(f"ðŸ”¹ Columns: {df.columns.tolist()}")
    print(f"ðŸ”¹ Column Data Types: {df.dtypes}")
    print(f"ðŸ”¹ Missing Values: {df.isnull().sum().sum()}")
    print(f"ðŸ”¹ Missing Values (Per Column) : {df.isnull().sum()}")
    print(f"ðŸ”¹ Missing Values (Per Column Percentage) : {(df.isnull().mean() * 100)}")
    print(f"ðŸ”¹ Unique Values: {df.nunique()}")


get_basic_info(train_df)


def plot_missing_bar(df):
    """Plot a bar chart showing the count of missing values per column."""

    missing_values = df.isnull().sum()
    missing_values = missing_values[missing_values > 0].sort_values(ascending=False)

    if missing_values.empty:
        print("âœ… No missing values to plot.")
        return

    missing_df = pd.DataFrame({
        'column': missing_values.index,
        'count': missing_values.values
    })

    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Verdana"]
    plt.figure(figsize=(12, 6), dpi=150)

    sns.barplot(
        data=missing_df,
        x='column',
        y='count',
        hue='column',  
        palette='magma',
        dodge=False
    )

    plt.xticks(rotation=90)
    plt.title("Missing Values Count per Column")
    plt.ylabel("Count")
    plt.xlabel("Columns")
    plt.tight_layout()
    plt.show()


plot_missing_bar(train_df)


missing_columns = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance', 'Going_outside', 'Drained_after_socializing', 'Friends_circle_size', 'Post_frequency']

for col in missing_columns:
    train_df[col] = train_df[col].fillna(train_df[col].mode()[0])


plot_missing_bar(train_df)


def classify_columns(df, categorical_threshold=20):
    """
    Classify columns into binary, numerical, and categorical types.
    Automatically treat integer columns with few unique values as categorical.
    """
    binary_cols = []
    numerical_cols = []
    categorical_cols = []
    
    for col in df.columns:
        unique_vals = df[col].nunique()
        dtype = df[col].dtype
        
        if unique_vals == 2:
            binary_cols.append(col)
        elif dtype == 'object' or dtype.name == 'category':
            categorical_cols.append(col)
        elif dtype == 'bool':
            binary_cols.append(col)
        elif dtype in ['int64', 'float64']:
            if unique_vals <= categorical_threshold:
                categorical_cols.append(col)
            else:
                numerical_cols.append(col)
    
    return {
        'binary': binary_cols,
        'numerical': numerical_cols,
        'categorical': categorical_cols
    }


classify_columns(train_df)


cols_classification = {'binary': ['Stage_fear', 'Drained_after_socializing', 'Personality'],
 'numerical': [],
 'categorical': ['Time_spent_Alone',
  'Social_event_attendance',
  'Going_outside',
  'Friends_circle_size',
  'Post_frequency']}


def plot_categorical_distributions(df, categorical_cols):
    """Plot count plots for all categorical and binary columns."""
    num_plots = len(categorical_cols)
    num_rows = (num_plots // 3) + (1 if num_plots % 3 != 0 else 0)  
    
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Verdana"]
    
    fig, axes = plt.subplots(num_rows, 3, figsize=(15, num_rows * 5), dpi=150)
    axes = axes.flatten()
    
    for i, col in enumerate(categorical_cols):
        sns.countplot(data=df, x=col, hue=col, palette='magma', ax=axes[i], dodge=False)
        axes[i].set_title(f"Count Plot of {col}")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Count")
        axes[i].tick_params(axis='x', rotation=90)
    
    for i in range(num_plots, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


all_categorical_cols = cols_classification['binary'] + cols_classification['categorical']
plot_categorical_distributions(train_df, all_categorical_cols)


train_df = train_df.drop(['id'], axis=1)


cols_classification


train_df['Stage_fear'] = train_df['Stage_fear'].map({'No':0, 'Yes':1})
train_df['Drained_after_socializing'] = train_df['Drained_after_socializing'].map({'No':0, 'Yes':1})
train_df['Personality'] = train_df['Personality'].map({'Introvert':0, 'Extrovert':1})


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


X_train, X_test, y_train, y_test = train_test_split_custom(train_df, "Personality", test_size=0.1)


X_train.head()


X_test.head()


y_train.head()


y_test.head()


def train_models_classification(X_train, X_test, y_train, y_test):
    """Train and compare multiple classification models"""
    models = {
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "XGBoost": xgb.XGBClassifier(),
        "SVM": SVC(),
        "Naive Bayes": GaussianNB()
    }

    print("ðŸ”¹ðŸ”¹ðŸ”¹ðŸ”¹ðŸ”¹ðŸ”¹ðŸ”¹ðŸ”¹ðŸ”¹ðŸ”¹ðŸ”¹ðŸ”¹ðŸ”¹ðŸ”¹ðŸ”¹ðŸ”¹ðŸ”¹ðŸ”¹ðŸ”¹ðŸ”¹ðŸ”¹ðŸ”¹")
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary')
        recall = recall_score(y_test, y_pred, average='binary')
        f1 = f1_score(y_test, y_pred, average='binary')

        print("ðŸ”¹ðŸ”¹ðŸ”¹ðŸ”¹ðŸ”¹ðŸ”¹ðŸ”¹ðŸ”¹ðŸ”¹ðŸ”¹ðŸ”¹ðŸ”¹ðŸ”¹ðŸ”¹ðŸ”¹ðŸ”¹ðŸ”¹ðŸ”¹ðŸ”¹ðŸ”¹ðŸ”¹ðŸ”¹")
        print(f"{name}")
        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1-Score : {f1:.4f}")
        print("ðŸ”¹ðŸ”¹ðŸ”¹ðŸ”¹ðŸ”¹ðŸ”¹ðŸ”¹ðŸ”¹ðŸ”¹ðŸ”¹ðŸ”¹ðŸ”¹ðŸ”¹ðŸ”¹ðŸ”¹ðŸ”¹ðŸ”¹ðŸ”¹ðŸ”¹ðŸ”¹ðŸ”¹ðŸ”¹")

    print("ðŸ”¹ðŸ”¹ðŸ”¹ðŸ”¹ðŸ”¹ðŸ”¹ðŸ”¹ðŸ”¹ðŸ”¹ðŸ”¹ðŸ”¹ðŸ”¹ðŸ”¹ðŸ”¹ðŸ”¹ðŸ”¹ðŸ”¹ðŸ”¹ðŸ”¹ðŸ”¹ðŸ”¹ðŸ”¹")


train_models_classification(X_train, X_test, y_train, y_test)


X__, _, y__, _ = train_test_split_custom(train_df, "Personality", test_size=0.0000001)


final_model = GradientBoostingClassifier()
final_model = final_model.fit(X__, y__)


joblib.dump(final_model, 'final_gradient_boosting_model.pkl')


get_basic_info(test_df)


test_df['Stage_fear'] = test_df['Stage_fear'].map({'No':0, 'Yes':1})
test_df['Drained_after_socializing'] = test_df['Drained_after_socializing'].map({'No':0, 'Yes':1})


fill_values = {}

for col in train_df.columns:
    fill_values[col] = train_df[col].mode()[0]


fill_values


test_df = test_df.fillna(fill_values)


plot_missing_bar(test_df)


test_df.head()


test_df = test_df.drop(['id'], axis=1)


preds = final_model.predict(test_df)


submission_df = sample_submission_df[["id"]].copy()
submission_df["Personality"] = preds


submission_df.to_csv("submission.csv", index=False)


submission_df = load_csv("/kaggle/working/submission.csv")


submission_df['Personality'].value_counts().plot(kind='bar', color='black', figsize=(8,5))
plt.title("Distribution of Personality in Submission Data")
plt.xlabel("Personality Type")
plt.ylabel("Count")
plt.show()


submission_df['Personality'] = submission_df['Personality'].map({0: 'Introvert', 1: 'Extrovert'})
submission_df.to_csv("submission.csv", index=False)


submission_df = load_csv("/kaggle/working/submission.csv")


submission_df.head()

