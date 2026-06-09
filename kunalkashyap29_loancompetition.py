# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import warnings
warnings.filterwarnings('ignore')


loan_df = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")


loan_df


loan_df.shape


loan_df.info()


loan_df.describe()


loan_df.isnull().sum()


loan_df.columns


loan_df


df = loan_df.copy()


df


df['loan_paid_back'] = df['loan_paid_back'].astype(int)


df['loan_paid_back'].value_counts()


df.info()


df['annual_income']


df['annual_income'].describe()


df['annual_income'].plot(kind="hist")


df['annual_income'].plot(kind="kde")


df['annual_income'].plot(kind="box")


df['annual_income'].skew()


df['annual_income'].isnull().sum()


import seaborn as sns
import matplotlib.pyplot as plt


def univariate_analysis(df, column_name):
    """
    Perform univariate analysis for a given numerical column.
    Shows:
      1. Summary statistics
      2. Null count
      3. Skewness
      4. Histogram
      5. KDE Plot
      6. Boxplot
    """
    print(f"ğŸ“Š Univariate Analysis for: {column_name}")
    print("="*60)
    
    # 1ï¸�âƒ£ Basic Info
    print("\nâ�¡ï¸� Summary Statistics:")
    print(df[column_name].describe())
    
    # 2ï¸�âƒ£ Missing values
    null_count = df[column_name].isnull().sum()
    print(f"\nğŸš« Missing Values: {null_count}")
    
    # 3ï¸�âƒ£ Skewness
    skew_value = df[column_name].skew()
    print(f"ğŸ“ˆ Skewness: {skew_value:.3f}")
    
    # 4ï¸�âƒ£ Plotting
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Histogram
    sns.histplot(df[column_name], kde=False, ax=axes[0], color='blue')
    axes[0].set_title(f"Histogram of {column_name}")
    
    # KDE Plot
    sns.kdeplot(df[column_name], ax=axes[1], color='orange')
    axes[1].set_title(f"KDE Plot of {column_name}")
    
    # Boxplot
    sns.boxplot(x=df[column_name], ax=axes[2], color='lightgreen')
    axes[2].set_title(f"Boxplot of {column_name}")
    
    plt.tight_layout()
    plt.show()


univariate_analysis(df,'debt_to_income_ratio')


univariate_analysis(df,'credit_score')


univariate_analysis(df,'loan_amount')


univariate_analysis(df,'interest_rate')


df['loan_paid_back']


df['loan_paid_back'].value_counts()


df['loan_paid_back'].value_counts().plot(kind="bar")


df['loan_paid_back'].value_counts().plot(kind="pie",autopct="%0.1f%%")


df['loan_paid_back'].isnull().sum()


def cat_summary(df, column):
    print(f"ğŸ“Š Univariate Analysis of '{column}'")
    print("=" * 60)

    # Step 1: Display the column data type
    print(f"Data Type: {df[column].dtype}\n")

    # Step 2: Value counts
    print("ğŸ”¹ Value Counts:")
    print(df[column].value_counts())
    
    print("\nğŸ”¹ Value Counts (in %):")
    print(df[column].value_counts(normalize=True) * 100)

    # Step 3: Missing values
    print("\nâ�Œ Missing Values:", df[column].isnull().sum())

    # Step 4: Plot Bar and Pie charts
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Bar Plot
    sns.countplot(x=df[column], ax=axes[0], palette="Set2")
    axes[0].set_title(f"Bar Plot of {column}")
    axes[0].set_xlabel(column)
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis='x', rotation=45)

    # Pie Chart
    df[column].value_counts().plot(
        kind="pie",
        autopct="%0.1f%%",
        ax=axes[1],
        colors=sns.color_palette("Set2", len(df[column].unique())),
        startangle=90,
        wedgeprops={'edgecolor': 'black'}
    )
    axes[1].set_ylabel("")  # remove y-label
    axes[1].set_title(f"Pie Chart of {column}")

    plt.tight_layout()
    plt.show()


cat_summary(df, 'gender')


cat_summary(df, 'marital_status')


cat_summary(df, 'education_level')


cat_summary(df, 'loan_purpose')


cat_summary(df, 'employment_status')


cat_summary(df, 'grade_subgrade')


df


pd.crosstab(df['loan_paid_back'],df['gender'])


sns.heatmap(pd.crosstab(df['loan_paid_back'],df['gender']))


def cat_bivariate(df, col1, col2, normalize=False):
    """
    Perform bivariate analysis between two categorical columns.
    Shows crosstab and heatmap.

    Parameters:
    df : DataFrame
    col1 : str -> First categorical column (e.g., target column)
    col2 : str -> Second categorical column (e.g., feature)
    normalize : bool -> If True, show proportions (%) instead of raw counts
    """

    print(f"ğŸ”� Bivariate Analysis between '{col1}' and '{col2}'")
    print("=" * 70)

    # Create cross-tab
    if normalize:
        ct = pd.crosstab(df[col1], df[col2], normalize='index') * 100
    else:
        ct = pd.crosstab(df[col1], df[col2])

    # Display table
    print("\nğŸ“Š Crosstab Table:")
    print(ct.round(2))

    # Plot heatmap
    plt.figure(figsize=(10, 5))
    sns.heatmap(ct, annot=True, cmap="YlGnBu", fmt=".1f" if normalize else "d")
    plt.title(f"Heatmap of {col1} vs {col2}")
    plt.xlabel(col2)
    plt.ylabel(col1)
    plt.show()


cat_bivariate(df, 'loan_paid_back', 'gender', normalize=True)


cat_bivariate(df, 'loan_paid_back', 'marital_status')


cat_bivariate(df, 'loan_paid_back', 'marital_status', normalize=True)


cat_bivariate(df, 'loan_paid_back', 'employment_status')


cat_bivariate(df, 'loan_paid_back', 'employment_status', normalize=True)


cat_bivariate(df, 'loan_paid_back', 'loan_purpose')


cat_bivariate(df, 'loan_paid_back', 'loan_purpose', normalize=True)


plt.figure(figsize=(8, 5))
sns.scatterplot(x='annual_income',y='loan_amount',data=df,
    alpha=0.5,            # transparency for dense data
    color='teal',          # pleasant color
    s=10                   # smaller points for large dataset
)
plt.title("Relationship between Annual Income and Loan Amount")
plt.xlabel("Annual Income")
plt.ylabel("Loan Amount")
plt.show()


df['annual_income'].corr(df['loan_amount'])


plt.figure(figsize=(8, 5))
sns.scatterplot(x='credit_score',y='loan_amount',data=df,
    alpha=0.5,            # transparency for dense data
    color='teal',          # pleasant color
    s=10                   # smaller points for large dataset
)
plt.title("Relationship between Credit Score and Loan Amount")
plt.xlabel("Credit Score")
plt.ylabel("Loan Amount")
plt.show()


df['credit_score'].corr(df['loan_amount'])


plt.figure(figsize=(8, 5))
sns.scatterplot(x='debt_to_income_ratio',y='loan_amount',data=df,
    alpha=0.5,            # transparency for dense data
    color='teal',          # pleasant color
    s=10                   # smaller points for large dataset
)
plt.title("Relationship between Credit Score and Loan Amount")
plt.xlabel("debt_to_income_ratio")
plt.ylabel("Loan Amount")
plt.show()


df['debt_to_income_ratio'].corr(df['loan_amount'])


df


sns.boxplot(x="loan_paid_back",y="credit_score",data=df)


sns.barplot(x="gender",y="credit_score",data=df)


corr_heatmap = df[['credit_score', 'loan_amount', 'annual_income', 'debt_to_income_ratio']].corr()


corr_heatmap


plt.figure(figsize=(8,5))
sns.heatmap(corr_heatmap, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()


sns.pairplot(df[['credit_score', 'loan_amount', 'annual_income', 'debt_to_income_ratio']], diag_kind="kde", plot_kws={'alpha':0.5, 's':20})


model_df = df.copy()


model_df


model_df = model_df.drop(columns=['id','marital_status','education_level'],axis=1)


model_df


from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from lightgbm import LGBMClassifier  # âœ… Fast & powerful


numeric_columns = ['annual_income','debt_to_income_ratio','credit_score','loan_amount','interest_rate'] 
categorical_columns = ['gender','employment_status','loan_purpose','grade_subgrade']


def expand_grade_subgrade(df):
    if 'grade_subgrade' in df.columns:
        df['grade'] = df['grade_subgrade'].astype(str).str[0]
        # handle entries like 'C3' -> 3, or missing gracefully
        df['subgrade'] = df['grade_subgrade'].astype(str).str[1:].replace({'': np.nan}).astype(float)
        # keep grade as categorical and subgrade as numeric
        df.drop(columns=['grade_subgrade'], inplace=True)
    return df


model_df = expand_grade_subgrade(model_df)


model_df


def cap_outliers_iqr(df, numeric_cols, factor=1.5, verbose=False):
    """
    Caps outliers using IQR method with np.where() instead of np.clip().

    Parameters:
        df (pd.DataFrame): Input DataFrame
        numeric_cols (list): Numeric columns to cap
        factor (float): IQR factor, default 1.5
        verbose (bool): Print limits info for each column
    
    Returns:
        pd.DataFrame: DataFrame with capped numeric columns
    """
    for col in numeric_cols:
        
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR

        # Replace values below lower bound
        df[col] = np.where(df[col] < lower, lower, df[col])
        # Replace values above upper bound
        df[col] = np.where(df[col] > upper, upper, df[col])
        
        if verbose:
            print(f"{col}: lower={lower:.2f}, upper={upper:.2f}, capped {((df[col] == lower) | (df[col] == upper)).sum()} values")
    
    return df


model_df = cap_outliers_iqr(model_df, numeric_columns, factor=1.5, verbose=True)


model_df


# Feature and target split
X = model_df.drop("loan_paid_back", axis=1)
y = model_df["loan_paid_back"]


# Define column types
numeric_cols = ['annual_income', 'debt_to_income_ratio', 'credit_score', 'loan_amount', 'interest_rate', 'subgrade']
categorical_cols = ['gender', 'employment_status', 'loan_purpose', 'grade']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# Numeric transformer
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])

# Categorical transformer
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Combine transformers
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ],
    n_jobs=-1  # parallel processing
)


lgb_model = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=10,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)


model_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", lgb_model)
])


model_pipeline.fit(X_train, y_train)


y_pred = model_pipeline.predict(X_test)

print("\nâœ… Accuracy:", accuracy_score(y_test, y_pred)*100)
print("\nğŸ“Š Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Paid (0)", "Paid (1)"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - LightGBM")
plt.show()


import joblib
joblib.dump(model_pipeline, "loan_model_lgbm.pkl")


test_df = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")


test_df


id = test_df['id']


model_pipeline = joblib.load("loan_model_lgbm.pkl")


# Extract 'grade' and 'subgrade' from 'grade_subgrade'
test_df['grade'] = test_df['grade_subgrade'].str[0]  # first character
test_df['subgrade'] = test_df['grade_subgrade'].str[1:].astype(float)  # remaining digits


# Drop extra columns not used in model
test_df = test_df.drop(['id', 'marital_status', 'education_level', 'grade_subgrade'], axis=1)


test_df


num_cols = ['annual_income','debt_to_income_ratio','credit_score','loan_amount','interest_rate']


test_df = cap_outliers_iqr(test_df, num_cols, factor=1.5, verbose=True)


test_df


prediction = model_pipeline.predict(test_df)


submission = pd.DataFrame({
    "id": id,             # make sure id column exists in test.csv
    "loan_paid_back": prediction    # predicted label (0 or 1)
})

# 6ï¸�âƒ£ Save as CSV file
submission.to_csv("submission.csv", index=False)

print("âœ… submission.csv file successfully created!")
submission.head()




