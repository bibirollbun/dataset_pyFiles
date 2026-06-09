import pandas as pd
import numpy as np
df = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
df = df.drop(columns=['id'])


df.head() # First 5 observations (rows 0-4)


df.tail() # Last 5 observations (rows 749995-749999)


df.shape # (750000, 17)


df.info()


df.select_dtypes('number').columns
df.select_dtypes('number').columns.tolist()


nan_count_by_column = df.isnull().sum() 
print("Count NaN for column:\n", nan_count_by_column) # No NaN in the DataFrame

# Count the total number of NaN in the DataFrame
total_nan = df.isnull().sum().sum()
print("\nTotal NaN in the DataFrame:", total_nan) # No NaN in the DataFrame


df.describe()


def freq_table(df, column):
    counts = df[column].value_counts()
    percentages = df[column].value_counts(normalize=True) * 100
    return pd.DataFrame({'count': counts, 'percentage': percentages.round(2)})


freq_table(df, 'job')


counts = df['job'].value_counts()

print(f"Category with the fewest observations: '{counts.idxmin()}' ({counts.min()} records)")
print(f"Category with the most observations: '{counts.idxmax()}' ({counts.max()} records)")


freq_table(df, 'marital')


freq_table(df, 'education')


freq_table(df, 'default')


freq_table(df, 'housing')


freq_table(df, 'loan')


freq_table(df, 'contact')


freq_table(df, 'month')


freq_table(df, 'poutcome')


freq_table(df, 'y')


import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Ignore the "use_inf_as_na" warning
warnings.filterwarnings("ignore", message=".*use_inf_as_na.*")

# Modern style configuration
sns.set_theme(style="whitegrid")

# Variable descriptions
descriptions = {
    "age": "Age of the client (in years).",
    "job": "Type of occupation of the client (e.g., admin, technician, unemployed).",
    "marital": "Marital status of the client (single, married, divorced).",
    "education": "Highest level of education attained by the client.",
    "default": "Indicates whether the client has any credit in default (non-payment).",
    "balance": "Average annual account balance of the client (in euros).",
    "housing": "Indicates whether the client has a housing loan.",
    "loan": "Indicates whether the client has a personal loan.",
    "contact": "Communication channel used (e.g., cellular, telephone, unknown).",
    "day": "Day of the month on which the last contact was made with the client.",
    "month": "Month in which the last contact was made with the client.",
    "duration": "Duration (in seconds) of the last call made.",
    "campaign": "Number of contacts performed during this campaign for the given client.",
    "pdays": "Number of days since the client was last contacted in a previous campaign (-1 if not previously contacted).",
    "previous": "Number of contacts made before this campaign.",
    "poutcome": "Outcome of the previous marketing campaign (e.g., success, failure, unknown).",
    "y": "Target variable â€” whether the client subscribed to a term deposit (1 = yes, 0 = no)."
}

def eda(df, target='y'):
    # Replace inf/-inf with NaN to avoid seaborn warnings
    df = df.replace([np.inf, -np.inf], np.nan)

    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    
    print("==== Exploratory Data Analysis ====\n")
    
    # --- Numerical variables ---
    for col in numeric_cols:
        plt.figure(figsize=(12,4))
        plt.subplot(1,2,1)
        sns.histplot(df[col], bins=30, kde=True, color='skyblue')
        plt.title(f"Histogram of {col}\n{descriptions.get(col,'')}")
        
        plt.subplot(1,2,2)
        sns.boxplot(x=df[col], color='lightgreen')
        plt.title(f"Boxplot of {col}")
        plt.tight_layout()
        plt.show()
        
        stats = df[col].describe()
        print(f"\nStatistics for {col}:")
        print(stats)
        
        # Skewness and outliers
        if df[col].skew() > 1:
            print(">> Insight: Right skew (high outliers).")
        elif df[col].skew() < -1:
            print(">> Insight: Left skew (low outliers).")
        
        outliers = df[(df[col] < stats['25%'] - 1.5*(stats['75%']-stats['25%'])) |
                      (df[col] > stats['75%'] + 1.5*(stats['75%']-stats['25%']))]
        if len(outliers) > 0:
            print(f">> Insight: There are {len(outliers)} possible outliers in {col}.")
        
        # Correlation with target
        if target in df.columns and df[target].dtype in ['int64','float64']:
            corr = df[col].corr(df[target])
            print(f">> Correlation with {target}: {corr:.2f}")
            if abs(corr) > 0.3:
                print(f">> Insight: {col} might be related to {target}.")
    
    # --- Categorical variables ---
    for col in categorical_cols:
        value_counts = df[col].value_counts(normalize=True).sort_values(ascending=True) * 100
        
        plt.figure(figsize=(10,4))
        sns.barplot(x=value_counts.values, y=value_counts.index, color='lightcoral')  # avoiding palette issue
        plt.xlabel("Percentage (%)")
        plt.ylabel(col)
        plt.title(f"Percentage distribution of {col}\n{descriptions.get(col,'')}")
        plt.tight_layout()
        plt.show()
        
        # Comparison with target
        if target in df.columns:
            cross = pd.crosstab(df[col], df[target], normalize='index')*100
            cross.plot(kind='barh', stacked=True, figsize=(10,5), colormap='Pastel1')
            plt.title(f"{col} vs Target ({target})")
            plt.xlabel("Percentage (%)")
            plt.ylabel(col)
            plt.legend(title=target, loc='upper right')
            plt.tight_layout()
            plt.show()
            
            high_diff = (cross.max(axis=1) - cross.min(axis=1)).max()
            if high_diff > 30:
                print(f">> Insight: {col} might be strongly related to {target}.")
    
    # --- Numerical correlation ---
    plt.figure(figsize=(12,8))
    sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation matrix of numerical variables")
    plt.show()
    
    print("\n==== End of EDA report ====")

# Run
eda(df)



def analyze_outliers(df, threshold=1.5, show_values=True):
    """
    Analyze outliers in numerical variables using IQR.
    
    Parameters:
    - df: DataFrame
    - threshold: IQR multiplier to consider a value as an outlier (default 1.5)
    - show_values: if True, prints the outlier values
    """
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    print("==== Outlier Analysis ====\n")
    
    outlier_summary = {}
    
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_count = outliers.shape[0]
        outlier_summary[col] = outlier_count
        
        print(f"Variable: {col}")
        print(f"Total outliers: {outlier_count}")
        if show_values and outlier_count > 0:
            print("Extreme values (first 10):")
            print(outliers[col].sort_values().head(10).tolist())
        print("-"*50)
        
        # Boxplot showing outliers
        plt.figure(figsize=(8,4))
        sns.boxplot(x=df[col], color='lightblue')
        plt.title(f"Boxplot with outliers for {col}")
        plt.show()
    
    # General summary
    summary_df = pd.DataFrame(list(outlier_summary.items()), columns=['Variable', 'Num_Outliers'])
    summary_df = summary_df.sort_values(by='Num_Outliers', ascending=False)
    print("\nOutlier Summary by Variable:")
    print(summary_df)
    
    return summary_df

# Run outlier analysis
outliers_report = analyze_outliers(df)



from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# ===============================
# Load data
# ===============================
train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')

# Save test IDs
test_ids = test["id"]

# Drop 'id' column from train and test
train = train.drop(columns=['id'])
X_test = test.drop(columns=['id'])

# ===============================
# Define variable types
# ===============================
binary_cols = ['default', 'housing', 'loan']          # binary variables 0/1
nominal_cols = ['job', 'marital', 'contact', 'month', 'poutcome']  # nominal variables
ordinal_cols = ['education']                          # ordinal variables

# Convert binary columns from 'yes'/'no' to 1/0
for col in binary_cols:
    train[col] = train[col].map({'yes': 1, 'no': 0})
    X_test[col] = X_test[col].map({'yes': 1, 'no': 0})


# ===============================
# Preprocessing for pipeline
# ===============================

# Ordinal encoder for 'education'
ordinal_transformer = OrdinalEncoder(categories=[['primary', 'secondary', 'tertiary']])

# Categorical transformer: imputer + one-hot encoder
nominal_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Other')),  # handle 'unknown'
    ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
])

# ColumnTransformer combining all
preprocessor = ColumnTransformer(
    transformers=[
        ('nominal', nominal_transformer, nominal_cols),
        ('ordinal', ordinal_transformer, ordinal_cols)
    ],
    remainder='passthrough'  # keep binary and numeric columns as they are
)

# ===============================
# Separate target variable
# ===============================
X = train.drop(columns=["y"])
y = train["y"]

# ===============================
# Note:
# Now the pipeline handles unknown categories automatically
# No manual removal or replacement of 'unknown' is required.
# ===============================




df.shape


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer



# ===============================
# Numeric columns (exclude binary)
# ===============================
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
numeric_cols = [col for col in numeric_cols if col not in binary_cols]

# Nominal and ordinal columns
categorical_cols = nominal_cols
ordinal_cols = ordinal_cols

# ===============================
# Transformers
# ===============================
numeric_transformer = StandardScaler()

# Nominal transformer: one-hot encode, keep 'unknown' as-is
categorical_transformer = OneHotEncoder(handle_unknown='ignore', drop='first')

# Ordinal transformer: include 'unknown' as a valid category
ordinal_transformer = OrdinalEncoder(categories=[['primary', 'secondary', 'tertiary', 'unknown']])

# ===============================
# ColumnTransformer
# ===============================
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols),
        ('ord', ordinal_transformer, ordinal_cols)
    ],
    remainder='passthrough'  # keep binary columns as-is
)



#Split for validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)



# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "XGBoost": XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,  
        eval_metric="auc",
        random_state=42
    )
}



# Train models, evaluate ROC AUC, and plot ROC curves
plt.figure(figsize=(10, 7))

for name, model in models.items():
    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])
    clf.fit(X_train, y_train)
    y_pred_proba = clf.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred_proba)
    print(f"{name} ROC AUC: {auc:.4f}")
    
    fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")

plt.plot([0,1], [0,1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves for Validation Set")
plt.legend(loc="lower right")
plt.show()



# ===============================
# Columns types
# ===============================
binary_cols = ['default', 'housing', 'loan']
nominal_cols = ['job', 'marital', 'contact', 'month', 'poutcome']
ordinal_cols = ['education']

# Convert binary columns to numeric
for col in binary_cols:
    X[col] = X[col].map({'yes': 1, 'no': 0})
    X_test[col] = X_test[col].map({'yes': 1, 'no': 0})

# Numeric columns (exclude binary)
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
numeric_cols = [col for col in numeric_cols if col not in binary_cols]

# ===============================
# Preprocessor
# ===============================
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), nominal_cols),
        ('ord', OrdinalEncoder(categories=[['primary', 'secondary', 'tertiary', 'unknown']]), ordinal_cols)
    ],
    remainder='passthrough'  # keep binary columns as-is
)

# ===============================
# Pipeline with preprocessor + XGB
# ===============================
final_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    ))
])

# Train model
final_model.fit(X, y)

# Predict test set
test_preds = final_model.predict_proba(X_test)[:, 1]

# Build submission
submission = pd.DataFrame({
    'id': test_ids,
    'y': test_preds
})
submission.to_csv("/kaggle/working/submission.csv", index=False)
print("âœ… submission.csv saved successfully in /kaggle/working/")

