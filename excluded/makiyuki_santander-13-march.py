import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Data manipulation and analysis
import numpy as np
import pandas as pd

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

# Statistical methods
from scipy import stats
from scipy.stats import mannwhitneyu, chi2_contingency

# Machine learning
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from scipy.stats import randint, uniform
from sklearn.model_selection import train_test_split

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold, GridSearchCV, cross_validate
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score, make_scorer
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import ConfusionMatrixDisplay
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import warnings

# Warnings
warnings.filterwarnings('ignore')



df = pd.read_csv('/kaggle/input/santander-product-recommendation/train_ver2.csv.zip')


product = df.iloc[:,24:]
product.head()


# Count the occurrences of '1' in each column
count_ones = (product == 1).sum()

# Sort the counts in descending order
sorted_counts = pd.DataFrame(count_ones.sort_values(ascending=False)).reset_index()
sorted_counts.columns = ['product','count']
# Display the sorted counts
print(sorted_counts)


sorted_counts['proportion'] =  sorted_counts['count']/13647309
sorted_counts


df.head()


df.info()


import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds

# âœ… Ensure renaming is correct
rename_dict = {
    "ind_cco_fin_ult1": "Current_Accounts",
    "ind_ahor_fin_ult1": "Saving_Account",
    "ind_nomina_ult1": "Payroll_Account",
    "ind_cno_fin_ult1": "Particular_Account",
    "ind_ecue_fin_ult1": "e_account",
    "ind_ctju_fin_ult1": "Junior_Account",
    "ind_ctpp_fin_ult1": "Particular_Plus_Account",
    "ind_reca_fin_ult1": "Taxes",
    "ind_deme_fin_ult1": "Long_term_deposits",
    "ind_ctma_fin_ult1": "Mas_Particular_Account",
    "ind_tjcr_fin_ult1": "Credit_Card",
    "ind_nom_pens_ult1": "Pensions",
    "ind_pres_fin_ult1": "Loans",
    "ind_recibo_ult1": "Direct_Debit",
    "ind_valo_fin_ult1": "Securities",
    "ind_fond_fin_ult1": "Funds",
    "ind_hip_fin_ult1": "Mortgage",
    "ind_deco_fin_ult1": "Short_term_deposits",
    "ind_viv_fin_ult1": "Home_Account",
    "ind_nomina_ult1": "Payroll",
    "ind_cder_fin_ult1": "Derivada_Account",
    "ind_dela_fin_ult1": "Medium_term_deposits",
    "ind_aval_fin_ult1": "Guarantees",
    "ind_ctop_fin_ult1": "Top_Products",   
    "ind_plan_fin_ult1": "Investment_Plan"
}

# âœ… Rename columns
df.rename(columns=rename_dict, inplace=True)

# âœ… Check available columns in df
print("Columns in df after renaming:")
print(df.columns.tolist())  

# âœ… Extract only the existing product columns
all_product_columns = list(rename_dict.values())  # Expected product columns
product_columns = [col for col in all_product_columns if col in df.columns]  # Only keep available ones

# âœ… Check selected columns
print(f"Total product columns found in df: {len(product_columns)}")
print("Selected product columns:", product_columns)



# Load dataset
selected_df = df.copy()

# Convert fecha_dato to datetime and sort values
selected_df["fecha_dato"] = pd.to_datetime(selected_df["fecha_dato"])
selected_df = selected_df.sort_values(by=["ncodpers", "fecha_dato"])

# Get first and last row per customer
first_last_df = selected_df.groupby("ncodpers").agg(["first", "last"])
first_last_df.columns = ['_'.join(col).strip() for col in first_last_df.columns.values]
first_last_df.reset_index(inplace=True)

# âœ… Debug: Print available columns in first_last_df
print("Available columns in first_last_df:")
print(first_last_df.columns.tolist())

# âœ… Ensure we only select columns that exist
available_columns = [col + "_first" for col in product_columns if col + "_first" in first_last_df.columns]

# âœ… Debugging info
print(f"Using {len(available_columns)} product columns for training.")

# Prepare training matrix (only product columns from the first row for factorization)
train_matrix = first_last_df[available_columns].values

# Handle NaNs and Infs
train_matrix = np.nan_to_num(train_matrix, nan=0.0, posinf=0.0, neginf=0.0)

# Perform matrix factorization using Singular Value Decomposition (SVD)
k = min(10, train_matrix.shape[1] - 1)  # Ensure k is valid
U, sigma, Vt = svds(train_matrix, k=k)
sigma = np.diag(sigma)

# Reconstruct matrix (approximated product preferences)
reconstructed_matrix = np.dot(np.dot(U, sigma), Vt)

# Convert back to DataFrame
recommendations = pd.DataFrame(reconstructed_matrix, index=first_last_df["ncodpers"], columns=available_columns)

# Exclude products already owned (ensure only new product recommendations)
owned_products = first_last_df[available_columns].values
new_recommendations = recommendations.copy()
new_recommendations[owned_products > 0] = 0  # Remove already owned products

# Rank recommendations for each customer (top 5 recommended products)
top_k = 5
top_recommendations = np.argsort(-new_recommendations.values, axis=1)[:, :top_k]

# Convert to human-readable format
final_recommendations = pd.DataFrame(
    [[available_columns[i].replace("_first", "") for i in row] for row in top_recommendations],
    index=first_last_df["ncodpers"],
    columns=[f"Top_{i+1}" for i in range(top_k)]
)

# Display final recommendations
print(final_recommendations.head())

# Evaluate precision using last row (actual customer purchases)
actual_purchases = first_last_df[[col.replace("_first", "_last") for col in available_columns]].values

def precision_at_k(recommended, actual, k=5):
    precision_scores = []
    for i in range(len(recommended)):
        recommended_items = set(recommended[i])
        relevant_items = set(np.where(actual[i] > 0)[0])  # Indices of actual purchases
        if relevant_items:
            precision_scores.append(len(recommended_items & relevant_items) / k)
    return np.mean(precision_scores) if precision_scores else 0.0

precision = precision_at_k(top_recommendations, actual_purchases, k=5)
print(f"Precision@5: {precision:.4f}")




# âœ… Verify df[product_columns] works without error
print("Shape of df before selecting product columns:", df.shape)
df_product_subset = df[product_columns]  # This should now work
print("Shape of df_product_subset:", df_product_subset.shape)


import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, accuracy_score, classification_report

# âœ… Ensure column renaming is applied
df.rename(columns=rename_dict, inplace=True)

# âœ… Convert age to numeric (fixing object type issue)
df["age"] = pd.to_numeric(df["age"], errors="coerce")  # Convert age to number, set errors to NaN
df["age"].fillna(df["age"].median(), inplace=True)  # Replace missing values with median

# âœ… Check available columns in df after renaming
print("Columns in df after renaming:")
print(df.columns.tolist())  

# âœ… Define Top 5 Products based on renamed names
top_products = ["Current_Accounts", "Top_Products", "Direct_Debit", "e_account", "Particular_Account"]

# âœ… Define Features (You can expand this list for better modeling)
features = ["age", "segmento", "renta"]  # Add more useful customer attributes

# âœ… Ensure features exist
features = [col for col in features if col in df.columns]

# âœ… Convert categorical features to category type
for col in ["segmento"]:  # Add more categorical features if needed
    df[col] = df[col].astype("category")

# âœ… Train-Test Split
X = df[features]
y = df[top_products]  # Multi-label dataset (each product = separate binary classification)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# âœ… Train 5 Models for Top 5 Products
models = {}
for product in top_products:
    print(f"\nðŸš€ Training Model for {product}...\n")

    # Define XGBoost Model
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        enable_categorical=True,  # âœ… Enable categorical support in XGBoost
        random_state=42
    )

    # Train Model
    model.fit(X_train, y_train[product])

    # Save Model
    models[product] = model

    # Make Predictions
    y_pred = model.predict(X_test)

    # Evaluate Performance
    precision = precision_score(y_test[product], y_pred)
    accuracy = accuracy_score(y_test[product], y_pred)

    print(f"ðŸ“Œ Precision for {product}: {precision:.4f}")
    print(f"ðŸ“Œ Accuracy for {product}: {accuracy:.4f}\n")
    print("ðŸ“Š Classification Report:")
    print(classification_report(y_test[product], y_pred))

print("\nâœ… Training Completed for All Models!")



df.info()


# import pandas as pd
# import numpy as np
# import xgboost as xgb
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import precision_score, accuracy_score, classification_report

# # âœ… Ensure column renaming is applied
# df.rename(columns=rename_dict, inplace=True)

# # âœ… Convert age to numeric (fixing object type issue)
# df["age"] = pd.to_numeric(df["age"], errors="coerce")  # Convert age to number, set errors to NaN
# df["age"].fillna(df["age"].median(), inplace=True)  # Replace missing values with median

# # âœ… Fix: Use correct column name "segmento" instead of "segment"
# if "segmento" in df.columns:
#     df.rename(columns={"segmento": "segment"}, inplace=True)

# # âœ… Convert categorical features to category type
# for col in ["segment"]:  # Ensure it exists before conversion
#     if col in df.columns:
#         df[col] = df[col].astype("category")

# # âœ… Define Top 5 Products based on renamed names
# top_products = ["Current_Accounts", "Top_Products", "Direct_Debit", "e_account", "Particular_Account"]

# # âœ… Define Features (You can expand this list for better modeling)
# features = ["age", "segment", "renta"]  # Add more useful customer attributes

# # âœ… Ensure features exist
# features = [col for col in features if col in df.columns]

# # âœ… Train-Test Split
# X = df[features]
# y = df[top_products]  # Multi-label dataset (each product = separate binary classification)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # âœ… Train 5 Models for Top 5 Products
# models = {}
# for product in top_products:
#     print(f"\nðŸš€ Training Model for {product}...\n")

#     # Define XGBoost Model
#     model = xgb.XGBClassifier(
#         objective="binary:logistic",
#         eval_metric="logloss",
#         use_label_encoder=False,
#         enable_categorical=True,  # âœ… Enable categorical support in XGBoost
#         random_state=42
#     )

#     # Train Model
#     model.fit(X_train, y_train[product])

#     # Save Model
#     models[product] = model

#     # Make Predictions
#     y_pred = model.predict(X_test)

#     # Evaluate Performance
#     precision = precision_score(y_test[product], y_pred)
#     accuracy = accuracy_score(y_test[product], y_pred)

#     print(f"ðŸ“Œ Precision for {product}: {precision:.4f}")
#     print(f"ðŸ“Œ Accuracy for {product}: {accuracy:.4f}\n")
#     print("ðŸ“Š Classification Report:")
#     print(classification_report(y_test[product], y_pred))

# print("\nâœ… Training Completed for All Models!")





