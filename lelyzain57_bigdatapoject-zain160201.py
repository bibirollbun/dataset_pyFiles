
#  Load the dataset
import pandas as pd

# Load the application_train.csv file
df = pd.read_csv("/kaggle/input/home-credit-default-risk/application_train.csv")

# Display shape and first few rows
print(f"Shape of dataset: {df.shape}")
df.head()



# Drop columns with >40% missing values
missing_ratio = df.isnull().sum() / len(df)
cols_to_drop = missing_ratio[missing_ratio > 0.4].index
df.drop(columns=cols_to_drop, inplace=True)
print(f"Dropped {len(cols_to_drop)} columns with >40% missing values.")

#  Smart fill numeric columns (mean or median based on skew)
import numpy as np

num_cols = df.select_dtypes(include=['int64', 'float64']).columns

for col in num_cols:
    if df[col].isnull().sum() > 0:
        skewness = df[col].skew()
        if abs(skewness) > 1:
            df[col].fillna(df[col].median(), inplace=True)
            print(f"{col}: filled with MEDIAN (skew={skewness:.2f})")
        else:
            df[col].fillna(df[col].mean(), inplace=True)
            print(f"{col}: filled with MEAN (skew={skewness:.2f})")

#  Fill categorical columns with mode
cat_cols = df.select_dtypes(include=['object']).columns
df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

print("Filled missing values in categorical columns using MODE.")



print("Remaining columns:", df.shape[1])
print("TARGET in dataset:", 'TARGET' in df.columns)
df.columns[:10]  # show first 10 column names



# Check class distribution before encoding
import seaborn as sns
import matplotlib.pyplot as plt

# Countplot for TARGET values
sns.countplot(x='TARGET', data=df)
plt.title("Class Distribution (Target)")
plt.xlabel("TARGET (0 = No Default, 1 = Default)")
plt.ylabel("Count")
plt.show()

# Print percentage breakdown
target_counts = df['TARGET'].value_counts()
print("\nClass Balance:")
print(target_counts)
print("Percentage:")
print(round(target_counts / target_counts.sum() * 100, 2))



# List all categorical (text/object) columns
cat_cols = df.select_dtypes(include='object').columns

print(f"Total categorical columns: {len(cat_cols)}\n")
for col in cat_cols:
    print(f"{col} ➜ {df[col].nunique()} unique values")



df['CODE_GENDER'].value_counts()



# Drop rows where CODE_GENDER is 'XNA'
df = df[df['CODE_GENDER'] != 'XNA']

# Confirm cleanup
print("Remaining gender values:", df['CODE_GENDER'].unique())



from sklearn.preprocessing import LabelEncoder

cat_cols = df.select_dtypes(include='object').columns
le = LabelEncoder()

for col in cat_cols:
    if df[col].nunique() == 2:
        df[col] = le.fit_transform(df[col])
        print(f"{col}: Label Encoded (binary)")
    else:
        df = pd.get_dummies(df, columns=[col], drop_first=True)
        print(f"{col}: One-Hot Encoded (multi-class)")



# Check dataset shape
print("Current shape:", df.shape)

# Check if TARGET is still in the data
print("TARGET present:", 'TARGET' in df.columns)

# Check if SK_ID_CURR is still available
print("SK_ID_CURR present:", 'SK_ID_CURR' in df.columns)

# Check for any remaining object (non-numeric) columns
print("Remaining non-numeric columns:", df.select_dtypes(include='object').columns.tolist())



import numpy as np

# Build correlation matrix
corr_matrix = df.corr().abs()

# Use upper triangle to avoid duplicates
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Identify columns to drop
to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]

# Drop them
df.drop(columns=to_drop, inplace=True)

print(f"Dropped {len(to_drop)} highly correlated features.")



from sklearn.feature_selection import VarianceThreshold

# Save current column names
columns_before = df.columns

# Remove columns with very low variance
selector = VarianceThreshold(threshold=0.01)
df_var = selector.fit_transform(df)

# Get names of remaining features
columns_after = columns_before[selector.get_support()]

# Rebuild df with selected features
df = pd.DataFrame(df_var, columns=columns_after)

print(f"Removed {len(columns_before) - len(columns_after)} low-variance features.")



print("Current shape:", df.shape)
print("TARGET present:", 'TARGET' in df.columns)
print("SK_ID_CURR present:", 'SK_ID_CURR' in df.columns)



import seaborn as sns
import matplotlib.pyplot as plt

# Compute correlation matrix
corr = df.corr()

# Focus on correlation with TARGET
plt.figure(figsize=(12, 1))
target_corr = corr[['TARGET']].sort_values(by='TARGET', ascending=False)

# Show strongest correlations (you can adjust slicing to top N)
sns.heatmap(target_corr.T, cmap='coolwarm', annot=True, fmt=".2f")
plt.title("Correlation with TARGET")
plt.show()



# Full correlation heatmap (this will be large, use for small feature sets only)
plt.figure(figsize=(16, 12))
sns.heatmap(df.corr(), cmap='coolwarm', center=0, linewidths=0.5)
plt.title("Full Feature Correlation Heatmap")
plt.show()



from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

# Split input/output
X = df.drop(columns=['TARGET', 'SK_ID_CURR'])
y = df['TARGET']

X_train, _, y_train, _ = train_test_split(X, y, test_size=0.7, stratify=y, random_state=42)

# Train random forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Feature importance ranking
importances = pd.Series(rf.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)

# Show top 30
top_features = importances.head(30)
print("Top 30 Features:\n", top_features)



from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Use only top 30 features
X_top = df[top_features.index]

#  Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_top)

#  Try KMeans with 2–5 clusters and plot inertia
inertias = []
for k in range(2, 6):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

# Elbow plot
plt.plot(range(2, 6), inertias, marker='o')
plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.grid(True)
plt.show()



# Apply KMeans with k=3
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Check how many samples in each cluster
print(df['Cluster'].value_counts())

# Compare cluster labels with TARGET
cluster_summary = pd.crosstab(df['Cluster'], df['TARGET'])
print("\nCluster vs Target:\n", cluster_summary)



from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

# Drop ID
X = df.drop(columns=['SK_ID_CURR', 'TARGET'])
y = df['TARGET']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression with class_weight
log_reg = LogisticRegression(max_iter=1000, class_weight='balanced')
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)

# Random Forest with class_weight
rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Evaluation Function
def evaluate(name, y_true, y_pred):
    print(f"\n{name} Performance:")
    print("Accuracy:", round(accuracy_score(y_true, y_pred), 4))
    print("Precision:", round(precision_score(y_true, y_pred), 4))
    print("Recall:", round(recall_score(y_true, y_pred), 4))
    print("F1 Score:", round(f1_score(y_true, y_pred), 4))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

# Results
evaluate("Logistic Regression", y_test, y_pred_log)
evaluate("Random Forest", y_test, y_pred_rf)



# Logistic Regression with class weights
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

log_reg = LogisticRegression(class_weight='balanced', random_state=42)
log_reg.fit(X_train, y_train)

# Predict
log_reg_pred = log_reg.predict(X_test)

# Logistic Regression Performance Evaluation
print("Logistic Regression Performance:")
print(f"Accuracy: {accuracy_score(y_test, log_reg_pred)}")
print(f"Precision: {precision_score(y_test, log_reg_pred)}")
print(f"Recall: {recall_score(y_test, log_reg_pred)}")
print(f"F1 Score: {f1_score(y_test, log_reg_pred)}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, log_reg_pred))

# Random Forest with class weights
from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(class_weight='balanced', random_state=42)
rf_clf.fit(X_train, y_train)

# Predict
rf_pred = rf_clf.predict(X_test)

# Random Forest Performance Evaluation
print("\nRandom Forest Performance:")
print(f"Accuracy: {accuracy_score(y_test, rf_pred)}")
print(f"Precision: {precision_score(y_test, rf_pred)}")
print(f"Recall: {recall_score(y_test, rf_pred)}")
print(f"F1 Score: {f1_score(y_test, rf_pred)}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, rf_pred))


