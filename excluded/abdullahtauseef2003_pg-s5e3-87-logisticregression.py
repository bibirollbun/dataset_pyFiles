import pandas as pd

train_df = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")


train_df.head()


train_df.info()


train_df.describe()


train_df.isnull().sum()


import numpy as np

np.isinf(train_df).sum()


train_df.nunique()


train_df.drop(["id", "day"], axis = 1, inplace = True)


cr = train_df.corr()


import seaborn as sns

sns.heatmap(cr, xticklabels = cr.columns, yticklabels = cr.columns, annot = True)


import matplotlib.pyplot as plt

def plot_all_distplots(df):
    plt.figure(figsize=(15, 12))  # Adjust figure size
    for i, col in enumerate(df.columns, 1):  # Loop through all columns
        plt.subplot(4, 3, i)  # Create subplots (adjust rows/cols as needed)
        sns.histplot(df[col], kde=True, bins=30)  # KDE + Histogram
        plt.title(f"Distribution of {col}")
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()

plot_all_distplots(train_df)



def plot_boxplots(df):
    num_features = df.select_dtypes(include=['number']).columns  # Select only numerical columns
    for feature in num_features:
        plt.figure(figsize=(6, 4))
        sns.catplot(x="rainfall", y=feature, data=df, kind="box", height=4, aspect=1.5)
        plt.title(f"Boxplot of {feature} by Rainfall")
        plt.grid()
        plt.show()

plot_boxplots(train_df)


def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

# Apply to relevant features
outlier_columns = ["pressure", "humidity", "cloud", "dewpoint", "windspeed", "maxtemp", "mintemp", "winddirection"]
for col in outlier_columns:
    train_df = remove_outliers_iqr(train_df, col)


# Log transformations
train_df["sunshine"] = np.log1p(train_df["sunshine"])  
train_df["windspeed"] = np.log1p(train_df["windspeed"])
train_df["winddirection"] = np.log1p(train_df["winddirection"])

# Square root transformation for skewed variables
train_df["cloud"] = np.sqrt(train_df["cloud"])


train_df["rainfall"] = train_df["rainfall"].astype("category")  # Convert to categorical


from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Standardization for normally distributed variables
scaler = StandardScaler()
train_df[["pressure", "temparature", "humidity", "windspeed", "maxtemp", "mintemp"]] = scaler.fit_transform(
    train_df[["pressure", "temparature", "humidity", "windspeed", "maxtemp", "mintemp"]]
)

# Min-Max Scaling for skewed variables
min_max_scaler = MinMaxScaler()
train_df[["cloud", "sunshine", "winddirection"]] = min_max_scaler.fit_transform(
    train_df[["cloud", "sunshine", "winddirection"]]
)


threshold = 0.85
to_remove = set()
cr = cr.abs()

for i in range(len(cr.columns)):
    for j in range(i):
        if cr.iloc[i, j] > threshold:
            colname = cr.columns[i]
            to_remove.add(colname)

train_df = train_df.drop(columns=to_remove)

print(f"Removed highly correlated features: {to_remove}")


X = train_df.drop(columns=["rainfall"])  # Features
y = train_df["rainfall"]  # Target variable


from sklearn.feature_selection import mutual_info_classif, RFE

mi_scores = mutual_info_classif(X, y)
mi_scores = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
print("\nMutual Information Scores:")
print(mi_scores)


selected_features_mi = mi_scores.index[:5].tolist()
print(f"\nTop Features Based on Mutual Information: {selected_features_mi}")


from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
rfe = RFE(model, n_features_to_select=5)
rfe.fit(X, y)

selected_features_rfe = X.columns[rfe.support_].tolist()
print(f"\nTop Features Based on RFE: {selected_features_rfe}")


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)
feature_importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

print("\nFeature Importance Scores (Random Forest):")
print(feature_importance)

# Select top 5 features
selected_features_rf = feature_importance.index[:5].tolist()
print(f"\nTop Features Based on Random Forest: {selected_features_rf}")

### FINAL SELECTED FEATURES ###
final_selected_features = list(set(selected_features_mi + selected_features_rfe + selected_features_rf))
print(f"\nFinal Selected Features: {final_selected_features}")



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


from sklearn.linear_model import LogisticRegression

model_lr = LogisticRegression()
model_lr.fit(X_train, y_train)
model_lr.score(X_test, y_test)


from sklearn.ensemble import RandomForestClassifier

model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)
model_rf.score(X_test, y_test)


from xgboost import XGBClassifier

model_xg = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model_xg.fit(X_train, y_train)
model_xg.score(X_test, y_test)


test = test_df.drop(["id", "day", "mintemp", "temparature", "dewpoint"], axis = 1)


test.head()


test.info()


test.describe()


test.isnull().sum()


test["winddirection"] = test["winddirection"].fillna(test["winddirection"].mean())


# Make predictions
y_pred = model_lr.predict(test)

# Create a DataFrame with id and predictions
output_df = pd.DataFrame({
    "id": test_df["id"],   # Assuming test dataset has an "id" column
    "prediction": y_pred
})

# Save to a new CSV file
output_df.to_csv("submission.csv", index=False)

# Print the predictions along with ID
print(output_df)


import pickle

# Save the model properly
with open("logistic_regression_model.pkl", "wb") as f:
    pickle.dump(model_lr, f)


import joblib

# Load the trained model
model_lr = joblib.load("logistic_regression_model.pkl")

# Ensure it's a valid model
print(type(model_lr))  # Should print something like <class 'sklearn.linear_model._logistic.LogisticRegression'>

