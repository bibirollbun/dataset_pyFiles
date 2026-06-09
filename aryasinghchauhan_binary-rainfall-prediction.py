import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import pandas as pd
train=pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
train


test=pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
test


train.info()


test.info()


import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(6,4))
sns.countplot(x=train['rainfall'], palette='bright')
plt.title("Rainfall Class Distribution")
plt.xlabel("Rainfall (0 = No Rain, 1 = Rain)")
plt.ylabel("Count")
plt.show()


rainfall_counts = train['rainfall'].value_counts(normalize=True) * 100
print("Rainfall Distribution:\n", rainfall_counts)


train.hist(figsize=(12, 10), bins=30, edgecolor='black')
plt.suptitle("Feature Distributions", fontsize=14)
plt.show()


from sklearn.utils import resample
df_majority = train[train['rainfall'] == 1]  
df_minority = train[train['rainfall'] == 0]  
df_majority_downsampled = resample(df_majority, 
                                   replace=False,   
                                   n_samples=int(len(df_majority) * 0.5),  
                                   random_state=42) 
train_balanced = pd.concat([df_majority_downsampled, df_minority])

train_balanced = train_balanced.sample(frac=1, random_state=42).reset_index(drop=True)


train_balanced['rainfall'].value_counts(normalize=True) * 100


num_cols = ['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 
            'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed']
plt.figure(figsize=(15, 12))
for i, col in enumerate(num_cols, 1):
    plt.subplot(3, 4, i)
    sns.boxplot(y=train_balanced[col], color='orange')
    plt.title(col)
plt.tight_layout()
plt.show()


corr_matrix = train_balanced.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()


continuous_features = ['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint',
                       'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed']

def remove_outliers_iqr(df, columns):
    df_clean = df.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean

train_cleaned = remove_outliers_iqr(train_balanced, continuous_features)

print(f"Original dataset size: {train_balanced.shape[0]} rows")
print(f"Cleaned dataset size: {train_cleaned.shape[0]} rows")
print(f"Rows removed: {train_balanced.shape[0] - train_cleaned.shape[0]}")



plt.figure(figsize=(12, 8))
sns.heatmap(train_cleaned.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap After Outlier Removal")
plt.show()


corr_before = train_balanced.corr() 
corr_after = train_cleaned.corr() 

corr_diff = corr_after - corr_before  

plt.figure(figsize=(10, 8))
sns.heatmap(corr_diff, annot=True, cmap="coolwarm", center=0, fmt=".2f")
plt.title("Correlation Difference Heatmap (After - Before)")
plt.show()



from statsmodels.stats.outliers_influence import variance_inflation_factor

numeric_features = train_balanced.select_dtypes(include=['number'])

vif_data = pd.DataFrame()
vif_data["Feature"] = numeric_features.columns
vif_data["VIF"] = [variance_inflation_factor(numeric_features.values, i) for i in range(len(numeric_features.columns))]

vif_data.sort_values(by="VIF", ascending=False)



train_balanced = train_balanced.drop(columns=['temparature'])

numeric_features = train_balanced.select_dtypes(include=['number'])
vif_data = pd.DataFrame()
vif_data["Feature"] = numeric_features.columns
vif_data["VIF"] = [variance_inflation_factor(numeric_features.values, i) for i in range(len(numeric_features.columns))]

vif_data.sort_values(by="VIF", ascending=False)



train_balanced = train_balanced.drop(columns=['mintemp'])

numeric_features = train_balanced.select_dtypes(include=['number'])
vif_data = pd.DataFrame()
vif_data["Feature"] = numeric_features.columns
vif_data["VIF"] = [variance_inflation_factor(numeric_features.values, i) for i in range(len(numeric_features.columns))]

vif_data.sort_values(by="VIF", ascending=False)



train_balanced = train_balanced.drop(columns=['maxtemp'])

numeric_features = train_balanced.select_dtypes(include=['number'])
vif_data = pd.DataFrame()
vif_data["Feature"] = numeric_features.columns
vif_data["VIF"] = [variance_inflation_factor(numeric_features.values, i) for i in range(len(numeric_features.columns))]

vif_data.sort_values(by="VIF", ascending=False)



train_balanced = train_balanced.drop(columns=['humidity'])

numeric_features = train_balanced.select_dtypes(include=['number'])
vif_data = pd.DataFrame()
vif_data["Feature"] = numeric_features.columns
vif_data["VIF"] = [variance_inflation_factor(numeric_features.values, i) for i in range(len(numeric_features.columns))]

vif_data.sort_values(by="VIF", ascending=False)



train_balanced = train_balanced.drop(columns=['pressure'])

numeric_features = train_balanced.select_dtypes(include=['number'])
vif_data = pd.DataFrame()
vif_data["Feature"] = numeric_features.columns
vif_data["VIF"] = [variance_inflation_factor(numeric_features.values, i) for i in range(len(numeric_features.columns))]


vif_data.sort_values(by="VIF", ascending=False)



train_balanced = train_balanced.drop(columns=['dewpoint'])

numeric_features = train_balanced.select_dtypes(include=['number'])
vif_data = pd.DataFrame()
vif_data["Feature"] = numeric_features.columns
vif_data["VIF"] = [variance_inflation_factor(numeric_features.values, i) for i in range(len(numeric_features.columns))]

vif_data.sort_values(by="VIF", ascending=False)




train_balanced = train_balanced.drop(columns=['cloud'])

numeric_features = train_balanced.select_dtypes(include=['number'])
vif_data = pd.DataFrame()
vif_data["Feature"] = numeric_features.columns
vif_data["VIF"] = [variance_inflation_factor(numeric_features.values, i) for i in range(len(numeric_features.columns))]

vif_data.sort_values(by="VIF", ascending=False)




from sklearn.preprocessing import StandardScaler

features = ['day', 'sunshine', 'winddirection', 'windspeed']
scaler = StandardScaler()
train_balanced[features] = scaler.fit_transform(train_balanced[features])

train_balanced.head()



from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
poly_features = poly.fit_transform(train_balanced[['sunshine', 'windspeed', 'winddirection']])

poly_feature_names = poly.get_feature_names_out(['sunshine', 'windspeed', 'winddirection'])
poly_df = pd.DataFrame(poly_features, columns=poly_feature_names)

train_balanced = pd.concat([train_balanced, poly_df], axis=1)

train_balanced.head()



from sklearn.feature_selection import mutual_info_classif

X_train = train_balanced.drop(columns=["rainfall"])  
y_train = train_balanced["rainfall"]

mi_scores = mutual_info_classif(X_train, y_train, random_state=42)

feature_names = X_train.columns.tolist()

print("Number of features:", len(feature_names))
print("Number of MI scores:", len(mi_scores))

if len(feature_names) == len(mi_scores):
    mi_df = pd.DataFrame({'Feature': feature_names, 'MI Score': mi_scores}).sort_values(by='MI Score', ascending=False)
    print(mi_df)
else:
    print("Error: Feature names and MI scores length mismatch!")




low_mi_features = [ "winddirection", "windspeed", "windspeed winddirection"]

train_selected = train_balanced.drop(columns=low_mi_features)

train_selected.head()




train_selected = train_selected.loc[:, ~train_selected.columns.duplicated()]

train_selected.head()



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = train_selected.drop(columns=['rainfall']) 
y = train_selected['rainfall']  

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

X_train.shape, X_test.shape, y_train.shape, y_test.shape



from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))



from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
    "SVM": SVC(kernel='rbf', random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{name} Accuracy: {acc:.4f}")
    print(f"Classification Report:\n{classification_report(y_test, y_pred)}")





