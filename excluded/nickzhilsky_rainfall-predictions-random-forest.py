import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support, mean_absolute_error, mean_squared_error, r2_score


df_train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')


df_train.shape,df_test.shape


missing_values_train = df_train.isnull().sum().sum()
missing_values_per_train = missing_values_train/df_train.size
print(f"Missing values - {round(missing_values_per_train, 10)}%")


missing_values_test = df_test.isnull().sum().sum()
missing_values_per_test = missing_values_test/df_test.size
print(f"Missing values - {round(missing_values_per_test, 10)}%")


missing_test = df_test.isnull().sum()


print("\nmissing values in test dataset df_test:")
print(missing_test[missing_test > 0])


df_test['winddirection'] = df_test['winddirection'].fillna(df_test['winddirection'].mean())


df_info = pd.DataFrame({
    "DataType": df_train.dtypes,
    "MissingValues": df_train.isnull().sum(),
    "UniqueValues": df_train.nunique()
}).sort_values(by="MissingValues", ascending=False)

df_info['MissingValuesRatio'] = round(df_info['MissingValues'] / len(df_train),2)

print(df_info)


df_test_info = pd.DataFrame({
    "DataType": df_test.dtypes,
    "MissingValues": df_test.isnull().sum(),
    "UniqueValues": df_test.nunique()
}).sort_values(by="MissingValues", ascending=False)

df_test_info['MissingValuesRatio'] = round(df_info['MissingValues'] / len(df_test),2)

print(df_test_info)


numerical_columns = ['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 
                     'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed']
binary_column = 'rainfall'  # Binary indicator (1 - Yes, 0 - No)


plt.figure(figsize=(16, 10))
for i, column in enumerate(numerical_columns, 1):
    plt.subplot(3, 4, i)
    ax = sns.boxplot(x=df_train[binary_column], y=df_train[column], hue=df_train[binary_column], palette="viridis")
    ax.get_legend().set_visible(False)
    plt.title(f'{column} vs Rainfall')
plt.tight_layout()
plt.show()


import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

for col in df_train.columns[:-1]:
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 2)
    sns.histplot(data=df_train, x=col, hue='rainfall', palette="bright", kde=True)
    plt.show()


plt.figure(figsize=(12, 8))
sns.heatmap(data=df_train.corr(), annot=True, linewidths=0.2, cmap="RdYlBu")


def detect_outliers(group):
    Q1 = group.quantile(0.25)
    Q3 = group.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (group < lower_bound) | (group > upper_bound)

outliers_dict = {}
for column in numerical_columns:
    outliers_dict[column] = df_train.groupby(binary_column)[column].transform(detect_outliers)

outliers_df = pd.DataFrame(outliers_dict)
outliers_df[binary_column] = df_train[binary_column]

outliers_summary = outliers_df.groupby(binary_column).sum(numeric_only=True)
print(outliers_summary)


def cramers_v(confusion_matrix):
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    
    denominator = min((kcorr - 1), (rcorr - 1))
    if denominator <= 0:  # Prevent division by zero
        return 0

    return np.sqrt(phi2corr / denominator)


def calculate_cramers_v(df_train, target):
    results = {}
    for col in df_train.columns:
        if col != target:
            confusion_matrix = pd.crosstab(df_train[col], df_train[target])
            results[col] = cramers_v(confusion_matrix)
    return results


cramers_v_results = calculate_cramers_v(df_train, binary_column)


sorted_cramers_v = sorted(cramers_v_results.items(), key=lambda x: x[1], reverse=True)
print("\nCramÃ©r's V Results (sorted by strength of association):")
for feature, value in sorted_cramers_v:
    print(f"{feature}: {value:.4f}")


df_cramer = pd.DataFrame(list(cramers_v_results.items()), columns=['Feature', 'Cramer_V'])
df_cramer = df_cramer.sort_values(by='Cramer_V', ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x="Cramer_V", y="Feature", data=df_cramer, hue="Feature", palette="coolwarm",)
plt.xlabel("CramÃ©r's V (Strength of Association)")
plt.ylabel("Feature")
plt.title("Strength of Association Between Features and Rainfall")
plt.xlim(0, 1)
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.show()


X_train = df_train.drop(columns=['rainfall'])
y_train = df_train['rainfall']
X_test = df_test
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)
rainfall_probabilities = model.predict_proba(X_test_scaled)[:, 1]  
df_results = pd.DataFrame({'id': df_test['id'], 'rainfall': rainfall_probabilities})
df_results.to_csv("rainfall_predictions.csv", index=False)
print(df_results.head())



cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
roc_auc_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc')

print(f"ğŸ”¹ Cross-Validation Accuracy (Mean Â± Std): {np.mean(cv_scores):.4f} Â± {np.std(cv_scores):.4f}")
print(f"ğŸ”¹ Cross-Validation ROC-AUC (Mean Â± Std): {np.mean(roc_auc_scores):.4f} Â± {np.std(roc_auc_scores):.4f}")


rainfall_predictions_train = model.predict(X_train_scaled)
rainfall_probabilities_train = model.predict_proba(X_train_scaled)[:, 1]

accuracy = accuracy_score(y_train, rainfall_predictions_train)
roc_auc = roc_auc_score(y_train, rainfall_probabilities_train)
precision, recall, f1, _ = precision_recall_fscore_support(y_train, rainfall_predictions_train, average='binary')

print(f"ğŸ”¹ Accuracy: {accuracy:.4f}")
print(f"ğŸ”¹ ROC-AUC Score: {roc_auc:.4f}")
print(f"ğŸ”¹ Precision: {precision:.4f}")
print(f"ğŸ”¹ Recall: {recall:.4f}")
print(f"ğŸ”¹ F1 Score: {f1:.4f}")


mae = mean_absolute_error(y_train, rainfall_probabilities_train)
mse = mean_squared_error(y_train, rainfall_probabilities_train)
rmse = mse ** 0.5
r2 = r2_score(y_train, rainfall_probabilities_train)

print(f"ğŸ”¹ Mean Absolute Error (MAE): {mae:.4f}")
print(f"ğŸ”¹ Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"ğŸ”¹ RÂ² Score: {r2:.4f}")


plt.figure(figsize = (8, 5))
sns.histplot(rainfall_probabilities_train, bins = 20, kde = True)
plt.title("Distribution of Predicted Probabilities (Training Data)")
plt.xlabel("Rainfall Probability")
plt.ylabel("Frequency")
plt.show()


df_new = df_train.drop(columns = ['id','day','maxtemp','mintemp','dewpoint','pressure','cloud' ])
df_test_new = df_test.drop(columns = ['id','day','maxtemp','mintemp','dewpoint','pressure','cloud' ])


plt.figure(figsize = (8, 6))
sns.heatmap(data = df_new.corr(), annot = True, linewidths = 0.2, cmap = "RdYlBu")


X_train_new = df_new.drop(columns=['rainfall'])
y_train_new = df_new['rainfall']
X_test_new = df_test_new
scaler = StandardScaler()
X_train_scaled_new = scaler.fit_transform(X_train_new)
X_test_scaled_new = scaler.transform(X_test_new)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled_new, y_train)
rainfall_probabilities_new = model.predict_proba(X_test_scaled_new)[:, 1]  
df_results_new = pd.DataFrame({'id': df_test['id'], 'rainfall': rainfall_probabilities_new})
df_results_new.to_csv("sample_submission.csv", index=False)
print(df_results_new.head())


cv_scores_new = cross_val_score(model, X_train_scaled_new, y_train_new, cv=5, scoring='accuracy')
roc_auc_scores_new = cross_val_score(model, X_train_scaled_new, y_train_new, cv=5, scoring='roc_auc')

print(f"ğŸ”¹ New cross-Validation Accuracy (Mean Â± Std): {np.mean(cv_scores_new):.4f} Â± {np.std(cv_scores_new):.4f}")
print(f"ğŸ”¹ New cross-Validation ROC-AUC (Mean Â± Std): {np.mean(roc_auc_scores_new):.4f} Â± {np.std(roc_auc_scores_new):.4f}")


rainfall_predictions_train_new = model.predict(X_train_scaled_new)
rainfall_probabilities_train_new = model.predict_proba(X_train_scaled_new)[:, 1]

accuracy_new = accuracy_score(y_train_new, rainfall_predictions_train_new)
roc_auc_new = roc_auc_score(y_train_new, rainfall_probabilities_train_new)
precision_new, recall_new, f1_new, _ = precision_recall_fscore_support(y_train_new, rainfall_predictions_train_new, average='binary')

print(f"ğŸ”¹ Accuracy: {accuracy_new:.4f}")
print(f"ğŸ”¹ ROC-AUC Score: {roc_auc_new:.4f}")
print(f"ğŸ”¹ Precision: {precision_new:.4f}")
print(f"ğŸ”¹ Recall: {recall_new:.4f}")
print(f"ğŸ”¹ F1 Score: {f1_new:.4f}")


mae_new = mean_absolute_error(y_train_new, rainfall_probabilities_train_new)
mse_new = mean_squared_error(y_train_new, rainfall_probabilities_train_new)
rmse_new = mse_new ** 0.5
r2_new = r2_score(y_train_new, rainfall_probabilities_train_new)

print(f"ğŸ”¹ Mean Absolute Error (MAE): {mae_new:.4f}")
print(f"ğŸ”¹ Root Mean Squared Error (RMSE): {rmse_new:.4f}")
print(f"ğŸ”¹ RÂ² Score: {r2_new:.4f}")


plt.figure(figsize=(8, 5))
sns.histplot(rainfall_probabilities_train_new, bins=20, kde=True)
plt.title("Distribution of Predicted Probabilities (Training Data) - Evaluated Model")
plt.xlabel("Rainfall Probability")
plt.ylabel("Frequency")
plt.show()

