import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.preprocessing import LabelEncoder

# Load your dataset
df = pd.read_csv("/kaggle/input/playground-series-s4e2/train.csv")

# Define categorical and numerical features
categorical_features = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
numerical_features = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
target_variable = 'NObeyesdad'

# Encode categorical features using Label Encoding
df_encoded = df.copy()
for col in categorical_features:
    df_encoded[col] = df_encoded[col].astype('category').cat.codes  # Convert categorical to numeric

# Encode the target variable
le = LabelEncoder()
df_encoded[target_variable] = le.fit_transform(df_encoded[target_variable])

# Compute Pearson correlation for numerical features
corr_matrix = df_encoded[numerical_features + categorical_features + [target_variable]].corr(method='pearson')

# Plot correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Matrix with Obesity Level")
plt.show()



import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np

df = pd.read_csv("/kaggle/input/playground-series-s4e2/train.csv")

# Define categorical and numerical features
categorical_features = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
numerical_features = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']

# Define feature matrix and target
X = df[categorical_features + numerical_features]  # Use only selected features
y = df['NObeyesdad']  # Target variable

# Encode categorical variables
X_encoded = X.copy()
for col in categorical_features:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X_encoded[col])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Apply SelectKBest and get both F-scores and p-values
k = min(10, X_train.shape[1])  # Select top 10 features or all if fewer
BestFeatures = SelectKBest(score_func=f_classif, k=k)  # Using f_classif for classification
fit = BestFeatures.fit(X_train, y_train)

# Create DataFrame for feature scores and p-values
df_scores = pd.DataFrame(fit.scores_, columns=['F-Score'])
df_pvalues = pd.DataFrame(fit.pvalues_, columns=['P-Value'])
df_columns = pd.DataFrame(X_train.columns, columns=['Feature'])

# Combine into a single DataFrame
f_Scores = pd.concat([df_columns, df_scores, df_pvalues], axis=1)

# Print the top 10 features
print("\nTop 10 Selected Features Based on Score:")
print(f_Scores.nsmallest(10, 'P-Value'))  # Features with lowest p-values are most significant

# Plot F-Scores and P-Values
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot F-Scores
color = 'tab:blue'
ax1.set_xlabel('Features')
ax1.set_ylabel('F-Score', color=color)
ax1.barh(f_Scores['Feature'], f_Scores['F-Score'], color=color, alpha=0.6, label='F-Score')
ax1.tick_params(axis='y', labelcolor=color)
ax1.invert_yaxis()  # Highest values on top

# Create second y-axis for P-Values
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('P-Value', color=color)
ax2.plot(f_Scores['Feature'], f_Scores['P-Value'], 'ro-', label='P-Value')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Feature Selection: F-Scores and P-Values')
plt.show()



import pandas as pd
import lightgbm as lgb
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

df = pd.read_csv("/kaggle/input/playground-series-s4e2/train.csv")

# Define categorical and numerical features
categorical_features = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
numerical_features = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']

# Define feature matrix and target
X = df[categorical_features + numerical_features]  # Use only selected features
y = df['NObeyesdad']  # Target variable

# Encode categorical variables before feature selection
X_encoded = X.copy()
for col in categorical_features:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X_encoded[col])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Apply SelectKBest to select top 10 best features
k = min(16, X_train.shape[1])  # Select top 10 features or all if less than 10
BestFeatures = SelectKBest(score_func=chi2, k=k)
fit = BestFeatures.fit(X_train, y_train)

# Create DataFrame to visualize feature scores
df_scores = pd.DataFrame(fit.scores_)
df_columns = pd.DataFrame(X_train.columns)
f_Scores = pd.concat([df_columns, df_scores], axis=1)  # Combine features with scores
f_Scores.columns = ['Feature', 'Score']

# Print feature ranking
print("\nFeature Ranking Based on SelectKBest Scores:")
print(f_Scores.nlargest(16, 'Score'))  # Print 10 best features in descending order

# Get selected feature names
selected_feature_names = X_train.columns[BestFeatures.get_support()]
print("\nSelected Features for Model Training:", selected_feature_names.tolist())

# Transform dataset using selected features
X_train_selected = BestFeatures.transform(X_train)
X_test_selected = BestFeatures.transform(X_test)

# Train LightGBM model with selected features
model = lgb.LGBMClassifier()
model.fit(X_train_selected, y_train)
accuracy = model.score(X_test_selected, y_test)

print(f"\nModel Accuracy with Selected Features: {accuracy}")



import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd

# Train LightGBM model
lgb_model = lgb.LGBMClassifier(random_state=0)
lgb_model.fit(X_train, y_train)

# Print feature importance values
print("Feature Importances:", lgb_model.feature_importances_)

# Create a DataFrame for better visualization
feature_importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': lgb_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Print the ranked feature importances
print("\nTop Features by Importance:\n", feature_importance_df)

# Plot feature importance
lgb.plot_importance(lgb_model, max_num_features=10, importance_type='split')  # Use 'gain' for information gain
plt.title("LightGBM Feature Importance")
plt.show()


