import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')


# Modeling imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.impute import SimpleImputer

# Clustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# Display Configs
plt.style.use('seaborn-whitegrid')
sns.set_palette('viridis')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 150)


print("âœ… Libraries Imported")


competitions = pd.read_csv('/kaggle/input/meta-kaggle/Competitions.csv')
users = pd.read_csv('/kaggle/input/meta-kaggle/Users.csv')
submissions = pd.read_csv('/kaggle/input/meta-kaggle/Submissions.csv')
team_members = pd.read_csv('/kaggle/input/meta-kaggle/TeamMemberships.csv')
teams = pd.read_csv('/kaggle/input/meta-kaggle/Teams.csv')

print("ğŸ“Š Data Loaded:")
print(f"Competitions: {competitions.shape}")
print(f"Users: {users.shape}")
print(f"Submissions: {submissions.shape}")
print(f"Teams: {teams.shape}")
print(f"Team Members: {team_members.shape}")


# Initial Cleaning
competitions.dropna(subset=['Id'], inplace=True)
competitions['DeadlineDate'] = pd.to_datetime(competitions['DeadlineDate'], errors='coerce')
competitions['EnabledDate'] = pd.to_datetime(competitions['EnabledDate'], errors='coerce')


# Preview top competitions
top_comps = competitions[['Id', 'Title', 'EnabledDate', 'DeadlineDate', 'RewardQuantity', 'RewardType']].sort_values(by='RewardQuantity', ascending=False).head(5)
print("\nğŸ’° Top 5 Reward Competitions:")
print(top_comps)


# Visual EDA example
plt.figure(figsize=(10, 6))
sns.histplot(competitions['RewardQuantity'].fillna(0), bins=50, color='skyblue')
plt.title("Reward Distribution across Competitions")
plt.xlabel("Reward Amount")
plt.ylabel("Frequency")
plt.show()



# Duration feature
competitions['DurationDays'] = (competitions['DeadlineDate'] - competitions['EnabledDate']).dt.days


# Plot 1: Duration distribution
plt.figure(figsize=(10, 5))
sns.histplot(competitions['DurationDays'].dropna(), bins=40, kde=True, color='darkcyan')
plt.title("â�±ï¸� Competition Duration (days)")
plt.xlabel("Duration")
plt.ylabel("Frequency")
plt.show()


# Plot 2: Competitions launched per year
competitions['EnabledYear'] = competitions['EnabledDate'].dt.year
plt.figure(figsize=(10, 5))
sns.countplot(data=competitions, x='EnabledYear', palette='mako')
plt.title("ğŸ“… Number of Competitions Launched Per Year")
plt.xticks(rotation=45)
plt.show()


# Plot 3: Monthly trend
competitions['EnabledMonth'] = competitions['EnabledDate'].dt.month
plt.figure(figsize=(10, 5))
sns.countplot(data=competitions, x='EnabledMonth', palette='rocket')
plt.title("ğŸ—“ï¸� Monthly Distribution of Competition Launches")
plt.xlabel("Month")
plt.ylabel("Count")
plt.show()


# Correlation Matrix
num_cols = ['RewardQuantity', 'DurationDays']
corr_matrix = competitions[num_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("ğŸ“ˆ Correlation Matrix")
plt.show()


# Encode categorical variables
competitions['RewardType_encoded'] = LabelEncoder().fit_transform(competitions['RewardType'].astype(str))
competitions['HostSegmentTitle_encoded'] = LabelEncoder().fit_transform(competitions['HostSegmentTitle'].astype(str))


# Fill missing numerical values
imputer = SimpleImputer(strategy='median')
numerical_features = ['MaxDailySubmissions', 'MaxTeamSize', 'NumPrizes', 'RewardQuantity', 'DurationDays']
competitions[numerical_features] = imputer.fit_transform(competitions[numerical_features])


# Create a binary target for popularity
competitions['TotalTeams'] = competitions['TotalTeams'].fillna(0)
competitions['IsPopular'] = (competitions['TotalTeams'] > 45).astype(int)


# Final modeling DataFrame
features = numerical_features + ['RewardType_encoded', 'HostSegmentTitle_encoded']
model_df = competitions[features + ['IsPopular']].dropna()


print("ğŸ§ª Feature matrix shape:", model_df.shape)
print(model_df.head())


# Split data
X = model_df[features]
y = model_df['IsPopular']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)



# XGBoost model with class weights
scale_pos_weight = (y == 0).sum() / (y == 1).sum()
xgb_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', scale_pos_weight=scale_pos_weight, use_label_encoder=False)
xgb_model.fit(X_train, y_train)


# Predictions
y_pred = xgb_model.predict(X_test)
y_prob = xgb_model.predict_proba(X_test)[:, 1]


from sklearn.metrics import roc_curve, roc_auc_score


# Evaluation
print("\nâœ… Accuracy:", accuracy_score(y_test, y_pred))
print("\nğŸ“Š Classification Report:\n", classification_report(y_test, y_pred))
print("\nğŸ“‰ ROC AUC Score:", roc_auc_score(y_test, y_prob))



from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"ğŸ“Š {name} Classification Report:\n")
    print(classification_report(y_test, y_pred))
    print("-" * 60)



from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Prepare data (assuming X_train, X_test, y_train, y_test already exist)
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

model_results = {}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    model_results[model_name] = {
        "Accuracy": acc,
        "F1 Score": f1,
        "Classification Report": classification_report(y_test, y_pred, output_dict=True)
    }

# Display model results
for model_name, result in model_results.items():
    print(f"ğŸ”¹ {model_name}")
    print(f"âœ… Accuracy: {result['Accuracy']:.4f}")
    print(f"âœ… F1 Score: {result['F1 Score']:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, models[model_name].predict(X_test)))
    print("-" * 60)



import shap
import xgboost # Assuming xgb_model is an XGBoost model


# SHAP for Interpretability
explainer = shap.Explainer(xgb_model, X_train)
shap_values = explainer(X_test[:100])
shap.plots.beeswarm(shap_values)


import shap

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Use SHAP for interpretation
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)

# Plot summary for class 1
shap.summary_plot(shap_values[1], X_test, plot_type="bar")



from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 6))

# ROC Curve
plt.subplot(1, 2, 1)
for model_name, model in models.items():
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ğŸ“Š ROC Curve')
plt.legend()

# Precision-Recall Curve
plt.subplot(1, 2, 2)
for model_name, model in models.items():
    y_prob = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    plt.plot(recall, precision, label=model_name)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('ğŸ“ˆ Precision-Recall Curve')
plt.legend()

plt.tight_layout()
plt.show()



# Standardize data for clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(model_df[features])


# Try 2 to 8 clusters and compute silhouette scores
silhouette_scores = []
for k in range(2, 9):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(X_scaled)
    silhouette = silhouette_score(X_scaled, labels)
    silhouette_scores.append(silhouette)
    print(f"K={k}, Silhouette Score: {silhouette:.4f}")


# Plot silhouette scores
plt.figure(figsize=(10, 5))
plt.plot(range(2, 9), silhouette_scores, marker='o', linestyle='-', color='navy')
plt.title("ğŸ§© Optimal Clusters via Silhouette Score")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.grid(True)
plt.show()


# Apply best clustering (let's assume 3)
kmeans_final = KMeans(n_clusters=3, random_state=42, n_init='auto')
model_df['Cluster'] = kmeans_final.fit_predict(X_scaled)


# Visualize clusters by Reward & Duration
plt.figure(figsize=(10, 6))
sns.scatterplot(data=model_df, x='RewardQuantity', y='DurationDays', hue='Cluster', palette='Set2')
plt.title("ğŸ“� Competition Clusters by Reward & Duration")
plt.xlabel("Reward Amount")
plt.ylabel("Competition Duration (Days)")
plt.legend()
plt.show()

