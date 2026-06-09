# SLOT 1: Imports and Loading with Warning Fix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import plotly.express as px
import warnings

# Silence runtime warnings temporarily
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ğŸ“¥ Load datasets
train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")

# âœ… Replace infinite values with NaN and handle them
train.replace([np.inf, -np.inf], np.nan, inplace=True)
test.replace([np.inf, -np.inf], np.nan, inplace=True)

# Optional: show count of NaNs
print("Missing in Train:\n", train.isnull().sum())
print("Missing in Test:\n", test.isnull().sum())

# Fill or drop if necessary (temporary for display)
train_display = train.fillna(0)
test_display = test.fillna(0)

# ğŸ”� Display safely
print("Train preview:")
print(train_display.head())
print("\nTrain Info:")
print(train_display.info())



# SLOT 2: Encode Labels
# Target: Personality â†’ Introvert/Extrovert
le = LabelEncoder()
train['Personality_encoded'] = le.fit_transform(train['Personality'])

# Remove any unnecessary columns
id_col = 'id'
target_col = 'Personality_encoded'
features = [col for col in train.columns if col not in [id_col, 'Personality', target_col]]

# Check for missing values
print("Missing in train:\n", train.isnull().sum())
print("Missing in test:\n", test.isnull().sum())

# Fill or drop nulls if necessary
train = train.dropna()
test = test.fillna(test.mean(numeric_only=True))  # fallback if missing in test



# SLOT 3: EDA (Improved with fixes)

# Make a copy to safely fill missing values just for visualization
train_vis = train.copy()
train_vis.replace([np.inf, -np.inf], np.nan, inplace=True)
train_vis.fillna(0, inplace=True)

# Double-check features list
features = [col for col in train.columns if col not in ['id', 'Personality', 'Personality_encoded']]


# 1. Personality Count Plot
if 'Personality' in train_vis.columns:
    sns.countplot(x='Personality', data=train_vis)
    plt.title("Personality Distribution")
    plt.show()


# 2. Feature Correlation Heatmap (only numeric features)
numeric_feats = train_vis[features + ['Personality_encoded']].select_dtypes(include=[np.number])
plt.figure(figsize=(10, 6))
sns.heatmap(numeric_feats.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()


# 3. Histograms (numeric only)
numeric_feats.hist(figsize=(14, 8), edgecolor='black')
plt.suptitle("Feature Distributions", fontsize=16)
plt.tight_layout()
plt.show()


# 4. Pairplot (only for small feature sets, max 5)
if 'Personality' in train_vis.columns and len(numeric_feats.columns) <= 5:
    pairplot_data = train_vis[numeric_feats.columns.tolist() + ['Personality']]
    sns.pairplot(pairplot_data, hue='Personality')
    plt.show()
else:
    print("Skipping pairplot: too many features or 'Personality' column not found.")


# SLOT 4: Model Training & Evaluation (Fixed without imports)

# Make copies to avoid modifying originals
train_model = train.copy()
test_model = test.copy()

# â�Œ Columns to exclude from features
exclude_cols = ['id', 'Personality', 'Personality_encoded']
features = [col for col in train_model.columns if col not in exclude_cols]

# ğŸ§¹ Handle missing values
train_model[features] = train_model[features].fillna("Missing")
test_model[features] = test_model[features].fillna("Missing")

# ğŸ”� Encode all object/categorical columns using pd.factorize (safe and reproducible)
for col in features:
    if train_model[col].dtype == 'object' or str(train_model[col].dtype).startswith('category'):
        train_model[col], uniques = pd.factorize(train_model[col])
        test_model[col] = test_model[col].apply(lambda x: np.where(uniques == x)[0][0] if x in uniques else -1)

# ğŸ�¯ Define input and target
X = train_model[features]
y = train_model['Personality_encoded']

# ğŸ”€ Train/Validation Split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# ğŸŒ³ Train Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# ğŸ”� Predict and evaluate
y_pred = clf.predict(X_valid)

print("âœ… Model trained successfully!")
print(classification_report(y_valid, y_pred))

# ğŸ“Š Confusion Matrix
sns.heatmap(confusion_matrix(y_valid, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()



# SLOT 6: Feature Importance
importances = clf.feature_importances_
feat_imp = pd.Series(importances, index=features).sort_values()

plt.figure(figsize=(10, 6))
feat_imp.plot(kind='barh', color='slateblue')
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.show()



# SLOT 7: Hyperparameter Tuning
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5],
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("ğŸ”§ Best Parameters:", grid_search.best_params_)
print("âœ… Best CV Accuracy:", grid_search.best_score_)

# Retrain with best parameters
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)



# SLOT 8: Model Explainability with SHAP
import shap

explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_valid)

# Plot SHAP summary
shap.summary_plot(shap_values[1], X_valid, plot_type="bar")

# Optional: Force plot for one sample
# shap.force_plot(explainer.expected_value[1], shap_values[1][0], X_valid.iloc[0])



# SLOT 9: Cross-Validation
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='accuracy')
print("ğŸ”� Cross-validation accuracy scores:", cv_scores)
print("ğŸ“Š Mean CV Accuracy: {:.2f}%".format(cv_scores.mean() * 100))



# SLOT 10: Exporting Model for Future Use
import joblib

joblib.dump(best_model, "best_personality_model.pkl")
print("âœ… Model saved as best_personality_model.pkl")



# SLOT 11: UMAP Clustering (Kaggle-Compatible)

# ğŸ“¦ Install UMAP if not already available in your Kaggle notebook
!pip install umap-learn --quiet

# ğŸ“š Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from sklearn.preprocessing import StandardScaler, LabelEncoder

# âœ… Assume train data is already loaded as `train`
# âœ… Assume features list is already defined (from earlier slots)

# Step 1: Separate categorical and numerical columns
categorical_cols = [col for col in features if train[col].dtype == "object"]
numerical_cols = [col for col in features if col not in categorical_cols]

# Step 2: Encode categorical columns (Label Encoding)
train_encoded = train.copy()
for col in categorical_cols:
    le = LabelEncoder()
    train_encoded[col] = le.fit_transform(train_encoded[col])

# Step 3: Combine final feature set
X_umap = train_encoded[numerical_cols + categorical_cols]

# Step 4: Scale numeric data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_umap)

# Step 5: Apply UMAP
umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
embedding = umap_model.fit_transform(X_scaled)

# Step 6: Visualize with Seaborn
umap_df = pd.DataFrame(embedding, columns=["UMAP1", "UMAP2"])
umap_df['Personality'] = train['Personality']

plt.figure(figsize=(10, 6))
sns.scatterplot(data=umap_df, x="UMAP1", y="UMAP2", hue="Personality", palette="coolwarm", s=60, alpha=0.8)
plt.title("ğŸ§  UMAP Projection of Personality Types")
plt.show()



# SLOT 12: Ensemble Model with Voting Classifier

from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Define base models
model1 = RandomForestClassifier(n_estimators=100, random_state=42)
model2 = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
model3 = AdaBoostClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
model4 = LogisticRegression(max_iter=1000)
model5 = SVC(probability=True)

# Combine in VotingClassifier
ensemble = VotingClassifier(
    estimators=[
        ('rf', model1),
        ('gb', model2),
        ('ada', model3),
        ('lr', model4),
        ('svc', model5)
    ],
    voting='soft'  # 'soft' uses predicted probabilities â€” better for AUC/log-loss
)

# Train on full training set
ensemble.fit(X_train, y_train)

# Evaluate
y_pred_ensemble = ensemble.predict(X_valid)

from sklearn.metrics import classification_report, confusion_matrix
print("âœ… Ensemble Model Evaluation")
print(classification_report(y_valid, y_pred_ensemble))
sns.heatmap(confusion_matrix(y_valid, y_pred_ensemble), annot=True, cmap='Blues')
plt.title("Ensemble Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()



# SLOT 5: Predict on Test Set (Safe for Kaggle Submission)

# ğŸ”� Copy and preprocess test set
test_encoded = test.copy()
test_encoded[features] = test_encoded[features].fillna("Missing")

# Encode test features using the same mapping from train_model
for col in features:
    train_uniques = train_model[col].unique()
    if train_model[col].dtype == 'int64':
        test_encoded[col] = test_encoded[col].apply(lambda x: x if x in train_uniques else -1)
    else:
        test_encoded[col] = test_encoded[col].apply(lambda x: np.where(train_uniques == x)[0][0] if x in train_uniques else -1)

# Extract features
X_test_final = test_encoded[features]

# ğŸ”� Predict using classifier
test_preds = clf.predict(X_test_final)

# ğŸ”„ Map encoded predictions back to label names
label_decoder = dict(zip(train['Personality_encoded'], train['Personality']))
test_preds_labels = pd.Series(test_preds).map(label_decoder)

# ğŸš« Handle any unmapped (null) predictions
test_preds_labels = test_preds_labels.fillna("Unknown")

# ğŸ“„ Create submission DataFrame
submission = pd.DataFrame({
    'id': test['id'],
    'Personality': test_preds_labels
})

# âœ… Save to CSV in strict Kaggle format
submission.to_csv("submission.csv", index=False, encoding='utf-8')
print("âœ… submission.csv generated successfully!")

# ğŸ”� Preview first few rows (if possible)
display(submission.head())


