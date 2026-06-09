# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


train_df=pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
train_df


test_df=pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
test_df


train_df.head()


train_df.info()


train_df.describe()


train_df.isnull().sum()


plt.figure(figsize=(10, 5))
sns.heatmap(train_df.isnull(), cbar=False, cmap="viridis")
plt.title("Missing Values Heatmap")
plt.show()



num_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside',
            'Friends_circle_size', 'Post_frequency']

train_df[num_cols].hist(bins=20, figsize=(12, 8))
plt.suptitle("Distributions of Numerical Features")
plt.show()


print(train_df['Personality'].unique())
print(train_df['Personality'].value_counts(dropna=False))


import matplotlib.pyplot as plt
import seaborn as sns

num_cols = [
    'Time_spent_Alone', 'Social_event_attendance',
    'Going_outside', 'Friends_circle_size', 'Post_frequency'
]

for col in num_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x='Personality', y=col, data=train_df)
    plt.title(f"{col} vs Personality (0 = Introvert, 1 = Extrovert)")
    plt.xlabel("Personality")
    plt.ylabel(col)
    plt.show()



print(train_df['Personality'].unique())
print(train_df['Personality'].value_counts(dropna=False))



print(train_df['Personality'].unique())
print(train_df['Personality'].value_counts(dropna=False))



import seaborn as sns
import matplotlib.pyplot as plt


train_df['Personality'] = train_df['Personality'].map({'Introvert': 0, 'Extrovert': 1})

cat_cols = ['Stage_fear', 'Drained_after_socializing']

# Drop rows with missing values in categorical columns 
plot_df = train_df.dropna(subset=cat_cols + ['Personality'])


for col in cat_cols:
    plt.figure(figsize=(6, 4))
    sns.countplot(x=col, hue='Personality', data=plot_df)
    plt.title(f"{col} vs Personality (0 = Introvert, 1 = Extrovert)")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.legend(title="Personality", labels=["Introvert", "Extrovert"])
    plt.tight_layout()
    plt.show()



# Fill all numeric NaNs with median
for col in ['Time_spent_Alone', 'Post_frequency', 'Friends_circle_size',
            'Social_event_attendance', 'Going_outside']:
    train_df[col].fillna(train_df[col].median(), inplace=True)



from sklearn.utils import resample

# Split classes
df_majority = train_df[train_df['Personality'] == 1]  # Extroverts
df_minority = train_df[train_df['Personality'] == 0]  # Introverts

# Upsample minority
df_minority_upsampled = resample(
    df_minority,
    replace=True,
    n_samples=len(df_majority),
    random_state=42
)

# Combine and shuffle
train_df_balanced = pd.concat([df_majority, df_minority_upsampled])
train_df_balanced = train_df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)



train_df_balanced


train_df_balanced.isnull().sum()


# Fill missing binary/categorical features with mode
train_df_balanced['Stage_fear'].fillna(train_df_balanced['Stage_fear'].mode()[0], inplace=True)
train_df_balanced['Drained_after_socializing'].fillna(train_df_balanced['Drained_after_socializing'].mode()[0], inplace=True)



train_df_balanced.isnull().sum()


print(train_df_balanced['Personality'].value_counts())



# Map numeric back to string labels for visualization
train_df_balanced['Personality_Label'] = train_df_balanced['Personality'].map({0: 'Introvert', 1: 'Extrovert'})



# Map string labels to numeric
train_df['Personality'] = train_df_balanced['Personality'].map({'Introvert': 0, 'Extrovert': 1})


# Fill missing binary/categorical features with mode
train_df_balanced['Stage_fear'].fillna(train_df_balanced['Stage_fear'].mode()[0], inplace=True)
train_df_balanced['Drained_after_socializing'].fillna(train_df_balanced['Drained_after_socializing'].mode()[0], inplace=True)



train_df_balanced.isnull().sum()


# Map Yes/No to 1/0 for relevant columns
binary_columns = ['Stage_fear', 'Drained_after_socializing']

for col in binary_columns:
    train_df_balanced[col] = train_df_balanced[col].map({'Yes': 1, 'No': 0})



train_df_balanced.drop(columns=['Personality_Label'], inplace=True)


train_df_balanced


from sklearn.preprocessing import StandardScaler
import pandas as pd

features = [
    'Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
    'Going_outside', 'Drained_after_socializing',
    'Friends_circle_size', 'Post_frequency'
]

# Extract features and target
X = train_df_balanced[features]
y = train_df_balanced['Personality']

# Normalize the features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_scaled_df = pd.DataFrame(X_scaled, columns=features)
X_scaled_df['Personality'] = y.values 



X_scaled_df


import statsmodels.api as sm

# Extract normalized X and target y
X = X_scaled_df[[
    'Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
    'Going_outside', 'Drained_after_socializing',
    'Friends_circle_size', 'Post_frequency'
]]
y = X_scaled_df['Personality'].astype(int)

# Add constant term for intercept
X = sm.add_constant(X)

# Fit the logistic regression model
logit_model = sm.Logit(y, X)
result = logit_model.fit()

# Display model summary
print(result.summary())



from sklearn.model_selection import train_test_split

# Separate X and y
X = X_scaled_df[features]
y = X_scaled_df['Personality']

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)



from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)

# K-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Support Vector Machine
svm = SVC(probability=True)
svm.fit(X_train, y_train)

# MLP Classifier (Neural Network)
mlp = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=300, random_state=42)
mlp.fit(X_train, y_train)

# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# XGBoost
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb.fit(X_train, y_train)

# Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)

# Linear Discriminant Analysis
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

# Quadratic Discriminant Analysis
qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train, y_train)



from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Create figure
plt.figure(figsize=(12, 8))

def plot_roc(model, name, color=None):
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_val)[:, 1]
    else:  # For models like SVM with no predict_proba unless specified
        y_prob = model.decision_function(X_val)
    fpr, tpr, _ = roc_curve(y_val, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})", color=color)

# Plot ROC curves for all models
plot_roc(lr, "Logistic Regression")
plot_roc(knn, "KNN")
plot_roc(svm, "SVM")
plot_roc(mlp, "MLP")
plot_roc(dt, "Decision Tree")
plot_roc(rf, "Random Forest")
plot_roc(xgb, "XGBoost")
plot_roc(nb, "Naive Bayes")
plot_roc(lda, "LDA")
plot_roc(qda, "QDA")

# Plot random chance line
plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')


plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()



from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Dictionary of all models
models = {
    "Logistic Regression": lr,
    "KNN": knn,
    "SVM": svm,
    "MLP": mlp,
    "Decision Tree": dt,
    "Random Forest": rf,
    "XGBoost": xgb,
    "Naive Bayes": nb,
    "LDA": lda,
    "QDA": qda
}

# Plot confusion matrix for each model
for name, model in models.items():
    y_pred = model.predict(X_val)
    cm = confusion_matrix(y_val, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Introvert", "Extrovert"])
    disp.plot(cmap='Blues')
    plt.title(f"Confusion Matrix: {name}")
    plt.grid(False)
    plt.show()



from sklearn.ensemble import VotingClassifier

voting_clf = VotingClassifier(
    estimators=[
        ('lr', lr), ('knn', knn), ('svm', svm), ('mlp', mlp)
    ],
    voting='soft'
)
voting_clf.fit(X_train, y_train)

# Evaluate
from sklearn.metrics import roc_auc_score
y_prob = voting_clf.predict_proba(X_val)[:, 1]
print("VotingClassifier AUC:", roc_auc_score(y_val, y_prob))



from sklearn.preprocessing import StandardScaler
import pandas as pd

# Step 0: Make a copy
test_df_cleaned = test_df.copy()

# Step 1: Convert Yes/No to 1/0
test_df_cleaned['Stage_fear'] = test_df_cleaned['Stage_fear'].map({'Yes': 1, 'No': 0})
test_df_cleaned['Drained_after_socializing'] = test_df_cleaned['Drained_after_socializing'].map({'Yes': 1, 'No': 0})

# Step 2: Fill NaN values with column-wise mean
test_df_cleaned.fillna(test_df_cleaned.mean(numeric_only=True), inplace=True)

# Step 3: Define feature columns
features = [
    'Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
    'Going_outside', 'Drained_after_socializing',
    'Friends_circle_size', 'Post_frequency'
]

# Step 4: Apply standard scaler (must be the same one fitted on training data)
X_test_scaled = scaler.transform(test_df_cleaned[features])

# Optional: Convert back to DataFrame
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=features)



X_test_scaled_df


import pandas as pd

# Define label map
label_map = {0: 'Introvert', 1: 'Extrovert'}

# Create list of models and names
models = [
    ('logistic_regression', lr),
    ('knn', knn),
    ('svm', svm),
    ('mlp', mlp),
    ('decision_tree', dt),
    ('random_forest', rf),
    ('xgboost', xgb),
    ('naive_bayes', nb),
    ('lda', lda),
    ('qda', qda)
]

# Predict and save submission for each model
for name, model in models:
    preds = model.predict(X_test_scaled_df[features])
    preds_labels = pd.Series(preds).map(label_map)

    submission_df = pd.DataFrame({
        'id': test_df['id'],
        'Personality': preds_labels
    })

    submission_df.to_csv(f"submission_{name}.csv", index=False)

# VotingClassifier separately with probability threshold
test_probs_voting = voting_clf.predict_proba(X_test_scaled_df[features])[:, 1]
test_preds_voting = (test_probs_voting >= 0.5).astype(int)
test_labels_voting = pd.Series(test_preds_voting).map(label_map)

submission_voting = pd.DataFrame({
    'id': test_df['id'],
    'Personality': test_labels_voting
})
submission_voting.to_csv("submission_voting_classifier.csv", index=False)



import seaborn as sns

# Feature importances from Random Forest
importances = rf.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Plot
plt.figure(figsize=(8, 5))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
plt.title('Feature Importance - Random Forest')
plt.tight_layout()
plt.show()



from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd

# List of models and their names
models = [
    ('lr', lr),
    ('knn', knn),
    ('svm', svm),
    ('mlp', mlp),
    ('dt', dt),
    ('rf', rf),
    ('xgb', xgb),
    ('nb', nb),
    ('lda', lda),
    ('qda', qda)
]


model_auc = {}
for name, model in models:
    try:
        y_val_proba = model.predict_proba(X_val)[:, 1]
        auc_score = roc_auc_score(y_val, y_val_proba)
        model_auc[name] = auc_score
    except AttributeError:
        print(f"Model '{name}' does not support predict_proba; skipping.")
        continue

# Normalize AUCs to create weights
total_auc = sum(model_auc.values())
model_weights = {k: v / total_auc for k, v in model_auc.items()}

# Weighted ensemble prediction on test data
weighted_sum = np.zeros(len(X_test_scaled_df))
for name, model in models:
    if name in model_weights:  # only use models that supported predict_proba
        prob = model.predict_proba(X_test_scaled_df[features])[:, 1]
        weighted_sum += prob * model_weights[name]

# Final prediction
final_preds = (weighted_sum >= 0.5).astype(int)
final_labels = pd.Series(final_preds).map({0: 'Introvert', 1: 'Extrovert'})

# Create and save submission file
submission_weighted = pd.DataFrame({
    'id': test_df['id'],
    'Personality': final_labels
})

submission_weighted.to_csv("submission_weighted_ensemble.csv", index=False)

# Optional: print weights
print("Model Weights based on AUC:")
for name, weight in model_weights.items():
    print(f"{name}: {weight:.3f}")


