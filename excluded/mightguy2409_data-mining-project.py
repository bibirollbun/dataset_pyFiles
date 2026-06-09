import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



data = pd.read_csv("/kaggle/input/playground-series-s4e11/train.csv")


data.head()


data.shape


print(data.isnull().values.any())


total_missing = data.isnull().sum().sum()
print("Total NaN values:", total_missing)


print(data.columns.tolist())



ws_data = data.groupby('Working Professional or Student')
data = ws_data.get_group('Working Professional').copy()



data['Working Professional or Student'].value_counts()


# these columns contribute to almost all nan values in the dataset so it would be better to drop them
columns=['CGPA','Academic Pressure','Study Satisfaction','Working Professional or Student']
data = data.drop(columns=['CGPA','Academic Pressure','Study Satisfaction','Working Professional or Student'],axis=1)


total = data.isnull().sum().sum()
print("Total NaN values after dropping the columns :", total)


#this tells us where the remaining nan values are
data.isnull().sum()[data.isnull().sum() > 0]



data['Profession'].unique()


replace_map = {
    'Finanancial Analyst': 'Financial Analyst',
    'Medical Doctor': 'Doctor',
    'MBBS': 'Doctor',
    'Research Analyst': 'Researcher',
    'City Manager': 'Manager',
}
data['Profession'] = data['Profession'].replace(replace_map)



valid_professions = {
    'Chef', 'Teacher', 'Business Analyst', 'Financial Analyst', 'Chemist',
    'Electrician', 'Software Engineer', 'Data Scientist', 'Plumber',
    'Marketing Manager', 'Accountant', 'Entrepreneur', 'HR Manager',
    'UX/UI Designer', 'Content Writer', 'Educational Consultant',
    'Civil Engineer', 'Manager', 'Pharmacist', 'Architect',
    'Mechanical Engineer', 'Customer Support', 'Consultant', 'Judge',
    'Researcher', 'Pilot', 'Graphic Designer', 'Travel Consultant',
    'Digital Marketer', 'Lawyer', 'Sales Executive', 'Doctor',
    'Investment Banker', 'Family Consultant'
}

special_values = {'Student', 'Unemployed', 'Working Professional', 'Academic', 'Profession'}


def clean_profession(x):
    if pd.isna(x):
        return np.nan
    x = x.strip()
    if x in special_values:
        return x
    return x if x in valid_professions else 'Other'


data['Profession'] = data['Profession'].apply(clean_profession)

# 3) handle missing
data['Profession'] = data['Profession'].fillna('Unknown')

# 4) optional: collapse rares
freq = data['Profession'].value_counts()
rare = freq[freq < 100].index
data.loc[data['Profession'].isin(rare), 'Profession'] = 'Other'


data['Degree'].head()


columns = ['Profession', 'Work Pressure', 'Job Satisfaction']

for col in columns:
    print(f"\nUnique values in '{col}':")
    print(data[col].dropna().unique())



# Fill missing values based on column type
#data['Profession'].fillna(data['Profession'].mode()[0], inplace=True)

data['Work Pressure'].fillna(data['Work Pressure'].median(), inplace=True)

data['Job Satisfaction'].fillna(data['Job Satisfaction'].median(), inplace=True)
data['Degree'].fillna(data['Degree'].mode()[0], inplace=True)
data['Dietary Habits'].fillna(data['Dietary Habits'].mode()[0], inplace=True)
data['Financial Stress'].fillna(data['Financial Stress'].mode()[0], inplace=True)


print(data.isnull().sum())



data['Financial Stress'].dropna(inplace=True)
data['Financial Stress'].isna().sum()


print(data.isnull().sum())


data.dtypes



print(data.dtypes)



# Univariate Analysis - Distribution of Key Variables
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Univariate Analysis: Distribution of Key Variables', fontsize=16, fontweight='bold', y=1.00)

# 1. Age Distribution
axes[0, 0].hist(data['Age'], bins=25, color='#3498db', edgecolor='black', alpha=0.7)
axes[0, 0].axvline(data['Age'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {data["Age"].mean():.1f}')
axes[0, 0].axvline(data['Age'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {data["Age"].median():.1f}')
axes[0, 0].set_title('Age Distribution', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Age')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].legend()

# 2. Work Pressure Distribution
axes[0, 1].hist(data['Work Pressure'], bins=5, color='#e74c3c', edgecolor='black', alpha=0.7)
axes[0, 1].set_title('Work Pressure Distribution', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Work Pressure (1-5)')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_xticks([1, 2, 3, 4, 5])

# 3. Financial Stress Distribution
axes[0, 2].hist(data['Financial Stress'], bins=5, color='#f39c12', edgecolor='black', alpha=0.7)
axes[0, 2].set_title('Financial Stress Distribution', fontsize=12, fontweight='bold')
axes[0, 2].set_xlabel('Financial Stress (1-5)')
axes[0, 2].set_ylabel('Frequency')
axes[0, 2].set_xticks([1, 2, 3, 4, 5])

# 4. Work/Study Hours Distribution
axes[1, 0].hist(data['Work/Study Hours'], bins=12, color='#9b59b6', edgecolor='black', alpha=0.7)
axes[1, 0].axvline(data['Work/Study Hours'].mean(), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {data["Work/Study Hours"].mean():.1f}')
axes[1, 0].set_title('Work/Study Hours Distribution', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Hours')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].legend()

# 5. Job Satisfaction Distribution
axes[1, 1].hist(data['Job Satisfaction'], bins=5, color='#2ecc71', edgecolor='black', alpha=0.7)
axes[1, 1].set_title('Job Satisfaction Distribution', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Job Satisfaction (1-5)')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_xticks([1, 2, 3, 4, 5])

# 6. Depression Distribution (Target Variable)
depression_counts = data['Depression'].value_counts()
axes[1, 2].bar(['No Depression', 'Depression'], depression_counts.values, 
               color=['#2ecc71', '#e74c3c'], edgecolor='black', alpha=0.7)
axes[1, 2].set_title('Depression Distribution (Target)', fontsize=12, fontweight='bold')
axes[1, 2].set_ylabel('Count')
for i, v in enumerate(depression_counts.values):
    axes[1, 2].text(i, v + 20, str(v), ha='center', fontweight='bold')

plt.tight_layout()
plt.show()

# Print summary statistics
print("=" * 60)
print("UNIVARIATE ANALYSIS - SUMMARY STATISTICS")
print("=" * 60)
numeric_cols = ['Age', 'Work Pressure', 'Job Satisfaction', 'Work/Study Hours', 'Financial Stress']
print(data[numeric_cols].describe().round(2))


fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Bivariate Analysis: Variable Relationships with Depression', fontsize=16, fontweight='bold', y=1.00)

# 1. Age vs Depression
sns.boxplot(x='Depression', y='Age', data=data, ax=axes[0, 0], palette=['#2ecc71', '#e74c3c'])
axes[0, 0].set_title('Age vs Depression Status', fontsize=12, fontweight='bold')
axes[0, 0].set_xticklabels(['No Depression', 'Depression'])

# 2. Work Pressure vs Depression
sns.violinplot(x='Depression', y='Work Pressure', data=data, ax=axes[0, 1], palette=['#2ecc71', '#e74c3c'])
axes[0, 1].set_title('Work Pressure vs Depression Status', fontsize=12, fontweight='bold')
axes[0, 1].set_xticklabels(['No Depression', 'Depression'])

# 3. Financial Stress vs Depression
sns.violinplot(x='Depression', y='Financial Stress', data=data, ax=axes[0, 2], palette=['#2ecc71', '#e74c3c'])
axes[0, 2].set_title('Financial Stress vs Depression Status', fontsize=12, fontweight='bold')
axes[0, 2].set_xticklabels(['No Depression', 'Depression'])

# 4. Job Satisfaction vs Depression
sns.boxplot(x='Depression', y='Job Satisfaction', data=data, ax=axes[1, 0], palette=['#2ecc71', '#e74c3c'])
axes[1, 0].set_title('Job Satisfaction vs Depression Status', fontsize=12, fontweight='bold')
axes[1, 0].set_xticklabels(['No Depression', 'Depression'])

# 5. Work/Study Hours vs Depression
sns.boxplot(x='Depression', y='Work/Study Hours', data=data, ax=axes[1, 1], palette=['#2ecc71', '#e74c3c'])
axes[1, 1].set_title('Work/Study Hours vs Depression Status', fontsize=12, fontweight='bold')
axes[1, 1].set_xticklabels(['No Depression', 'Depression'])

# 6. Scatter: Age vs Financial Stress colored by Depression
for dep_status in [0, 1]:
    subset = data[data['Depression'] == dep_status]
    axes[1, 2].scatter(subset['Age'], subset['Financial Stress'], 
                      alpha=0.5, s=30, 
                      c='#2ecc71' if dep_status == 0 else '#e74c3c',
                      label='No Depression' if dep_status == 0 else 'Depression')
axes[1, 2].set_title('Age vs Financial Stress by Depression', fontsize=12, fontweight='bold')
axes[1, 2].set_xlabel('Age')
axes[1, 2].set_ylabel('Financial Stress')
axes[1, 2].legend()

plt.tight_layout()
plt.show()

# Statistical comparison
print("\n" + "=" * 60)
print("BIVARIATE ANALYSIS - GROUP COMPARISONS")
print("=" * 60)
for col in numeric_cols:
    print(f"\n{col} by Depression Status:")
    print(data.groupby('Depression')[col].describe()[['mean', 'std', 'min', 'max']].round(2))


import numpy as np
# Multivariate Analysis - Complex relationships (FULLY FIXED)
fig = plt.figure(figsize=(16, 10))
fig.suptitle('Multivariate Analysis: Complex Variable Interactions', fontsize=16, fontweight='bold', y=0.98)

# 1. Correlation Heatmap
ax1 = plt.subplot(2, 3, 1)
numeric_data = data[['Age', 'Work Pressure', 'Job Satisfaction', 'Work/Study Hours', 'Financial Stress', 'Depression']]
correlation = numeric_data.corr()
sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax1)
ax1.set_title('Correlation Matrix', fontsize=12, fontweight='bold')

# 2. Stacked Area Chart - Distribution across groups
ax2 = plt.subplot(2, 3, 2)
wp_bins = data.groupby(['Work Pressure', 'Depression']).size().unstack(fill_value=0)
wp_bins.plot(kind='bar', stacked=True, ax=ax2, color=['#2ecc71', '#e74c3c'], alpha=0.7)
ax2.set_title('Work Pressure Distribution by Depression', fontsize=12, fontweight='bold')
ax2.set_xlabel('Work Pressure Level')
ax2.set_ylabel('Count')
ax2.legend(['No Depression', 'Depression'])
ax2.tick_params(axis='x', rotation=0)

# 3. Heatmap: Age Groups vs Financial Stress (REPLACED 3D PLOT)
ax3 = plt.subplot(2, 3, 3)
# Create age bins
data_temp = data.copy()
data_temp['Age_Group'] = pd.cut(data_temp['Age'], bins=5, labels=['Very Young', 'Young', 'Middle', 'Senior', 'Very Senior'])
pivot_table = pd.crosstab(data_temp['Age_Group'], data_temp['Financial Stress'], 
                          values=data_temp['Depression'], aggfunc='mean')
sns.heatmap(pivot_table, annot=True, fmt='.2f', cmap='RdYlGn_r', ax=ax3, cbar_kws={"shrink": 0.8})
ax3.set_title('Depression Rate by Age & Financial Stress', fontsize=12, fontweight='bold')
ax3.set_xlabel('Financial Stress Level')
ax3.set_ylabel('Age Group')

# 4. Scatter: Work Pressure vs Job Satisfaction
ax4 = plt.subplot(2, 3, 4)
for dep_status in [0, 1]:
    subset = data[data['Depression'] == dep_status]
    if len(subset) > 0:
        ax4.scatter(subset['Work Pressure'], subset['Job Satisfaction'],
                   alpha=0.4, s=30,
                   c='#2ecc71' if dep_status == 0 else '#e74c3c',
                   label='No Depression' if dep_status == 0 else 'Depression')
ax4.set_xlabel('Work Pressure')
ax4.set_ylabel('Job Satisfaction')
ax4.set_title('Work Pressure vs Job Satisfaction', fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. Bubble chart: Age, Financial Stress, sized by Work Hours
ax5 = plt.subplot(2, 3, 5)
for dep_status in [0, 1]:
    subset = data[data['Depression'] == dep_status]
    if len(subset) > 0:
        ax5.scatter(subset['Age'], subset['Financial Stress'],
                   s=subset['Work/Study Hours'] * 10,
                   alpha=0.4,
                   c='#2ecc71' if dep_status == 0 else '#e74c3c',
                   label='No Depression' if dep_status == 0 else 'Depression',
                   edgecolors='black', linewidth=0.5)
ax5.set_xlabel('Age')
ax5.set_ylabel('Financial Stress')
ax5.set_title('Age vs Financial Stress\n(bubble size = Work Hours)', fontsize=12, fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. Grouped bar chart: Multiple factors by Depression
ax6 = plt.subplot(2, 3, 6)
depression_groups = data.groupby('Depression')[['Work Pressure', 'Job Satisfaction', 'Financial Stress']].mean()
x = np.arange(len(depression_groups.columns))
width = 0.35
ax6.bar(x - width/2, depression_groups.iloc[0], width, label='No Depression', 
        color='#2ecc71', alpha=0.7, edgecolor='black')
ax6.bar(x + width/2, depression_groups.iloc[1], width, label='Depression', 
        color='#e74c3c', alpha=0.7, edgecolor='black')
ax6.set_ylabel('Average Score')
ax6.set_title('Average Scores by Depression Status', fontsize=12, fontweight='bold')
ax6.set_xticks(x)
ax6.set_xticklabels(depression_groups.columns, rotation=45, ha='right')
ax6.legend()
ax6.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# Print multivariate insights
print("\n" + "=" * 60)
print("MULTIVARIATE ANALYSIS - KEY INSIGHTS")
print("=" * 60)
print("\nTop 3 Correlations with Depression:")
depression_corr = correlation['Depression'].abs().sort_values(ascending=False)[1:4]
for feature, corr_value in depression_corr.items():
    print(f"  {feature}: {correlation.loc[feature, 'Depression']:.3f}")

print("\nAverage values by Depression Status:")
numeric_cols = ['Age', 'Work Pressure', 'Job Satisfaction', 'Work/Study Hours', 'Financial Stress']
print(data.groupby('Depression')[numeric_cols].mean().round(2))


# we wont be using name for prediction so i just dropped it
data.drop(columns=['Name'], inplace=True)


# we have a lot of categorical var so we will fix that first
# depression column has no and yes values so i converted them to 0 and 1
le = LabelEncoder()
data['Depression'] = le.fit_transform(data['Depression'])


print(data['Depression'].unique())


# i am doing one hot encoding to handle cat attributes this can slow down our trainingslows down training, increases memory usage, and can make the model less effective by making the data mostly 0s.
#maybe for checkpoint 2 we can come up with some different approach her and make use of an effective technique
categorical_cols = [
    'Gender', 'City', 'Profession',
    'Sleep Duration', 'Dietary Habits', 'Degree',
    'Have you ever had suicidal thoughts ?', 'Family History of Mental Illness'
]

for i in categorical_cols:
    data[i] = le.fit_transform(data[i])
    


# now we have numeric columns which we should std:['Age', 'Work Pressure', 'Job Satisfaction', 'Work/Study Hours', 'Financial Stress']
scaler = StandardScaler()
num_cols = ['Age', 'Work Pressure', 'Job Satisfaction', 'Work/Study Hours', 'Financial Stress']
data[num_cols] = scaler.fit_transform(data[num_cols])



X = data.drop(columns=['Depression'])
y = data['Depression']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify =y)


print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")


X_train.head()


sns.set(style="whitegrid", palette="coolwarm")
plt.rcParams['figure.figsize'] = (8, 5)


sns.countplot(x='Depression', data=data)
plt.title('Depression Distribution')
plt.show()


sns.boxplot(x='Depression', y='Work Pressure', data=data)
plt.title('Work Pressure vs Depression')
plt.show()


sns.boxplot(x='Depression', y='Job Satisfaction', data=data)
plt.title('Job Satisfaction vs Depression')
plt.show()



sns.histplot(data=data, x='Age', kde=True, bins=20, hue='Depression')
plt.title('Age Distribution and Depression')
plt.show()


# plt.figure(figsize=(10, 8))
# sns.heatmap(data.corr(numeric_only=True), cmap='coolwarm', annot=True, fmt='.2f')
# plt.title('Correlation Heatmap')
# plt.show()


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)




print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_logreg))


from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier(random_state=42)
dtree.fit(X_train, y_train)
y_pred_dtree = dtree.predict(X_test)

print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dtree))


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)




print('Random Forrest Accuracy: ',accuracy_score(y_test,y_pred_rf))





from sklearn.ensemble import AdaBoostClassifier

ada = AdaBoostClassifier(n_estimators=100, random_state=42)
ada.fit(X_train, y_train)
y_pred_ada = ada.predict(X_test)



print("AdaBoost accuracy:" ,accuracy_score(y_test,y_pred_ada))


from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=200, random_state=42)
mlp.fit(X_train, y_train)
y_pred_mlp = mlp.predict(X_test)



print('Neural Network accuracy:' ,accuracy_score(y_test,y_pred_mlp))


models = {
    'Logistic Regression': logreg,
    'Decision Tree': dtree,
    'Random Forest': rf,
    'MLP Neural Network': mlp,
    'AdaBoost': ada,
    
}

accuracies = {}

for name, model in models.items():
    # Some models (like CatBoost) may return predictions as float32; cast them to match y_test if needed
    preds = model.predict(X_test)
    if preds.dtype != y_test.dtype:
        preds = preds.astype(y_test.dtype)
    accuracies[name] = accuracy_score(y_test, preds)

# Create and display the accuracy table
accuracy_df = pd.DataFrame(list(accuracies.items()), columns=['Model', 'Accuracy'])
print(accuracy_df)


from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

for name, model in models.items():
    print(f"\n===== Model: {name} =====")

    # Generate predictions and predicted probabilities
    y_pred = model.predict(X_test)
    # For ROC AUC, need probability estimates
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        # fallback: use decision_function, if available
        y_proba = model.decision_function(X_test)
    
    # Confusion Matrix
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Classification Report: precision, recall, f1-score, support
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # ROC AUC Score
    print("ROC AUC Score:", roc_auc_score(y_test,y_proba))


from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import roc_auc_score, average_precision_score

def get_scores(model, X, needs_sigmoid=False):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        z = model.decision_function(X)
        proba = 1/(1+np.exp(-z)) if needs_sigmoid else (z - z.min())/(z.max()-z.min()+1e-9)
    else:
        raise ValueError("Model has neither predict_proba nor decision_function")
    pred = (proba >= 0.5).astype(int)
    return pred, proba

def plot_evaluation(model, X_test, y_test, name="model"):
    y_pred, y_proba = get_scores(model, X_test, needs_sigmoid=True)

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Evaluation — {name}", fontsize=14, weight="bold")

    # 1) Confusion Matrix (normalized)
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, normalize='true', ax=axs[0,0], cmap='Blues')
    axs[0,0].set_title("Confusion Matrix (normalized)")

    # 2) ROC Curve
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=axs[0,1])
    axs[0,1].set_title(f"ROC (AUC = {roc_auc_score(y_test, y_proba):.3f})")

    # 3) Precision–Recall Curve (better for imbalance)
    PrecisionRecallDisplay.from_predictions(y_test, y_proba, ax=axs[1,0])
    aps = average_precision_score(y_test, y_proba)
    axs[1,0].set_title(f"Precision–Recall (AP = {aps:.3f})")

    # 4) Calibration (reliability) curve
    CalibrationDisplay.from_predictions(y_test, y_proba, n_bins=10, ax=axs[1,1])
    axs[1,1].set_title("Calibration Curve")

    plt.tight_layout(); plt.show()


for name, model in models.items():
    print(f"\n===== {name} =====")
    plot_evaluation(model, X_test, y_test, name=name)



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
for name, model in models.items():
    _, proba = get_scores(model, X_test, needs_sigmoid=True)
    RocCurveDisplay.from_predictions(y_test, proba, name=name, ax=ax1)
    PrecisionRecallDisplay.from_predictions(y_test, proba, name=name, ax=ax2)
ax1.set_title("ROC Comparison"); ax2.set_title("PR Comparison")
plt.tight_layout(); plt.show()



from sklearn.metrics import precision_recall_curve, f1_score
prec, rec, th = precision_recall_curve(y_test, y_proba)
f1 = 2*prec*rec/(prec+rec+1e-9)
plt.figure(figsize=(6,4)); plt.plot(th, prec[:-1], label="Precision"); plt.plot(th, rec[:-1], label="Recall"); plt.plot(th, f1[:-1], label="F1")
plt.xlabel("Threshold"); plt.legend(); plt.title("Threshold vs Precision/Recall/F1"); plt.show()



from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

logreg = LogisticRegression(max_iter=2000, random_state=42)
param_grid_logreg = {
    'penalty': ['l1', 'l2'],
    'C': [0.01, 0.1, 1, 10],
    'solver': ['liblinear', 'saga']
}
# Perform 5-fold grid search
grid_search_lr = GridSearchCV(logreg, param_grid_logreg, cv=5, scoring='accuracy')
grid_search_lr.fit(X_train, y_train)

# Retrieve the best model and evaluate
best_logreg = grid_search_lr.best_estimator_
y_pred_lr = best_logreg.predict(X_test)
print("Best LogisticRegression params:", grid_search_lr.best_params_)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))



from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

dtree = DecisionTreeClassifier(random_state=42)
param_grid = {
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5],
    'criterion': ['gini', 'entropy']
}
grid_search_dt = GridSearchCV(dtree, param_grid, cv=5, scoring='accuracy')
grid_search_dt.fit(X_train, y_train)

best_dtree = grid_search_dt.best_estimator_
y_pred_dt = best_dtree.predict(X_test)
print("Best DecisionTree params:", grid_search_dt.best_params_)
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))




# Parameter grid for Random Forest
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'max_features': ['sqrt', 'log2'],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Perform grid search with 5-fold CV
grid_search_rf = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid=param_grid_rf,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
grid_search_rf.fit(X_train, y_train)
print("Best Random Forest parameters:", grid_search_rf.best_params_,verbose=2)



best_rf = grid_search_rf.best_params_
y_pred_rf = best_rf.predict(X_test)
print("Random Forrest Accuracy:", accuracy_score(y_test, y_pred_dt))


mlp = MLPClassifier(max_iter=500, random_state=42)
param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'sgd'],
    'alpha': [0.0001, 0.001],
    'learning_rate': ['constant', 'adaptive']
}
grid_search_mlp = GridSearchCV(mlp, param_grid, cv=5, scoring='accuracy',verbose = 2,n_jobs=-1
                )
grid_search_mlp.fit(X_train, y_train)

best_mlp = grid_search_mlp.best_estimator_
y_pred_mlp = best_mlp.predict(X_test)
print("Best MLPClassifier params:", grid_search_mlp.best_params_)
print("Neural Network Accuracy:", accuracy_score(y_test, y_pred_mlp))



ada = AdaBoostClassifier(random_state=42)
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 1.0]
}
grid_search_ada = GridSearchCV(ada, param_grid, cv=5, scoring='accuracy',verbose=2)
grid_search_ada.fit(X_train, y_train)

best_ada = grid_search_ada.best_estimator_
y_pred_ada = best_ada.predict(X_test)
print("Best AdaBoost params:", grid_search_ada.best_params_)
print("AdaBoost Accuracy:", accuracy_score(y_test, y_pred_ada))





