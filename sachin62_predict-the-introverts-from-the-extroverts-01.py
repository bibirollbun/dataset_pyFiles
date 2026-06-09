import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import warnings  
warnings.simplefilter('ignore')


train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
sub = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')


train.head()





train.info()


train.isnull().sum()


train.duplicated().sum()



train['Stage_fear'] = train['Stage_fear'].fillna(train['Stage_fear'].mode()[0])
train['Stage_fear'] = train['Stage_fear'].map({'Yes': 1, 'No': 0})

train['Drained_after_socializing'] = train['Drained_after_socializing'].fillna(train['Drained_after_socializing'].mode()[0])
train['Drained_after_socializing'] = train['Drained_after_socializing'].map({'Yes': 1, 'No': 0})

train['Personality'] = train['Personality'].map({'Extrovert': 1, 'Introvert': 0})


test['Stage_fear'] = test['Stage_fear'].fillna(test['Stage_fear'].mode()[0])
test['Stage_fear'] = test['Stage_fear'].map({'Yes': 1, 'No': 0})

test['Drained_after_socializing'] = test['Drained_after_socializing'].fillna(test['Drained_after_socializing'].mode()[0])
test['Drained_after_socializing'] = test['Drained_after_socializing'].map({'Yes': 1, 'No': 0})



cols_with_missing = [
    'Time_spent_Alone', 
    'Social_event_attendance',
    'Going_outside', 
    'Friends_circle_size', 
    'Post_frequency'
]

for col in cols_with_missing:
    median_val = train[col].median()
    train[col] = train[col].fillna(median_val)




for col in cols_with_missing:
    median_val = test[col].median()
    test[col] = test[col].fillna(median_val)



train.isnull().sum()


train.to_csv('train_p.csv', index=False)
test.to_csv('test_p.csv', index=False)


sns.set(style='whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
target = 'Personality'

print("Basic Dataset Info")
display(train.info())


display(train.describe())
numerical_cols = train.drop(columns=['id', target]).columns

print("Feature Distributions")
for col in numerical_cols:
    sns.histplot(train[col], kde=True, bins=30, color='skyblue')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()

# Target Distribution
print("Target Class Distribution")
sns.countplot(data=train, x=target, palette='Set2')
plt.title('Distribution of Personality Classes')
plt.show()

#  Correlation Heatmap
print(" Correlation Matrix")
plt.figure(figsize=(10, 8))
sns.heatmap(train.drop(columns='id').corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Boxplots: Features vs. Target
print("Boxplots: Features vs Personality")
for col in numerical_cols:
    sns.boxplot(x=target, y=col, data=train, palette='Set3')
    plt.title(f'{col} by Personality')
    plt.show()

#  Grouped Statistics by Personality
print("Grouped Summary by Personality")
display(train.groupby('Personality')[numerical_cols].mean().T.style.background_gradient(cmap='YlGnBu'))

# Violin plots (optional advanced visualization)
print(" Violin Plots (optional)")
for col in numerical_cols:
    sns.violinplot(x=target, y=col, data=train, palette='pastel')
    plt.title(f'{col} vs Personality')
    plt.show()



from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


X = train.drop(columns=['id', 'Personality'])
y = train['Personality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))



from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

X = train.drop(columns=['id', 'Personality'])
y = train['Personality']  

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

xgb = XGBClassifier(
    scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
    eval_metric='mlogloss',
    random_state=42
)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

grid_search = GridSearchCV(xgb, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

print("✅ Best Parameters:", grid_search.best_params_)
best_xgb = grid_search.best_estimator_
y_pred = best_xgb.predict(X_test)

print("\n✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
plt.title("XGBoost Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()






X_test_real = test.drop(columns=['id'])

X_test_scaled = scaler.transform(X_test_real)

# Predict using best_xgb
y_test_pred = best_xgb.predict(X_test_scaled)

label_map = {1: 'Extrovert', 0: 'Introvert'}

predicted_labels = pd.Series(y_test_pred).map(label_map)
output = pd.DataFrame({
    'id': test['id'],
    'Personality': predicted_labels
})

# Display or save
print(output.head())

# Save to CSV
output.to_csv('submission.csv', index=False)











