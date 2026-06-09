






import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier


from sklearn.pipeline import Pipeline
from sklearn import metrics



import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train_df = pd.read_csv('/kaggle/input/multiclassificationtask/train.csv', index_col=0)
print("Train data loaded successfully.")


print("ğŸ”� Dastlabki qatorlar:")
display(train_df.head())


print("\nğŸ“Š Ma'lumotlar shakli (shape):", train_df.shape)



print("\nğŸ§¾ Ustunlar va ularning turlari:")
print(train_df.dtypes)


print("\nâ�“ Null qiymatlar soni:")
print(train_df.isnull().sum())



print("\nğŸ�·ï¸� 'Status' ustunidagi sinflar taqsimoti:")
print(train_df['Status'].value_counts())


plt.figure(figsize=(6,4))
sns.countplot(data=train_df, x='Status', palette='viridis')
plt.title("Status ustunidagi sinflar soni")
plt.xlabel("Status sinflari")
plt.ylabel("Soni")
plt.show()



numeric_columns = train_df.select_dtypes(include=np.number).columns

train_df[numeric_columns].hist(figsize=(15, 10), bins=30, color='skyblue', edgecolor='black')
plt.suptitle("ğŸ“Š Sonli ustunlar uchun histogramlar", fontsize=16)
plt.tight_layout()
plt.show()



from sklearn.preprocessing import LabelEncoder

train_df = train_df[train_df['Status'] != 'Y']

X = train_df.drop(columns=['Status'])
y = train_df['Status']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

print("Encoded target values:", set(y_encoded))
print("Original classes:", le.classes_)

print("\nTarget distribution after encoding:")
print(pd.Series(y_encoded).value_counts())

print("\nFeatures info:")
print(X.info())



from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

categorical_cols = X.select_dtypes(include=['object']).columns
numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),  
    ('onehot', OneHotEncoder(handle_unknown='ignore'))   
])

numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),          
    ('scaler', StandardScaler())                            
])

preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, numeric_cols),
    ('cat', categorical_pipeline, categorical_cols)
])

X_prepared = preprocessor.fit_transform(X)

print("Data preprocessing completed. Prepared data shape:", X_prepared.shape)



from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, classification_report

X_train, X_test, y_train, y_test = train_test_split(X_prepared, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

def estimate_model(model, X_test, y_test):
    y_pred = model.predict(X_test) 
    y_proba = model.predict_proba(X_test)  
    
    acc = accuracy_score(y_test, y_pred)  
    logloss = log_loss(y_test, y_proba)  
    report = classification_report(y_test, y_pred, zero_division=0)
    
    print(f"Accuracy: {acc:.4f}")
    print(f"Log Loss: {logloss:.4f}")
    print("Classification Report:\n", report)


def evaluate_model(name, model, X_test, y_test):
    print(f"Model: {name}")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    logloss = log_loss(y_test, y_proba)
    report = classification_report(y_test, y_pred, zero_division=0)
    
    print(f"Accuracy: {acc:.4f}")
    print(f"Log Loss: {logloss:.4f}")
    print("Classification Report:\n", report)
    print("-" * 50)

models = {
    "SVM": SVC(probability=True, random_state=42),
    "RandomForest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "NeuralNetwork": MLPClassifier(max_iter=500, random_state=42),
    "DecisionTree": DecisionTreeClassifier(random_state=42)
}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    evaluate_model(name, model, X_test, y_test)



from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

estimators = [
    ('rf', RandomForestClassifier(random_state=42)),
    ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)),
    ('svm', SVC(probability=True, random_state=42)),
    ('mlp', MLPClassifier(max_iter=500, random_state=42))
]

final_estimator = LogisticRegression(max_iter=1000, random_state=42)

stack = StackingClassifier(
    estimators=estimators,
    final_estimator=final_estimator,
    passthrough=True,  
    cv=5,
    n_jobs=-1
)

print("Training Stacking Ensemble model...")
stack.fit(X_train, y_train)

y_pred = stack.predict(X_test)
y_proba = stack.predict_proba(X_test)

acc = accuracy_score(y_test, y_pred)
logloss = log_loss(y_test, y_proba)
report = classification_report(y_test, y_pred, zero_division=0)

print(f"Stacking Ensemble Model Accuracy: {acc:.4f}")
print(f"Stacking Ensemble Model Log Loss: {logloss:.4f}")
print("Classification Report:\n", report)



import pandas as pd
from xgboost import XGBClassifier
from sklearn import metrics

test_df = pd.read_csv('/kaggle/input/multiclassificationtask/test.csv')

test_df_prepared = preprocessor.transform(test_df)

model = XGBClassifier(
    objective='multi:softprob',
    num_class=3,
    eval_metric='mlogloss',
    use_label_encoder=False,
    random_state=42
)
model.fit(X_train, y_train)
y_proba = model.predict_proba(test_df_prepared)


submission = pd.DataFrame(y_proba, columns=['Status_C', 'Status_CL', 'Status_D'])
submission['id'] = test_df['id']

submission = submission[['id', 'Status_C', 'Status_CL', 'Status_D']]
submission.to_csv('submission.csv', index=False)


