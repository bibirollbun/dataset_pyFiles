# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


#1: What is the goal of this competition? Is it regression or classification? What evaluation metric is used?
#✅ Phân loại (Classification)

#❌ Không phải hồi quy (Regression)


#2: Load train and test datasets. Display:
import pandas as pd
train_df = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")


train_df.shape


train_df.head(2)


train_df.info()


#3: Check for missing values. Propose and apply a strategy to handle them.
train_df.isna().sum()


test_df.shape


#4: Analyze target variable distribution. Is it balanced or skewed?
import seaborn as sns
import matplotlib.pyplot as plt
sns.countplot(data=train_df,x='Personality')
plt.title('Value Counts: Personality')
plt.show()
train_df['Personality'].value_counts()


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score


# Separate features and target
X = train_df.drop(['id', 'Personality'], axis=1)
y = train_df['Personality'].map({'Introvert': 0, 'Extrovert': 1})  # Encode target
X_test = test_df.drop(['id'], axis=1)

# Handle categorical variables
categorical_cols = ['Stage_fear', 'Drained_after_socializing']
for col in categorical_cols:
    X[col] = X[col].map({'No': 0, 'Yes': 1})
    X_test[col] = X_test[col].map({'No': 0, 'Yes': 1})
    
    # Impute missing values in categorical columns with mode
    imputer = SimpleImputer(strategy='most_frequent')
    X[col] = imputer.fit_transform(X[[col]]).ravel()
    X_test[col] = imputer.transform(X_test[[col]]).ravel()

# Handle numerical columns
numerical_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
                  'Friends_circle_size', 'Post_frequency']

imputer_num = SimpleImputer(strategy='median')
X[numerical_cols] = imputer_num.fit_transform(X[numerical_cols])
X_test[numerical_cols] = imputer_num.transform(X_test[numerical_cols])


#5: Visualize histograms and boxplots for numeric features. Identify outliers or unusual patterns.

# Lọc các cột số
numeric_features = train_df.select_dtypes(include=['int64', 'float64']).columns
# Vẽ histogram và boxplot cho từng biến
for col in numeric_features:
    plt.figure(figsize=(12, 4))

    # Histogram
    plt.subplot(1, 2, 1)
    sns.histplot(train_df[col], kde=True, color='skyblue')
    plt.title(f'Histogram of {col}')


# Boxplot
for col in numeric_features:
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 2)
    sns.boxplot(x=train_df[col], color='salmon')
    plt.title(f'Boxplot of {col}')

    plt.tight_layout()
    plt.show()


#6: Encode categorical variables properly (Label Encoding or One-Hot Encoding). Explain your choice.


#7: Check for multicollinearity among features using correlation matrix or VIF.
# Tính toán ma trận tương quan
corr_matrix = train_df[numeric_features].corr()

# Vẽ heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()


#8: Split data into train-validation sets (e.g. 80-20 split) with random_state for reproducibility.
# Split training data for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


#9: Scale numeric features if needed. Which models benefit from scaling and why?
#scale chi co loi cho 2 model do la: Linear logistis va SVM
# Scale numerical features
scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])


#1: Train a Logistic Regression model as baseline.
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
# Train Random Forest model
model = LogisticRegression(max_iter=200,random_state=42)
model.fit(X_train, y_train)

# Train model
train_pred = model.predict(X_train)
f1 = f1_score(y_train, train_pred)
print(f'Train F1 Accuracy: {f1:.4f}')
accuracy = accuracy_score(y_train, train_pred)
print(f'Train Accuracy: {accuracy:.4f}')

# Validate model
val_pred = model.predict(X_val)
f1 = f1_score(y_val, val_pred)
print(f'F1 Accuracy: {f1:.4f}')
accuracy = accuracy_score(y_val, val_pred)
print(f'Validation Accuracy: {accuracy:.4f}')


# Predict on test set
test_pred = model.predict(X_test)
test_pred_labels = np.where(test_pred == 0, 'Introvert', 'Extrovert')


# Create submission file
submission = pd.DataFrame({'id': test_df['id'], 'Personality': test_pred_labels})
submission.to_csv('submission.csv', index=False)
print("Submission file created: submission.csv")


print(submission.shape)


from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
# Train Random Forest model
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
param_grid = [
    {'kernel': ['linear'], 'C': [10]},  # chỉ chọn 1-2 giá trị tốt
    {'kernel': ['rbf'], 'C': [10, 100], 'gamma': [0.1, 0.01]},
    {'kernel': ['poly'], 'C': [10], 'gamma': [0.1], 'degree': [2, 3]}
]

model = GridSearchCV(SVC() , param_grid , refit=True , verbose=3)
model.fit(X_train, y_train)

# Train model
train_pred = model.predict(X_train)
f1 = f1_score(y_train, train_pred)
print(f'Train F1 Accuracy: {f1:.4f}')
accuracy = accuracy_score(y_train, train_pred)
print(f'Train Accuracy: {accuracy:.4f}')

# Validate model
val_pred = model.predict(X_val)
f1 = f1_score(y_val, val_pred)
print(f'F1 Accuracy: {f1:.4f}')
accuracy = accuracy_score(y_val, val_pred)
print(f'Validation Accuracy: {accuracy:.4f}')


# Predict on test set
test_pred = model.predict(X_test)
test_pred_labels = np.where(test_pred == 0, 'Introvert', 'Extrovert')


# Create submission file
svc = pd.DataFrame({'id': test_df['id'], 'Personality': test_pred_labels})
svc.to_csv('svc.csv', index=False)
print("Submission file created: SVC.csv")


print(svc.shape)


#3: Plot learning curves for the best SVM configuration.
from sklearn.model_selection import learning_curve
best_svm = model.best_estimator_
train_sizes, train_scores, val_scores = learning_curve(
    best_svm, X_train, y_train, cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy', n_jobs=-1
)

train_scores_mean = train_scores.mean(axis=1)
val_scores_mean = val_scores.mean(axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_mean, 'o-', color='blue', label='Training accuracy')
plt.plot(train_sizes, val_scores_mean, 'o-', color='green', label='Validation accuracy')
plt.title('Learning Curve for Best SVM Model')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.grid(True)
plt.show()

'''Nếu khoảng cách giữa training và validation là nhỏ: Mô hình học tốt và tổng quát tốt.

Nếu khoảng cách lớn: Mô hình đang overfit hoặc underfit:

Training cao – Validation thấp: Overfitting.

Cả hai đều thấp: Underfitting.

'''


#1: Train a Decision Tree Classifier or Regressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)
# Train model
train_pred = model.predict(X_train)
f1 = f1_score(y_train, train_pred)
print(f'Train F1 Accuracy: {f1:.4f}')
accuracy = accuracy_score(y_train, train_pred)
print(f'Train Accuracy: {accuracy:.4f}')

# Validate model
val_pred = model.predict(X_val)
f1 = f1_score(y_val, val_pred)
print(f'F1 Accuracy: {f1:.4f}')
accuracy = accuracy_score(y_val, val_pred)
print(f'Validation Accuracy: {accuracy:.4f}')


#Plot the tree structure and interpret splits.
plt.figure(figsize=(16, 8))
plot_tree(model, 
          feature_names=X_train.columns, 
          class_names=[str(c) for c in model.classes_] if hasattr(model, 'classes_') else None,
          filled=True, 
          rounded=True, 
          fontsize=10)
plt.title("Decision Tree")
plt.show()


#2: Tune max_depth and min_samples_split. How does depth affect bias-variance trade-off?
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
param_grid = {
    'max_depth': [2, 4, 6, 8, 10, 12, None],
    'min_samples_split': [2, 5, 10, 20]
}

model = GridSearchCV(DecisionTreeClassifier(random_state=42),
                           param_grid, cv=5, scoring='accuracy')
model.fit(X_train, y_train)

print("Best params:", model.best_params_)
print("Best CV accuracy:", model.best_score_)

# Train model
train_pred = model.predict(X_train)
f1 = f1_score(y_train, train_pred)
print(f'Train F1 Accuracy: {f1:.4f}')
accuracy = accuracy_score(y_train, train_pred)
print(f'Train Accuracy: {accuracy:.4f}')

# Validate model
val_pred = model.predict(X_val)
f1 = f1_score(y_val, val_pred)
print(f'F1 Accuracy: {f1:.4f}')
accuracy = accuracy_score(y_val, val_pred)
print(f'Validation Accuracy: {accuracy:.4f}')

'''max_depth ↓ → Bias ↑, Variance ↓ → underfit.

max_depth ↑ → Bias ↓, Variance ↑ → overfit.'''


# Predict on test set
test_pred = model.predict(X_test)
test_pred_labels = np.where(test_pred == 0, 'Introvert', 'Extrovert')


# Create submission file
DecisionTree = pd.DataFrame({'id': test_df['id'], 'Personality': test_pred_labels})
DecisionTree.to_csv('DecisionTree.csv', index=False)
print("Submission file created: DecisionTree.csv")


print(DecisionTree.shape)


#3: Plot learning curves for Decision Tree. Does the model overfit when depth is unrestricted?
best_decisionT = model.best_estimator_
train_sizes, train_scores, val_scores = learning_curve(
    best_decisionT, X_train, y_train, cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy', n_jobs=-1
)

train_scores_mean = train_scores.mean(axis=1)
val_scores_mean = val_scores.mean(axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_mean, 'o-', color='blue', label='Training accuracy')
plt.plot(train_sizes, val_scores_mean, 'o-', color='green', label='Validation accuracy')
plt.title('Learning Curve for Best Decision Tree Model')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.grid(True)
plt.show()


#1: Train a BaggingClassifier (or BaggingRegressor) with Decision Tree as base estimator.
from sklearn.ensemble import BaggingClassifier
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(random_state=42),
    random_state=42,
    bootstrap=True,
    n_jobs=-1
)

param_grid = {
    'n_estimators': [50, 100, 150],
    'estimator__max_depth': [3, 5, 7, None],
    'max_features': [0.5, 0.7, 1.0]
}

model = GridSearchCV(
    estimator=bagging,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)


model.fit(X_train, y_train)

print("Best params:", model.best_params_)
print("Best CV accuracy:", model.best_score_)

# Train model
train_pred = model.predict(X_train)
f1 = f1_score(y_train, train_pred)
print(f'Train F1 Accuracy: {f1:.4f}')
accuracy = accuracy_score(y_train, train_pred)
print(f'Train Accuracy: {accuracy:.4f}')

# Validate model
val_pred = model.predict(X_val)
f1 = f1_score(y_val, val_pred)
print(f'F1 Accuracy: {f1:.4f}')
accuracy = accuracy_score(y_val, val_pred)
print(f'Validation Accuracy: {accuracy:.4f}')


# Predict on test set
test_pred = model.predict(X_test)
test_pred_labels = np.where(test_pred == 0, 'Introvert', 'Extrovert')


# Create submission file
BaggingClassifier = pd.DataFrame({'id': test_df['id'], 'Personality': test_pred_labels})
BaggingClassifier.to_csv('BaggingClassifier.csv', index=False)
print("Submission file created: BaggingClassifier.csv")


print(BaggingClassifier.shape)


#2: Train a Random Forest Classifier or Regressor.
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'max_features': ['sqrt', 'log2', 1.0]
}

model = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid,
    cv=5,
    scoring='accuracy'
)
model.fit(X_train, y_train)

print("Best params:", model.best_params_)
print("Best CV accuracy:", model.best_score_)

# Train model
train_pred = model.predict(X_train)
f1 = f1_score(y_train, train_pred)
print(f'Train F1 Accuracy: {f1:.4f}')
accuracy = accuracy_score(y_train, train_pred)
print(f'Train Accuracy: {accuracy:.4f}')

# Validate model
val_pred = model.predict(X_val)
f1 = f1_score(y_val, val_pred)
print(f'F1 Accuracy: {f1:.4f}')
accuracy = accuracy_score(y_val, val_pred)
print(f'Validation Accuracy: {accuracy:.4f}')


# Predict on test set
test_pred = model.predict(X_test)
test_pred_labels = np.where(test_pred == 0, 'Introvert', 'Extrovert')


# Create submission file
RandomForestClassifier = pd.DataFrame({'id': test_df['id'], 'Personality': test_pred_labels})
RandomForestClassifier.to_csv('RandomForestClassifier.csv', index=False)
print("Submission file created: RandomForestClassifier.csv")


print(RandomForestClassifier.shape)


#4: Plot learning curves for Random Forest.
best_RFC = model.best_estimator_
train_sizes, train_scores, val_scores = learning_curve(
    best_RFC, X_train, y_train, cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy', n_jobs=-1
)

train_scores_mean = train_scores.mean(axis=1)
val_scores_mean = val_scores.mean(axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_mean, 'o-', color='blue', label='Training accuracy')
plt.plot(train_sizes, val_scores_mean, 'o-', color='green', label='Validation accuracy')
plt.title('Learning Curve for Best RFC Model')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.grid(True)
plt.show()


#1: Train an AdaBoostClassifier or AdaBoostRegressor with Decision Tree stumps.
from sklearn.ensemble import AdaBoostClassifier
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(random_state=42),
    random_state=42
)

param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.5, 1.0, 1.5]
}

model = GridSearchCV(
    estimator=ada,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)


model.fit(X_train, y_train)

print("Best params:", model.best_params_)
print("Best CV accuracy:", model.best_score_)

# Train model
train_pred = model.predict(X_train)
f1 = f1_score(y_train, train_pred)
print(f'Train F1 Accuracy: {f1:.4f}')
accuracy = accuracy_score(y_train, train_pred)
print(f'Train Accuracy: {accuracy:.4f}')

# Validate model
val_pred = model.predict(X_val)
f1 = f1_score(y_val, val_pred)
print(f'F1 Accuracy: {f1:.4f}')
accuracy = accuracy_score(y_val, val_pred)
print(f'Validation Accuracy: {accuracy:.4f}')


# Predict on test set
ada_test_pred = model.predict(X_test)
ada_test_pred_labels = np.where(ada_test_pred == 0, 'Introvert', 'Extrovert')


# Create submission file
AdaBoostClassifier = pd.DataFrame({'id': test_df['id'], 'Personality': ada_test_pred_labels})
AdaBoostClassifier.to_csv('AdaBoostClassifier.csv', index=False)
print("Submission file created: AdaBoostClassifier.csv")


print(AdaBoostClassifier.shape)


#2: Train a Gradient Boosting Classifier or Regressor
from sklearn.ensemble import GradientBoostingClassifier
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 5],
    'subsample': [0.8, 1.0]
}

model = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
model.fit(X_train, y_train)

print("Best params:", model.best_params_)
print("Best CV accuracy:", model.best_score_)

# Train model
train_pred = model.predict(X_train)
f1 = f1_score(y_train, train_pred)
print(f'Train F1 Accuracy: {f1:.4f}')
accuracy = accuracy_score(y_train, train_pred)
print(f'Train Accuracy: {accuracy:.4f}')

# Validate model
val_pred = model.predict(X_val)
f1 = f1_score(y_val, val_pred)
print(f'F1 Accuracy: {f1:.4f}')
accuracy = accuracy_score(y_val, val_pred)
print(f'Validation Accuracy: {accuracy:.4f}')


# Predict on test set
gra_test_pred = model.predict(X_test)
gra_test_pred_labels = np.where(gra_test_pred == 0, 'Introvert', 'Extrovert')


# Create submission file
GradientBoosting = pd.DataFrame({'id': test_df['id'], 'Personality': gra_test_pred_labels})
GradientBoosting.to_csv('GradientBoosting.csv', index=False)
print("Submission file created: GradientBoosting.csv")


#5: Plot training vs. validation errors for Gradient Boosting. Does it overfit?
best_GB = model.best_estimator_
train_sizes, train_scores, val_scores = learning_curve(
    best_GB, X_train, y_train, cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy', n_jobs=-1
)

train_scores_mean = train_scores.mean(axis=1)
val_scores_mean = val_scores.mean(axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_mean, 'o-', color='blue', label='Training accuracy')
plt.plot(train_sizes, val_scores_mean, 'o-', color='green', label='Validation accuracy')
plt.title('Learning Curve for Best GB Model')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.grid(True)
plt.show()


#2: Use StackingClassifier or StackingRegressor to combine them
#Với stacking, quá trình grid search sẽ tốn nhiều thời gian hơn đáng kể vì nhiều lớp mô hình
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
# Các mô hình base
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
estimators = [
    ('lr', LogisticRegression(max_iter=1000)),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42))
]

# Mô hình stacking với Logistic Regression làm final estimator
model = StackingClassifier(
    estimators=estimators,
    final_estimator=RandomForestClassifier(), 
    cv=5,                  # Sử dụng cross-validation để huấn luyện mô hình meta
    n_jobs=-1
)
model.fit(X_train, y_train)

# Train model
train_pred = model.predict(X_train)
f1 = f1_score(y_train, train_pred)
print(f'Train F1 Accuracy: {f1:.4f}')
accuracy = accuracy_score(y_train, train_pred)
print(f'Train Accuracy: {accuracy:.4f}')

# Validate model
val_pred = model.predict(X_val)
f1 = f1_score(y_val, val_pred)
print(f'F1 Accuracy: {f1:.4f}')
accuracy = accuracy_score(y_val, val_pred)
print(f'Validation Accuracy: {accuracy:.4f}')


#1: Implement VotingClassifier (hard voting) combining Logistic Regression, Decision Tree, and SVM.
from sklearn.ensemble import VotingClassifier
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
# Khởi tạo các mô hình base learners
lr = LogisticRegression(max_iter=1000, random_state=42)
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
svm = SVC(kernel='linear', probability=False, random_state=42)  # Với hard voting, không cần probability=True

# Kết hợp trong VotingClassifier (hard voting là mặc định)
model = VotingClassifier(
    estimators=[('lr', lr), ('dt', dt), ('svm', svm)],
    voting='hard',  # 'hard' là mặc định
    n_jobs=-1
)
model.fit(X_train, y_train)

# Train model
train_pred = model.predict(X_train)
f1 = f1_score(y_train, train_pred)
print(f'Train F1 Accuracy: {f1:.4f}')
accuracy = accuracy_score(y_train, train_pred)
print(f'Train Accuracy: {accuracy:.4f}')

# Validate model
val_pred = model.predict(X_val)
f1 = f1_score(y_val, val_pred)
print(f'F1 Accuracy: {f1:.4f}')
accuracy = accuracy_score(y_val, val_pred)
print(f'Validation Accuracy: {accuracy:.4f}')


#2: Implement VotingClassifier (soft voting). Which performs better?
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
# Khởi tạo các mô hình
lr = LogisticRegression(max_iter=1000, random_state=42)
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
svm = SVC(kernel='rbf', probability=True, random_state=42)  # BẮT BUỘC phải có probability=True cho soft voting

# Tạo mô hình VotingClassifier với soft voting
soft_voting_clf = VotingClassifier(
    estimators=[('lr', lr), ('dt', dt), ('svm', svm)],
    voting='soft',
    n_jobs=-1
)
model.fit(X_train, y_train)

# Train model
train_pred = model.predict(X_train)
f1 = f1_score(y_train, train_pred)
print(f'Train F1 Accuracy: {f1:.4f}')
accuracy = accuracy_score(y_train, train_pred)
print(f'Train Accuracy: {accuracy:.4f}')

# Validate model
val_pred = model.predict(X_val)
f1 = f1_score(y_val, val_pred)
print(f'F1 Accuracy: {f1:.4f}')
accuracy = accuracy_score(y_val, val_pred)
print(f'Validation Accuracy: {accuracy:.4f}')


#1: Perform k-fold cross-validation for top models. Which is most stable across folds?
# Tạo các mô hình
import time
log_reg = LogisticRegression(max_iter=1000, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

# Voting
voting = VotingClassifier(
    estimators=[('lr', log_reg), ('rf', rf), ('gb', gb)],
    voting='soft',
    n_jobs=-1
)

# Stacking
stacking = StackingClassifier(
    estimators=[('lr', log_reg), ('rf', rf), ('gb', gb)],
    final_estimator=LogisticRegression(),
    cv=5,
    n_jobs=-1
)

models = {
    'Logistic Regression': log_reg,
    'Random Forest': rf,
    'Gradient Boosting': gb,
    'Voting': voting,
    'Stacking': stacking
}

# Cross-validation score
for name, model in models.items():
    start = time.time()
    scores = cross_val_score(model, X, y, cv=5, scoring='f1', n_jobs=-1)
    end = time.time()
    print(f"{name}: F1 = {scores.mean():.4f} ± {scores.std():.4f} | Time = {end - start:.2f} sec\n")



'''Model	Accuracy	Std Dev	Time (s)	Ghi chú
Logistic Regression:	0.9687	0.0031	0.09	Nhanh nhất, đơn giản, ít overfit
Random Forest:	0.9639	0.0046	2.68	Chính xác tốt, nhưng chậm hơn
Gradient Boosting:	0.9684	0.0036	2.49	Chính xác cao, chậm
Voting:	0.9687	0.0035	4.33	Kết hợp tốt, nhưng tốn thời gian
Stacking:	0.9686	0.0037	23.86	Cao nhất về độ phức tạp & thời gian
'''




