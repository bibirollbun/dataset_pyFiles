# ðŸ›³ Titanic Survival Prediction â€” Kaggle Capstone Project

## ðŸ“Œ Objective
Build a machine-learning model to predict Titanic passenger survival using the Kaggle Titanic dataset.  
This single-page notebook includes **report + full working code**.

## ðŸ“Œ Workflow
1. Load and inspect data  
2. Perform EDA  
3. Clean & preprocess data  
4. Encode categorical features  
5. Train/validate model  
6. Predict on test set  
7. Create submission file for Kaggle  

## ðŸ“Œ Modelling Approach
- **RandomForestClassifier** (stable, strong baseline)
- 300 trees, max_depth=10
- 80/20 train-validation split

## ðŸ“Œ Expected Accuracy
The model typically achieves **80â€“85% accuracy**, sufficient for Kaggle leaderboard entry.

---

# ==============================
# 1. IMPORT LIBRARIES
# ==============================
import numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# ==============================
# 2. LOAD DATA
# ==============================
train = pd.read_csv('/kaggle/input/titanic/train.csv')
test  = pd.read_csv('/kaggle/input/titanic/test.csv')

# ==============================
# 3. QUICK EDA
# ==============================
sns.countplot(x='Survived', data=train); plt.title("Survival Count"); plt.show()
sns.countplot(x='Sex', hue='Survived', data=train); plt.title("Survival by Gender"); plt.show()

# ==============================
# 4. CLEAN + PREPROCESS
# ==============================
train['Age'].fillna(train['Age'].median(), inplace=True)
test['Age'].fillna(test['Age'].median(), inplace=True)
train['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)
test['Embarked'].fillna(test['Embarked'].mode()[0], inplace=True)
test['Fare'].fillna(test['Fare'].median(), inplace=True)

train['Cabin'].fillna('Unknown', inplace=True)
test['Cabin'].fillna('Unknown', inplace=True)

drop_cols = ['PassengerId','Name','Ticket','Cabin']
test_ids = test['PassengerId']
train = train.drop(columns=drop_cols)
test  = test.drop(columns=drop_cols)

le = LabelEncoder()
for col in ['Sex','Embarked']:
    train[col] = le.fit_transform(train[col])
    test[col]  = le.transform(test[col])

# ==============================
# 5. SPLIT TRAIN/VALIDATION
# ==============================
X = train.drop('Survived',axis=1)
y = train['Survived']
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# 6. MODEL TRAINING
# ==============================
model = RandomForestClassifier(
    n_estimators=300, max_depth=10, random_state=42
)
model.fit(X_train, y_train)

# ==============================
# 7. EVALUATION
# ==============================
preds = model.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, preds))

# ==============================
# 8. TEST SET PREDICTIONS
# ==============================
test_preds = model.predict(test)

# ==============================
# 9. SUBMISSION FILE
# ==============================
submission = pd.DataFrame({
    'PassengerId': test_ids,
    'Survived': test_preds
})
submission.to_csv('submission.csv', index=False)
submission.head()
 



