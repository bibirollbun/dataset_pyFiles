import pandas as pd
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

df = pd.read_csv("/kaggle/input/playground-series-s4e10/train.csv")

df.info()

df.head()


print("**Kiá»…m tra Missing Value")
print(df.isnull().sum())

print("**Kiá»…m tra Duplicate Value")
print(df.duplicated().sum())


cate_cols = ["person_home_ownership", "loan_intent" , "loan_grade", "cb_person_default_on_file"]

for col in cate_cols:
    df[col] = df[col].astype('category')


df.describe()


import matplotlib.pyplot as plt
import seaborn as sns

corr = df.corr(numeric_only=True)

plt.figure(figsize=(14, 7))  # Ä�áº·t kÃ­ch thÆ°á»›c figure
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")  # ThÃªm annot cho dá»… nhÃ¬n náº¿u muá»‘n
plt.title("Heatmap Correlation")
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

numerical_cols = df.select_dtypes(include='number').columns.tolist()

for col in numerical_cols:
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    # Histogram
    sns.histplot(df[col], ax=ax[0], kde=True)
    ax[0].set_title(f"Histogram of {col}")

    # Boxplot
    sns.boxplot(y=df[col], ax=ax[1])
    ax[1].set_title(f"Boxplot of {col}")

    plt.tight_layout()
    plt.show()



import matplotlib.pyplot as plt
import seaborn as sns

categorical_cols = df.select_dtypes(include='category').columns.tolist()

for col in categorical_cols:
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    # Countplot
    sns.countplot(x=df[col], ax=ax)
    ax.set_title(f"Countplot of {col}")
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()



import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.countplot(x='person_home_ownership', hue='loan_status', data=df)
plt.title('Loan Status by Home Ownership')
plt.xlabel('Home Ownership')
plt.ylabel('Count')
plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(x='person_home_ownership', y='cb_person_cred_hist_length', data=df)
plt.title('Credit History Length by Home Ownership')
plt.xlabel('Home Ownership')
plt.ylabel('Credit History Length (years)')
plt.show()


df['loan_age_bin'] = pd.cut(
    df['person_age'],
    bins=[20, 30, 40, 56, 120],
    labels=['20-29', '30-39', '40-55', '56+'],
    right=False 
)

plt.figure(figsize=(10, 6))
sns.countplot(x='person_home_ownership', hue='loan_age_bin', data=df)
plt.title('Home Ownership by Loan Age Group')
plt.xlabel('Home Ownership')
plt.ylabel('Count')
plt.legend(title='Loan Age Group', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()




plt.figure(figsize=(10, 6))
sns.countplot(x='loan_intent', hue='loan_status', data=df)
plt.title('Loan Status by Loan Intent')
plt.xlabel('Loan Intent')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='loan_intent', y='loan_percent_income', data=df)
plt.title('Loan Percent Income Distribution by Loan Intent')
plt.xlabel('Loan Intent')
plt.ylabel('Loan Percent Income')
plt.xticks(rotation=45)
plt.show()




plt.figure(figsize=(10, 6))
sns.countplot(x='loan_grade', hue='loan_status', data=df)
plt.title('Loan Status Count by Loan Grade')
plt.xlabel('Loan Grade')
plt.ylabel('Count')
plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(x='loan_grade', y='loan_percent_income', data=df)
plt.title('Loan Percent Income Distribution by Loan Intent')
plt.xlabel('Loan Intent')
plt.ylabel('Loan Percent Income')
plt.xticks(rotation=45)
plt.show()



plt.figure(figsize=(10, 6))
sns.boxplot(x='loan_grade', y='person_age', data=df)
plt.title('Person Age Distribution by Loan Grade')
plt.xlabel('Loan Grade')
plt.ylabel('Person Age')
plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(x='loan_grade', y='person_income', data=df)
plt.title('Person Income Distribution by Loan Grade')
plt.xlabel('Loan Grade')
plt.ylabel('Person Income')
plt.show()


plt.figure(figsize=(10, 6))
sns.countplot(x='loan_grade', hue='person_home_ownership', data=df)
plt.title('Home Ownership by Loan Grade')
plt.xlabel('Loan Grade')
plt.ylabel('Count')
plt.show()



plt.figure(figsize=(8, 5))
sns.countplot(x='cb_person_default_on_file', hue='loan_status', data=df)
plt.title('Loan Status by Default History')
plt.xlabel('Default on File')
plt.ylabel('Count')
plt.show()


plt.figure(figsize=(8, 5))
sns.countplot(x='cb_person_default_on_file', hue='loan_grade', data=df)
plt.title('Loan Grade by Default History')
plt.xlabel('Default on File')
plt.ylabel('Count')
plt.show()


plt.figure(figsize=(8, 5))
sns.boxplot(x='cb_person_default_on_file', y='loan_int_rate', data=df)
plt.title('Loan Interest Rate by Default History')
plt.xlabel('Default on File')
plt.ylabel('Loan Interest Rate (%)')
plt.show()



drop_cols = ["id","loan_age_bin"]

df = df.drop(drop_cols, axis=1)

df.head()


grade_list = df['loan_grade'].unique().tolist()

grade_list.sort(reverse=True)

print(grade_list)

grade_mapping = {grade: idx for idx, grade in enumerate(grade_list)}

print(grade_mapping)

df['loan_grade'] = df['loan_grade'].map(grade_mapping)
df['loan_grade'] = df['loan_grade'].astype('int8')
df.head()


df_dummies = pd.get_dummies(df, drop_first=True)

df_dummies.head()


from sklearn.model_selection import train_test_split

X = df_dummies.drop("loan_status",axis=1)
y = df_dummies["loan_status"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Train size")
print(y_train.value_counts(normalize=True))

print("Test size")
print(y_test.value_counts(normalize=True))


import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def objective(trial, model_name):
    if model_name == 'LogisticRegression':
        C = trial.suggest_loguniform('C', 0.001, 100)
        model = LogisticRegression(C=C, max_iter=1000, solver='liblinear')
    
    elif model_name == 'RandomForest':
        n_estimators = trial.suggest_int('n_estimators', 100, 300)
        max_depth = trial.suggest_int('max_depth', 5, 15)
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, verbose=0)
    
    elif model_name == 'XGBoost':
        n_estimators = trial.suggest_int('n_estimators', 100, 300)
        max_depth = trial.suggest_int('max_depth', 3, 9)
        learning_rate = trial.suggest_loguniform('learning_rate', 0.01, 0.1)
        model = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, 
                              learning_rate=learning_rate, verbosity=0, use_label_encoder=False)
    
    elif model_name == 'CatBoost':
        iterations = trial.suggest_int('iterations', 100, 300)
        depth = trial.suggest_int('depth', 3, 9)
        learning_rate = trial.suggest_loguniform('learning_rate', 0.01, 0.1)
        model = CatBoostClassifier(iterations=iterations, depth=depth, 
                                   learning_rate=learning_rate, verbose=0)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return acc

models_list = ['LogisticRegression', 'RandomForest', 'XGBoost', 'CatBoost']

for model_name in models_list:
    print(f"\nğŸ”¹ Hyperparameter tuning for: {model_name}")
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, model_name), n_trials=30)

    print(f"Best params: {study.best_params}")
    print(f"Best accuracy: {study.best_value:.4f}")

    best_trial = study.best_params
    # Khá»Ÿi táº¡o láº¡i model vá»›i best params
    if model_name == 'LogisticRegression':
        model = LogisticRegression(C=best_trial['C'], max_iter=1000, solver='liblinear')
    elif model_name == 'RandomForest':
        model = RandomForestClassifier(n_estimators=best_trial['n_estimators'],
                                       max_depth=best_trial['max_depth'], verbose=0)
    elif model_name == 'XGBoost':
        model = XGBClassifier(n_estimators=best_trial['n_estimators'],
                              max_depth=best_trial['max_depth'],
                              learning_rate=best_trial['learning_rate'],
                              verbosity=0, use_label_encoder=False)
    elif model_name == 'CatBoost':
        model = CatBoostClassifier(iterations=best_trial['iterations'],
                                   depth=best_trial['depth'],
                                   learning_rate=best_trial['learning_rate'],
                                   verbose=0)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    best_model_ = model
    print("Classification report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix: {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()



from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

models = {
    'LogisticRegression': {
        'model': LogisticRegression(max_iter=1000, solver='liblinear'),
        'params': {
            'C': [0.01, 0.1, 1, 10]
        }
    },
    'RandomForest': {
        'model': RandomForestClassifier(),
        'params': {
            'n_estimators': [100, 200,300],
            'max_depth': [5, 10, 15]
        }
    },
    'XGBoost': {
        'model': XGBClassifier(verbosity=0, use_label_encoder=False),
        'params': {
            'n_estimators': [100, 200,300],
            'max_depth': [3, 6 , 9],
            'learning_rate': [0.1, 0.05, 0.01]
        }
    },
    'CatBoost': {
        'model': CatBoostClassifier(verbose=0),
        'params': {
            'iterations': [100, 200,300],
            'depth': [3, 6,9],
            'learning_rate': [0.1, 0.05, 0.01]
        }
    }
}

for name, cfg in models.items():
    print(f"\nğŸ”¹ Hyperparameter tuning for: {name}")
    
    grid = GridSearchCV(cfg['model'], cfg['params'], cv=3, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    
    best_model = grid.best_estimator_
    
    print(f"Best params: {grid.best_params_}")
    best_model_ = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    
    print("Classification report:")
    print(classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix: {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()



from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

counter = Counter(y_train)
n_negative = counter[0]
n_positive = counter[1]
scale_pos_weight = n_negative / n_positive

models = {
    'LogisticRegression': {
        'model': LogisticRegression(max_iter=1000, solver='liblinear', class_weight='balanced'),
        'params': {
            'C': [0.01, 0.1, 1, 10]
        }
    },
    'RandomForest': {
        'model': RandomForestClassifier(class_weight='balanced'),
        'params': {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, 15]
        }
    },
    'XGBoost': {
        'model': XGBClassifier(verbosity=0, use_label_encoder=False, scale_pos_weight=scale_pos_weight),
        'params': {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.1, 0.05, 0.01]
        }
    },
    'CatBoost': {
        'model': CatBoostClassifier(verbose=0, scale_pos_weight=scale_pos_weight),
        'params': {
            'iterations': [100, 200, 300],
            'depth': [3, 6, 9],
            'learning_rate': [0.1, 0.05, 0.01]
        }
    }
}

for name, cfg in models.items():
    print(f"\nğŸ”¹ Hyperparameter tuning for: {name}")
    
    grid = GridSearchCV(cfg['model'], cfg['params'], cv=3, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    
    best_model = grid.best_estimator_
    
    print(f"Best params: {grid.best_params_}")
    
    y_pred = best_model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    
    print("Classification report:")
    print(classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix: {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()



X_train = X_train.drop('loan_grade',axis= 1)
X_test = X_test.drop('loan_grade',axis= 1)

X_train.head()


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

models = {
    'LogisticRegression': {
        'model': LogisticRegression(max_iter=1000, solver='liblinear'),
        'params': {
            'C': [0.01, 0.1, 1, 10]
        }
    },
    'RandomForest': {
        'model': RandomForestClassifier(),
        'params': {
            'n_estimators': [100, 200,300],
            'max_depth': [5, 10, 15]
        }
    },
    'XGBoost': {
        'model': XGBClassifier(verbosity=0, use_label_encoder=False),
        'params': {
            'n_estimators': [100, 200,300],
            'max_depth': [3, 6 , 9],
            'learning_rate': [0.1, 0.05, 0.01]
        }
    },
    'CatBoost': {
        'model': CatBoostClassifier(verbose=0),
        'params': {
            'iterations': [100, 200,300],
            'depth': [3, 6,9],
            'learning_rate': [0.1, 0.05, 0.01]
        }
    }
}

for name, cfg in models.items():
    print(f"\nğŸ”¹ Hyperparameter tuning for: {name}")
    
    grid = GridSearchCV(cfg['model'], cfg['params'], cv=3, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    
    best_model = grid.best_estimator_
    
    print(f"Best params: {grid.best_params_}")
    
    y_pred = best_model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    
    print("Classification report:")
    print(classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix: {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()



from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

counter = Counter(y_train)
n_negative = counter[0]
n_positive = counter[1]
scale_pos_weight = n_negative / n_positive

models = {
    'LogisticRegression': {
        'model': LogisticRegression(max_iter=1000, solver='liblinear', class_weight='balanced'),
        'params': {
            'C': [0.01, 0.1, 1, 10]
        }
    },
    'RandomForest': {
        'model': RandomForestClassifier(class_weight='balanced'),
        'params': {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, 15]
        }
    },
    'XGBoost': {
        'model': XGBClassifier(verbosity=0, use_label_encoder=False, scale_pos_weight=scale_pos_weight),
        'params': {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.1, 0.05, 0.01]
        }
    },
    'CatBoost': {
        'model': CatBoostClassifier(verbose=0, scale_pos_weight=scale_pos_weight),
        'params': {
            'iterations': [100, 200, 300],
            'depth': [3, 6, 9],
            'learning_rate': [0.1, 0.05, 0.01]
        }
    }
}

for name, cfg in models.items():
    print(f"\nğŸ”¹ Hyperparameter tuning for: {name}")
    
    grid = GridSearchCV(cfg['model'], cfg['params'], cv=3, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    
    best_model = grid.best_estimator_
    
    print(f"Best params: {grid.best_params_}")
    
    y_pred = best_model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    
    print("Classification report:")
    print(classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix: {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()



df_test = pd.read_csv("/kaggle/input/playground-series-s4e10/test.csv")

cate_cols = ["person_home_ownership", "loan_intent" , "cb_person_default_on_file"]

for col in cate_cols:
    df[col] = df[col].astype('category')



grade_list = df['loan_grade'].unique().tolist()

grade_list.sort(reverse=True)

grade_mapping = {grade: idx for idx, grade in enumerate(grade_list)}

df_test['loan_grade'] = df_test['loan_grade'].map(grade_mapping)



df_test_dummies = pd.get_dummies(df_test, drop_first=True)

df_test_dummies.head()


y_pred = best_model_.predict_proba(df_test_dummies)[:,1]

submission = pd.DataFrame({"id":df_test["id"], "loan_status":y_pred})

submission.to_csv("submission.csv",index=False)

