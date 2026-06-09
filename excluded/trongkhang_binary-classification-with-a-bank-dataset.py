import pandas as pd

df = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
df.info() # Xem cÆ¡ báº£n vá»� thÃ nh pháº§n dá»¯ liá»‡u
df.head() # Xem 5 dÃ²ng dá»¯ liá»‡u xem cÃ³ cÃ¡c giÃ¡ trá»‹ cá»Ÿ báº£n gÃ¬


print("** Duplicated Value: ",df.duplicated().sum())


import matplotlib.pyplot as plt
import seaborn as sns

corr = df.corr(numeric_only=True)
sns.heatmap(data=corr, annot=True, fmt=".2f", cmap='coolwarm')
plt.show()


numeric_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64']]

for col in numeric_cols:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f'Biá»ƒu Ä‘á»“ cho {col}', fontsize=14)

    # Histogram
    sns.histplot(df[col], kde=True, ax=axes[0], color='skyblue')
    axes[0].set_title('Histogram')

    # Boxplot
    sns.boxplot(x=df[col], ax=axes[1], color='lightgreen')
    axes[1].set_title('Boxplot')

    plt.tight_layout()
    plt.show()


categorical_cols = [col for col in df.columns if df[col].dtype in ['object', 'category']]

for col in categorical_cols:
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x=col)
    plt.title(f"Count Plot: {col}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()



# Chuyá»ƒn cÃ¡c cá»™t object sang category
object_cols = [col for col in df.columns if df[col].dtype == 'object']
for col in object_cols:
    df[col] = df[col].astype('category')


# Káº¿t quáº£ cá»§a convert datatype
df.info()


# Kiá»…m tra cÃ¡c giÃ¡ trá»‹ cá»§a cÃ¡c cá»™t category
category_cols = [col for col in df.columns if df[col].dtype == 'category']

count = 1
for col in category_cols:
    print(f"{count:<5}{df[col].name}: {df[col].unique()}")
    count += 1


category_cols = [col for col in df.columns if df[col].dtype.name == 'category']
# Tiáº¿n hÃ nh onehot
df = pd.get_dummies(df, columns=category_cols, drop_first=False)


# Xem káº¿t quáº£
df.head()


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# LÃ½ do drop duration bá»Ÿi vÃ¬ duration chá»‰ cÃ³ khi cuá»™c gá»�i káº¿t thÃºc -> khÃ³ Ã¡p dá»¥ng mÃ´ hÃ¬nh
# vÃ o thá»±c táº¿

X = df.drop(["y", "id"], axis=1)
y = df["y"]

# Chia train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Xem káº¿t quáº£
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test :", X_test.shape)
print("y_test :", y_test.shape)

print("\nTá»‰ lá»‡ y trong train:")
print(y_train.value_counts(normalize=True))

print("\nTá»‰ lá»‡ y trong test:")
print(y_test.value_counts(normalize=True))


import optuna
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import joblib

# ===============================
# Giáº£ sá»­ X_train_res, y_train_res, X_test, y_test Ä‘Ã£ cÃ³ sáºµn
# ===============================

# HÃ m objective cho Optuna
def objective(trial):
    C = trial.suggest_float('C', 1e-3, 1e1, log=True)
    solver = trial.suggest_categorical('solver', ['lbfgs', 'liblinear'])
    penalty = 'l2' if solver == 'lbfgs' else trial.suggest_categorical('penalty', ['l1', 'l2'])
    max_iter = trial.suggest_int('max_iter', 100, 300)

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=C, solver=solver, penalty=penalty, max_iter=max_iter,
            class_weight='balanced', random_state=42
        )
    )

    # Sá»­ dá»¥ng backend threading Ä‘á»ƒ trÃ¡nh lá»—i pickle trÃªn sklearn má»›i
    with joblib.parallel_backend('threading'):
        score = cross_val_score(model, X_train, y_train,
                                cv=5, scoring='accuracy', n_jobs=-1).mean()
    return score


# ===============================
# Tuning tham sá»‘ vá»›i Optuna
# ===============================
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=15)

print("Best params:", study.best_params)

# ===============================
# Huáº¥n luyá»‡n model tá»‘t nháº¥t
# ===============================
best_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(**study.best_params, class_weight='balanced', random_state=42)
)

best_model.fit(X_train, y_train)

# ===============================
# Ä�Ã¡nh giÃ¡
# ===============================
y_pred = best_model.predict(X_test)
print("Train accuracy:", best_model.score(X_train, y_train))
print("Test accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ===============================
# Confusion matrix
# ===============================
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=best_model.classes_, yticklabels=best_model.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()



import optuna
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# ===============================
# Giáº£ sá»­ X_train_res, y_train_res, X_test, y_test Ä‘Ã£ tá»“n táº¡i
# ===============================

# HÃ m objective cho Optuna
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 6),
        'gamma': trial.suggest_float('gamma', 0, 2),
        'scale_pos_weight': sum(y_train == 0) / sum(y_train == 1),
        'random_state': 42,
        'n_jobs': -1,
        'use_label_encoder': False,
        'eval_metric': 'logloss'
    }

    model = xgb.XGBClassifier(**params)

    # DÃ¹ng threading backend Ä‘á»ƒ trÃ¡nh lá»—i pickle
    with joblib.parallel_backend('threading'):
        score = cross_val_score(model, X_train, y_train,
                                cv=5, scoring='accuracy', n_jobs=-1).mean()
    return score


# ===============================
# Tá»‘i Æ°u tham sá»‘
# ===============================
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)  # giáº£m trial cho nhanh

print("Best parameters:", study.best_params)
print("Best CV accuracy:", study.best_value)

# ===============================
# Huáº¥n luyá»‡n model tá»‘t nháº¥t
# ===============================
best_params = study.best_params.copy()
best_params.update({
    'use_label_encoder': False,
    'eval_metric': 'logloss',
    'n_jobs': -1
})

best_model = xgb.XGBClassifier(**best_params)
best_model.fit(X_train, y_train)

# ===============================
# Ä�Ã¡nh giÃ¡
# ===============================
y_pred = best_model.predict(X_test)
print("Train accuracy:", best_model.score(X_train, y_train))
print("Test accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ===============================
# Confusion matrix
# ===============================
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=best_model.classes_, yticklabels=best_model.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


from imblearn.over_sampling import SMOTE

# Ã�p dá»¥ng SMOTE lÃªn táº­p train
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)


# Kiá»ƒm tra káº¿t quáº£
print("TrÆ°á»›c SMOTE:", y_train.value_counts())
print("Sau SMOTE :", y_train_res.value_counts())


import optuna
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import joblib

# ===============================
# Giáº£ sá»­ X_train_res, y_train_res, X_test, y_test Ä‘Ã£ cÃ³ sáºµn
# ===============================

# HÃ m objective cho Optuna
def objective(trial):
    C = trial.suggest_float('C', 1e-3, 1e1, log=True)
    solver = trial.suggest_categorical('solver', ['lbfgs', 'liblinear'])
    penalty = 'l2' if solver == 'lbfgs' else trial.suggest_categorical('penalty', ['l1', 'l2'])
    max_iter = trial.suggest_int('max_iter', 100, 300)

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=C, solver=solver, penalty=penalty, max_iter=max_iter,
            class_weight='balanced', random_state=42
        )
    )

    # Sá»­ dá»¥ng backend threading Ä‘á»ƒ trÃ¡nh lá»—i pickle trÃªn sklearn má»›i
    with joblib.parallel_backend('threading'):
        score = cross_val_score(model, X_train_res, y_train_res,
                                cv=5, scoring='accuracy', n_jobs=-1).mean()
    return score


# ===============================
# Tuning tham sá»‘ vá»›i Optuna
# ===============================
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=15)

print("Best params:", study.best_params)

# ===============================
# Huáº¥n luyá»‡n model tá»‘t nháº¥t
# ===============================
best_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(**study.best_params, class_weight='balanced', random_state=42)
)

best_model.fit(X_train_res, y_train_res)

# ===============================
# Ä�Ã¡nh giÃ¡
# ===============================
y_pred = best_model.predict(X_test)
print("Train accuracy:", best_model.score(X_train_res, y_train_res))
print("Test accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ===============================
# Confusion matrix
# ===============================
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=best_model.classes_, yticklabels=best_model.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()



import optuna
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# ===============================
# Giáº£ sá»­ X_train_res, y_train_res, X_test, y_test Ä‘Ã£ tá»“n táº¡i
# ===============================

# HÃ m objective cho Optuna
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 6),
        'gamma': trial.suggest_float('gamma', 0, 2),
        'scale_pos_weight': sum(y_train_res == 0) / sum(y_train_res == 1),
        'random_state': 42,
        'n_jobs': -1,
        'use_label_encoder': False,
        'eval_metric': 'logloss'
    }

    model = xgb.XGBClassifier(**params)

    # DÃ¹ng threading backend Ä‘á»ƒ trÃ¡nh lá»—i pickle
    with joblib.parallel_backend('threading'):
        score = cross_val_score(model, X_train_res, y_train_res,
                                cv=5, scoring='accuracy', n_jobs=-1).mean()
    return score


# ===============================
# Tá»‘i Æ°u tham sá»‘
# ===============================
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)  # giáº£m trial cho nhanh

print("Best parameters:", study.best_params)
print("Best CV accuracy:", study.best_value)

# ===============================
# Huáº¥n luyá»‡n model tá»‘t nháº¥t
# ===============================
best_params = study.best_params.copy()
best_params.update({
    'use_label_encoder': False,
    'eval_metric': 'logloss',
    'n_jobs': -1
})

best_model = xgb.XGBClassifier(**best_params)
best_model.fit(X_train_res, y_train_res)

# ===============================
# Ä�Ã¡nh giÃ¡
# ===============================
y_pred = best_model.predict(X_test)
print("Train accuracy:", best_model.score(X_train_res, y_train_res))
print("Test accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ===============================
# Confusion matrix
# ===============================
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=best_model.classes_, yticklabels=best_model.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()



import numpy as np

df['balance_log'] = np.log(df['balance'] - df['balance'].min() + 1)


# Xem káº¿t quáº£
df.head()


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X = df.drop(["y", "id", "balance"], axis=1)
y = df["y"]

# Chia train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Ã�p dá»¥ng SMOTE lÃªn táº­p train
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)


import optuna
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import joblib

# ===============================
# Giáº£ sá»­ X_train_res, y_train_res, X_test, y_test Ä‘Ã£ cÃ³ sáºµn
# ===============================

# HÃ m objective cho Optuna
def objective(trial):
    C = trial.suggest_float('C', 1e-3, 1e1, log=True)
    solver = trial.suggest_categorical('solver', ['lbfgs', 'liblinear'])
    penalty = 'l2' if solver == 'lbfgs' else trial.suggest_categorical('penalty', ['l1', 'l2'])
    max_iter = trial.suggest_int('max_iter', 100, 300)

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=C, solver=solver, penalty=penalty, max_iter=max_iter,
            class_weight='balanced', random_state=42
        )
    )

    # Sá»­ dá»¥ng backend threading Ä‘á»ƒ trÃ¡nh lá»—i pickle trÃªn sklearn má»›i
    with joblib.parallel_backend('threading'):
        score = cross_val_score(model, X_train_res, y_train_res,
                                cv=5, scoring='accuracy', n_jobs=-1).mean()
    return score


# ===============================
# Tuning tham sá»‘ vá»›i Optuna
# ===============================
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=15)

print("Best params:", study.best_params)

# ===============================
# Huáº¥n luyá»‡n model tá»‘t nháº¥t
# ===============================
best_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(**study.best_params, class_weight='balanced', random_state=42)
)

best_model.fit(X_train_res, y_train_res)

# ===============================
# Ä�Ã¡nh giÃ¡
# ===============================
y_pred = best_model.predict(X_test)
print("Train accuracy:", best_model.score(X_train_res, y_train_res))
print("Test accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ===============================
# Confusion matrix
# ===============================
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=best_model.classes_, yticklabels=best_model.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()



import optuna
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# ===============================
# Giáº£ sá»­ X_train_res, y_train_res, X_test, y_test Ä‘Ã£ tá»“n táº¡i
# ===============================

# HÃ m objective cho Optuna
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 6),
        'gamma': trial.suggest_float('gamma', 0, 2),
        'scale_pos_weight': sum(y_train_res == 0) / sum(y_train_res == 1),
        'random_state': 42,
        'n_jobs': -1,
        'use_label_encoder': False,
        'eval_metric': 'logloss'
    }

    model = xgb.XGBClassifier(**params)

    # DÃ¹ng threading backend Ä‘á»ƒ trÃ¡nh lá»—i pickle
    with joblib.parallel_backend('threading'):
        score = cross_val_score(model, X_train_res, y_train_res,
                                cv=5, scoring='accuracy', n_jobs=-1).mean()
    return score


# ===============================
# Tá»‘i Æ°u tham sá»‘
# ===============================
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)  # giáº£m trial cho nhanh

print("Best parameters:", study.best_params)
print("Best CV accuracy:", study.best_value)

# ===============================
# Huáº¥n luyá»‡n model tá»‘t nháº¥t
# ===============================
best_params = study.best_params.copy()
best_params.update({
    'use_label_encoder': False,
    'eval_metric': 'logloss',
    'n_jobs': -1
})

best_model = xgb.XGBClassifier(**best_params)
best_model.fit(X_train_res, y_train_res)

# ===============================
# Ä�Ã¡nh giÃ¡
# ===============================
y_pred = best_model.predict(X_test)
print("Train accuracy:", best_model.score(X_train_res, y_train_res))
print("Test accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ===============================
# Confusion matrix
# ===============================
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=best_model.classes_, yticklabels=best_model.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()



# Ä�á»�c dá»¯ liá»‡u
test_df = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")


# Chuyá»ƒn kiá»ƒu dá»¯ liá»‡u
object_cols = [col for col in test_df.columns if test_df[col].dtype == 'object']
for col in object_cols:
    test_df[col] = test_df[col].astype('category')


# One-hot encoding
category_cols = [col for col in test_df.columns if test_df[col].dtype.name == 'category']
# Tiáº¿n hÃ nh onehot
test_df = pd.get_dummies(test_df, columns=category_cols, drop_first=False)


# Engineering data
import numpy as np

test_df['balance_log'] = np.log(test_df['balance'] - test_df['balance'].min() + 1)
preidict_df = test_df.drop(['id', 'balance'], axis = 1)


# Dá»± Ä‘oÃ¡n trÃªn táº­p test
y_proba = best_model.predict_proba(preidict_df)[:, 1]  # XÃ¡c suáº¥t thuá»™c lá»›p 1

submission = pd.DataFrame({
    'id': test_df["id"] ,       # Cá»™t ID cá»§a táº­p test (cáº§n cÃ³ trong dá»¯ liá»‡u)
    'prediction': y_proba # Káº¿t quáº£ dá»± Ä‘oÃ¡n
})

submission.to_csv('submission.csv', index=False)


from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# Logistic Regression vá»›i best params
log_model = LogisticRegression(
    C=0.0011211496483690068,
    solver='lbfgs',
    max_iter=254,
    random_state=42
)

# XGBoost vá»›i best params
xgb_model = XGBClassifier(
    n_estimators=175,
    max_depth=7,
    learning_rate=0.03728671523473948,
    subsample=0.8872251483230372,
    colsample_bytree=0.7005232140482939,
    min_child_weight=1,
    gamma=1.409213413854455,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)



from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

# Khá»Ÿi táº¡o model vá»›i best params
log_model = LogisticRegression(
    C=0.0011211496483690068,
    solver='lbfgs',
    max_iter=254
)

xgb_model = XGBClassifier(
    n_estimators=175,
    max_depth=7,
    learning_rate=0.03728671523473948,
    subsample=0.8872251483230372,
    colsample_bytree=0.7005232140482939,
    min_child_weight=1,
    gamma=1.409213413854455,
    use_label_encoder=False,
    eval_metric='logloss'
)

# Train tá»«ng model riÃªng Ä‘á»ƒ trÃ¡nh lá»—i sample_weight
log_model.fit(X_train_res, y_train_res)
xgb_model.fit(X_train_res, y_train_res)

# Voting Hard & Soft
voting_hard = VotingClassifier(
    estimators=[('logistic', log_model), ('xgboost', xgb_model)],
    voting='hard',
)

voting_soft = VotingClassifier(
    estimators=[('logistic', log_model), ('xgboost', xgb_model)],
    voting='soft',
)

# Train Voting
voting_hard.fit(X_train_res, y_train_res)
voting_soft.fit(X_train_res, y_train_res)

# HÃ m váº½ confusion matrix
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Evaluate
for name, model in [('Hard Voting', voting_hard), ('Soft Voting', voting_soft)]:
    y_pred = model.predict(X_test)
    print(f"\n=== {name} ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    plot_confusion_matrix(y_test, y_pred, f"Confusion Matrix - {name}")


