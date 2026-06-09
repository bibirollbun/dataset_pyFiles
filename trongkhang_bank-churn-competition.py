import pandas as pd
import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv("/kaggle/input/bank-churn-competition-by-ipii-hs-ex-mts/train.csv")

df.info()

df.head()


print("**Kiá»ƒm tra duplicate value")

df.duplicated().sum()


df = df.drop(columns=['id','CustomerId','Surname'],axis=1)


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


corr_matrix = df.corr(numeric_only=True)
plt.figure(figsize=(12, 8))  # tÃ¹y chá»‰nh kÃ­ch thÆ°á»›c
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()



df.info()


import matplotlib.pyplot as plt
import seaborn as sns

numerical_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']

# TÃ­nh sá»‘ hÃ ng cáº§n (má»—i hÃ ng 2 biá»ƒu Ä‘á»“)
n_cols = 2
n_rows = (len(numerical_cols) + 1) // n_cols

# Táº¡o grid figure
fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
axes = axes.flatten()  # Chuyá»ƒn thÃ nh 1D Ä‘á»ƒ dá»… xá»­ lÃ½

# Váº½ tá»«ng biá»ƒu Ä‘á»“ vÃ o tá»«ng subplot
for i, col in enumerate(numerical_cols):
    sns.histplot(df[col], kde=True, bins=30, color='skyblue', ax=axes[i])
    axes[i].set_title(f"Distribution of {col}")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Frequency")
    axes[i].grid(True)

# XÃ³a subplot thá»«a náº¿u cÃ³
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()



import matplotlib.pyplot as plt
import seaborn as sns

corr_cols = ['Age', 'Balance', 'NumOfProducts', 'IsActiveMember']
target_col = 'Exited'

# Thiáº¿t láº­p lÆ°á»›i 2 cá»™t
n_cols = 2
n_rows = (len(corr_cols) + 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
axes = axes.flatten()

# Váº½ boxplot cho tá»«ng cá»™t
for i, col in enumerate(corr_cols):
    sns.boxplot(x=target_col, y=col, data=df, ax=axes[i], palette='Set2')
    axes[i].set_title(f'{col} vs {target_col}')
    axes[i].set_xlabel(target_col)
    axes[i].set_ylabel(col)

# XoÃ¡ subplot thá»«a náº¿u cÃ³
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()




import matplotlib.pyplot as plt
import seaborn as sns

corr_cols = ['Tenure', 'EstimatedSalary']
target_col = 'Exited'

# Thiáº¿t láº­p lÆ°á»›i 2 cá»™t
n_cols = 2
n_rows = (len(corr_cols) + 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
axes = axes.flatten()

# Váº½ boxplot cho tá»«ng cá»™t
for i, col in enumerate(corr_cols):
    sns.boxplot(x=target_col, y=col, data=df, ax=axes[i], palette='Set2')
    axes[i].set_title(f'{col} vs {target_col}')
    axes[i].set_xlabel(target_col)
    axes[i].set_ylabel(col)

# XoÃ¡ subplot thá»«a náº¿u cÃ³
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()



df['Exited'].value_counts().plot(kind='bar')
plt.title("Customer Exit Count")
plt.xlabel("Exited (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.show()


df.info()


category_cols = ['Geography','Gender','HasCrCard','IsActiveMember']

for col in category_cols:
    df[col] = df[col].astype('category')

df.info()


for col in category_cols:
    print(f"**{col}    {df[col].unique()}")


dummies_df = pd.get_dummies(df, drop_first=True)


dummies_df.head()


from sklearn.model_selection import train_test_split

drop_cols = ['Exited']

X = dummies_df.drop(drop_cols, axis=1)

y = dummies_df["Exited"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# 1. Ä�á»‹nh nghÄ©a cÃ¡c model vÃ  cÃ¡c hyperparameters tÆ°Æ¡ng á»©ng
models = {
    'Logistic Regression': {
        'pipeline': Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(solver='liblinear', class_weight='balanced'))
        ]),
        'params': {
            'model__penalty': ['l1', 'l2'],
            'model__C': [0.01, 0.1, 1, 10, 100],
            'model__max_iter': [100, 200, 500]
        }
    },
    'Random Forest': {
        'pipeline': Pipeline([
            ('model', RandomForestClassifier(random_state=42, class_weight='balanced'))
        ]),
        'params': {
            'model__n_estimators': [100, 200, 500],
            'model__max_depth': [None, 10, 20, 30],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4],
            'model__bootstrap': [True, False]
        }
    },
    'XGBoost': {
        'pipeline': Pipeline([
            ('model', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
        ]),
        'params': {
            'model__n_estimators': [100, 200, 300],
            'model__max_depth': [3, 4, 5, 6, 10],
            'model__learning_rate': [0.01, 0.1, 0.2, 0.3],
            'model__subsample': [0.5, 0.7, 1.0],
            'model__colsample_bytree': [0.5, 0.7, 1.0],
            'model__scale_pos_weight': [1, 2, 3, 4, 5]
        }
    }
}

# 2. Duyá»‡t tá»«ng model Ä‘á»ƒ tÃ¬m best hyperparameters vÃ  Ä‘Ã¡nh giÃ¡ káº¿t quáº£
for name, config in models.items():
    print(f"\nğŸ”� Tuning hyperparameters for: {name}")
    search = RandomizedSearchCV(
        config['pipeline'],
        config['params'],
        cv=10,
        scoring='accuracy',
        n_iter=15,
        random_state=42,
        n_jobs=-1
    )
    search.fit(X_train, y_train)

    print(f"âœ… Best params for {name}: {search.best_params_}")
    best_model = search.best_estimator_
    # Predict & Evaluate
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"ğŸ“ˆ Accuracy on test set: {acc:.4f}")
    print("ğŸ“Š Classification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title(f"Confusion Matrix - {name}")
    plt.show()

    print("=" * 60)



from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 1. Pipeline cho Logistic Regression vá»›i best params
logistic_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(
        solver='liblinear',
        penalty='l1',
        C=0.1,
        class_weight='balanced',
        max_iter=200
    ))
])

# 2. XGBoost vá»›i best params
xgboost_model = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
    n_estimators=200,
    max_depth=3,
    learning_rate=0.2,
    subsample=0.7,
    colsample_bytree=0.5,
    scale_pos_weight=1
)


stack_model = StackingClassifier(
    estimators=[
        ('logistic', logistic_pipe),
        ('xgb', xgboost_model)
    ],
    final_estimator=LogisticRegression(),
    passthrough=False,
    cv=5,
    n_jobs=-1
)

stack_model.fit(X_train, y_train)

# 5. Dá»± Ä‘oÃ¡n & Ä‘Ã¡nh giÃ¡
y_pred = stack_model.predict(X_test)

print("ğŸ“Š Classification Report:")
print(classification_report(y_test, y_pred))

# 6. Ma tráº­n nháº§m láº«n
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix - Stacking (Tuned Logistic + XGBoost)")
plt.show()



from sklearn.metrics import precision_recall_curve, average_precision_score, PrecisionRecallDisplay

# Láº¥y xÃ¡c suáº¥t dá»± Ä‘oÃ¡n lá»›p dÆ°Æ¡ng (label=1)
y_scores = stack_model.predict_proba(X_test)[:, 1]

# TÃ­nh precision, recall
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
avg_precision = average_precision_score(y_test, y_scores)

# Váº½ biá»ƒu Ä‘á»“
disp = PrecisionRecallDisplay(precision=precision, recall=recall, average_precision=avg_precision)
disp.plot()
plt.title("Precision-Recall Curve - Stacking Model")
plt.grid(True)
plt.show()



df_test = pd.read_csv("/kaggle/input/bank-churn-competition-by-ipii-hs-ex-mts/test.csv")

df_id = df_test["id"]

df_test = df_test.drop(columns=['id','CustomerId','Surname'],axis=1)

category_cols = ['Geography','Gender','HasCrCard','IsActiveMember']

for col in category_cols:
    df_test[col] = df_test[col].astype('category')

test_dummies_df = pd.get_dummies(df_test, drop_first=True)

test_dummies_df 


# Dá»± Ä‘oÃ¡n xÃ¡c suáº¥t thuá»™c lá»›p "Exited" (class 1)
y_pred_proba = stack_model.predict_proba(test_dummies_df)[:, 1]

# Táº¡o DataFrame submission
submission_df = pd.DataFrame({
    "id": df_id,
    "Exited": y_pred_proba
})

# (TÃ¹y chá»�n) Ghi ra file CSV
submission_df.to_csv("submission.csv", index=False)


