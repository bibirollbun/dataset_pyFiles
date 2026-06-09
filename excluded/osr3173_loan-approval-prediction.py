import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


train = pd.read_csv("/kaggle/input/playground-series-s4e10/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s4e10/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s4e10/sample_submission.csv")


train.info()


test.info()


x=train.drop(columns=['id','loan_status'])
y=train['loan_status']


# loan_status의 분포
import matplotlib.pyplot as plt
sns.countplot(x='loan_status', data=train)
plt.show()



# 범주형 변수 수치화
X_encoded=pd.get_dummies(x)
test_encoded=pd.get_dummies(test.drop(columns=['id']))


# loan_status와 다른 변수 간의 상관관계 계산
correlation_with_loan_status = train.corr(numeric_only=True)['loan_status'].sort_values(ascending=False)


# 상관관계 시각화
plt.figure(figsize=(10, 6))
sns.heatmap(pd.DataFrame(correlation_with_loan_status), annot=True, cmap='coolwarm', fmt=".3f")
plt.title('Correlation with Loan Status')
plt.show()

# 상관관계 출력
print(correlation_with_loan_status)


# id 컬럼을 제외한 상위 10개 변수 선택
top_10_features = train.columns[1:11]

# 숫자형 변수만 선택
numeric_features = train[top_10_features].select_dtypes(include=np.number)

# Heatmap으로 상관관계 시각화
correlation_matrix = numeric_features.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Top 10 Features (excluding ID)')
plt.show()


# 수치형 변수 파악

# person_age(나이)

# person_income(연간 소득)

# person_emp_length(근속 연수)

# loan amnt(대출 금액)

# loan_int_rate(대출 이자율)

# loan_percent_income(연간소득에 대출금액이 차지하는 비율)

# cb_person_cred_hist_length(신용 유지 기간[년])


# 수치형 변수 분포
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

# Age group distribution (나이)
train['age_group'] = (train['person_age'] // 10) * 10
sns.countplot(x='age_group', data=train, ax=axes[0, 0])
axes[0, 0].set_title('Distribution of Age Groups')
axes[0, 0].set_xlabel('Age Group (10-year intervals)')
axes[0, 0].set_ylabel('Count')

# Income distribution (연간 소득)
income_bins = [0, 50000, 100000, 150000, 200000, float('inf')]
income_labels = ['0-50K', '50K-100K', '100K-150K', '150K-200K', '200K+']
train['income_category'] = pd.cut(train['person_income'], bins=income_bins, labels=income_labels, right=False)
sns.countplot(x='income_category', data=train, order=income_labels, ax=axes[0, 1])
axes[0, 1].set_title('Distribution of Person Income ')
axes[0, 1].set_xlabel('Income')
axes[0, 1].set_ylabel('Count')

# Employment length distribution (근속 연수)
emp_length_bins = [-1, 0, 5, 10, 15, 20, float('inf')]
emp_length_labels = ['<1 year', '1-5 years', '6-10 years', '11-15 years', '16-20 years', '20+ years']
train['emp_length_category'] = pd.cut(train['person_emp_length'], bins=emp_length_bins, labels=emp_length_labels, right=False)
sns.countplot(x='emp_length_category', data=train, order=emp_length_labels, ax=axes[0, 2])
axes[0, 2].set_title('Distribution of Employment Length ')
axes[0, 2].set_xlabel('Employment Length')
axes[0, 2].set_ylabel('Count')
axes[0, 2].set_xticklabels(emp_length_labels, rotation=45)

# Loan amount distribution (대출 금액)
loan_amnt_bins = [0, 5000, 10000, 15000, 20000, 25000, float('inf')]
loan_amnt_labels = ['0-5K', '5K-10K', '10K-15K', '15K-20K', '20K-25K', '25K+']
train['loan_amnt_category'] = pd.cut(train['loan_amnt'], bins=loan_amnt_bins, labels=loan_amnt_labels, right=False)
sns.countplot(x='loan_amnt_category', data=train, order=loan_amnt_labels, ax=axes[1, 0])
axes[0, 3].set_title('Distribution of Loan Amount ')
axes[0, 3].set_xlabel('Loan Amount Category')
axes[0, 3].set_ylabel('Count')

# cb_person_cred_hist_length (신용 유지 기간[년])
sns.histplot(train['cb_person_cred_hist_length'], bins=20, ax=axes[0, 3])
axes[1, 0].set_title('Distribution of cb_person_cred_hist_length')
axes[1, 0].set_xlabel('cb_person_cred_hist_length (Years)')
axes[1, 0].set_ylabel('Count')


# Loan interest rate distribution (대출 이자율)
sns.histplot(train['loan_int_rate'], bins=20, ax=axes[1, 1])
axes[1, 1].set_title('Distribution of Loan Interest Rate')
axes[1, 1].set_xlabel('Loan Interest Rate')
axes[1, 1].set_ylabel('Count')

# Loan percent of income distribution (소득 대비 대출 비율)
sns.histplot(train['loan_percent_income'], bins=20, ax=axes[1, 2])
axes[1, 2].set_title('Distribution of Loan Amount Percentage of Income')
axes[1, 2].set_xlabel('Loan Amount Percentage of Income')
axes[1, 2].set_ylabel('Count')

# Hide the last subplot (if unused)
axes[1, 3].axis('off')

plt.tight_layout()
plt.show()



# 범주형 변수 분포

import matplotlib.pyplot as plt
categorical_cols = train.select_dtypes(include='object').columns

num_plots = len(categorical_cols)
rows = (num_plots + 3) // 4
cols = min(num_plots, 4)


plt.figure(figsize=(16, rows * 4))

for i, col in enumerate(categorical_cols):
  plt.subplot(rows, cols, i + 1)
  sns.countplot(x=train[col])
  plt.title(f'Distribution of {col}')
  plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()



# 범주형 변수파악

# person_home_ownership(대출 소유 상태) : Rent와 주택 담보의 비율이 가장 많은 편이며, 자가는 소수임

# loan_intent(대출 목적) : 교육 목적이 가장 많음

# loan_grade(대출 등급) : A 와 B의 비율이 가장 많음

# cb_person_default_on_file(대출 상환 연체 여부) : 연체 기록이 있는지에 대한 유무 / Y : 연체 기록 있음, N : 연체 기록 없음, 연체 없는 비율이 높음


test_encoded.info()


test_encoded=test_encoded.reindex(columns=X_encoded.columns,fill_value=0)


# 데이터 나누기
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X_encoded,y,test_size=0.2,stratify=y)


# SMOTE 적용
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from sklearn.metrics import roc_curve, auc, roc_auc_score

# 데이터 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_smote)
X_test_scaled = scaler.transform(X_test)

# 모델 정의 수정
models = {
    'Logistic Regression': LogisticRegression(max_iter=5000, solver='liblinear'),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(n_jobs=-1,max_depth=10,n_estimators=100),
    'Support Vector Classifier': SVC(probability=True),
    'K-Nearest Neighbors': KNeighborsClassifier(n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Xgboost Classifier': XGBClassifier(n_jobs=-1, eval_metric='auc',max_depth=10,n_estimators=100)
}

def train_and_evaluate(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_pred_proba = model.predict_proba(X_test)[:, 1]
    
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    accuracy = accuracy_score(y_test, y_test_pred)
    auc_score = roc_auc_score(y_test, y_test_pred_proba)
    
    return {
        'Model': name,
        'Train Score': train_score,
        'Test Score': test_score,
        'Accuracy Score': accuracy,
        'AUC Score': auc_score
    }

# 병렬 처리로 모델 학습 및 평가
results = Parallel(n_jobs=-1)(
    delayed(train_and_evaluate)(
        name, model, X_train_scaled, y_train_smote, X_test_scaled, y_test
    ) for name, model in models.items()
)

results_df = pd.DataFrame(results)
print(results_df)


# ROC 곡선 그리기
plt.figure(figsize=(10, 8))
for name, model in models.items():
    model.fit(X_train_scaled, y_train_smote)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Different Models')
plt.legend(loc="lower right")
plt.show()


# XGBoost 모델 병렬 처리로 학습 및 평가
def train_and_evaluate_xgb(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_pred_proba)
    return model, auc_score

xgb = XGBClassifier(n_jobs=-1, eval_metric='auc', max_depth=10, n_estimators=100)
results = Parallel(n_jobs=-1)(
    delayed(train_and_evaluate_xgb)(
        xgb, X_train_smote, y_train_smote, X_test_scaled, y_test
    ) for _ in range(1)
)

best_model, auc_score = results[0]
print(f"XGBoost 모델의 AUC-ROC 점수: {auc_score:.4f}")

# 제출 데이터 예측
submission_pred = best_model.predict(test_encoded)
submission_df = pd.DataFrame({'id': test['id'], 'loan_status': submission_pred})
submission_df.to_csv('submission.csv', index=False)
sub = pd.read_csv('submission.csv')
sub.head()

