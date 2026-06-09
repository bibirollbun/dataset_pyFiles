# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
import warnings
warnings.filterwarnings("ignore")


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
print(df.shape)
display(df.sample(5))



df.isnull().sum()


df['loan_paid_back'].mean()


df['credit_score'].describe()


df['credit_score_bin'] = pd.qcut(df['credit_score'],q=10)
df.groupby(['credit_score_bin']).agg({'loan_paid_back':[len,'mean']})





df['loan_purpose'].value_counts(1)


df.groupby(['loan_purpose']).agg({'loan_paid_back':[len,'mean']})


df.groupby(['credit_score_bin','loan_purpose']).agg({'loan_paid_back':['mean']}).unstack()





df.sample(2)


df['loan_amount_bin'] = pd.qcut(df['loan_amount'],q=10)
df['debt_to_income_ratio_bin'] = pd.qcut(df['debt_to_income_ratio'],q=10)


df.groupby(['loan_amount_bin']).agg({'loan_paid_back':['mean']})


df.groupby(['debt_to_income_ratio_bin']).agg({'loan_paid_back':['mean']})


df.groupby(['debt_to_income_ratio_bin','loan_amount_bin']).agg({'loan_paid_back':'mean'}).unstack()


df.columns





df_test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')
print(df_test.shape)


df.sample(3)


df_test.sample(3)


def financial_burden_stability(dff):
    dff['debt_to_loan_ratio'] = (dff['debt_to_income_ratio'] * dff['annual_income']) / dff['loan_amount']
    dff['interest_to_income_ratio'] = (dff['interest_rate'] * dff['loan_amount']) / dff['annual_income']
    dff['loan_per_credit_score'] = dff['loan_amount'] / dff['credit_score']
    return dff


df = financial_burden_stability(df)
df_test = financial_burden_stability(df_test)


df.sample(2)


df_test.sample(2)





def credit_quality(dff):
    bins = [0, 580, 670, 740, 800, np.inf]
    labels = ['Poor', 'Fair', 'Good', 'Very Good', 'Excellent']
    dff['credit_score_binned'] = pd.cut(dff['credit_score'], bins=bins, labels=labels, right=False)
    return dff


df = credit_quality(df)
df_test = credit_quality(df_test)


df.sample(2)


df_test.sample(2)





df['grade_subgrade'].unique()


def grade_encode(dff):
    def grade_to_numeric(grade_subgrade):
        # Base value for A=0, B=5, C=10, D=15, E=20, F=25, G=30
        grade_map = {'A': 0, 'B': 5, 'C': 10, 'D': 15, 'E': 20, 'F': 25, 'G': 30} 
        try:
            grade = grade_subgrade[0].upper()
            subgrade = int(grade_subgrade[1])
            # A1 = 0 + 1 = 1
            # C3 = 10 + 3 = 13
            # F5 = 25 + 5 = 30
            return grade_map.get(grade, np.nan) + subgrade 
        except:
            return np.nan
        
    dff['grade_subgrade_numeric'] = dff['grade_subgrade'].apply(grade_to_numeric)
    
    return dff


df = grade_encode(df)
df_test = grade_encode(df_test)


df['grade_subgrade_numeric'].value_counts(dropna=False)


df.sample(2)


df_test.sample(2)





df.sample(2)


df.columns


FLAG = 'loan_paid_back'

FEATURES = [
    'debt_to_income_ratio',
    'loan_amount', 
    'interest_rate', 
    'gender', 
    'marital_status',
    'education_level', 
    'employment_status', 
    'loan_purpose',
    'grade_subgrade_numeric',
    'debt_to_loan_ratio',
    'interest_to_income_ratio', 
    'loan_per_credit_score',
    'credit_score_binned'
    
]


df = (df.reset_index(drop=True)
    .sample(frac=1, random_state=512)
    .reset_index(drop=True)
)

train = df.iloc[: int(0.75 * df.shape[0])]
val = df.iloc[int(0.75 * df.shape[0]) : int(0.85 * df.shape[0])]
test = df.iloc[int(0.85 * df.shape[0]) : ]


print(train.shape)
print(val.shape)
print(test.shape)


print(train[FLAG].mean())
print(val[FLAG].mean())
print(test[FLAG].mean())





FEATURES


df.sample(2)


from catboost import CatBoostClassifier

cat_features = ["gender","marital_status","education_level",
               "employment_status","loan_purpose","credit_score_binned"]

model = CatBoostClassifier(early_stopping_rounds=50, random_state=0, iterations=7000)

model = model.fit(
    train[FEATURES],
    train[FLAG],
    cat_features=cat_features,
    eval_set=(val[FEATURES], val[FLAG]),
)





from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score

def get_gini(a, b):
    try:
        return round(2 * roc_auc_score(a, b) - 1, 5)
    except:
        return None


train["score"] = model.predict_proba(train[FEATURES])[:, 1]
val["score"] = model.predict_proba(val[FEATURES])[:, 1]
test["score"] = model.predict_proba(test[FEATURES])[:, 1]


print('gini: ',get_gini(test['loan_paid_back'], test.score))

print('------------------------')

print('log loss : ',log_loss(test['loan_paid_back'], test.score))


test['score_bin'] = pd.qcut(test['score'],q=10)
display(test.groupby('score_bin').agg({'score':[len,'mean'],'loan_paid_back':['mean']}))


test.sample(2)


for crd in test['credit_score_binned'].unique().tolist():
    print(crd)
    pf = test[test['credit_score_binned']==crd]
    pf['score_bin'] = pd.qcut(pf['score'],q=10)
    display(pf.groupby('score_bin').agg({'score':[len,'mean'],'loan_paid_back':['mean']}))





sample = pd.read_csv('/kaggle/input/playground-series-s5e11/sample_submission.csv')
sample.sample(3)


# Find the Optimal Threshold (Maximizing F1-score) ---
y_true = np.round(test['loan_paid_back']).astype(int)
y_proba = model.predict_proba(test[FEATURES])[:, 1]


thresholds = np.linspace(0.01, 0.99, 100) # Check thresholds from 0.01 to 0.99
f1_scores = []

for threshold in thresholds:
    # Convert probabilities to binary predictions
    y_pred = (y_proba >= threshold).astype(int)
    
    # Calculate F1-score
    f1 = f1_score(y_true, y_pred)
    f1_scores.append(f1)

# Find the threshold that yields the maximum F1-score
optimal_threshold_index = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_threshold_index]
max_f1_score = f1_scores[optimal_threshold_index]

# --- 5. Calculate Metrics at Optimal Threshold ---
y_pred_optimal = (y_proba >= optimal_threshold).astype(int)

# Calculate confusion matrix components and overall accuracy
tn, fp, fn, tp = confusion_matrix(y_true, y_pred_optimal).ravel()

# Other metrics
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

print(f"\n--- Optimal Threshold Results (Maximizing F1-Score) ---")
print(f"The best threshold to apply is: {optimal_threshold:.4f}")
print(f"Maximum F1-Score: {max_f1_score:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall (Sensitivity): {recall:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"Confusion Matrix (TN, FP, FN, TP): {tn}, {fp}, {fn}, {tp}")

# --- 6. Visualization (Optional) ---
plt.figure(figsize=(10, 6))
plt.plot(thresholds, f1_scores, label='F1-Score')
plt.axvline(optimal_threshold, color='red', linestyle='--', 
            label=f'Optimal Threshold: {optimal_threshold:.4f}')
plt.xlabel('Probability Threshold')
plt.ylabel('F1-Score')
plt.title('F1-Score vs. Probability Threshold')
plt.legend()
plt.grid(True)
plt.show()


df_test["score"] = model.predict_proba(df_test[FEATURES])[:, 1]
df_test["loan_paid_back"] = (df_test["score"] > 0.4852).astype(int)
df_test.shape


df_test.sample(3)


df_test2 = df_test[['id','loan_paid_back']]
df_test2.sample(3)




