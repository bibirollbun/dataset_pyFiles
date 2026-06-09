!git clone https://github.com/priorlabs/tabpfn-extensions.git


!pip install -q -e tabpfn-extensions


!pip install -q mrmr-selection tabpfn


import pandas as pd
from mrmr import mrmr_classif
from tabpfn import TabPFNClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split


kaggle = True
main_dir = '/kaggle/input/widsdatathon2025/' if kaggle else 'widsdatathon2025/'

train_connectome = pd.read_csv(f"{main_dir}/TRAIN_NEW/TRAIN_FUNCTIONAL_CONNECTOME_MATRICES_new_36P_Pearson.csv")
test_connectome = pd.read_csv(f"{main_dir}/TEST/TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv")

train_solutions = pd.read_excel(f"{main_dir}/TRAIN_NEW/TRAINING_SOLUTIONS.xlsx")
sample_submission = pd.read_excel(f'{main_dir}/SAMPLE_SUBMISSION.xlsx')


X = train_connectome.drop('participant_id', axis=1)
solutions = train_solutions.set_index('participant_id')
solutions = solutions.reindex(train_connectome['participant_id']).reset_index()


selected_features_adhd = mrmr_classif(X=X, y=solutions['ADHD_Outcome'].values, K=250)

X_adhd = X[selected_features_adhd]


selected_features_sex = mrmr_classif(X=X, y=solutions['Sex_F'].values, K=250)

X_sex = X[selected_features_sex]


def train_evaluate_tabpfn(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y
    )
    
    clf = TabPFNClassifier(device='cuda')
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='micro')
    auc_roc = roc_auc_score(y_test, y_pred_proba[:, 1])
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {auc_roc:.4f}")
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'auc_roc': auc_roc,
        'model': clf
    }


print("ADHD_Outcome prediction results:\n")
adhd_results = train_evaluate_tabpfn(
    X_adhd,
    solutions['ADHD_Outcome'].values,
)


print("Sex_F prediction results:\n")
sex_results = train_evaluate_tabpfn(
    X_sex,
    solutions['Sex_F'].values,
)


def train_tabpfn(X, y):
    clf = TabPFNClassifier(device='cuda')
    clf.fit(X, y)
    
    return clf


adhd_model = train_tabpfn(
    X_adhd,
    solutions['ADHD_Outcome'].values,
)

sex_model = train_tabpfn(
    X_sex,
    solutions['Sex_F'].values,
)


submission_set = test_connectome.drop('participant_id', axis=1)
submission_set_adhd = submission_set[selected_features_adhd]
submission_set_sex = submission_set[selected_features_sex]

adhd_pred = adhd_model.predict(submission_set_adhd)
sex_pred = sex_model.predict(submission_set_sex)


submission = pd.DataFrame({
    'participant_id': test_connectome['participant_id'],
    'ADHD_Outcome': adhd_pred,
    'Sex_F': sex_pred,
})

submission.to_csv('submission.csv', index=False)




