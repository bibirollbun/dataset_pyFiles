!pip install -q lazypredict


from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import StratifiedKFold
import pandas as pd


kaggle = True
main_dir = '/kaggle/input/widsdatathon2025/' if kaggle else '../widsdatathon2025/'

train_connectome = pd.read_csv(f"{main_dir}/TRAIN_NEW/TRAIN_FUNCTIONAL_CONNECTOME_MATRICES_new_36P_Pearson.csv")
test_connectome = pd.read_csv(f"{main_dir}/TEST/TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv")
train_quantitative = pd.read_excel(f"{main_dir}/TRAIN_NEW/TRAIN_QUANTITATIVE_METADATA_new.xlsx")
test_quantitative = pd.read_excel(f"{main_dir}/TEST/TEST_QUANTITATIVE_METADATA.xlsx")
train_solutions = pd.read_excel(f"{main_dir}/TRAIN_NEW/TRAINING_SOLUTIONS.xlsx")
sample_submission = pd.read_excel(f'{main_dir}/SAMPLE_SUBMISSION.xlsx')

X_solutions = pd.merge(train_solutions, train_connectome, on='participant_id')

X_adhd_outcome = X_solutions.drop(['participant_id', 'ADHD_Outcome', 'Sex_F'], axis=1)
y_adhd_outcome = X_solutions['ADHD_Outcome']

X_sex_f = X_solutions.drop(['participant_id', 'Sex_F', 'ADHD_Outcome'], axis=1)
y_sex_f = X_solutions['Sex_F']


n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=77)

results = []

for train_index, test_index in skf.split(X_adhd_outcome, y_adhd_outcome):
    X_train, X_test = X_adhd_outcome.iloc[train_index], X_adhd_outcome.iloc[test_index]
    y_train, y_test = y_adhd_outcome.iloc[train_index], y_adhd_outcome.iloc[test_index]

    clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
    models, predictions = clf.fit(X_train, X_test, y_train, y_test)
    results.append(models)

metrics_df = pd.concat(results)
metrics_summary = metrics_df.groupby(metrics_df.index).agg(['mean', 'std'])
metrics_summary


n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=77)

results = []

for train_index, test_index in skf.split(X_sex_f, y_sex_f):
    X_train, X_test = X_sex_f.iloc[train_index], X_sex_f.iloc[test_index]
    y_train, y_test = y_sex_f.iloc[train_index], y_sex_f.iloc[test_index]

    clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
    models, predictions = clf.fit(X_train, X_test, y_train, y_test)
    results.append(models)


metrics_df = pd.concat(results)
metrics_summary = metrics_df.groupby(metrics_df.index).agg(['mean', 'std'])
metrics_summary


from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, RidgeClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB



adhd_models = [
    ('GaussianNB', GaussianNB()),
    ('RandomForest', RandomForestClassifier()),
    ('Perceptron', Perceptron()),
    ('RidgeClassifier', RidgeClassifier()),
]

sex_models = [
    ('GaussianNB', GaussianNB()),
    ('BernoulliNB', BernoulliNB()),
    ('Perceptron', Perceptron()),
    ('RidgeClassifier', RidgeClassifier()),
]


adhd_stack = StackingClassifier(
    estimators=adhd_models,
    final_estimator=LogisticRegression(max_iter=1000),
    stack_method='auto'
)

sex_stack = StackingClassifier(
    estimators=sex_models,
    final_estimator=LogisticRegression(max_iter=1000),
    stack_method='auto'
)


X_sex_f = X_solutions.drop(['participant_id', 'Sex_F', 'ADHD_Outcome'], axis=1)
y_sex_f = X_solutions['Sex_F']

X_adhd = X_solutions.drop(['participant_id', 'ADHD_Outcome', 'Sex_F'], axis=1)
y_adhd = X_solutions['ADHD_Outcome']


adhd_stack.fit(X_adhd, y_adhd)


sex_stack.fit(X_sex_f, y_sex_f)


test_features_adhd = test_connectome.drop('participant_id', axis=1)
test_features_sex = test_connectome.drop('participant_id', axis=1)

adhd_pred = adhd_stack.predict(test_features_adhd)
sex_pred = sex_stack.predict(test_features_sex)


submission = pd.DataFrame({
    'participant_id': test_connectome['participant_id'],
    'ADHD_Outcome': adhd_pred,
    'Sex_F': sex_pred,
})

submission['ADHD_Outcome'] = submission['ADHD_Outcome'].astype(int)
submission['Sex_F'] = submission['Sex_F'].astype(int)

submission.to_csv('submission.csv', index=False)

