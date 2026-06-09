import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, GradientBoostingClassifier,  VotingClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC


train_data = pd.DataFrame(pd.read_csv('train.csv'))
test_data = pd.DataFrame(pd.read_csv('test.csv'))
display(train_data.head())
display(test_data.head())
display(train_data.info())


categorical_features = ['job','marital','education','default','housing','loan','contact','month','poutcome']
encoder = OneHotEncoder(sparse_output=False)
encoded_train = encoder.fit_transform(train_data[categorical_features])
encoded_test = encoder.transform(test_data[categorical_features])

df_encoded_train = pd.DataFrame(encoded_train, columns=encoder.get_feature_names_out(categorical_features))
df_encoded_test = pd.DataFrame(encoded_test, columns=encoder.get_feature_names_out(categorical_features))
df_encoded_train.index = train_data.index
df_encoded_test.index = test_data.index
df_train = pd.concat([train_data.drop(columns=categorical_features), df_encoded_train], axis=1)
df_test = pd.concat([test_data.drop(columns=categorical_features), df_encoded_test], axis=1)
display(df_train.head())
display(df_test.head())
display(df_train.info())


X = df_train.drop(columns=['y','id'])
y = df_train['y']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
test_scaled = scaler.transform(df_test.drop(columns=['id']))

X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)


base_models = [
    ('xgb', XGBClassifier(
        n_estimators=100,
        tree_method='hist', 
        eval_metric='auc', 
        random_state=42
        )),
    ('lgb', LGBMClassifier(
        n_estimators=100, 
        verbose=-1, 
        random_state=42
        )),
    ('cat', CatBoostClassifier(
        iterations=100,
        verbose=1, 
        random_state=42
        )),
    ('rf', RandomForestClassifier(
        n_estimators=100, 
        random_state=42,
        )),
    ('gb', GradientBoostingClassifier(
        n_estimators= 100, 
        random_state=42
        )),
    ('lr', LogisticRegression(
        max_iter=200, 
        random_state=42
        )),
    ('svc', SVC(
        probability=True, 
        random_state=42
        ))
    
]

# for name, model in tqdm(base_models, desc="Fitting base models"):
#     model.fit(X_train_res, y_train_res)

voting_model = VotingClassifier(
    estimators=base_models,
    voting = 'soft',
    n_jobs=-1,
    verbose= True
)
voting_model.fit(X_train_res, y_train_res)



y_val_pred = voting_model.predict(X_val)
print("ROC:", roc_auc_score(y_val, y_val_pred))


test_preds = voting_model.predict(test_scaled)


submission = pd.DataFrame({
    "id": df_test['id'],  # or the column name Kaggle requires (check sample_submission.csv)
    "y": test_preds
})


submission.to_csv('submission_voting.csv', index=False)
print("Submission file created")

