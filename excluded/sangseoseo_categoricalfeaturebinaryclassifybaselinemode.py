import pandas as pd
#data_path
data_path = '/kaggle/input/cat-in-the-dat/'
train   = pd.read_csv(data_path + 'train.csv', index_col='id')
test    = pd.read_csv(data_path +'test.csv', index_col='id')
submission = pd.read_csv(data_path + 'sample_submission.csv', index_col='id')


all_data = pd.concat([train, test])
all_data = all_data.drop('target', axis =1)
all_data


from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder()
all_data_encoded = encoder.fit_transform(all_data)


num_train = len(train)

X_train = all_data_encoded[:num_train]
X_test  = all_data_encoded[num_train:]

y = train['target']


from sklearn.model_selection import train_test_split

# test_size = 0.1 means test dataset accounts for 10 percent of the entire training dataset
# stratify = y means : Distribution of target values evenly scattered across the training ad test dataset
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y, test_size=0.1, stratify=y, random_state=0)


from sklearn.linear_model import LogisticRegression

#max_iter - The number of iterations for updating the model's regression coefficients.
logistic_model = LogisticRegression(max_iter = 1000, random_state=42)
logistic_model.fit(X_train, y_train)


logistic_model.predict_proba(X_valid)  # Probabilities where the target value 1 or 0.
#logistic_model.predict(X_valid)  # target value itself 0 0r 1,


logistic_model.predict(X_valid)


# containing the probabilities that the target value 1.
y_valid_preds = logistic_model.predict_proba(X_valid)[:, 1]


from sklearn.metrics import roc_auc_score

roc_auc = roc_auc_score(y_valid, y_valid_preds)

print(f"ROC AUC Score : {roc_auc}")


y_preds = logistic_model.predict_proba(X_test)[:, 1]

submission['target'] = y_preds
submission.to_csv('submission.csv')


submission.head()

