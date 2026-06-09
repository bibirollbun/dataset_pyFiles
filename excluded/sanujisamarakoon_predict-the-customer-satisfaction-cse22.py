Loading Data


import pandas as pd

train = pd.read_csv('/kaggle/input/Predict-the-Customer-Satisfaction-CSE-22/train_dataset.csv')
test = pd.read_csv('/kaggle/input/Predict-the-Customer-Satisfaction-CSE-22/test_dataset.csv')   
sample_submission = pd.read_csv('/kaggle/input/Predict-the-Customer-Satisfaction-CSE-22/sample_submission.csv') 


Understanding Data


train


train.dtypes


train.isnull().sum()


train['customer_experience'].value_counts()


Data Cleaning


train['Date_Registered'] = pd.to_datetime(train['Date_Registered'])
train['payment_datetime'] = pd.to_datetime(train['payment_datetime'])
train['purchased_datetime'] = pd.to_datetime(train['purchased_datetime'])
train['released_date'] = pd.to_datetime(train['released_date'])
train['estimated_delivery_date'] = pd.to_datetime(train['estimated_delivery_date'])
train['received_date'] = pd.to_datetime(train['received_date'])


Adding new features


train['purchase_hour'] = train['payment_datetime'].dt.hour
train['purchase_day'] = train['payment_datetime'].dt.day_name()
train['purchase_month'] = train['payment_datetime'].dt.month
train['days_since_registration'] = (train['purchased_datetime'] - train['Date_Registered']).dt.days
train['estimated_delivery_day'] = train['estimated_delivery_date'].dt.day_name()
train['received_day'] = train['received_date'].dt.day_name()


# Convert date columns to Unix timestamps
date_columns = ['Date_Registered', 'payment_datetime', 'purchased_datetime', 
                'released_date', 'estimated_delivery_date', 'received_date']

for col in date_columns:
    train[col] = pd.to_datetime(train[col], errors='coerce').astype(int) / 10**9  


Categorical Encoding


train['customer_experience'] = pd.Categorical(train['customer_experience'], categories=['bad', 'neutral', 'good'], ordered=True)
train['customer_experience'] = train['customer_experience'].cat.codes


One Hot Encoding


from sklearn.preprocessing import LabelEncoder

categorical_columns = [ 'Gender', 'Is_current_loyalty_program_member', 
                       'product_category',
                       'payment_method', 'purchase_medium', 'shipping_method', 'purchase_day','estimated_delivery_day','received_day']

label_encoder = LabelEncoder()

for col in categorical_columns:
    train[col] = label_encoder.fit_transform(train[col].astype(str))  # Convert categorical to numeric


Adding more new features


train['Delivery_time'] = train['received_date'] - train['released_date']
train['Delivery_delay'] = train['received_date'] - train['estimated_delivery_date']
train['Waiting_time'] = train['received_date'] - train['payment_datetime']
train['Additional_charge'] = train['final_payment'] - train['Product_value']
train['Waiting_percentage'] = (train['received_date'] - train['estimated_delivery_date'])/(train['received_date'] - train['payment_datetime'])
train['Processing_time'] = train['released_date'] - train['payment_datetime']
train['Loyalty_engagement'] = train['loyalty_points_redeemed'] / train['Product_value']


import numpy as np

train.replace(r'[^0-9]+', np.nan, regex=True, inplace=True)


train.fillna(0, inplace=True)


train = train.apply(pd.to_numeric)


X = train.drop('customer_experience', axis=1) 
y = train['customer_experience'] 


from sklearn.feature_selection import mutual_info_classif

mi_scores = mutual_info_classif(X, y,random_state=42)
mi_scores_df = pd.DataFrame({'Feature': X.columns, 'MI Score': mi_scores})

print(mi_scores_df.sort_values(by='MI Score', ascending=False))


Dropping least important columns


X = train.drop(['customer_experience','tracking_number', 'user_id', 'loyalty_tier','purchase_medium' ,'shipping_method','Gender','order_id', 'Received_tier_discount_percentage','Is_current_loyalty_program_member', 'transaction_id'],axis=1)
y = train['customer_experience']


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)


Training the model


from sklearn.ensemble import RandomForestClassifier

# model = RandomForestClassifier(class_weight='balanced', random_state=42)
# model = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42, class_weight='balanced')
model = RandomForestClassifier(
    n_estimators=300, 
    max_depth=None, 
    min_samples_split=5, 
    min_samples_leaf=2, 
    max_features='sqrt', 
    random_state=42, 
    class_weight='balanced'
)


# # Define a list of hyperparameter combinations
# param_list = [
#     {'n_estimators': 100, 'max_depth': 10},
#     {'n_estimators': 200, 'max_depth': None},
#     {'n_estimators': 300, 'max_depth': 20, 'min_samples_split': 5},
#     {'n_estimators': 100, 'max_depth': 15, 'min_samples_split': 10}
# ]

# best_model = None
# best_score = 0

# # Iterate through parameters and test each combination
# for params in param_list:
#     model = RandomForestClassifier(**params, class_weight='balanced', random_state=42)
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     score = accuracy_score(y_test, y_pred)
#     print(f"Params: {params}, Accuracy: {score}")
#     if score > best_score:
#         best_model = model
#         best_score = score

# print("Best Parameters:", best_model.get_params())


model.fit(X_train, y_train)


y_pred = model.predict(X_test)


from sklearn.metrics import classification_report, accuracy_score

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


from sklearn.metrics import f1_score

f1 = f1_score(y_test, y_pred, average='weighted')
print(f'Weighted F1 Score: {f1}')


# import optuna
# from lightgbm import LGBMClassifier
# from sklearn.model_selection import cross_val_score

# def objective(trial):
#     params = {
#         'n_estimators': trial.suggest_int('n_estimators', 100, 500),
#         'max_depth': trial.suggest_int('max_depth', -1, 15),
#         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
#         'num_leaves': trial.suggest_int('num_leaves', 31, 100),
#         'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
#         'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
#         'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 10.0),
#         'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 10.0),
#     }

#     model = LGBMClassifier(random_state=42, **params)
#     scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
#     return scores.mean()

# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=20)

# print("Best Parameters:", study.best_params)
# print("Best Score:", study.best_value)


from lightgbm import LGBMClassifier

model = LGBMClassifier(
    n_estimators=100,
    max_depth=9,
    learning_rate=0.1,
    num_leaves=57,
    feature_fraction=0.7233,
    bagging_fraction=0.7492,
    lambda_l1=1.9796,
    lambda_l2=8.1072,
    random_state=42
)


model.fit(X_train, y_train)


y_pred = model.predict(X_test)


from sklearn.metrics import classification_report, accuracy_score

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


from sklearn.metrics import f1_score

f1 = f1_score(y_test, y_pred, average='weighted')
print(f'Weighted F1 Score: {f1}')


test


test['Date_Registered'] = pd.to_datetime(test['Date_Registered'])
test['payment_datetime'] = pd.to_datetime(test['payment_datetime'])
test['purchased_datetime'] = pd.to_datetime(test['purchased_datetime'])
test['released_date'] = pd.to_datetime(test['released_date'])
test['estimated_delivery_date'] = pd.to_datetime(test['estimated_delivery_date'])
test['received_date'] = pd.to_datetime(test['received_date'])


test['purchase_hour'] = test['payment_datetime'].dt.hour
test['purchase_day'] = test['payment_datetime'].dt.day_name()
test['purchase_month'] = test['payment_datetime'].dt.month
test['days_since_registration'] = (test['purchased_datetime'] - test['Date_Registered']).dt.days
test['estimated_delivery_day'] = test['estimated_delivery_date'].dt.day_name()
test['received_day'] = test['received_date'].dt.day_name()


date_columns = ['Date_Registered', 'payment_datetime', 'purchased_datetime', 
                'released_date', 'estimated_delivery_date', 'received_date']

for col in date_columns:
    test[col] = pd.to_datetime(test[col], errors='coerce').astype(int) / 10**9


from sklearn.preprocessing import LabelEncoder

categorical_columns = [ 'Gender', 'Is_current_loyalty_program_member', 
                       'product_category', 
                       'payment_method', 'purchase_medium', 'shipping_method', 'purchase_day','estimated_delivery_day','received_day']

label_encoder = LabelEncoder()

for col in categorical_columns:
    test[col] = label_encoder.fit_transform(test[col].astype(str))


test['Delivery_time'] = test['received_date'] - test['released_date']
test['Delivery_delay'] = test['received_date'] - test['estimated_delivery_date']
test['Waiting_time'] = test['received_date'] - test['payment_datetime']
test['Additional_charge'] = test['final_payment'] - test['Product_value']
test['Waiting_percentage'] = (test['received_date'] - test['estimated_delivery_date'])/(test['received_date'] - test['payment_datetime'])
test['Processing_time'] = test['released_date'] - test['payment_datetime']
test['Loyalty_engagement'] = test['loyalty_points_redeemed'] / test['Product_value']


test.replace(r'[^0-9]+', np.nan, regex=True, inplace=True)


test.fillna(0,inplace=True)


test = test.apply(pd.to_numeric)


new_test=test.drop(['tracking_number', 'user_id', 'loyalty_tier','purchase_medium' ,'shipping_method','Gender','order_id', 'Received_tier_discount_percentage', 'Is_current_loyalty_program_member', 'transaction_id'], axis=1)  


final_model = LGBMClassifier(
    n_estimators=100,
    max_depth=9,
    learning_rate=0.1,
    num_leaves=57,
    feature_fraction=0.7233,
    bagging_fraction=0.7492,
    lambda_l1=1.9796,
    lambda_l2=8.1072,
    random_state=42
)


final_model.fit(X, y)


y_pred = final_model.predict(new_test)


label_mapping = {0: "bad", 1: "neutral", 2: "good"}
y_pred_labels = [label_mapping[label] for label in y_pred]

submission = pd.DataFrame({"id": new_test["id"], "customer_experience": y_pred_labels})
submission.to_csv("submission.csv", index=False)


submission = pd.read_csv("submission.csv")


submission 

