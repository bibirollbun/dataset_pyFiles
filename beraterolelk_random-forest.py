import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer



train_df  = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test_df  = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
orginal = pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")


X = train_df.drop(columns=['rainfall', 'id'])
y = train_df['rainfall']


test_id = test_df['id']

X_test = test_df.drop(columns=['id'])



imputer = SimpleImputer(strategy='mean')

X_train_imputed = imputer.fit_transform(X)
X_test_imputed = imputer.transform(X_test)


model = RandomForestClassifier(n_estimators=100, random_state=42)  
model.fit(X_train_imputed, y) 
y_pred_prob = model.predict_proba(X_test_imputed)[:, 1]



average_prediction = y_pred_prob.mean()
print(f" Average Prediction: {average_prediction}")





