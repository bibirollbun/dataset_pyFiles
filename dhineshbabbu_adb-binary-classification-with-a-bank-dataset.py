import pandas as pd
from sklearn.preprocessing import LabelEncoder , StandardScaler
from sklearn.model_selection import train_test_split 
from sklearn.feature_selection import chi2
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score , classification_report , roc_auc_score 


train_df = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')


train_df


train_df.isnull().sum()


train_df.info()


categorical_features = train_df.select_dtypes(exclude = 'int')
numerical_features = train_df.select_dtypes(include = 'int')


print(numerical_features.shape[-1] , categorical_features.shape[-1])


labelencoder = LabelEncoder()
label_mappings = {}


def object_to_int(df, columns):
    for col in columns:
        if df[col].dtype == 'object':
            df[col] = labelencoder.fit_transform(df[col])
            # Store the label mapping for each column
            label_mappings[col] = dict(zip(labelencoder.classes_, labelencoder.transform(labelencoder.classes_)))
    return df


train_data = object_to_int(train_df,categorical_features)
train_data.drop(['id'],axis=1,inplace=True)


data_corr = train_data.corr()
data_corr
plt.figure(figsize=(20,14))
sns.heatmap(data_corr,annot=True)
plt.show()


# # 'job': 85,
#  'poutcome': 77,
#  'education': 64,
#  'loan': 50,
#  'marital': 47,
#  'previous': 9,
#  'default': 6}
train_data.drop(['default'],axis=1,inplace=True)


X = train_data.drop(['y'],axis=1)
y = train_data['y']


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


clf = LGBMClassifier(max_depth=20, n_estimators=500, num_leaves=70, random_state=42,
               subsample=0.3)
clf.fit(X_train,y_train)


print('Training Accuracy :',accuracy_score(y_train,clf.predict(X_train)))


print("Testing Accuracy :",accuracy_score(y_test,clf.predict(X_test)))


y_pred = clf.predict(X_test)


print(classification_report(y_test ,y_pred))


print('ROC AUC Score:',roc_auc_score(y_test,y_pred))


y_pred_prob = clf.predict_proba(X_test)[:,1]


fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.plot(fpr, tpr, label='ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()



# # Define the parameter grid
# param_grid = {
#     'learning_rate': [0.01, 0.1,0.2],
#     'n_estimators': [ 200,300],
#     'max_depth': [ 10,20],
#     'num_leaves': [50,70],
#     'subsample': [0.3,0.6],
# }


# # Initialize the model
# lgb_model = LGBMClassifier(random_state=42)
# grid_search = GridSearchCV(estimator=lgb_model, param_grid=param_grid, cv=3, scoring='roc_auc', n_jobs=-1,verbose=2)
# grid_search.fit(X_train, y_train)  


# print("Best parameters:", grid_search.best_params_)
# print("Best ROC-AUC score:", grid_search.best_score_)
# best_model = grid_search.best_estimator_
# print(best_model)


test_df = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
test_df


# map the converetd categorical features into the test data
for cat_col in categorical_features:
    test_df[cat_col] = test_df[cat_col].map(label_mappings[cat_col])


result = pd.DataFrame(columns = ['id'])
result['id'] = test_df['id']


test_df.drop(['id','default'],axis=1,inplace=True)
test_df = scaler.transform(test_df)


y_test_pred = clf.predict_proba(test_df)[:,1]


result['y'] = y_test_pred


result.to_csv('submission.csv',index=False)

