import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,roc_auc_score,confusion_matrix,ConfusionMatrixDisplay,classification_report


train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
sample = pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')


train.shape,test.shape


train.head(3)


test.head(3)


train.isna().sum()


test.isna().sum()


train.describe()


train.info()


numerical_cols = train.select_dtypes(include=['number']).columns.tolist()
print("Numerical Columns:", len(numerical_cols))


for col in numerical_cols:
    plt.figure(figsize=(10, 4))
    sns.histplot(train[col], bins=30, kde=True)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()


categorical_cols = train.select_dtypes(include=['object']).columns.tolist()
print("Categorical Columns:", len(categorical_cols))


for col in categorical_cols:
    plt.figure(figsize=(8, 4))
    sns.countplot(data=train, x=col)
    plt.title(f'Count of {col}')
    plt.xticks(rotation=45)
    plt.show()


lb = LabelEncoder()
for col in categorical_cols:
    train[col] = lb.fit_transform(train[col])
    test[col] = lb.transform(test[col])


X = train.drop(columns=['id','y'])
y = train['y']
Xt = test.drop(columns=['id'])


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)


model = RandomForestClassifier()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)


y_predt = model.predict_proba(Xt)


print(f'Accuracy: {accuracy_score(y_test,y_pred)*100:.2f}')
print(f'ROC Score: {roc_auc_score(y_test,y_pred):.2f}')


cm = confusion_matrix(y_test,y_pred)
disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.title('Confusion Matrix')
plt.show()


print('classification report:\n',classification_report(y_test,y_pred))


sample['y'] = y_predt[:, 1]
sample.to_csv('submission.csv',index=False)


sub = pd.read_csv('submission.csv')
sub.head(2)

