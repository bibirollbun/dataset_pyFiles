import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler


train_df = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
train_df.head()


test_df = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
test_df.head()


train_df['rainfall'].value_counts()


train_df.shape


test_df.shape


train_df.isna().sum()


train_df.describe()


#shuffling train df 
train_df = train_df.sample(frac=1)
train_df.head()


#scaling data and oversampling (to get equal distribution of targets)

def scale_df(df,types='',oversample=False):
    
    def scale(df):
        scaler = StandardScaler()
        X = scaler.fit_transform(df)
        return X
        
    if types == 'train':
        X = df[df.columns[2:-1]].values
        y = df[df.columns[-1]].values
        X = scale(X)
        if oversample:
            ros = RandomOverSampler()
            X, y = ros.fit_resample(X,y)

        
        return X,y
    else:
        X = df[df.columns[2:]].values
        X = scale(X)

        return X    


X_train, y_train = scale_df(train_df,'train',oversample=True)
X_test = scale_df(test_df)


print(X_train.shape)
print(y_train.shape)


np.unique(y_train,return_counts=True)


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.impute import SimpleImputer


np.unique(X_test,return_counts=True)


imputer = SimpleImputer(strategy='mean')
X_test = imputer.fit_transform(X_test)


np.unique(X_test,return_counts=True)


#from sklearn.tree import DecisionTreeClassifier

from sklearn.tree import ExtraTreeClassifier

dt2_model = ExtraTreeClassifier(random_state=2)



dt2_model.fit(X_train,y_train)

#knn_model = fit


y_pred = dt2_model.predict(X_test)
#y_pred[:5]


submission = test_df[['id']].copy()
submission['rainfall'] = y_pred
# Include 'id' and the predicted target column
submission.to_csv('submission.csv', index=False)

print("Submission file created: submission.csv")


submission




