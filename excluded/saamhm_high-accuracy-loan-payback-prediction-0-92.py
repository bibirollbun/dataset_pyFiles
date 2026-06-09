# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier

from category_encoders.target_encoder import TargetEncoder

from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

# -------------------------
# Load Data
# -------------------------
train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")
sub = pd.read_csv("/kaggle/input/playground-series-s5e11/sample_submission.csv")


#  Target  
target = "loan_paid_back"
y=train[target]

train.drop(['id',target],axis=1,inplace=True)
test.drop('id',axis=1,inplace=True)


def add_features(df):
    df = df.copy()
    
   
    df["dti"] = df["loan_amount"] / df["annual_income"]
    df["annam_int"] = df["loan_amount"] * df["interest_rate"]
    df["burden"] = df["interest_rate"] / df["annual_income"]
    df["interest_b"] = (df["loan_amount"] * df["interest_rate"]) / df["annual_income"]
    
   
    df["income_to_loan_ratio"] = df["annual_income"] / df["loan_amount"]
    
    # Interaction features
    df["credit_income_interaction"] = df["credit_score"] * df["annual_income"]
    
    # Binning numerical features
    df['credit_score_binned'] = pd.cut(df['credit_score'], bins=5, labels=False)
    
    df['annual_income_binned'] = pd.cut(df['annual_income'], bins=5, labels=False)
    
    return df

train = add_features(train)
test = add_features(test)


from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel

# Update column lists after new features
cat_col = train.select_dtypes(include="object").columns.tolist()
num_col = train.select_dtypes(include="number").columns.tolist()


# Enhanced preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ("target_encoder", TargetEncoder(cols=cat_col, smoothing=15.0), cat_col),
#        ("num_imputer", SimpleImputer(strategy='median'), num_col),
        ("scaler", StandardScaler(), num_col),
    ],
    remainder="passthrough"  # Keep the binned features
)


trans_train=preprocessor.fit_transform(train,y)
feature_name=preprocessor.get_feature_names_out()
transform_train=pd.DataFrame(trans_train,columns=feature_name)
transform_train


trans_test=preprocessor.transform(test)
transform_test=pd.DataFrame(trans_test,columns=feature_name)
transform_test


from sklearn.model_selection import StratifiedKFold

def create_metafeature(model,train,test,target):
    oof_train=np.zeros(len(train))
    oof_test=np.zeros(len(test))
    kf=StratifiedKFold(n_splits=5,shuffle=True,random_state=00)
    x=train
    y=target

    for train_ind,test_ind in kf.split(x,y):
        x_train,x_test=x.iloc[train_ind],x.iloc[test_ind]
        y_train,y_test=y.iloc[train_ind],y.iloc[test_ind]
        model.fit(x_train,y_train)
        oof_train[test_ind]=model.predict_proba(x_test)[:,1]
        oof_test+=model.predict_proba(test)[:,1]/5
        
    return oof_train,oof_test


meta=transform_train.copy()
meta_t=transform_test.copy()


lgbm=LGBMClassifier(
        metric="auc",n_estimators=2000,learning_rate=0.03,max_depth=6,
        num_leaves=50,colsample_bytree=0.8,subsample=0.8,min_child_samples=20,
        reg_alpha=0.05,reg_lambda=0.1,random_state=42,n_jobs=-1,verbose=-1
)



catb = CatBoostClassifier(
    iterations=3000,learning_rate=0.03,depth=8,loss_function="Logloss", eval_metric="AUC",
    random_seed=42,auto_class_weights="Balanced", verbose=0,task_type="GPU", devices="0"             
)


xgb = XGBClassifier(
    objective="binary:logistic",eval_metric="auc",learning_rate=0.01,max_depth=6,
    min_child_weight=3,colsample_bytree=0.3,subsample=0.6,reg_alpha=0.5,
    reg_lambda=2.0,n_estimators=2000,random_state=42,n_jobs=-1,tree_method="hist",
   device='cuda'
)


from sklearn.neural_network import MLPClassifier

mlp_clf = MLPClassifier(
    hidden_layer_sizes=(100,),  # one hidden layer with 100 neurons
    activation="relu",          # activation: ‘identity’, ‘logistic’, ‘tanh’, ‘relu’
    solver="adam",              # optimizer: ‘lbfgs’, ‘sgd’, ‘adam’
    alpha=0.0001,               # L2 regularization
    learning_rate="adaptive",   # ‘constant’, ‘invscaling’, ‘adaptive’
    max_iter=300,               # number of epochs
    random_state=42
)




meta['lgb'],meta_t['lgb']=create_metafeature(lgbm,transform_train,transform_test,y)

meta['cat'],meta_t['cat']=create_metafeature(catb,transform_train,transform_test,y)

meta['xgb'],meta_t['xgb']=create_metafeature(xgb,transform_train,transform_test,y)

meta['mlp'],meta_t['mlp']=create_metafeature(mlp_clf,transform_train,transform_test,y)


print(meta.shape)
print(meta_t.shape)


pip install protobuf==3.20.3 --force-reinstall



import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
import warnings
warnings.filterwarnings('ignore')



model.get_weights()


model=Sequential([
    Dense(256,activation='relu',input_shape=(23,)),
    
    Dense(128,activation='relu'),
    BatchNormalization(),
    Dropout(.3),

    Dense(64,activation='relu'),
    BatchNormalization(),
    Dropout(.2),

    Dense(32,activation='relu'),
    Dropout(.2),

    Dense(1,activation='sigmoid')
])


from tensorflow.keras.optimizers import SGD

# Apply NAG with momentum
optimizer = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
#model.compile(optimizer=optimizer, loss='categorical_crossentropy')



model.compile(optimizer=optimizer,
             loss='binary_crossentropy',metrics=['accuracy','AUC'])

callbacks=[
    EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True,verbose=1),
    ReduceLROnPlateau(monitor='val_loss',factor=.5,patience=5,verbose=1)
]

history=model.fit(meta,y,batch_size=128,epochs=550,
                  callbacks=callbacks,verbose=1,shuffle=True)


import matplotlib.pyplot as plt


plt.plot(history.history['loss'],label='training loss')


plt.plot(history.history['accuracy'],label='training loss')
plt.plot(history.history['AUC'],label='training loss')




# 1. Predict probabilities on the test meta-features
test_preds = model.predict(meta_t)   

# 2. Flatten predictions (Keras outputs are 2D arrays)
test_preds = test_preds.ravel()

# 3. Build submission DataFrame
submission = sub.copy()
submission[target] = test_preds
submission.to_csv("submission.csv", index=False)

print("Submission file created successfully!")






