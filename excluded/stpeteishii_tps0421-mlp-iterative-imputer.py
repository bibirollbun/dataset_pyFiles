import lightgbm as lgb
import numpy as np
import pandas as pd
import random
import optuna
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Activation
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer # after import enable_iterative_imputer


train0 = pd.read_csv("../input/tabular-playground-series-apr-2021/train.csv")
test0 = pd.read_csv("../input/tabular-playground-series-apr-2021/test.csv")


cols=train0.columns.tolist()
data0=pd.concat([train0,test0],axis=0)


from sklearn.preprocessing import LabelEncoder

def labelencoder(df):
    for c in df.columns:
        if df[c].dtype=='object':
            df[c] = df[c].fillna('N')
            lbl = LabelEncoder()
            lbl.fit(list(df[c].values))
            df[c] = lbl.transform(df[c].values)
    return df


data1=labelencoder(data0)
display(data1.info())


multi_imp = IterativeImputer(
    max_iter=9, random_state=42, verbose=0, 
    skip_complete=True, n_nearest_features=10, tol=0.001)

multi_imp.fit(data0)
data2 = multi_imp.transform(data1)
data3=pd.DataFrame(data=data2,columns=cols)
display(data3.info())


train=data3.iloc[0:len(train0)]
test=data3.iloc[len(train0):]


trainY = train['Survived']
trainX = train.drop(['Survived','PassengerId','Name','Ticket'],axis=1)
testX = test.drop(['Survived','PassengerId','Name','Ticket'],axis=1)


columns=trainX.columns.to_list()
print(columns)


n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

train_oof = np.zeros((trainX.shape[0]))
test_preds = np.zeros((testX.shape[0]))


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Define the MLP architecture using Keras
# Num layers is 5.
def build_keras_mlp(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(400),
        BatchNormalization(),
        Activation('relu'),
        Dense(100),
        BatchNormalization(),
        Activation('relu'),
        Dense(300),
        BatchNormalization(),
        Activation('relu'),
        Dense(100),
        BatchNormalization(),
        Activation('relu'),
        Dense(30),
        BatchNormalization(),
        Activation('relu'),
        Dense(1, activation='sigmoid') 
    ])
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# Cross-validation loop
for fold_num, (train_index, val_index) in enumerate(kf.split(trainX)):
    print(f"Fitting fold {fold_num+1}/{n_splits}")
    
    # Split data
    train_features = trainX.iloc[train_index]
    train_target = trainY.iloc[train_index]
    val_features = trainX.iloc[val_index]
    val_target = trainY.iloc[val_index]

    # Build model
    model = build_keras_mlp(input_dim=train_features.shape[1])
    
    # Early stopping
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=100,
        restore_best_weights=True,
        verbose=0
    )
    
    # Train the model
    model.fit(
        train_features, train_target,
        validation_data=(val_features, val_target),
        epochs=1000,
        batch_size=256,
        callbacks=[early_stop],
        verbose=0
    )
    
    # Predict on validation set
    val_pred = model.predict(val_features).ravel()
    train_oof[val_index] = val_pred
    
    # Evaluation metrics
    val_pred_labels = (val_pred > 0.5).astype(int)
    
    acc = accuracy_score(val_target, val_pred_labels)
    prec = precision_score(val_target, val_pred_labels)
    rec = recall_score(val_target, val_pred_labels)
    f1 = f1_score(val_target, val_pred_labels)
    auc = roc_auc_score(val_target, val_pred)
    
    print(f"Fold {fold_num+1} - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1 Score: {f1:.4f}, AUC: {auc:.4f}")
    
    # Predict on test set (average predictions across folds)
    test_pred_fold = model.predict(testX).ravel()
    test_preds += test_pred_fold / n_splits





submit=pd.read_csv('/kaggle/input/tabular-playground-series-apr-2021/sample_submission.csv')
test_preds2=[]
for p in test_preds:
    test_preds2+=[int(round(p,0))]
submit['Survived']=test_preds2
display(submit)
submit.to_csv('submission.csv',index=False)


submit['Survived'].value_counts()




