import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier

from category_encoders.target_encoder import TargetEncoder

from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

!pip uninstall protobuf -y
!pip install --upgrade --force-reinstall protobuf==3.20.3

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization,LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
import warnings
warnings.filterwarnings('ignore')


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
    
    # Avoid division-by-zero
    df["annual_income"].replace(0, np.nan, inplace=True)
    df["loan_amount"].replace(0, np.nan, inplace=True)

    # Basic ratios
    df["dti"] = df["loan_amount"] / df["annual_income"]
    df["annam_int"] = df["loan_amount"] * df["interest_rate"]
    df["burden"] = df["interest_rate"] / df["annual_income"]
    df["interest_b"] = (df["loan_amount"] * df["interest_rate"]) / df["annual_income"]

    df["income_to_loan_ratio"] = df["annual_income"] / df["loan_amount"]

    # Clip extremely large values to avoid exploding gradients
    ratio_cols = ["dti", "burden", "interest_b", "income_to_loan_ratio"]
    for col in ratio_cols:
        df[col] = df[col].clip(0, df[col].quantile(0.99))

    # Interaction features
    df["credit_income_interaction"] = df["credit_score"] * np.log1p(df["annual_income"])

    # Non-linear transform (helps neural networks)
    df["log_annual_income"] = np.log1p(df["annual_income"])
    df["log_loan_amount"] = np.log1p(df["loan_amount"])

    # Binning – using quantile based instead of fixed bins
    df['credit_score_binned'] = pd.qcut(df['credit_score'], q=5, labels=False, duplicates="drop")
    df['annual_income_binned'] = pd.qcut(df['annual_income'], q=5, labels=False, duplicates="drop")

    # Fill small NaNs created after qcut
    df = df.fillna(0)

    return df


train = add_features(train)
test  = add_features(test)



from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel

# Update column lists after new features
cat_col = train.select_dtypes(include="object").columns.tolist()
num_col = train.select_dtypes(include="number").columns.tolist()


# Enhanced preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ("target_encoder", TargetEncoder(cols=cat_col, smoothing=15.0), cat_col),
        ("scaler", StandardScaler(), num_col),
    ],
    remainder="drop"  
)


trans_train=preprocessor.fit_transform(train,y)
feature_name=preprocessor.get_feature_names_out()
transform_train=pd.DataFrame(trans_train,columns=feature_name)


trans_test=preprocessor.transform(test)
transform_test=pd.DataFrame(trans_test,columns=feature_name)


from sklearn.model_selection import StratifiedKFold

def create_metafeature(model,train,test,target):
    oof_train=np.zeros(len(train))
    oof_test=np.zeros(len(test))
    kf=StratifiedKFold(n_splits=3,shuffle=True,random_state=00)
    x=train
    y=target

    for train_ind,test_ind in kf.split(x,y):
        x_train,x_test=x.iloc[train_ind],x.iloc[test_ind]
        y_train,y_test=y.iloc[train_ind],y.iloc[test_ind]
        model.fit(x_train,y_train)
        oof_train[test_ind]=model.predict_proba(x_test)[:,1]
        oof_test+=model.predict_proba(test)[:,1]/3
        
    return oof_train,oof_test


meta=transform_train.copy()
meta_t=transform_test.copy()


lgbm = LGBMClassifier(
    objective="binary",
    metric="auc",
    n_estimators=3500,
    learning_rate=0.02,

    max_depth=6,          # let num_leaves control complexity
    num_leaves=64,

    min_child_samples=30,
    min_data_in_leaf=40,   # more stable

    colsample_bytree=0.85,
    subsample=0.8,
    subsample_freq=1,

    reg_alpha=0.05,
    reg_lambda=0.2,        # stronger regularization = better generalization

    n_jobs=-1,
    random_state=42,
)

meta['lgb'],meta_t['lgb']=create_metafeature(lgbm,transform_train,transform_test,y)


catb = CatBoostClassifier(
    iterations=3500,
    learning_rate=0.03,
    depth=7,
    l2_leaf_reg=6,

    loss_function="Logloss",     # VALID on GPU
    eval_metric="Logloss",       # use Logloss during training (GPU supported)

    random_seed=42,
    auto_class_weights="Balanced",

    task_type="GPU",
    devices="0",
    verbose=0
)

meta['cat'],meta_t['cat']=create_metafeature(catb,transform_train,transform_test,y)


xgb = XGBClassifier(
    objective="binary:logistic",
    eval_metric="auc",

    # GPU algorithm
    tree_method="gpu_hist",
    device="cuda",

    # Core hyperparameters
    n_estimators=3000,
    learning_rate=0.02,
    max_depth=6,
    min_child_weight=3,

    # Regularization
    reg_alpha=0.5,
    reg_lambda=2.0,

    # Sampling
    colsample_bytree=0.6,
    subsample=0.7,

    # GPU optimization
    max_bin=256,

    random_state=42,
    n_jobs=-1
)
meta['xgb'],meta_t['xgb']=create_metafeature(xgb,transform_train,transform_test,y)


meta.shape[1]


model = Sequential([
    Dense(256, kernel_regularizer='l2', activation='relu',input_shape=(meta.shape[1],)),
    BatchNormalization(),
    Dropout(0.25),

    Dense(128, kernel_regularizer='l2'),
    BatchNormalization(),
    LeakyReLU(),
    

    Dense(64, kernel_regularizer='l2'),
    BatchNormalization(),
    LeakyReLU(),
    Dropout(0.20),

    Dense(32, kernel_regularizer='l2'),
    BatchNormalization(),
    LeakyReLU(),
    Dropout(0.15),

    Dense(1, activation='sigmoid')
])



model.get_weights()



from tensorflow.keras.optimizers import SGD

# SGD with momentum
#optimizer = SGD(learning_rate=0.01, momentum=0.9,nesterov=True)
optimizer = Adam(learning_rate=1e-3)

model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy', 'AUC']
)

callbacks = [
    EarlyStopping(monitor='val_auc', patience=15, mode='max',
                  restore_best_weights=True, verbose=1),

    ReduceLROnPlateau(monitor='val_auc', mode='max',
                      factor=0.5, patience=5, verbose=1)
]

history = model.fit(
    meta, y,
    validation_split=0.2,
    epochs=300,
    batch_size=128,
    shuffle=True,
    callbacks=callbacks,
    verbose=1
)


test_preds = model.predict(meta_t)   

# 2. Flatten predictions (Keras outputs are 2D arrays)
test_preds = test_preds.ravel()

# 3. Build submission DataFrame
submission = sub.copy()
submission[target] = test_preds
submission.to_csv("submission.csv", index=False)

print("Submission file created successfully!")

