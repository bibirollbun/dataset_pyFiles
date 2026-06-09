import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# read csv_file

train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")

print("--------train---------")
print(train.info())
print(train.describe())
print("- - - -欠損値の個数- - - -")
print(train.isnull().sum())

print("---------test---------")
print(test.info())
print(test.describe())
print("- - - - 欠損値の個数- - - -")
print(test.isnull().sum())



#It seems that there are no missing values in both the train and test data.
#Sexのみがカテゴリー変数、その他は量的変数、idは不要なので学習する際は削除

#量的変数が少なく、Age,Height,Weightなどは完全な独立変数とも言い難い(相関あり)
 #そこで2変数同士の交互作用を含めた特徴を含めて学習する

numerical_features = ["Age","Height","Weight","Duration","Heart_Rate","Body_Temp"] 

def add_feature_cross_terms(df, numerical_features):
    df_add = df.copy()
    for i in range(len(numerical_features)):
        for j in range(i+1, len(numerical_features)):
            feature1 = numerical_features[i]
            feature2 = numerical_features[j]
            cross_term_name = f"{feature1}_x_{feature2}"
            df_add[cross_term_name] = df_add[feature1]*df_add[feature2]
    return df_add

# update dataset with cross_term_name
train = add_feature_cross_terms(train,numerical_features)
test = add_feature_cross_terms(test,numerical_features)

print("-------new train dataset-------")
print(train.info())
print("-------new test dataset-------")
print(test.info())



num_features = train.select_dtypes(include='number')

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
train['Sex'] = le.fit_transform(train['Sex'])
test['Sex'] = le.transform(test['Sex'])

#この2行の処理はしなくても機械学習自体は成立するが、実行していたほうが正確で効率的な処理となり、「誤解」や「最適性の欠如」といったリスクを低減することができる。処理をしないと単体の数値として認識される
train['Sex'] = train['Sex'].astype("category")
test['Sex'] = test['Sex'].astype("category")

X = train.drop(columns = ['id','Calories'])
y = np.log1p(train['Calories'])
X_test = test.drop(columns = ['id'])


train.describe()


from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import time


FOLDS = 5 
FEATURES = X.columns.tolist()

#KFold setup
kf = KFold(n_splits = FOLDS, shuffle = True, random_state=42)

#Array for sotring Prediction values
oof = np.zeros(len(train))
pred = np.zeros(len(test))

# start CV loop
for i, (train_idx, valid_idx) in enumerate(kf.split(X,y)):
    print(f"\n{'#'*10} Fold {i+1} {'#'*10}")
    
    x_train = X.iloc[train_idx].copy()
    y_train = y.iloc[train_idx]
    x_valid = X.iloc[valid_idx].copy()
    y_valid = y.iloc[valid_idx]
    x_test = X_test.copy()
    
    start = time.time()
    #train model
    model = XGBRegressor(
        device = "cuda" if XGBRegressor().get_params().get("device") == "cuda" else "cpu",
        max_depth = 12,
        colsample_bytree = 0.75,
        subsample=0.85,
        n_estimators = 2000,
        learning_rate = 0.015,
        gamma = 0.01,
        max_delta_step = 2,
        early_stopping_rounds = 100,
        eval_metric = "rmse",
        enable_categorical = True
    )

    model.fit(
        x_train,y_train,
        eval_set = [(x_valid,y_valid)],
        verbose = 100
    )

    # Predict OOF and test
    oof[valid_idx] = model.predict(x_valid)
    pred += model.predict(x_test)

    rmse = np.sqrt(mean_squared_error(y_valid, oof[valid_idx]))
    print(f"Fold {i+1} RMSE:{rmse:4f}")
    print(f"Feature engineering & training time:{time.time() - start:.1f} sec")

# Average test predictions
pred /= FOLDS

# Final RMSE
full_rmse = np.sqrt(mean_squared_error(y, oof))
print(f"\nFinal CV RMSE: {full_rmse:.4f}")


y_preds = np.expm1(pred)
print('predict mean :', y_preds.mean())
print('predict median :', np.median(y_preds))

y_preds = np.clip(y_preds,1,314)
print('predict mean after clip:', y_preds.mean())
print('predict median after clip:', np.median(y_preds))

submission["Calories"] = y_preds
submission.to_csv("submission.csv", index = False)
submission.head()




