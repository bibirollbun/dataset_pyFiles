import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder


train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


# show train data sample
train_df.head(5) 


encoders = [LabelEncoder() for le in range(3)]
def preprocessing(df,is_train):
    # label encoding
    col_names = ['Stage_fear', 'Drained_after_socializing']
    for i, col in enumerate(col_names):
        df[col] = df[col].fillna('None')
        if is_train:
            df[col] = encoders[i].fit_transform(df[col])
        else:
            df[col] = encoders[i].transform(df[col])
    if is_train:
        df['Personality']= encoders[2].fit_transform(df['Personality'])

    #mean
    cols = ['Social_event_attendance','Going_outside','Friends_circle_size','Post_frequency']
    for col in cols:
        df[col] = df[col].fillna(df[col].mean())

    #median
    df['Time_spent_Alone'] = df['Time_spent_Alone'].fillna(df['Time_spent_Alone'].median())
    return df


# preprocessing
train_df=preprocessing(train_df,True)
test_df=preprocessing(test_df,False)


X_features = train_df.drop(['id','Personality'],axis=1)
y_target = train_df['Personality']

# split train and test set
X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.2, random_state=0)


# get best parameters using optuna

# def objective(trial):
#     params = {
#         "n_estimators": trial.suggest_int("n_estimators", 50, 300),
#         "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
#         "max_depth": trial.suggest_int("max_depth", 2, 10),
#         "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
#         "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
#         "subsample": trial.suggest_float("subsample", 0.5, 1.0),
#         "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
#     }

#     model = GradientBoostingClassifier(**params)

#     # K-Fold Cross Validation
#     cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#     score = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=cv)
#     return np.mean(score)

# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=100)
# print("Best hyperparameters:", study.best_params)
# print("Best CV accuracy:", study.best_value)


model = GradientBoostingClassifier(n_estimators= 185, learning_rate=0.035976250614921844,max_depth=3,
                                   min_samples_split=13, min_samples_leaf=6, subsample=0.8027485837107697, max_features='log2')
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(accuracy_score(y_test, pred))


# predict data
X_result = test_df.drop(['id'], axis=1, inplace=False)
test_pred = model.predict(X_result)
pred_result = encoders[2].inverse_transform(test_pred)

# make submission file
test_pd = pd.DataFrame({'id':test_df['id'],'Personality':pred_result})
test_pd.to_csv("submission.csv",index=False)
test_pd.head(10)

