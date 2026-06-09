import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


base_dir = "/kaggle/input/playground-series-s5e7"
os.listdir(base_dir)


train_df = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
submission_df = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")


train_df.head(10)


test_df


print(f"Size of train data {train_df.shape}")
print(f"Size of test data {test_df.shape}")


train_df.info()


test_df.info()


train_df.describe()


train_extro = train_df[train_df["Personality"] =="Extrovert"]
train_intro = train_df[train_df["Personality"] =="Introvert"]


train_extro.describe()


train_intro.describe()


train_df.isna().sum()


test_df.isna().sum()


train_df.isna().sum() / train_df.shape[0] *100


test_df.isna().sum() / test_df.shape[0] *100


train_df.info()


numeric_cols = [
    "Time_spent_Alone" , 
    "Social_event_attendance",
    "Going_outside",
    "Friends_circle_size",
    "Post_frequency" 
]


for i in numeric_cols:
    
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.histplot(train_df[i], kde=True , color = "green")
    plt.title(f"Histogram of {i}")
    plt.xlabel(i)
    plt.ylabel("Frequency")


for i in numeric_cols:
    
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.boxplot(train_df[i], color = "green")
    plt.title(f"Barplot of {i}")
    plt.xlabel(i)
    plt.ylabel("Frequency")


for i in numeric_cols:
    
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.kdeplot(train_df[i], fill = True)
    plt.title(f"Kde of {i}")
    plt.xlabel(i)
    plt.ylabel("Frequency")


count_stage = train_df["Stage_fear"].value_counts().sort_values(ascending = False)
plt.figure(figsize=(6, 6))
plt.pie(count_stage, labels=count_stage.index, autopct='%1.2f%%')
plt.title("Distribution of stage fear" , size = 22)
plt.show()



train_df["Personality"].value_counts() / train_df.shape[0] *100


count_Personality = train_df["Personality"].value_counts().sort_values(ascending = False)
plt.figure(figsize=(6, 6))
plt.pie(count_Personality, labels=count_Personality.index, autopct='%1.2f%%')
plt.title("Distribution of count_Personality" , size = 22)
plt.show()



train_df[numeric_cols] = train_df[numeric_cols].fillna(train_df[numeric_cols].mean())


train_df[numeric_cols].isna().sum()


test_df[numeric_cols] = test_df[numeric_cols].fillna(test_df[numeric_cols].mean())


test_df[numeric_cols].isna().sum()


from sklearn.impute import SimpleImputer


imputer = SimpleImputer(strategy='most_frequent')

imputed_values = imputer.fit_transform(train_df["Stage_fear"].values.reshape(-1, 1))

train_df["Stage_fear"] = imputed_values.ravel()



imputed_values2 = imputer.fit_transform(train_df["Drained_after_socializing"].values.reshape(-1, 1))

train_df["Drained_after_socializing"] = imputed_values2.ravel()


imputed_values = imputer.fit_transform(test_df["Stage_fear"].values.reshape(-1, 1))

test_df["Stage_fear"] = imputed_values.ravel()



imputed_values2 = imputer.fit_transform(test_df["Drained_after_socializing"].values.reshape(-1, 1))

test_df["Drained_after_socializing"] = imputed_values2.ravel()





from sklearn.model_selection import train_test_split
new_train_df , val_df = train_test_split(train_df , test_size =  0.2 , random_state = 42)


print(f"Size of training data {new_train_df.shape}")
print(f"Size of validation data {val_df.shape}")


train_df.columns


inputs_cols = [
    'Time_spent_Alone', 
    'Stage_fear', 
    'Social_event_attendance',
    'Going_outside', 
    'Drained_after_socializing', 
    'Friends_circle_size',
    'Post_frequency'
]
target_cols = ['Personality']


train_inputs = new_train_df[inputs_cols]
train_target = new_train_df[target_cols]


# train_inputs = train_df[inputs_cols]
# train_target = train_df[target_cols]


train_target


val_inputs = val_df[inputs_cols]
val_target = val_df[target_cols]


test_inputs = test_df[inputs_cols]


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()



train_inputs["Stage_fear"]  = encoder.fit_transform(train_inputs["Stage_fear"])


train_inputs["Stage_fear"]


train_inputs["Drained_after_socializing"] = encoder.fit_transform(train_inputs["Drained_after_socializing"])


train_inputs["Drained_after_socializing"]


val_inputs["Stage_fear"]  = encoder.fit_transform(val_inputs["Stage_fear"])
val_inputs["Drained_after_socializing"] = encoder.fit_transform(val_inputs["Drained_after_socializing"])


val_inputs.head(5)


test_inputs["Stage_fear"]  = encoder.fit_transform(test_inputs["Stage_fear"])
test_inputs["Drained_after_socializing"] = encoder.fit_transform(test_inputs["Drained_after_socializing"])


test_inputs.head(5)


train_target = encoder.fit_transform(train_target)


val_target = encoder.fit_transform(val_target)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# train_inputs = scaler.fit_transform(train_inputs)
test_inputs = scaler.fit_transform(test_inputs)


print(train_inputs.shape)
print(train_target.shape)
print(val_inputs.shape)
print(val_target.shape)


from xgboost import XGBClassifier


# model = XGBClassifier(
#     random_state = 42 , 
#     n_jobs = -1  ,
#     n_estimator = 20 ,
#     max_depth = 4
# )


# model = XGBClassifier( n_jobs=-1, random_state=42,
#                                n_estimators=500, max_depth=5, learning_rate=0.1,
#                                subsample=0.8, colsample_bytree=0.8)


model = XGBClassifier(
    objective =  "binary:logistic",
    eval_metric= "logloss",
    max_depth = 4,
    eta = 0.1,
    subsample =  0.8,
    colsample_bytree = 0.8,
    random_state = 42 , 
    n_estimators = 300
)


model.fit(train_inputs , train_target)


val_preds = model.predict(val_inputs)


from sklearn.metrics import accuracy_score
acc = accuracy_score(val_target , val_preds)
print(f" Accuracy: {acc:.4f}")


from sklearn.ensemble import RandomForestClassifier


rf = RandomForestClassifier(n_estimators = 100 , random_state = 42 , max_depth=8 , oob_score = True , n_jobs = -1)
rf.fit(train_inputs , train_target)


val_preds2 = rf.predict(val_inputs)


acc = accuracy_score(val_target , val_preds2)
print(f" Accuracy: {acc:.4f}")


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(random_state=42)
logreg.fit(train_inputs , train_target)


val_preds3 = logreg.predict(val_inputs)


acc = accuracy_score(val_target , val_preds3)
print(f" Accuracy: {acc:.4f}")


def predict_and_submit(model, test_inputs, fname):

    test_predictions = model.predict(test_inputs)


    label_map = {0: 'Introvert', 1: 'Extrovert'}
    mapped_predictions = [label_map[int(p)] for p in test_predictions]


    sub_df = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")


    if len(mapped_predictions) != len(sub_df):
        raise ValueError(f"Prediction count ({len(mapped_predictions)}) doesn't match submission rows ({len(sub_df)})")


    sub_df['Personality'] = mapped_predictions


    sub_df.to_csv(fname, index=False)
    return sub_df





predict_and_submit(rf , test_inputs,  "ran.csv")



def test_params(modelclass , **params):

      model = modelclass(**params).fit(train_inputs , train_target)
      train_score = model.predict(train_inputs , train_target)
      val_score = model.predict(val_inputs , val_target)

      return train_score , val_score
      

def test_param_and_plot(ModelClass, param_name, param_values, **other_params):

    train_errors, val_errors = [], []
    for value in param_values:
        params = dict(other_params)
        params[param_name] = value
        train_score, val_score = test_params(ModelClass, **params)
        train_errors.append(train_score)
        val_errors.append(val_score)

    plt.figure(figsize=(10,6))
    plt.title('Overfitting curve: ' + param_name)
    plt.plot(param_values, train_errors, 'b-o')
    plt.plot(param_values, val_errors, 'r-o')
    plt.xlabel(param_name)
    plt.ylabel('RMSE')
    plt.legend(['Training', 'Validation'])



best_params = {
    "random_state" : 42,
    "n_jobs":-1,
    "objective" :  "binary:logistic"
}


# test_param_and_plot(XGBClassifier , "n_estimators" , [100 , 200, 500] , **best_params)




