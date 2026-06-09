import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer
from sklearn.base import BaseEstimator
from catboost import CatBoostClassifier
from scipy.stats import randint, uniform


train_csv = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test_csv = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
submission_csv = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")


train_csv.head()


train_csv.tail()


def Temp_Humidity_Index(data):
    THI = data["Temparature"] - ((0.55 - (0.0055 * data["Humidity"])) * (data["Temparature"] - 14.5))
    data["THI"] = THI
    return data


def mapk(actual, predicted, k=3):
    def apk(a, p, k):
        if len(p) > k:
            p = p[:k]

        score = 0.0
        for i, pred in enumerate(p):
            if pred == a and pred not in p[:i]:
                score += 1.0 / (i + 1.0)
                break  # only one true label per sample

        return score

    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


def mapk_scorer_func(estimator, X, y):
    proba = estimator.predict_proba(X)
    topk_preds = np.argsort(proba, axis=1)[:, ::-1][:, :3]  # top-k
    topk_preds = topk_preds.tolist()
    return mapk(y, topk_preds, k=3)


#Basic Information
train_csv.info()


test_csv.info()


train_csv.describe()

test_csv.describe()


#Correlation Heatmap
heatmap = train_csv[["Temparature", "Humidity", "Moisture", "Nitrogen", "Potassium", "Phosphorous"]].corr()
sns.heatmap(heatmap, annot = True)
plt.show()


df1 = train_csv.copy()
df1 = Temp_Humidity_Index(df1)
df1.head()


#Different Soil Type

unq_soil = df1["Soil Type"].unique().tolist()
unq_soil


df1["Soil Type"].value_counts().plot(kind = "bar", cmap = "coolwarm")
plt.title("Count of each Soil Type")
plt.ylabel("Count")
plt.show()


df1["Fertilizer Name"].value_counts().plot(kind = "bar", cmap = "coolwarm")
plt.ylabel("Count")
plt.title("Count of Each Fertilizer")
plt.show()


df1["Crop Type"].value_counts().plot(kind = "bar", cmap = "coolwarm")
plt.title("Count of each Soil Type")
plt.ylabel("Count")
plt.show()


def preprocessing(data):
    df = data.copy()
    df = Temp_Humidity_Index(df)

    columns = ["Soil Type", "Crop Type"]

    df = pd.get_dummies(df,
                        prefix = "OHE",
                        prefix_sep = "_",
                        columns = columns,
                       )

    df = df.drop("id" , axis = 1)

    return df


def preprocess_spilt(df):
    fertilizer_map = {
        "14-35-14" : 0,
        "10-26-26" : 1,
        "17-17-17" : 2,
        "28-28" : 3,
        "20-20" : 4,
        "DAP" : 5,
        "Urea" : 6
    }

    df["Fertilizer Name"] = df["Fertilizer Name"].map(fertilizer_map)

    X_data = df.drop("Fertilizer Name", axis = 1)
    y_data = df["Fertilizer Name"]

    X_train, X_test, y_train, y_test = train_test_split(X_data,
                                                       y_data,
                                                       test_size = 0.2,
                                                       random_state = 42)

    return X_train, y_train, X_test, y_test


def xgb_classification():
    data = preprocessing(train_csv)

    fertilizer_map = {
        "14-35-14" : 0,
        "10-26-26" : 1,
        "17-17-17" : 2,
        "28-28" : 3,
        "20-20" : 4,
        "DAP" : 5,
        "Urea" : 6
    }
    #X_train, y_train, X_test, y_test = preprocess_spilt(data)

    data["Fertilizer Name"] = data["Fertilizer Name"].map(fertilizer_map)

    X_train = data.drop("Fertilizer Name", axis = 1)
    y_train = data["Fertilizer Name"]

    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=7,
        subsample= 0.8,
        reg_lambda= 1.5,
        reg_alpha= 1, 
        n_estimators= 300, 
        max_depth= 7, 
        learning_rate= 0.2, 
        gamma= 0.1, 
        colsample_bytree= 0.6
    )

    model.fit(X_train, y_train)

    return model


def catboost_classification():
    
    data = preprocessing(train_csv)
    
    fertilizer_map = {
        "14-35-14" : 0,
        "10-26-26" : 1,
        "17-17-17" : 2,
        "28-28" : 3,
        "20-20" : 4,
        "DAP" : 5,
        "Urea" : 6
    }
    
    data["Fertilizer Name"] = data["Fertilizer Name"].map(fertilizer_map)
    
    X_train = data.drop("Fertilizer Name", axis = 1)
    
    y_train = data["Fertilizer Name"]

    

    
    #X_train, y_train, X_test, y_test = preprocess_spilt(data)
    
    model = CatBoostClassifier(
        border_count= 221, 
        depth = 7, 
        iterations = 257, 
        l2_leaf_reg = 8.473201101373808, 
        learning_rate = 0.17190763971672393
    )

    model.fit(X_train, y_train)

    return model


def randomize_serach_cv(model, X_train, y_train):

    param_grid = {
        'iterations': randint(50, 300),
        'learning_rate': uniform(0.01, 0.3),
        'depth': randint(3, 10),
        'l2_leaf_reg': uniform(1, 10),
        'border_count': randint(32, 255)
    }


    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=20,
        scoring=mapk_scorer_func,
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )

    search.fit(X_train, y_train)
    
    return search


def predicts(model1, model2, X_test, y_test):
    
    proba1 = model1.predict_proba(X_test)
    proba2 = model2.predict_proba(X_test)
    
    avg_proba = (proba1 + proba2) / 2

    top_3_preds = np.argsort(avg_proba, axis=1)[:, -3:][:, ::-1]

    map3 = mapk(y_test, top_3_preds, k=3)
    
    return map3


def make_submission(model1, model2, X_test, submission_df, output_path="submission.csv"):
    fertilizer_map = {
        "14-35-14": 0,
        "10-26-26": 1,
        "17-17-17": 2,
        "28-28": 3,
        "20-20": 4,
        "DAP": 5,
        "Urea": 6
    }
    inverse_map = {v: k for k, v in fertilizer_map.items()}

    # Predict probabilities from both models
    proba1 = model1.predict_proba(X_test)
    proba2 = model2.predict_proba(X_test)

    # Average the predicted probabilities
    avg_proba = (proba1 + proba2) / 2

    # Get top 3 predicted class indices
    top_3_preds = np.argsort(avg_proba, axis=1)[:, -3:][:, ::-1]

    # Map numeric predictions to fertilizer names
    top_3_fertilizers = [
        " ".join([inverse_map[p] for p in preds])
        for preds in top_3_preds
    ]

    # Assign to submission DataFrame
    submission_df["Fertilizer Name"] = top_3_fertilizers

    # Save submission
    submission_df.to_csv(output_path, index=False)
    print(f"✅ Submission file saved as: {output_path}")


model1 = xgb_classification()
model2 = catboost_classification()
data = preprocessing(train_csv)
X_train, y_train, X_test, y_test = preprocess_spilt(data)



map3 = predicts(model1, model2, X_test, y_test)
print(map3)


#searching = randomize_serach_cv(model,X_train, y_train)
#print("Best parameters:", searching.best_params_)
#print("Best score (MAP@k):", searching.best_score_)


X_test = preprocessing(test_csv)


make_submission(model1, model2, X_test, submission_csv, output_path="submission.csv")

