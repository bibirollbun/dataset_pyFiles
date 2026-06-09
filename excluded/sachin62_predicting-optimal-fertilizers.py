import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools   import add_constant
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection   import train_test_split
from sklearn.ensemble          import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier
from sklearn.linear_model      import LogisticRegression
from sklearn.svm               import SVC
from sklearn.tree              import DecisionTreeClassifier
from sklearn.naive_bayes       import GaussianNB
from sklearn.neighbors         import KNeighborsClassifier
from xgboost                   import XGBClassifier
from lightgbm                  import LGBMClassifier
from sklearn.metrics           import accuracy_score,recall_score,precision_score,f1_score
from scipy.stats               import randint
from sklearn.preprocessing     import LabelEncoder
from sklearn.preprocessing     import StandardScaler
import warnings
warnings.simplefilter("ignore")


df = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
df.head()


test.head()


print(df.columns)
print(test.columns)


df.drop(columns=['id'], inplace = True)
test.drop(columns=['id'], inplace = True)
df.head()


print(df.shape)
print(test.shape)


print(df.isnull().sum())
print(test.isnull().sum())


print(df.duplicated().sum())
print(test.duplicated().sum())


df.info()


df.head()


print(df['Soil Type'].value_counts())
print(df['Crop Type'].value_counts())


df['Fertilizer Name'].value_counts()


cat_cols = ['Soil Type','Crop Type']
num_cols = ['Temparature', 'Humidity', 'Moisture','Nitrogen', 'Potassium', 'Phosphorous']
target_cols = ['Fertilizer Name']


data = df.copy()


def num_plot_dist(df , num_features):
    fig , axes = plt.subplots(len(num_features),2,figsize=(15,len(num_features)*5))
    if len(num_features)==1:
        axes=[axes]
    
    for i,column in enumerate(num_features):
        sns.histplot(data=df , x=column , ax=axes[i][0] , kde=True , palette="Blues" )
        axes[i][0].set_title(f"Histogram for {column}")

        sns.boxplot(data=df , x=column , ax=axes[i][1] , palette="Blues")
        axes[i][1].set_title(f"Box Plot for {column}")
    
    plt.tight_layout()
    plt.show()


num_plot_dist(data , num_cols)


for cat_feature in cat_cols:
    plt.figure(figsize=(10,6))
    data[cat_feature].value_counts().plot(kind='bar' , color='skyblue')
    plt.title(f"{cat_feature} Distribution")
    plt.xlabel(cat_feature)
    plt.ylabel("Count")
    plt.show()


def plot_bivariate_num(df , target , num_features):
    num_plots = len(num_features)
    num_rows = (num_plots+1)//2

    fig , axes = plt.subplots(num_rows , 2 , figsize=(15 , num_rows*5))
    axes = axes.flatten()

    for i , column in enumerate(num_features):
        sns.boxplot(x=target,y=column,ax=axes[i] , data=df , palette="Blues")
        axes[i].set_title(f"{column}  VS {target}")

    plt.tight_layout()
    plt.show()


plot_bivariate_num(data , 'Fertilizer Name' , num_cols)


def plot_bivaraite_cat(df , target , cat_features):

    num_features = len(cat_features)
    num_rows = (num_features+1)//2

    fig , axes = plt.subplots(num_rows , 2 , figsize=(15 , num_rows*5))
    axes = axes.flatten()

    for i,feature in enumerate(cat_features):
        sns.countplot(x=feature , hue=target ,data=df , palette="Set2" , ax = axes[i])
        axes[i].set_title(f"{feature} VS {target}")
        axes[i].tick_params(axis='x' , rotation=90)

    plt.tight_layout()
    plt.show()


plot_bivaraite_cat(data , 'Fertilizer Name' , cat_cols)


label_encoder = LabelEncoder()
mappings={}
for col in cat_cols:
    df[col] = label_encoder.fit_transform(df[col])
    mappings[col] = {label:code for label,code in zip(label_encoder.classes_ , label_encoder.transform(label_encoder.classes_))}


## doing on test data
for col in cat_cols:
    test[col] = test[col].map(mappings[col])
    # Optional: handle unseen categories
    test[col] = test[col].fillna(-1).astype(int)


mappings


df.head()


df.info()


## handle target columns 
label_encoder = LabelEncoder()
df['target_encoded'] = label_encoder.fit_transform(df[target_cols])

label_to_code = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
code_to_label = dict(zip(label_encoder.transform(label_encoder.classes_), label_encoder.classes_))



df.drop(columns=['Fertilizer Name'], inplace = True)


df.info()


X = add_constant(df)
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values,i) for i in range(X.shape[1])]


vif_data


corr = df.corr()
corr


sns.heatmap(corr , linewidths=0.5)


skewness  = df.skew()
skewness


df['target_encoded'].value_counts()


X = df.drop(columns='target_encoded')
y = df["target_encoded"]


X_train , X_test , y_train , y_test = train_test_split(X,y , test_size=0.1 , random_state=42)


# # Initialize the scaler
# scaler = StandardScaler()

# # Fit on training data and transform both train and test data
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # Apply the same scaler to the separate test dataset
# test_scaled = scaler.transform(test)





classifiers = {
    # "Random Forest" : RandomForestClassifier(random_state=42),
    # "LogisticRegression" : LogisticRegression(random_state=42),
    # "Gradient Boosting" : GradientBoostingClassifier(random_state=42),
    # "Suuport vector classifier" : SVC(random_state=42),
    # "Decsion Tree" : DecisionTreeClassifier(random_state=42),
    # "KNN" : KNeighborsClassifier(),
    # "Naive Bayes" : GaussianNB(),
    "XGboost" : XGBClassifier(random_state=42),
    # "Adaboost" : AdaBoostClassifier(random_state=42),
    "LGBM" : LGBMClassifier(random_state=42) 
}


def apk(actual, predicted, k=3):
    if len(predicted) > k:
        predicted = predicted[:k]
    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p == actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
            break
    return score

def mapk(actuals, preds, k=3):
    return np.mean([apk(a, p, k) for a, p in zip(actuals, preds)])



metrics = {
    "Model" : [],
    "Accuracy" : [],
    "Precision" : [],
    "Recall" : [],
    "F1 Score" : [],
    "MAP@3": []
}



for model_name, classifier in classifiers.items():
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

    # Calculate MAP@3 if model supports predict_proba
    if hasattr(classifier, "predict_proba"):
        y_prob = classifier.predict_proba(X_test)
        top3_preds = np.argsort(y_prob, axis=1)[:, -3:][:, ::-1]
        map3 = mapk(y_test, top3_preds, k=3)
    else:
        map3 = np.nan  
    
    print(map3 , accuracy)
    
    metrics["Model"].append(model_name)
    metrics["Accuracy"].append(accuracy)
    metrics["Precision"].append(precision)
    metrics["Recall"].append(recall)
    metrics["F1 Score"].append(f1)
    metrics["MAP@3"].append(map3)


metrics_df= pd.DataFrame(metrics)
metrics_df


X_train , X_test , y_train , y_test = train_test_split(X,y , test_size=0.1 , random_state=42)



# Define classifier
classifiers = {
    "XGboost": XGBClassifier(random_state=42)
}

# Extract the model
model = classifiers["XGboost"]

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)




test1 = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')


y_predict = model.predict(test)

fertilizer_names = [code_to_label[code] for code in y_predict]

output_df = pd.DataFrame({
    'id': test1['id'],
    'Fertilizer Name': fertilizer_names
})

output_df.to_csv('submission.csv', index=False)




