import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV




import warnings
warnings.filterwarnings('ignore')


raw_data = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")


raw_data.head()


raw_data.describe()


raw_data.isnull().sum()


for col in raw_data.columns:
    if col!="id":
        print(col.center(50))
        print(raw_data[col].value_counts())
        print("-"*50)


def preproccess_data(data):

    data = data.copy() 
    
    # COLUMN TYPES
    cat_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # FILLING NA
    for col in cat_cols:
        data[col].fillna(data[col].mode()[0], inplace=True)
    
    data[num_cols] = data[num_cols].fillna(data[num_cols].mean())

    # ENCODING
    # BOTH CATEGORICAL COLUMS WERE YES,NO AND THERE WERE NO OTHER VALUES SO MAPPİNG İS SAFE
    data["Stage_fear"] = data["Stage_fear"].map({"Yes":1,"No":0})
    data["Drained_after_socializing"] = data["Drained_after_socializing"].map({"Yes":1,"No":0})
    if "Personality" in data.columns:
        data["Personality"] = data["Personality"].map({"Extrovert":1,"Introvert":0})
    
    # FE
    # CHECK CORRELATION TO UNDERSTAND
    data["alone_x_fear"] = data["Time_spent_Alone"] * data["Stage_fear"]
    data["goout_x_social"] = data["Going_outside"] * data["Social_event_attendance"]

    return data


data = preproccess_data(raw_data)


data


num_cols = [col for col in data.columns if col not in ['id', 'Personality'] and data[col].dtype in ['int64', 'float64']]
for col in num_cols:
    plt.figure(figsize=(6,4))
    sns.boxplot(x='Personality', y=col, data=data)
    plt.title(f'{col} by Personality')
    plt.xlabel("Personality(Introvert 0, Extrovert 1)")
    plt.show()

num_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
corr_matrix = data[num_cols].corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()


X = data.drop(["Personality","id"],axis=1)
y = data["Personality"]


X_train,X_valid,y_train,y_valid = train_test_split(X,y,test_size = 0.2,stratify=y,random_state=42)


# CHECK TUNING TO SEE HOW I GOT THESE PARAMETERS
model = RandomForestClassifier(
    bootstrap=True,
    max_depth=10,
    max_features='sqrt',
    min_samples_leaf=4,
    min_samples_split=10,
    n_estimators=500,
    random_state=42
)
model.fit(X_train,y_train)
preds = model.predict(X_valid)


def evaluate_model(y_valid,y_pred,show_report=True):
    acc = accuracy_score(y_valid, y_pred)
    print(f"Accuracy: {acc:.4f}")
    
    if show_report:
        print("\nClassification Report:")
        print(classification_report(y_valid, y_pred))

    return acc


evaluate_model(y_valid,preds)


importances = model.feature_importances_

feature_names = X_train.columns

feat_importances = pd.Series(importances, index=feature_names)
feat_importances = feat_importances.sort_values(ascending=False)
print(feat_importances)

plt.figure(figsize=(10, 6))
feat_importances.head(20).plot(kind='bar')
plt.title("Feature Importances")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()


raw_test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")


test = preproccess_data(raw_test)


y_final = model.predict(test.drop(["id"],axis=1))


submission_df = pd.DataFrame({
    "id": test["id"],   
    "Personality": y_final    
})

submission_df["Personality"] = submission_df["Personality"].map({0: "Introvert", 1: "Extrovert"})



submission_df.to_csv("/kaggle/working/submission.csv", index=False)




