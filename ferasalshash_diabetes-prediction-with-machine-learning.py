import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


data_train=pd.read_csv("/kaggle/input/diabetes-prediction-from-medicalrecords/train.csv")
data_test=pd.read_csv("/kaggle/input/diabetes-prediction-from-medicalrecords/test.csv")


sub=pd.read_csv("/kaggle/input/diabetes-prediction-from-medicalrecords/sample_submission.csv")


sub


data_train.info()


data_test.info()


data_train["Outcome"].value_counts()


data_train.head()


data_test.head()


data_train.drop("Id", axis=1, inplace=True)


plt.figure(figsize=(15, 10))
colors = ['#ff9999', "#ffc300", '#66b3ff',"#40916c" ,'#99ff99', '#ffcc99', '#00b4d8', '#80ed99',"#ae2012"]

for i, j in enumerate(data_train):
    plt.subplot(3, 3, i+1)
    sns.boxplot(x=data_train[j], color=colors[i])
    plt.title(f'Distribution of {j}', fontsize=12)
    plt.xlabel('')

plt.tight_layout()
plt.suptitle('Outlier Analysis for Features', fontsize=16, y=1.02)
plt.show()


def HndlingOutlier(Cname):
  Q1=np.quantile(Cname, 0.25)
  Q3=np.quantile(Cname, 0.75)
  IQR=Q3-Q1
  Lower=Q1-1.5*IQR
  Upper=Q3+1.5*IQR
  Outlier=Cname[(Cname<Lower) | (Cname>Upper)]
  return Outlier


outlirs=[]
for i in data_train:
 outlier_train=HndlingOutlier(data_train[i])
 outlier_count=len(outlier_train)
 outlirs.append({"column": i , "Outliers": outlier_train.values ,"Count": outlier_count })

outlier_df=pd.DataFrame(outlirs)


outlier_df


data_train=data_train[data_train["Glucose"]>0]
data_train=data_train[data_train["BMI"]>0]
data_train=data_train[data_train["BloodPressure"]>0]


correlation=data_train.corr()
plt.figure(figsize=(8, 5))
sns.heatmap(data=correlation , cmap="gnuplot" , annot=True)
plt.title("correlation matrix")
plt.show()


plt.figure(figsize=(8,5))
sns.histplot(data=data_train, x="Glucose", hue="Outcome" , multiple="dodge" ,  palette="seismic_r")
plt.title("Correlation between Glucose and Diabetes")
plt.show()


plt.figure(figsize=(8,5))
sns.histplot(data=data_train, x="BMI", hue="Outcome"  ,multiple="dodge",  palette="seismic_r")
plt.title("Correlation between BMI and Diabetes")
plt.show()


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay,roc_curve, precision_recall_curve, average_precision_score, auc, classification_report


X=data_train.drop("Outcome", axis=1)
y=data_train["Outcome"]


from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)


X_train,X_val, y_train, y_val= train_test_split(X,y, test_size=0.2 , random_state=42)


scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_val=scaler.transform(X_val)


LR_model=LogisticRegression(class_weight='balanced')
LR_model


grid_param_LR=[
    {"solver": ["lbfgs"], "penalty": ["l2"], "C": [0.1, 0.5, 1], "max_iter": [ 200, 300, 500, 1000]},
    {"solver": ["liblinear"], "penalty": ["l1", "l2"], "C": [0.1, 0.5, 1], "max_iter": [ 200, 300, 500, 1000]}
]


Grid_Search_LR=GridSearchCV(LR_model,grid_param_LR, cv=5)
Grid_Search_LR.fit(X_train, y_train)


print(f"The Best Parameters for LogisticRegression are: {Grid_Search_LR.best_params_}")


LR_best=Grid_Search_LR.best_estimator_


LR_best_score=LR_best.score(X_val, y_val)
LR_best_score


y_pred_LR=LR_best.predict(X_val)


print(f"Classification Report is:\n{classification_report(y_val,y_pred_LR)}")
Accuracy=accuracy_score(y_val,y_pred_LR)
Precision=precision_score(y_val, y_pred_LR)
Recall=recall_score(y_val, y_pred_LR)
F1=f1_score(y_val, y_pred_LR)

print(f"The accuracy of LogisticRegression model is: {Accuracy}")
print(f"The Precision of LogisticRegression model is: {Precision}")
print(f"The Recall of LogisticRegression model is: {Recall}")
print(f"The F1 of LogisticRegression model is: {F1}")


cm_LR=confusion_matrix(y_val,y_pred_LR)
disp_LR=ConfusionMatrixDisplay(confusion_matrix=cm_LR)
disp_LR.plot(cmap="Paired_r")
plt.title("Confusion Matrix for Logistic Regreesion")
plt.show()


fpr_LR, tpr_LR, _=roc_curve(y_val, y_pred_LR)
AUC_LR=auc(fpr_LR, tpr_LR)
plt.plot(fpr_LR, tpr_LR, label=f"Logestig Regression AUC: {AUC_LR:.2f}" , lw=3 , color = "#b388eb")
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc='lower right')
plt.show()


Recall_LR_curve, Precision_LR_curve,_ =precision_recall_curve(y_val,y_pred_LR)
PR_LR=average_precision_score(y_val,y_pred_LR)
Recall_LR_curve, Precision_LR_curve,_ =precision_recall_curve(y_val,y_pred_LR)
plt.plot(Recall_LR_curve, Precision_LR_curve, label=f"Logestig Regression Precion_Rcall: {PR_LR:.2f}", lw=3, color="#b388eb" )
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PR- Curve')
plt.legend(loc='lower left')
plt.show()


RF_model=RandomForestClassifier(class_weight='balanced')
RF_model


Grid_param_RF=[{
    "n_estimators":[100,200],
    "max_depth": [None,10,20],
    "min_samples_split":[2,5],
    "max_features":["sqrt","log2", None]
}]


grid_search_RF=GridSearchCV(RF_model,Grid_param_RF,cv=5)
grid_search_RF.fit(X_train,y_train)


print(f"best parameters for Random Forest Classifier are: {grid_search_RF.best_params_}")


RF_best=grid_search_RF.best_estimator_


RF_best_score=RF_best.score(X_val,y_val)
RF_best_score


y_pred_RF=RF_best.predict(X_val)


print(f"Classification Report is:\n{classification_report(y_val,y_pred_RF)}")
Accuracy=accuracy_score(y_val,y_pred_RF)
Precision=precision_score(y_val,y_pred_RF)
Recall=recall_score(y_val,y_pred_RF)
F1=f1_score(y_val,y_pred_RF)

print(f"The accuracy of Random Forest Classifier model is: {Accuracy}")
print(f"The Precision of Random Forest Classifier model is: {Precision}")
print(f"The Recall of Random Forest Classifier model is: {Recall}")
print(f"The F1 of Random Forest Classifier model is: {F1}")


cm_RF=confusion_matrix(y_val,y_pred_RF)
disp_vo=ConfusionMatrixDisplay(confusion_matrix=cm_RF)
disp_vo.plot(cmap="crest")
plt.title("Confusion Matrix for Random Forest Classifier")
plt.show()


from sklearn.ensemble import VotingClassifier


voting_clf = VotingClassifier(estimators=[('logreg', LR_best), ('rf', RF_best)], voting='soft')
voting_clf.fit(X_train, y_train)



y_pred_v= voting_clf.predict(X_val)



print(f"Classification Report is:\n{classification_report(y_val,y_pred_v)}")
Accuracy=accuracy_score(y_val,y_pred_v)
Precision=precision_score(y_val,y_pred_v)
Recall=recall_score(y_val,y_pred_v)
F1=f1_score(y_val,y_pred_v)

print(f"The accuracy of Voting  Classifier model is: {Accuracy}")
print(f"The Precision of Voting  Classifier model is: {Precision}")
print(f"The Recall of Voting  Classifier model is: {Recall}")
print(f"The F1 of Voting  Classifier model is: {F1}")


cm_vo=confusion_matrix(y_val,y_pred_v)
disp_vo=ConfusionMatrixDisplay(confusion_matrix=cm_vo)
disp_vo.plot(cmap="crest")
plt.title("Confusion Matrix for Voting Classifier")
plt.show()



ID=data_test["Id"]
data_test=data_test.iloc[ : , :-1]



data_test_scaled = scaler.transform(data_test)

predictions = voting_clf.predict(data_test_scaled)


sub["Outcome"]=predictions


sub


sub.to_csv("sample_submission.csv", index=False)

