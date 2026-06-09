import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,confusion_matrix,roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold,RandomizedSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.utils.multiclass import type_of_target
from catboost import CatBoostClassifier


df_train=pd.read_csv('/kaggle/input/playground-series-s3e7/train.csv')
df_test=pd.read_csv('/kaggle/input/playground-series-s3e7/test.csv')
df_suv=pd.read_csv('/kaggle/input/playground-series-s3e7/sample_submission.csv')


df_suv.head()


print(df_train.shape,df_test.shape)


df_train.head()


df_train.isna().sum()


df_train.info()


sns.countplot(x='booking_status',data=df_train)


df_train.describe()


sns.histplot(df_train.no_of_adults)


df_train=df_train[df_train.no_of_adults>0]


sns.histplot(df_train.no_of_children)


df_train=df_train[df_train.no_of_children<4]


sns.countplot(x='type_of_meal_plan',data=df_train)


df_train.type_of_meal_plan.value_counts()


df_train=df_train[df_train.type_of_meal_plan<3]


sns.histplot(df_train.lead_time)


sns.countplot(x='repeated_guest',data=df_train)


sns.histplot(df_train.no_of_previous_cancellations)


df_train=df_train[df_train.no_of_previous_cancellations<=11]


df_train.head()


sns.histplot(df_train.avg_price_per_room,bins=20,kde=True)


plt.figure(figsize=(10,10))
sns.heatmap(df_train.corr(),annot=True,fmt='.1f',vmin=-1,vmax=1,cmap='Blues')


X=df_train.drop(columns=['arrival_year','id','booking_status'],axis=1)
y=df_train['booking_status']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
models = {
    'Logistic Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
    ]),
    'Decision Tree': DecisionTreeClassifier(max_depth=10, class_weight='balanced', random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=15, class_weight='balanced', random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', max_depth=6, learning_rate=0.1, random_state=42),
    'CatBoost': CatBoostClassifier(verbose=0, depth=6, learning_rate=0.1, random_state=42)
}

# Train and evaluate
for name, model in models.items():
    print(f"\nğŸ”� {name}")
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

    print("\nğŸ“Š Classification Report:")
    print(classification_report(y_test, y_pred))

    if y_proba is not None:
        if type_of_target(y_test) == 'binary':
            roc_auc = roc_auc_score(y_test, y_proba[:, 1])
        else:
            roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
        print(f"ğŸ“ˆ ROC AUC Score: {roc_auc:.4f}")


final_model=CatBoostClassifier(verbose=0, depth=6, learning_rate=0.1, random_state=42)


final_model.fit(X_train,y_train)


X_test_df=df_test.drop(columns=['id','arrival_year'],axis=1)


final_model.predict(X_test_df)


y_pred_prob=model.predict_proba(X_test_df)[:, 1]


np.round(y_pred_prob,3)


submission = pd.DataFrame({
    'id': df_test['id'],               
    'booking_status': np.round(y_pred_prob,3)  
})

# Save to CSV
submission.to_csv("submission.csv", index=False)
print("âœ… Submission file created: submission.csv")




