!pip install lofo-importance


import pandas as pd

df = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
df


from sklearn.preprocessing import LabelEncoder

df["Personality"] = LabelEncoder().fit_transform(df["Personality"])

for col in df.columns:
    if df[col].dtype == object:
        print(col)
        df[col] = LabelEncoder().fit_transform(df[col].fillna("nan").astype(str))
        df[col] = df[col].astype("category")


from lofo import LOFOImportance, Dataset, plot_importance
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

lofo_dataset = Dataset(df=df, 
                       target="Personality", 
                       features=[col for col in df.columns if col not in ["id", "Personality"]],
                       auto_group_threshold=0.85)


model = XGBClassifier(
    device="cuda",
    max_depth=3,  
    colsample_bytree=0.5, 
    subsample=0.8, 
    n_estimators=400,  
    learning_rate=0.1,
    enable_categorical=True,
    min_child_weight=5
)

lofo_imp = LOFOImportance(dataset=lofo_dataset, model=model, scoring="roc_auc")
importance_df = lofo_imp.get_importance()

plot_importance(importance_df)

