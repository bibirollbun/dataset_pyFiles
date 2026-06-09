import pandas as pd

df = pd.read_csv("/kaggle/input/space-ship/train.csv")
display(df)


df = df.drop(['Name'], axis=1)
display(df)


import numpy as np

nan_rows = df[df['RoomService'].isna() & df['FoodCourt'].isna() & df['ShoppingMall'].isna() & df['Spa'].isna() & df['VRDeck'].isna()].astype(object)
display(nan_rows)


result = df.query("FoodCourt != 0 and CryoSleep == True").value_counts()
print(result)


nan_rows = df[df['CryoSleep'].isna() & df['FoodCourt'].isna()].astype(object)
display(nan_rows)


indices = [3232, 5370, 7948]

df.loc[indices, 'CryoSleep'] = [True, False, False]


result = df.query("FoodCourt == 0 and CryoSleep == False and Age > 20")
display(result)


df['Group'] = df['PassengerId'].str.split('_').str[0]
groups_with_spending = df.groupby('Group')[['FoodCourt', 'RoomService', 'Spa', 'VRDeck']].sum().any(axis=1)
zero_spenders_in_active_groups = df[df['Group'].isin(groups_with_spending[groups_with_spending].index) & (df[['FoodCourt', 'RoomService', 'Spa', 'VRDeck']].sum(axis=1) == 0)]



#Feature: Total spending
df['TotalSpending'] = df[['FoodCourt', 'RoomService', 'ShoppingMall', 'Spa', 'VRDeck']].sum(axis=1)
#df = df.drop(['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck'], axis=1)
display(df)


output = df.query("TotalSpending == 0 and CryoSleep == False")
display(output)


group_stats = df.groupby('Group').agg(
    Total_Spending=('TotalSpending', 'sum'),
    Average_Spending=('TotalSpending', 'mean'),
    Passenger_Count=('PassengerId', 'count'),
    CryoSleep_Ratio=('CryoSleep', lambda x: x.eq(True).mean())
).sort_values('Total_Spending', ascending=False)

#Show top 10 groups by spending
display(group_stats.head(10))


group_stats = df.groupby('Group').agg(
    Total_Spending=('TotalSpending', 'sum'),
    All_Not_In_CryoSleep=('CryoSleep', lambda x: x.eq(False).all())
)


zero_spend_awake_groups = group_stats[
    (group_stats['Total_Spending'] == 0) & 
    (group_stats['All_Not_In_CryoSleep'])
].reset_index()

print(f"Found {len(zero_spend_awake_groups)} groups with 0 spending and all members awake.")
display(zero_spend_awake_groups)


suspect_groups = zero_spend_awake_groups['Group'].tolist()
suspect_passengers = df[df['Group'].isin(suspect_groups)][
    ['Group', 'Age', 'CryoSleep', 'TotalSpending', 'HomePlanet', 'VIP', 'Destination'] 
]

print("\nPassengers in these groups:")
display(suspect_passengers.sort_values('Group'))


df.loc[df['TotalSpending'] > 0, 'CryoSleep'] = False


remainings = df[df['CryoSleep'].isna()].astype(object)
display(remainings)


print(remainings["VIP"].value_counts())
print(remainings["VIP"].isna().sum())


remainings['GroupSize'] = remainings.groupby('Group')['PassengerId'].transform('count')

print("Group size distribution:")
print(remainings['GroupSize'].value_counts())

solo_passengers = remainings[remainings['GroupSize'] == 1]
print(f"\nSolo passengers with NaN CryoSleep: {len(solo_passengers)}")
display(solo_passengers[['Group', 'Age', 'HomePlanet', 'VIP']])


df[['deck', 'num', 'side']] = df['Cabin'].str.split('/', expand=True)
df.drop("Cabin", axis=1, inplace=True)

display(df)


cabin_counts = df['num'].value_counts().reset_index()
cabin_counts.columns = ['num', 'passenger_count']

shared_cabins = cabin_counts[cabin_counts['passenger_count'] > 1]['num']
print(f"Cabins shared by multiple passengers: {len(shared_cabins)}")


remainings[['deck', 'num', 'side']] = remainings['Cabin'].str.split('/', expand=True)
remainings.drop("Cabin", axis=1, inplace=True)

solo_zero_cabins = remainings['num'].unique()

for cabin in solo_zero_cabins:
    cabin_mates = df[df['num'] == cabin]
    if len(cabin_mates) > 1: 
        total_spending = cabin_mates['TotalSpending'].sum()
        
        if total_spending > 0:
            print(f"\nCabin {cabin} has spending (total: {total_spending}):")
            display(cabin_mates[['PassengerId', 'CryoSleep', 'TotalSpending', 'deck', 'side']])
            
            df.loc[
                (df['num'] == cabin) & 
                (df['PassengerId'].isin(remainings['PassengerId'])), 
                'CryoSleep'
            ] = False


print("Remaining NaN in CryoSleep:", df['CryoSleep'].isna().sum())


df['CryoSleep'] = df['CryoSleep'].fillna(True).infer_objects(copy=False)


display(df)


cat_cols = df[['HomePlanet', 'Destination', 'deck', 'side']]


for col in cat_cols:  
    mode_val = df[col].mode()[0]   
    df[col].fillna(mode_val, inplace=True)  


import seaborn as sns
import matplotlib.pyplot as plt

corr_matrix = df[['Age', 'TotalSpending']].corr()
sns.heatmap(corr_matrix, annot=True)
plt.show()



df['Age'].fillna(df['Age'].median(), inplace=True)


df['num'].fillna(df['num'].mode()[0], inplace=True)  


df["VIP"].fillna(df["VIP"].mode()[0], inplace=True)


spending_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

df.loc[df['CryoSleep'] == True, spending_cols] = 0

for col in spending_cols:
    median_val = df[col].mode()[0]  
    df[col] = np.where(
        (df['CryoSleep'] == False) & (df[col].isna()), 
        median_val,
        df[col]
    )


print(df.dtypes)


df = df.drop(['PassengerId', 'Group'], axis=1)


df = pd.get_dummies(df, columns=["HomePlanet"])
df = pd.get_dummies(df, columns=["Destination"])


df["CryoSleep"] = df["CryoSleep"].map({True: 1, False: 0}).astype(int)
df["VIP"] = df["VIP"].map({True: 1, False: 0}).astype(int)

print(df.dtypes)


#df[['deck', 'num', 'side']] = df['Cabin'].str.split('/', expand=True)
df = pd.get_dummies(df, columns=['deck', 'side'], drop_first=True)


bool_cols = df.select_dtypes(include='bool').columns
df[bool_cols] = df[bool_cols].astype(int)
print(df.dtypes)


#df.drop("Cabin", axis=1, inplace=True)

df['num'] = pd.to_numeric(df['num'], errors='coerce')
print(df.dtypes)


display(df)


from sklearn.model_selection import train_test_split

X = df.drop("Transported", axis=1)    
y = df["Transported"]             

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier


models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "SVM": SVC(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "CatBoost": CatBoostClassifier(random_seed=42, verbose=0)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc


results_df = pd.DataFrame(results.items(), columns=["Model", "Accuracy"])
results_df.sort_values("Accuracy", ascending=False, inplace=True)
print(results_df)


cat_model = CatBoostClassifier(
    random_seed=42,
    verbose=0 
)

cat_model.fit(X_train, y_train)

y_pred = cat_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"CatBoost Baseline Accuracy: {acc:.4f}")


params = {
    'iterations': [100, 200],
    'depth': [4, 6],
    'learning_rate': [0.01, 0.1],
    'l2_leaf_reg': [1, 3]  
}

from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(cat_model, params, cv=3, scoring='accuracy')
grid.fit(X_train, y_train)

print("Best Params:", grid.best_params_)
print("Best Accuracy:", grid.best_score_)


feat_importances = cat_model.get_feature_importance()
feature_names = X_train.columns
plt.barh(feature_names, feat_importances)
plt.title("CatBoost Feature Importance")
plt.show()

