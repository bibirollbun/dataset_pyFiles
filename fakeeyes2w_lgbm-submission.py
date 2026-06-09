import pandas as pd


drop_list = ["IDTeams","WTeamID","LTeamID","Team1","Team2","ST","SeedSum","HigherSeed","Phase","PhaseSeedInteraction","MatchID","SeedTier1","SeedTier2"]

games = pd.read_csv("/kaggle/input/afterfeatureengineer/games.csv")
sub = pd.read_csv("/kaggle/input/afterfeatureengineer/sub.csv")
games.drop(columns=drop_list,inplace=True)

common_columns = list(set(games.columns).intersection(set(sub.columns)))

games_common = games[common_columns]
sub_common = sub[common_columns]

filtered_columns = [col for col in games_common.columns if not col.strip().endswith("c_score")]
print(filtered_columns)
filtered_columns = [col for col in games.columns if not col.strip().endswith("c_score")]
print(filtered_columns)


# print("Common Columns:", common_columns)
# print("\nGames Dataframe with Common Columns:")
# print(games_common.head())
# print("\nSub Dataframe with Common Columns:")
# print(sub_common.head())
X_total = games_common.drop(columns=["Pred", "Season","ID"])
y_total = games_common['Pred']
X_val = sub_common.drop(columns=["Pred", "Season","ID"])
# Train Season <= 2023
train_data = games_common.query("Season <= 2023")
X_train = train_data.drop(columns=["Pred", "Season", "ID"])
y_train = train_data['Pred']

# test：Season == 2024
test_data = games_common.query("Season > 2023")
X_test = test_data.drop(columns=["Pred", "Season", "ID"])
y_test = test_data['Pred']





import lightgbm as lgb
from sklearn.metrics import brier_score_loss
from matplotlib import pyplot as plt


lgb_model = lgb.LGBMClassifier()
lgb_model.fit(X_train, y_train)

y_pred_prob = lgb_model.predict_proba(X_test)[:, 1]  # positive prob
brier_score = brier_score_loss(y_test, y_pred_prob)

print(f"Brier Score: {brier_score}")

lgb.plot_importance(lgb_model, max_num_features=50, figsize=(10, 8), importance_type='split')
plt.show()



del lgb_model
ID = sub['ID']
lgb_model = lgb.LGBMClassifier()
lgb_model.fit(X_total,y_total)
y_val_pred = lgb_model.predict_proba(X_val)[:,1]
submission = pd.DataFrame({'ID': ID, 'Pred': y_val_pred})
print(submission.head())
submission.to_csv("submission.csv",index=False)




