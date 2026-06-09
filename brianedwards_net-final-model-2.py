!nvcc --version
# !pip3 install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu120
# !pip3 install -U numpy pandas scikit-learn catboost
# !pip3 install -U --extra-index-url=https://pypi.nvidia.com "cudf-cu12==25.2.*" "cuml-cu12==25.2.*"
print("net, as in neural net and basketball net ;-)")


import warnings

warnings.simplefilter("ignore")

import os
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import catboost as ctb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import cudf
from cuml.preprocessing import StandardScaler as CuStandardScaler

pd.set_option("display.expand_frame_repr", False)
pd.set_option("display.max_columns", None)
pd.set_option('display.max_rows', 6)
pd.set_option("display.width", None)


train = pd.read_csv("../input/net-final-dataset/train-net-final-pre-1.csv")

train = pd.concat([
    train.select_dtypes("int64").astype("int32"),
    train.select_dtypes("float64").astype("float32"),
], axis=1)

train["SeasonsAgo"] = (2025 - train["Season"]).astype("float32")
train["DayNum"] = train["DayNum"].astype("float32")

train.loc[train["TeamID"] < train["OppID"], "ID"] = \
    train["Season"].astype("str") + "_" + train["TeamID"].astype("str") + "_" + train["OppID"].astype("str")

train.loc[train["OppID"] < train["TeamID"], "ID"] = \
    train["Season"].astype("str") + "_" + train["OppID"].astype("str") + "_" + train["TeamID"].astype("str")

stage1 = pd.read_csv("../input/march-machine-learning-mania-2025/SampleSubmissionStage1.csv")
test1 = train[train['ID'].isin(stage1['ID'])].reset_index(drop=True)
train1 = train[~train['ID'].isin(stage1['ID'])].reset_index(drop=True)

sx = StandardScaler()
sx.fit(train.select_dtypes("float32"))
sy = StandardScaler()
sy.fit(train[["Margin"]])

print(f"train {train.shape}")
print(train.select_dtypes("int32"))
# print()
# print(f"test {test.shape}")
# print(test.select_dtypes("int32"))


# verify that we train in both directions
print(train[(train["Season"]==2003) & (train["TeamID"]==1411) & (train["OppID"]==1421)].iloc[:, :10])
print(train[(train["Season"]==2003) & (train["TeamID"]==1421) & (train["OppID"]==1411)].iloc[:, :10])


print(train)


tensor_kwargs = dict(dtype=torch.float32, device="cuda")

def tensor(data):
    return torch.tensor(data, **tensor_kwargs)

def to_X(df):
    if "T_margin_pred" in df:
        save = df[["T_margin_pred", "O_margin_pred"]]
        df = df.drop(columns=["T_margin_pred", "O_margin_pred"])
    new = pd.concat([
            pd.DataFrame(
                sx.transform(df.select_dtypes("float32")),
                index=df.index,
                columns=df.select_dtypes("float32").columns
            ),
            df[["Women"]],
        ],
        axis=1,
    )
    if "T_margin_pred" in df:
        new[["T_margin_pred", "O_margin_pred"]] = save
    return new

def X_tensor(df):
    return tensor(to_X(df).values)

def weight(*size):
    fan_in = size[0]
    std = np.sqrt(2.0 / fan_in)
    return nn.Parameter(std * torch.randn(*size, **tensor_kwargs))

def zeros(*size):
    return torch.zeros(*size, **tensor_kwargs)

def bias(*size):
    return nn.Parameter(zeros(*size))

def forward(m, X_i):
    y_pred = X_i
    for j, (w, b) in enumerate(m):
        y_pred = y_pred @ w + b
        if j < (len(m)-1):
            y_pred = F.leaky_relu(y_pred, negative_slope=0.1)
    return y_pred

mse_ = torch.nn.MSELoss()

def mse(y_pred_epoch, y_i):
    return  mse_(y_pred_epoch, y_i.view(-1, 1))

def aslist(param):
    return param.cpu().detach().numpy().tolist()

def aspy(m):
    return [(aslist(w), aslist(b)) for w, b in m]

def scale_back_to_margin(sy, y_pred):
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    return sy.inverse_transform(y_pred.reshape(-1, 1)).flatten()

def brier_score(margin_true, margin_pred):
    win_true = (margin_true > 0).astype("int32")
    win_prob_pred = 1 / (1 + np.exp(-margin_pred * 0.175))
    return np.mean((win_prob_pred - win_true) ** 2)

def train_nn(train):
    X = X_tensor(train)
    print(f"X {X.shape}")
    y = tensor(sy.transform(train[["Margin"]]))
    print(f"y {y.shape}")
    d = [X.shape[1], 64, 32, 16, y.shape[1]]
    n_epochs = 10_000
    patience = 60
    kfold = KFold(shuffle=True, random_state=42)
    y_pred_oof = torch.zeros(y.shape[0], dtype=torch.float32, device="cuda")
    models = []

    for fold_n, (i_fold, i_oof) in enumerate(kfold.split(X), 1):
        print(f"  fold {fold_n}")
        start = datetime.now()

        m = [(  weight(d[i], d[i+1]),
                bias(d[i+1]),
            )
            for i in range(len(d)-1)]

        optim = torch.optim.Adam(
            [h[0] for h in m] + [h[1] for h in m],
            weight_decay=1e-4)

        for epoch_n in range(1, n_epochs + 1):
            y_pred_epoch_fold = forward(m, X[i_fold])
            mse_epoch_fold = mse(y_pred_epoch_fold, y[i_fold])
            optim.zero_grad()
            mse_epoch_fold.backward()
            optim.step()

            with torch.no_grad():
                y_pred_epoch_oof = forward(m, X[i_oof])
                mse_epoch_oof = mse(y_pred_epoch_oof, y[i_oof])

            if epoch_n == 1 or m_best[0] > mse_epoch_oof:
                m_best = (mse_epoch_oof, 0, aspy(m))
            else:
                m_best = (m_best[0], m_best[1]+1, m_best[2])

            if ((epoch_n % (n_epochs // 100) == 0)
                    or (epoch_n > (n_epochs - 3))
                    or (m_best[1] > patience)):
                print(
                    f"    epoch {epoch_n:>6}: "
                    f"fold={mse_epoch_fold.item():.4f} "
                    f"oof={mse_epoch_oof.item():.4f}"
                )

            if m_best[1] > patience:
                print(f"    out of patience: oof={m_best[0]:.4f}")
                break

        with torch.no_grad():
            m = [(tensor(w), tensor(b)) for w, b in m_best[2]]
            y_pred_oof[i_oof] = forward(m, X[i_oof]).flatten()

        models.append(aspy(m))
        t = (datetime.now() - start).total_seconds()
        print(f"  done fold {fold_n} {t} seconds")

    margin_pred_oof = scale_back_to_margin(sy, y_pred_oof)
    score = brier_score(train["Margin"], margin_pred_oof)
    print(f"nn oof brier score: {score:.4f}")
    return models, margin_pred_oof

def test_nn(test, models):
    X = X_tensor(test)
    y_pred = zeros(X.shape[0])

    for m_py in models:
        m = [(tensor(w), tensor(b)) for w, b in m_py]
        with torch.no_grad():
            y_pred += forward(m, X).flatten()

    margin_pred = scale_back_to_margin(sy, y_pred/len(models))
    score = brier_score(test["Margin"], margin_pred)
    print(f"nn test brier score: {score:.4f}")
    return margin_pred


def predict_nn(test, models):
    X = X_tensor(test)
    y_pred = zeros(X.shape[0])

    for m_py in models:
        m = [(tensor(w), tensor(b)) for w, b in m_py]
        with torch.no_grad():
            y_pred += forward(m, X).flatten()

    margin_pred = scale_back_to_margin(sy, y_pred/len(models))
    return margin_pred


nn_models1, margin_pred_oof1 = train_nn(train1)
nn_margin_pred1 = test_nn(test1, nn_models1)


def train_ctb(train):
    X = cudf.DataFrame.from_pandas(to_X(train))
    print(f"X {X.shape}")
    
    y = cudf.DataFrame.from_pandas(pd.DataFrame(
            sy.transform(train[["Margin"]]),
            index=train.index,
            columns=["Margin"],
    ))["Margin"]

    print(f"y {y.shape}")
    
    kfold = KFold(shuffle=True, random_state=42)
    y_pred_oof = cudf.Series(np.zeros(len(y), dtype=np.float32))
    models = []

    for fold_n, (i_fold, i_oof) in enumerate(kfold.split(X), 1):
        print(f"  fold {fold_n}")
        start = datetime.now()

        m = ctb.CatBoostRegressor(
            task_type="GPU",
            devices="0",
            
            cat_features=["Women"],
            
            learning_rate=0.02,
            depth=8,
            min_data_in_leaf=20,
            l2_leaf_reg=3.0,

            # bootstrap_type="Bernoulli",
            # bootstrap_type="Bayesian",
            
            # subsample=0.8,
            # colsample_bylevel=0.8,

            iterations=10000,
            
            # early_stopping_rounds=60,
            od_type="Iter",
            od_wait=60,
            
            random_seed=42,
        )
        
        m.fit(
            X.iloc[i_fold].to_pandas(), y[i_fold].to_pandas(),
            eval_set=[(X.iloc[i_oof].to_pandas(), y[i_oof].to_pandas())],
            verbose=100
        )
        
        y_pred_oof.iloc[i_oof] = m.predict(X.iloc[i_oof].to_pandas())
        models.append(m)
        print(f"    best iteration: {m.best_iteration_}, oof rmse: {m.best_score_}")
        t = (datetime.now() - start).total_seconds()
        print(f"  done fold {fold_n} {t} seconds")


    margin_pred_oof = scale_back_to_margin(sy, y_pred_oof.values.get())
    score = brier_score(train["Margin"].values, margin_pred_oof)
    print(f"ctb oof brier score: {score:.4f}")
    return models, margin_pred_oof

def test_ctb(test, models):
    X = to_X(test)
    y_pred = np.zeros(len(X), dtype=np.float32)
    
    for m in models:
        y_pred += m.predict(X)
    
    margin_pred = scale_back_to_margin(sy, y_pred/len(models))
    score = brier_score(test["Margin"].values, margin_pred)
    print(f"ctb test brier score: {score:.4f}")
    return margin_pred


def predict_ctb(test, models):
    X = to_X(test)
    y_pred = np.zeros(len(X), dtype=np.float32)
    
    for m in models:
        y_pred += m.predict(X)
    
    margin_pred = scale_back_to_margin(sy, y_pred/len(models))
    return margin_pred


ctb_models1, ctb_margin_pred_oof1 = train_ctb(train1)
ctb_margin_pred1 = test_ctb(test1, ctb_models1)


nn_margin_pred1 = test_nn(test1, nn_models1)
ctb_margin_pred1 = test_ctb(test1, ctb_models1)
margin_pred1 = np.mean([nn_margin_pred1, ctb_margin_pred1], axis=0)
score1 = brier_score(test1["Margin"].values, margin_pred1)
print(f"ensemble test brier score: {score1:.4f}")


test = pd.read_csv("../input/net-final-dataset/test-net-final-pre-1.csv")

test = pd.concat([
    test.select_dtypes("int64").astype("int32"),
    test.select_dtypes("float64").astype("float32"),
], axis=1)

test["SeasonsAgo"] = 0
test["SeasonsAgo"] = test["SeasonsAgo"].astype("float32")
test["DayNum"] = test["DayNum"].astype("float32")

test_flipped = test.rename(columns=dict(
    TeamID="OppID",
    OppID="TeamID",
    **{c: f"O_{c[2:]}" for c in test if c[:2] == "T_"},
    **{c: f"T_{c[2:]}" for c in test if c[:2] == "O_"},
))

test = pd.concat([
    test,
    test_flipped,
])

print(f"test {test.shape}")
print(test.select_dtypes("int32"))

test_ = test.copy()
test = test[to_X(train).columns]


# verify that we test in both directions
print(test_[(test_["TeamID"]==1411) & (test_["OppID"]==1421)].iloc[:, :10])
print()
print(test_[(test_["TeamID"]==1421) & (test_["OppID"]==1411)].iloc[:, :10])
print()
print(test_[(test_["TeamID"]==1411) & (test_["OppID"]==1421)].iloc[:, -10:])
print()
print(test_[(test_["TeamID"]==1421) & (test_["OppID"]==1411)].iloc[:, -10:])


nn_models, _ = train_nn(train)
# nn_margin_pred = predict_nn(test, nn_models)


nn_margin_pred = predict_nn(test, nn_models)


ctb_models, _ = train_ctb(train)
# ctb_margin_pred = predict_ctb(test, ctb_models)


ctb_margin_pred = predict_ctb(test, ctb_models)


mp = test_.copy()

# ensemble
mp["margin_pred"] = np.mean([nn_margin_pred, ctb_margin_pred], axis=0)

# average results for TeamID<OppID and OppID<TeamID
mp_unflipped = mp[mp["OppID"] < mp["TeamID"]]

mp_unflipped = mp_unflipped.rename(columns=dict(
    TeamID="OppID",
    OppID="TeamID",
    **{c: f"O_{c[2:]}" for c in test if c[:2] == "T_"},
    **{c: f"T_{c[2:]}" for c in test if c[:2] == "O_"},
))

mp_unflipped["margin_pred"] *= -1

mp = pd.concat([
    mp[mp["TeamID"] < mp["OppID"]],
    mp_unflipped,
])

mp = mp.groupby(["Season", "TeamID", "OppID"])["margin_pred"].mean().reset_index()
print(f"mp {mp.shape}")
print(mp)


sub = pd.DataFrame()
sub["ID"] = "2025_" + mp["TeamID"].astype("str") + "_" + mp["OppID"].astype("str")
sub["Pred"] = 1 / (1 + np.exp(-mp["margin_pred"] * 0.175))
sub = sub.sort_values("ID").reset_index(drop=True)
sub.to_csv("submission.csv", float_format='%.8f', index=False)


def do_foo(nn_margin_pred, ctb_margin_pred):
    # foo = test_[[c for c in test_ if c[:2] not in ("T_", "O_")]]
    # foo = foo.drop(columns=["Season", "SeasonsAgo", "DayNum", "Women"])
    # foo["nn_margin_pred"] = nn_margin_pred
    # foo["ctb_margin_pred"] = ctb_margin_pred
    # foo["margin_pred"] = np.mean([nn_margin_pred, ctb_margin_pred], axis=0)

    foo = mp.drop(columns=["Season"])
    
    quux = foo.copy()
    print(foo.shape)
    bar = foo.rename(columns={"TeamID": "OppID", "OppID": "TeamID"})
    bar["margin_pred"] *= -1
    foo = pd.concat([foo, bar])
    foo = foo.drop(columns=["OppID"])
    # foo = foo[foo["margin_pred"] > 0]
    foo = foo.groupby("TeamID")["margin_pred"].sum().reset_index().sort_values("margin_pred", ascending=False).reset_index(drop=True)

    sfoo = foo.copy()
    mask = sfoo["TeamID"]<3000
    sfoo.loc[mask, "margin_pred"] = StandardScaler().fit_transform(sfoo.loc[mask, ["margin_pred"]])
    sfoo.loc[~mask, "margin_pred"] = StandardScaler().fit_transform(sfoo.loc[~mask, ["margin_pred"]])
    # mfoo = foo[foo["TeamID"]<3000].reset_index(names=["Rank"])
    # mfoo["Rank"] = StandardScaler().fit_transform(mfoo[["Rank"]])
    # wfoo = foo[foo["TeamID"]>=3000].reset_index(names=["Rank"])
    # wfoo["Rank"] = StandardScaler().fit_transform(wfoo[["Rank"]])

    teams = pd.concat([
        pd.read_csv("../input/march-machine-learning-mania-2025/MTeams.csv", usecols=["TeamID", "TeamName"]),
        pd.read_csv("../input/march-machine-learning-mania-2025/WTeams.csv", usecols=["TeamID", "TeamName"])
    ])

    foo = pd.merge(foo, teams, on=["TeamID"])
    m25 = foo[foo["TeamID"]<3000]["TeamName"].head(80).values
    w25 = foo[foo["TeamID"]>=3000]["TeamName"].head(80).values

    print("Rank  Men              Women")
    print("------------------------------------")
    for rank, (mteam, wteam) in enumerate(zip(m25, w25), 1):
        print(f"{rank:>4}  {mteam:<15} {wteam}")
    return quux
        
quux = do_foo(nn_margin_pred, ctb_margin_pred)


spread = pd.read_csv("../input/net-final-dataset/point-spread.csv")
mask = spread["Fav_TeamID"] < spread["Dog_TeamID"]
listing1 = pd.merge(spread[mask], quux, left_on=["Fav_TeamID", "Dog_TeamID"], right_on=["TeamID", "OppID"])
listing2 = pd.merge(spread[~mask], quux, left_on=["Dog_TeamID", "Fav_TeamID"], right_on=["TeamID", "OppID"])
listing2["margin_pred"] *= -1
floo = pd.concat([listing1, listing2])
floo = floo[["Fav_TeamID", "Fav_TeamName", "Dog_TeamName", "PointSpread", "margin_pred"]]
floo["PctChange"] = np.abs((floo["margin_pred"] - np.abs(floo["PointSpread"])) / np.abs(floo["PointSpread"]))
floo = floo.sort_values("PctChange", ascending=False).reset_index(drop=True)

print("Vegas                                                     Brian's Pick         Boost (Spread)")
print("---------------------------------------------------------------------------------------------")

for row in list(floo.values):
    team_id, fav, dog, spread_, margin, pct_change = tuple(row)
    gender = "W" if team_id >= 3000 else "M"
    pick = fav if margin - spread_ > 0 else dog
    if pick == dog:
        margin *= -1
    margin = round(margin * 2) / 2
    spread_ *= -1
    margin *= -1
    print(f"{fav+' ('+gender+')':<19} {spread_:>5.1f} over {dog:<17}         {pick:<20}  {pct_change:>4.1f}  ({margin:>5.1f})")


# vegas_win_prob = spread[spread["Fav_TeamID"] < spread["Dog_TeamID"]]
# vegas_win_prob_flipped = spread[spread["Dog_TeamID"] < spread["Fav_TeamID"]]
# vegas_win_prob_flipped["PointSpread"] *= -1
# vegas_win_prob = pd.concat([vegas_win_prob, vegas_win_prob_flipped])
# vegas_win_prob = vegas_win_prob.rename(columns={"Fav_TeamID": "TeamID", "Dog_TeamID": "OppID"})
# vegas_win_prob["ID"] = "2025_" + vegas_win_prob["TeamID"].astype("str") + "_" + vegas_win_prob["OppID"].astype("str")
# vegas_win_prob["Pred"] = 1 / (1 + np.exp(-vegas_win_prob["PointSpread"] * 0.175))
# vegas_win_prob = vegas_win_prob[["ID", "Pred"]]
# vegas_win_prob = vegas_win_prob.sort_values(["ID"]).reset_index(drop=True)
# vegas_win_prob

# First, get matchups with point spreads in the right format
matchups = []

# Process each spread entry to ensure ID format has lower team ID first
for _, row in spread.iterrows():
   fav_id = row['Fav_TeamID']
   dog_id = row['Dog_TeamID']
   point_spread = row['PointSpread']
   
   # Determine which ID goes first in our ID format
   if fav_id < dog_id:
       # Favorite has lower ID, keep point spread as is
       id_str = f"2025_{fav_id}_{dog_id}"
       adjusted_spread = point_spread
   else:
       # Underdog has lower ID, flip the point spread
       id_str = f"2025_{dog_id}_{fav_id}"
       adjusted_spread = -point_spread
   
   # Calculate win probability using logistic function
   win_prob = 1 / (1 + np.exp(-adjusted_spread * 0.175))
   matchups.append({"ID": id_str, "Pred": win_prob})

# Convert to DataFrame and sort
vegas_win_prob = pd.DataFrame(matchups)
vegas_win_prob = vegas_win_prob.sort_values("ID").reset_index(drop=True)
vegas_win_prob.values


sub2 = sub.copy()

vegas_dict = dict(zip(vegas_win_prob['ID'], vegas_win_prob['Pred']))

for idx in sub2.index:
    if sub2.at[idx, 'ID'] in vegas_dict:
        sub2.at[idx, 'Pred'] = vegas_dict[sub2.at[idx, 'ID']]

sub2 = sub2.sort_values("ID").reset_index(drop=True)

sub2.to_csv("submission2.csv", float_format='%.8f', index=False)
print(sub2)


print(sub2[sub2["ID"]=="2025_1104_1352"])
print(vegas_win_prob[vegas_win_prob["ID"]=="2025_1104_1352"])


match_count = sum(sub['ID'].isin(vegas_win_prob['ID']))
print(f"Matching IDs: {match_count}")


# Check how many unique IDs are in sub
print(f"Unique IDs in sub: {sub['ID'].nunique()}")

# Check how many unique IDs are in vegas_win_prob
print(f"Unique IDs in vegas_win_prob: {vegas_win_prob['ID'].nunique()}")

# Check how many IDs match between the two
matching_ids = set(sub['ID']).intersection(set(vegas_win_prob['ID']))
print(f"Matching IDs count: {len(matching_ids)}")

# Look at some sample IDs that don't match
non_matching = set(vegas_win_prob['ID']) - set(sub['ID'])
print(f"Sample non-matching IDs: {list(non_matching)[:5] if non_matching else 'All match'}")

# Verify ID format in vegas_win_prob
for id_val in list(vegas_win_prob['ID'])[:5]:
    parts = id_val.split('_')
    if len(parts) == 3:
        team_id, opp_id = int(parts[1]), int(parts[2])
        print(f"ID: {id_val}, TeamID: {team_id}, OppID: {opp_id}, TeamID < OppID: {team_id < opp_id}")


sub3 = sub2.copy()
sub3['Pred'] = (np.random.random(len(sub3)) < sub3['Pred']).astype(int)
sub3 = sub3.sort_values("ID").reset_index(drop=True)

sub3.to_csv("submission3.csv", float_format='%.8f', index=False)
print(sub3.head())
print(sub3.tail())


sub4 = pd.read_csv("../input/net-final-dataset/submission-net-final-model-1-c.csv")
sub4[["Season", "TeamID", "OppID"]] = sub4["ID"].str.split("_", expand=True).astype("int32")

v = vegas_win_prob.copy()
v[["Season", "TeamID", "OppID"]] = v["ID"].str.split("_", expand=True).astype("int32")
v = v[v["TeamID"] < 3000]

teams = pd.read_csv("../input/march-machine-learning-mania-2025/MTeams.csv", usecols=["TeamID", "TeamName"])

blammo = pd.merge(v, sub4, how="left", on=["TeamID", "OppID"], suffixes=["_v", "_sub4"])
blammo = pd.merge(blammo, teams[["TeamID", "TeamName"]], how="left", on=["TeamID"])
blammo = pd.merge(blammo, teams[["TeamID", "TeamName"]].rename(columns={"TeamID": "OppID", "TeamName": "OppName"}), how="left", on=["OppID"])

mask = blammo["Pred_sub4"]==0
blammo["Pick"] = blammo["TeamName"]
blammo.loc[mask, "Pick"] = blammo.loc[mask, "OppName"]

blammo["Opp"] = blammo["OppName"]
blammo.loc[mask, "Opp"] = blammo.loc[mask, "TeamName"]

blammo["Pred_v_"] = blammo["Pred_v"]
blammo.loc[mask, "Pred_v_"] = 1 - blammo.loc[mask, "Pred_v_"]

blammo["Pred_sub4_"] = blammo["Pred_sub4"]
blammo.loc[mask, "Pred_sub4_"] = 1 - blammo.loc[mask, "Pred_sub4_"]

blammo["Pct_change"] = (blammo["Pred_sub4_"] - blammo["Pred_v_"]) / blammo["Pred_v_"]

blammo = blammo.sort_values("Pct_change", ascending=False).reset_index(drop=True)

blammo = blammo[["Pick", "Opp", "Pred_v_", "Pred_sub4_", "Pct_change"]]


for row in blammo.values:
    Pick, Opp, Pred_v_, Pred_sub4_, Pct_change = tuple(row)
    if Pred_v_ > 0.5:
        break
    print(f"{Pick:<15} {Pred_v_:.2f}  {Pct_change:>4.1f}  ({Opp})")

print("-"*40)

for row in blammo.values:
    Pick, Opp, Pred_v_, Pred_sub4_, Pct_change = tuple(row)
    if Pred_v_ <= 0.5:
        continue 
    print(f"{Pick:<15} {Pred_v_:.2f}  {Pct_change:>4.1f}  ({Opp})")




