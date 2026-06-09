import warnings

warnings.simplefilter("ignore")

import os
import numpy as np
import pandas as pd

pd.set_option("display.expand_frame_repr", False)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 6)
pd.set_option("display.width", None)

data_dir = f"../input/march-machine-learning-mania-2025"


def create_raw():
    raw = pd.DataFrame()
    
    for fn in ["MNCAATourney", "MRegularSeason", "WNCAATourney", "WRegularSeason"]:
        raw = pd.concat(
            [
                raw,
                pd.read_csv(
                    f"{data_dir}/{fn}DetailedResults.csv",
                ),
            ]
        ).reset_index(drop=True)
    
    for c in raw.select_dtypes("int64"):
        raw[c] = raw[c].astype("int32")

    return raw

raw = create_raw()

print(f"raw {raw.shape}")


def create_tar(raw):
    tar = raw[["Season", "DayNum", "WTeamID", "LTeamID"]]
    tar["WMargin"] = raw["WScore"] - raw["LScore"]
    tar["LMargin"] = -tar["WMargin"]
    
    return pd.concat(
        [
            tar.rename(
                columns={
                    "WTeamID": "TeamID",
                    "LTeamID": "OppID",
                    "WMargin": "Margin",
                }
            ).drop(columns="LMargin"),
            tar.rename(
                columns={
                    "LTeamID": "TeamID",
                    "WTeamID": "OppID",
                    "LMargin": "Margin",
                }
            ).drop(columns="WMargin"),
        ]
    ).reset_index(drop=True)

tar = create_tar(raw)

print(f"tar {tar.shape}\n")
print(tar)


def create_gam(raw):
    gam = raw.copy()
    gam.loc[gam["WLoc"] == "A", "WLoc_"] = -1
    gam.loc[gam["WLoc"] == "N", "WLoc_"] = 0
    gam.loc[gam["WLoc"] == "H", "WLoc_"] = 1
    gam = gam.drop(columns="WLoc").rename(columns={"WLoc_": "WLoc"})
    gam["WLoc"] = gam["WLoc"].astype("int32")
    gam["LLoc"] = -gam["WLoc"]
    
    gam = pd.concat(
        [
            gam.rename(columns={c: f"{c[1:]}_gu" for c in gam if c[0] == "W"}).rename(
                columns={c: f"{c[1:]}_gt" for c in gam if c[0] == "L"}
            ),
            gam.rename(columns={c: f"{c[1:]}_gu" for c in gam if c[0] == "L"}).rename(
                columns={c: f"{c[1:]}_gt" for c in gam if c[0] == "W"}
            ),
        ]
    ).reset_index(drop=True)
    
    gam = gam.rename(columns={"TeamID_gu": "TeamID", "TeamID_gt": "OppID"})
    gam["NumOT_gu"] = gam["NumOT"]
    return gam.drop(columns=["NumOT", "Loc_gt"])

gam = create_gam(raw)

print(f"gam {gam.shape}\n")
print(gam[[c for c in gam if c[-3:-1] != "_g"]])
print()
print(gam[[c for c in gam if c[-3:] == "_gu"]])
print()
print(gam[[c for c in gam if c[-3:] == "_gt"]])


def create_sea(gam):
    sea = (
        gam.groupby(["Season", "TeamID"])
        .agg(
            dict(
                **{
                    c: "sum"
                    for c in gam.drop(
                        columns=[
                            "Season",
                            "TeamID",
                            "OppID",
                            "DayNum",
                        ]
                    )
                },
                DayNum="count",
            )
        )
        .reset_index()
    )
    
    sea["DayNum"] = sea["DayNum"].astype("int32")
    
    sea = sea.rename(columns={c: f"{c[:-2]}s{c[-1]}" for c in sea if c[-3:-1] == "_g"})
    
    return sea.rename(columns={"DayNum": "Games_su"})

sea = create_sea(gam)

print(f"sea {sea.shape}\n")
print(sea[[c for c in sea if c[-3:-1] != "_s"]])
print()
print(sea[[c for c in sea if c[-3:] == "_su"]])
print()
print(sea[[c for c in sea if c[-3:] == "_st"]])


def create_gsx(gam, sea):
    gsx = pd.merge(gam, sea, on=["Season", "TeamID"])
    
    gsx = gsx.rename(columns={c: f"T_{c}" for c in gsx if c[-3:-1] == "_s"})
    
    gsx = pd.merge(
        gsx,
        sea,
        left_on=["Season", "OppID"],
        right_on=["Season", "TeamID"],
        suffixes=["", "_fromseason"],
    )
    
    gsx = gsx.drop(columns="TeamID_fromseason")
    
    return gsx.rename(
        columns={c: f"O_{c}" for c in gsx if c[:2] != "T_" and c[-3:-1] == "_s"}
    )

gsx = create_gsx(gam, sea)

print(f"gsx {gsx.shape}\n")
print(gsx[[c for c in gsx if c[-3:-1] not in ("_g", "_s")]])
print()
print(gsx[[c for c in gsx if c[-3:] == "_gu"]])
print()
print(gsx[[c for c in gsx if c[-3:] == "_gt"]])
print()
print(gsx[[c for c in gsx if c[:2] == "T_" and c[-3:] == "_su"]])
print()
print(gsx[[c for c in gsx if c[:2] == "T_" and c[-3:] == "_st"]])
print()
print(gsx[[c for c in gsx if c[:2] == "O_" and c[-3:] == "_su"]])
print()
print(gsx[[c for c in gsx if c[:2] == "O_" and c[-3:] == "_st"]])


def create_aop(gsx):
    aop = (
        gsx.groupby(["Season", "TeamID"])[[c for c in gsx if c[:2] == "O_"]]
        .sum()
        .reset_index()
    )
    
    return aop.rename(columns={c: f"{c[2:-2]}a{c[-1]}" for c in aop if c[-3:-1] == "_s"})

aop = create_aop(gsx)

print(f"aop {aop.shape}\n")
print(aop[[c for c in aop if c[-3:-1] != "_a"]])
print()
print(aop[[c for c in aop if c[-3:] == "_au"]])
print()
print(aop[[c for c in aop if c[-3:] == "_at"]])


stats = [c[:-3] for c in gam if c[-3:] == "_gu" and c not in ("Loc_gu", "NumOT_gu")]

def create_gsa(gsx, aop, remove_game):
    gsa = pd.merge(gsx, aop, on=["Season", "TeamID"])
    gsa = gsa.rename(columns={c: f"T_{c}" for c in gsa if c[-3:-1] == "_a"})
    gsa = pd.merge(gsa, aop, left_on=["Season", "OppID"], right_on=["Season", "TeamID"], suffixes=["", "_"])
    gsa = gsa.drop(columns=["TeamID_"])
    gsa = gsa.rename(columns={c: f"O_{c}" for c in gsa if c[:2] != "T_" and c[-3:-1] == "_a"})
    
    if remove_game:
        for stat in stats:
            gsa[f"T_{stat}_su"] = gsa[f"T_{stat}_su"] - gsa[f"{stat}_gu"]
            gsa[f"T_{stat}_st"] = gsa[f"T_{stat}_st"] - gsa[f"{stat}_gt"]
        
            gsa[f"O_{stat}_su"] = gsa[f"O_{stat}_su"] - gsa[f"{stat}_gt"]
            gsa[f"O_{stat}_st"] = gsa[f"O_{stat}_st"] - gsa[f"{stat}_gu"]
        
            gsa[f"T_{stat}_au"] = gsa[f"T_{stat}_au"] - gsa[f"{stat}_gt"]
            gsa[f"T_{stat}_at"] = gsa[f"T_{stat}_at"] - gsa[f"{stat}_gu"]

            gsa[f"T_{stat}_au"] = gsa[f"T_{stat}_au"] - gsa[f"{stat}_gt"]
            gsa[f"T_{stat}_at"] = gsa[f"T_{stat}_at"] - gsa[f"{stat}_gu"]
        
        stat = "Loc"
        gsa[f"T_{stat}_su"] = gsa[f"T_{stat}_su"] - gsa[f"{stat}_gu"]
        gsa[f"O_{stat}_su"] = gsa[f"O_{stat}_su"] + gsa[f"{stat}_gu"]
        gsa[f"T_{stat}_au"] = gsa[f"T_{stat}_au"] + gsa[f"{stat}_gu"]
        gsa[f"O_{stat}_au"] = gsa[f"O_{stat}_au"] - gsa[f"{stat}_gu"]

        stat = "NumOT"
        gsa[f"T_{stat}_su"] = gsa[f"T_{stat}_su"] - gsa[f"{stat}_gu"]
        gsa[f"O_{stat}_su"] = gsa[f"O_{stat}_su"] - gsa[f"{stat}_gu"]
        gsa[f"T_{stat}_au"] = gsa[f"T_{stat}_au"] - gsa[f"{stat}_gu"]
        gsa[f"O_{stat}_au"] = gsa[f"O_{stat}_au"] - gsa[f"{stat}_gu"]
        
        gsa[f"T_Games_su"] -= 1
        gsa[f"O_Games_su"] -= 1
        gsa[f"T_Games_au"] -= 1
        gsa[f"O_Games_au"] -= 1
    
    return gsa.drop(columns=[c for c in gsa if c[-3:-1] == "_g"])

gsa = create_gsa(gsx, aop, True)

print(f"gsa {gsa.shape}\n")
print(gsa[[c for c in gsa if c[-3:-1] not in ("_s", "_a")]])
print()
print(gsa[[c for c in gsa if c[:2] == "T_" and c[-3:] == "_su"]])
print()
print(gsa[[c for c in gsa if c[:2] == "T_" and c[-3:] == "_st"]])
print()
print(gsa[[c for c in gsa if c[:2] == "O_" and c[-3:] == "_su"]])
print()
print(gsa[[c for c in gsa if c[:2] == "O_" and c[-3:] == "_st"]])
print()
print(gsa[[c for c in gsa if c[-3:] == "_au"]])
print()
print(gsa[[c for c in gsa if c[-3:] == "_at"]])


def create_pg(gsa):
    pg = gsa.copy()
    
    for stat in stats:
        pg[f"T_{stat}_su"] = (pg[f"T_{stat}_su"] / pg[f"T_Games_su"]).astype("float32")
        pg[f"T_{stat}_st"] = (pg[f"T_{stat}_st"] / pg[f"T_Games_su"]).astype("float32")
    
        pg[f"O_{stat}_su"] = (pg[f"O_{stat}_su"] / pg[f"O_Games_su"]).astype("float32")
        pg[f"O_{stat}_st"] = (pg[f"O_{stat}_st"] / pg[f"O_Games_su"]).astype("float32")
    
        pg[f"T_{stat}_au"] = (pg[f"T_{stat}_au"] / pg[f"T_Games_au"]).astype("float32")
        pg[f"T_{stat}_at"] = (pg[f"T_{stat}_at"] / pg[f"T_Games_au"]).astype("float32")

        pg[f"O_{stat}_au"] = (pg[f"O_{stat}_au"] / pg[f"O_Games_au"]).astype("float32")
        pg[f"O_{stat}_at"] = (pg[f"O_{stat}_at"] / pg[f"O_Games_au"]).astype("float32")
    
    for stat in ["Loc", "NumOT"]:
        pg[f"T_{stat}_su"] = (pg[f"T_{stat}_su"] / pg[f"T_Games_su"]).astype("float32")
        pg[f"O_{stat}_su"] = (pg[f"O_{stat}_su"] / pg[f"O_Games_su"]).astype("float32")
        pg[f"T_{stat}_au"] = (pg[f"T_{stat}_au"] / pg[f"T_Games_au"]).astype("float32")
        pg[f"O_{stat}_au"] = (pg[f"O_{stat}_au"] / pg[f"O_Games_au"]).astype("float32")
    
    return pg.drop(columns=["T_Games_su", "O_Games_su", "T_Games_au", "O_Games_au"])

pg = create_pg(gsa)

print(f"pg {pg.shape}\n")
print(pg[[c for c in pg if c[-3:-1] not in ("_s", "_a")]])
print()
print(pg[[c for c in pg if c[:2] == "T_" and c[-3:] == "_su"]])
print()
print(pg[[c for c in pg if c[:2] == "T_" and c[-3:] == "_st"]])
print()
print(pg[[c for c in pg if c[:2] == "O_" and c[-3:] == "_su"]])
print()
print(pg[[c for c in pg if c[:2] == "O_" and c[-3:] == "_st"]])
print()
print(pg[[c for c in pg if c[-3:] == "_au"]])
print()
print(pg[[c for c in pg if c[-3:] == "_at"]])


# df = gam
# na_counts = df.isna().sum()
# columns_with_na_counts = na_counts[na_counts > 0]
# print(columns_with_na_counts)


def create_train_test(tar, pg):
    df = pd.merge(tar, pg, on=["Season", "DayNum", "TeamID"], suffixes=["", "_"])
    df = df.drop(columns="OppID_")
    df["Women"] = pd.Series(0, index=df.index, dtype="int32")
    df.loc[df["TeamID"] >= 3000, "Women"] = 1
    print(df["Margin"])
    df.to_csv(f"train-net-final-pre-1.csv", index=False)
    return df

train = create_train_test(tar, pg)

print(f"train {train.shape}")
print(train)


stage2 = pd.read_csv("../input/march-machine-learning-mania-2025/SampleSubmissionStage2.csv", usecols=["ID"])
stage2[["Season", "TeamID", "OppID"]] = stage2["ID"].str.split("_", expand=True).astype("int32")
stage2["DayNum"] = pg["DayNum"].max()

print(stage2)
# gsx = create_gsx(gam, sea)
# aop = create_aop(gsx)
gsa_test = create_gsa(gsx, aop, False)
gsa_test = gsa_test[gsa_test["Season"] == 2025]
pg_test = create_pg(gsa_test)
# df_test = create_train_test(None, pg_test)
df_test = pg_test.copy()
df_test["Women"] = pd.Series(0, index=df_test.index, dtype="int32")
df_test.loc[df_test["TeamID"] >= 3000, "Women"] = 1

print(f"df_test {df_test.shape}")


print(df_test.columns.to_list())


foo = df_test[
    ["Season", "TeamID", "Women"] + 
    [c for c in df_test if c[:2] == "T_"]
]
foo = foo.drop_duplicates()
foo = foo.sort_values("TeamID").reset_index(drop=True)
print(f"foo {foo.shape}\n{foo}\n")

bar = df_test[
    ["Season", "OppID", "Women"] + 
    [c for c in df_test if c[:2] == "O_"]
]
# bar = bar.rename(columns=dict(OppID="TeamID", **{c: f"T_{c[2:]}" for c in bar if c[:2] == "O_"}))
bar = bar.drop_duplicates()
bar = bar.sort_values("OppID").reset_index(drop=True)

print(f"bar {bar.shape}\n{bar}")


print(foo["TeamID"].nunique())
print(bar["OppID"].nunique())
print()
print(foo[["Season", "TeamID"]].value_counts())
print()
print(bar[["Season", "OppID"]].value_counts())


test = pd.merge(stage2, foo, on=["Season", "TeamID"])
test = pd.merge(test, bar.drop(columns=["Women"]), on=["Season", "OppID"])
test.to_csv(f"test-net-final-pre-1.csv", index=False)
display(test.head(100))


test




