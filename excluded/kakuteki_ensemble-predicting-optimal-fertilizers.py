import pandas as pd
from collections import Counter

# 読み込み元パス
path_ds = '/kaggle/input/16-june-2025-fertilizer-18'

# モデル名（列名）を定義
solut_names = ['Gaurav_Dutta','M_Naumov','Mahog','Mahog_GEN']

def load_submissions(path=path_ds):
    # 各提出ファイルの読み込み
    df1  = pd.read_csv(f"{path}/submission__LB_0_36995__v01__Gaurav_Dutta.csv")
    df2  = pd.read_csv(f"{path}/submission__LB_0_36964__v03__M_Naumov.csv")
    df3  = pd.read_csv(f"{path}/submission__LB_0_37228__v01__Mahog.csv")
    df3g = pd.read_csv(f"{path}/submission__LB_0_37228__v01__Mahog__GEN.csv")

    # 列名変更（統一）
    df1.rename(columns={'Fertilizer Name': 'Gaurav_Dutta'}, inplace=True)
    df2.rename(columns={'Fertilizer Name': 'M_Naumov'}, inplace=True)
    df3.rename(columns={'Fertilizer Name': 'Mahog'}, inplace=True)
    df3g.rename(columns={'Fertilizer Name': 'Mahog_GEN'}, inplace=True)

    # 結合
    df = df1[['id', 'Gaurav_Dutta']].merge(
        df2[['id', 'M_Naumov']], on='id'
    ).merge(
        df3[['id', 'Mahog']], on='id'
    ).merge(
        df3g[['id', 'Mahog_GEN']], on='id'
    )
    return df

def majority_vote(row):
    # すべての予測値を取得
    preds = [row[col] for col in solut_names]
    # 最頻値を取得（多数決）
    count = Counter(preds)
    # 同率トップが複数ある場合もあるのでソートして上位をとる
    top = count.most_common()
    if top[0][1] >= 2:
        return top[0][0]  # 2票以上ある最頻値
    else:
        return row['Mahog_GEN']  # バックアップ戦略としてMahog_GENを使う

# サブミッションを読み込み
df = load_submissions()

# 多数決を実行し、新しい予測列を生成
df['Fertilizer Name'] = df.apply(majority_vote, axis=1)

# 必要な形式に変換して保存
df_final = df[['id', 'Fertilizer Name']]
df_final.to_csv("submission.csv", index=False)


