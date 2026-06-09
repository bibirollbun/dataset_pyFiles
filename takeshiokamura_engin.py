%%time
try:
    from lifelines.utils import concordance_index
except ModuleNotFoundError:
    print('Installing lifelines...')
    !pip install -q /kaggle/input/pip-install-lifelines/autograd-1.7.0-py3-none-any.whl
    !pip install -q /kaggle/input/pip-install-lifelines/autograd-gamma-0.5.0.tar.gz
    !pip install -q /kaggle/input/pip-install-lifelines/interface_meta-1.3.0-py3-none-any.whl
    !pip install -q /kaggle/input/pip-install-lifelines/formulaic-1.0.2-py3-none-any.whl
    !pip install -q /kaggle/input/pip-install-lifelines/lifelines-0.30.0-py3-none-any.whl


import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter, PercentFormatter
import numpy as np
import xgboost
import catboost
import warnings
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from scipy.stats import rankdata

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, quantile_transform, FunctionTransformer, PolynomialFeatures, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

all_model_scores = {}

pd.options.display.max_columns = 1000


train = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/train.csv')
test = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/test.csv')
data_dictionary = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/data_dictionary.csv')
train.tail()


print(train.shape)
print(test.shape)
train['flag'] = 'train'
test['flag'] = 'test'
df = pd.concat([train, test])
print(df.shape)


# 数値型の列を自動検出
num_cols = df.select_dtypes(include=[np.number]).columns

# 各数値列について、欠損値フラグを作成
for col in num_cols:
    df[col + '_missing'] = df[col].isna().astype(int)
    df[col] = df[col].fillna(0)

print(df.shape)





df['dri_score'].value_counts()


# 小児患者に関してはリスク指標が適用されない
df.loc[df['dri_score'] == 'N/A - pediatric', 'pediatric_flag'] = 1
# 非悪性疾患の場合、従来の悪性疾患向けリスク指標が適用されない
df.loc[df['dri_score'] == 'N/A - non-malignant indicationc', 'non-malignant indicationc'] = 1
# 疾患分類ができなかったケース,欠損扱いにする
df.loc[df['dri_score'] == 'N/A - disease not classifiable', 'dri_score_missing'] = 1
# 細胞遺伝学情報未確定は別カテゴリ扱い
df.loc[df['dri_score'] == 'TBD cytogenetics', 'cytogenetics未確定'] = 1
# 急性骨髄性白血病で細胞遺伝学情報が欠損しているとき、リスク評価が難しい
# cytogenetics欠損フラグを付与する
df.loc[df['dri_score'] == 'High - TED AML case <missing cytogenetics', 'cytogenetics_missing'] = 1
df.loc[df['dri_score'] == 'Intermediate - TED AML case <missing cytogenetics', 'cytogenetics_missing'] = 1
# 病状なしはcytogenetics欠損と同じように処理
df.loc[df['dri_score'] == 'Very high' 'Missing disease status', 'cytogenetics_missing'] = 1
# 最後に欠損は欠損扱い
df.loc[df['dri_score'] == 'Missing disease status', 'dri_score_missing'] = 1
df.loc[df['dri_score'].isna(), 'dri_score_missing'] = 1

# fillna
df['pediatric_flag'] = df['pediatric_flag'].fillna(0)
df['non-malignant indicationc'] = df['non-malignant indicationc'].fillna(0)
df['dri_score_missing'] = df['dri_score_missing'].fillna(0)
df['cytogenetics未確定'] = df['cytogenetics未確定'].fillna(0)
df['cytogenetics_missing'] = df['cytogenetics_missing'].fillna(0)

dri_score_dict = {
    'Low':1,
    'Intermediate':2,
    'High':3,
    'Very high':4,
    'N/A - pediatric':0,
    'N/A - non-malignant indication':0,
    'N/A - disease not classifiable':0,
    'TBD cytogenetics':0,
    'High - TED AML case <missing cytogenetics':3,
    'Very high' 'Missing disease status':4,
    'Intermediate - TED AML case <missing cytogenetics':2,
    'Missing disease status':0
}
# rename
df['dri_score'] = df['dri_score'].map(dri_score_dict)
df['dri_score'] = df['dri_score'].fillna(0)


df['psych_disturb'].value_counts()


# Not doneは欠損(0)とし、それ以外は01でフラグ付
df.loc[df['psych_disturb'] == 'Yes', 'psych_disturb'] = 1
df.loc[df['psych_disturb'] == 'No', 'psych_disturb'] = 0
df.loc[df['psych_disturb'] == 'Not done', 'psych_disturb_missing'] = 1
df.loc[df['psych_disturb'] == 'Not done', 'psych_disturb'] = 0
df.loc[df['psych_disturb'].isna(), 'psych_disturb_missing'] = 1
df.loc[df['psych_disturb'].isna(), 'psych_disturb'] = 0
df['psych_disturb_missing'] = df['psych_disturb_missing'].fillna(0)


df['cyto_score'].value_counts()


# 欠損、不明情報は欠損として扱う
df.loc[df['cyto_score'] == 'TBD', 'cyto_score_missing'] = 1
df.loc[df['cyto_score'] == 'Other', 'cyto_score_missing'] = 1
df.loc[df['cyto_score'] == 'Not tested', 'cyto_score_missing'] = 1
df.loc[df['cyto_score'].isna(), 'cyto_score_missing'] = 1
df.loc[df['cyto_score'] == 'TBD', 'cyto_score'] = 0
df.loc[df['cyto_score'] == 'Other', 'cyto_score'] = 0
df.loc[df['cyto_score'] == 'Not tested', 'cyto_score'] = 0
df.loc[df['cyto_score'].isna(), 'cyto_score'] = 0
df['cyto_score_missing'] = df['cyto_score_missing'].fillna(0)

# マッピング
cyto_score_dict = {
    'Poor':3,
    'Intermediate':2,
    'Normal':2,
    'Favorable':1,
}
df['cyto_score'] = df['cyto_score'].map(cyto_score_dict)
df['cyto_score'] = df['cyto_score'].fillna(0)


# 糖尿病について
# Not doneは欠損(0)とし、それ以外は01でフラグ付
df.loc[df['diabetes'] == 'Yes', 'diabetes'] = 1
df.loc[df['diabetes'] == 'No', 'diabetes'] = 0
df.loc[df['diabetes'] == 'Not done', 'diabetes_missing'] = 1
df.loc[df['diabetes'] == 'Not done', 'diabetes'] = 0
df.loc[df['diabetes'].isna(), 'diabetes_missing'] = 1
df.loc[df['diabetes'].isna(), 'diabetes'] = 0
df['diabetes_missing'] = df['diabetes_missing'].fillna(0)


df['hla_match_c_high'].value_counts()
# 2が完全一致


df['hla_high_res_8'].value_counts()
# 希少値についてどう扱うか、が今後の宿題
# 値は大きい方が良い


df['tbi_status'].value_counts()


# TBI実施有無をフラグか
df.loc[df['tbi_status'] == 'No TBI', 'TBI_flag'] = 0
df.loc[df['tbi_status'] != 'No TBI', 'TBI_flag'] = 1

# 併用薬使用をフラグ化
df.loc[df['tbi_status'] == 'No TBI', '併用薬_flag'] = 0
df.loc[df['tbi_status'] != 'No TBI', '併用薬_flag'] = 1

# 放射方法の違いを抽出
df.loc[df['tbi_status'] == 'TBI +- Other, <=cGy', '放射線量'] = 1
df.loc[df['tbi_status'] == 'TBI +- Other, -cGy, single', '放射線量'] = 2
df.loc[df['tbi_status'] == 'TBI +- Other, -cGy, fractionated', '放射線量'] = 2
df.loc[df['tbi_status'] == 'TBI +- Other, >cGy', '放射線量'] = 3
df.loc[df['tbi_status'] == 'TBI +- Other, -cGy, unknown dose', '放射線量'] = 0
df.loc[df['tbi_status'] == 'TBI +- Other, -cGy, unknown dose', '放射線量_missing'] = 1
df.loc[df['tbi_status'] == 'TBI +- Other, unknown dose', '放射線量'] = 0
df.loc[df['tbi_status'] == 'TBI +- Other, unknown dose', '放射線量_missing'] = 1
df.loc[df['tbi_status'] == 'TBI + Cy +- Other', '放射線量'] = 0
df.loc[df['tbi_status'] == 'TBI + Cy +- Other', '放射線量_missing'] = 1

# 回数フラグ
df.loc[df['tbi_status'] == 'TBI +- Other, -cGy, single', '放射回数'] = 1
df.loc[df['tbi_status'] == 'TBI +- Other, -cGy, fractionated', '放射回数'] = 2

df['TBI_flag'] = df['TBI_flag'].fillna(0)
df['併用薬_flag'] = df['併用薬_flag'].fillna(0)
df['放射線量'] = df['放射線量'].fillna(0)
df['放射線量_missing'] = df['放射線量_missing'].fillna(0)
df['放射回数'] = df['放射回数'].fillna(0)

df = df.drop(columns = ['tbi_status'])


df['arrhythmia'].value_counts()


# 不整脈について
# Not doneは欠損(0)とし、それ以外は01でフラグ付
df.loc[df['arrhythmia'] == 'Yes', 'arrhythmia'] = 1
df.loc[df['arrhythmia'] == 'No', 'arrhythmia'] = 0
df.loc[df['arrhythmia'] == 'Not done', 'arrhythmia_missing'] = 1
df.loc[df['arrhythmia'] == 'Not done', 'arrhythmia'] = 0
df.loc[df['arrhythmia'].isna(), 'arrhythmia_missing'] = 1
df.loc[df['arrhythmia'].isna(), 'arrhythmia'] = 0
df['arrhythmia_missing'] = df['arrhythmia_missing'].fillna(0)


df['hla_low_res_6'].value_counts()


df['graft_type'].value_counts()


df.loc[df['graft_type'] == 'Peripheral blood', 'graft_type'] = 1
df.loc[df['graft_type'] == 'Bone marrow', 'graft_type'] = 0


df['vent_hist'].value_counts()


# 人工呼吸器使用歴について
# nullは欠損(0)とし、それ以外は01でフラグ付
df.loc[df['vent_hist'] == 'Yes', 'vent_hist'] = 1
df.loc[df['vent_hist'] == 'No', 'vent_hist'] = 0
df.loc[df['vent_hist'].isna(), 'vent_hist_missing'] = 1
df.loc[df['vent_hist'].isna(), 'vent_hist'] = 0
df['vent_hist_missing'] = df['vent_hist_missing'].fillna(0)


df['renal_issue'].value_counts()


# 腎機能障害について
# Not doneは欠損(0)とし、それ以外は01でフラグ付
df.loc[df['renal_issue'] == 'Yes', 'renal_issue'] = 1
df.loc[df['renal_issue'] == 'No', 'renal_issue'] = 0
df.loc[df['renal_issue'] == 'Not done', 'renal_issue_missing'] = 1
df.loc[df['renal_issue'] == 'Not done', 'renal_issue'] = 0
df.loc[df['renal_issue'].isna(), 'renal_issue_missing'] = 1
df.loc[df['renal_issue'].isna(), 'renal_issue'] = 0
df['renal_issue_missing'] = df['renal_issue_missing'].fillna(0)


df['pulm_severe'].value_counts()


# 肺障害について
# Not doneは欠損(0)とし、それ以外は01でフラグ付
df.loc[df['pulm_severe'] == 'Yes', 'pulm_severe'] = 1
df.loc[df['pulm_severe'] == 'No', 'pulm_severe'] = 0
df.loc[df['pulm_severe'] == 'Not done', 'pulm_severe_missing'] = 1
df.loc[df['pulm_severe'] == 'Not done', 'pulm_severe'] = 0
df.loc[df['pulm_severe'].isna(), 'pulm_severe_missing'] = 1
df.loc[df['pulm_severe'].isna(), 'pulm_severe'] = 0
df['pulm_severe_missing'] = df['pulm_severe_missing'].fillna(0)


df['prim_disease_hct'].value_counts()


# 各疾患をまとめて分類する辞書を作る
# 造血器腫瘍（blood_tumor）=1/0, 急性白血病（acute_leukemia）=1/0, リスク（risk）='high'/'intermediate'/'low' など
disease_groups = {
    'ALL': {
        'blood_tumor': 1,
        'acute_leukemia': 1,
        'risk': 3  # 急性リンパ性白血病は一般にリスク高め
    },
    'AML': {
        'blood_tumor': 1,
        'acute_leukemia': 1,
        'risk': 3  # 急性骨髄性白血病も高リスク扱い
    },
    'MDS': {
        'blood_tumor': 1,
        'acute_leukemia': 0,
        'risk': 2  # 骨髄異形成症候群はリスクが幅広いので中間とした
    },
    'MPN': {
        'blood_tumor': 1,
        'acute_leukemia': 0,
        'risk': 2  # 骨髄増殖性腫瘍の中でも真性多血症や骨髄線維症など様々
    },
    'Other acute leukemia': {
        'blood_tumor': 1,
        'acute_leukemia': 1,
        'risk': 3
    },
    'AI': {
        'blood_tumor': 0,
        'acute_leukemia': 0,
        'risk': 1  # Autoimmune系なら移植適応の場合でも悪性度は低めと仮定
    },
    'SAA': {
        'blood_tumor': 0,
        'acute_leukemia': 0,
        'risk': 2  # 重症再生不良性貧血は悪性ではないけど重症度高い
    },
    'IEA': {
        'blood_tumor': 0,
        'acute_leukemia': 0,
        'risk': 1  # 不明だが非悪性疾患と仮定
    },
    'NHL': {
        'blood_tumor': 1,
        'acute_leukemia': 0,
        'risk': 3  # 非ホジキンリンパ腫は種類によるが高リスク例も多い
    },
    'PCD': {
        'blood_tumor': 1,
        'acute_leukemia': 0,
        'risk': 3  # Plasma cell dyscrasia (多発性骨髄腫など) として高リスク扱い
    },
    'IIS': {
        'blood_tumor': 0,
        'acute_leukemia': 0,
        'risk': 1  # 先天性免疫不全などを想定し低〜中リスクで仮定
    },
    'HIS': {
        'blood_tumor': 1,
        'acute_leukemia': 0,
        'risk': 3  # Hemophagocytic syndrome(血球貪食症候群)など重症例多し
    },
    'Other leukemia': {
        'blood_tumor': 1,
        'acute_leukemia': 0,
        'risk': 2  # 明細不明の白血病なので中間にしておく
    },
    'Solid tumor': {
        'blood_tumor': 0,
        'acute_leukemia': 0,
        'risk': 3  # 固形腫瘍でHCTする場合は概して難治例が多い
    },
    'IMD': {
        'blood_tumor': 0,
        'acute_leukemia': 0,
        'risk': 1  # 遺伝性代謝疾患などを仮定
    },
    'HD': {
        'blood_tumor': 1,
        'acute_leukemia': 0,
        'risk': 3  # ホジキンリンパ腫(Hodgkin lymphoma)
    },
    'CML': {
        'blood_tumor': 1,
        'acute_leukemia': 0,
        'risk': 1  # 慢性期CMLを想定
    },
    'IPA': {
        'blood_tumor': 0,
        'acute_leukemia': 0,
        'risk': 1  # 正体不明につき非悪性と想定
    }
}

# 上記の辞書を使って新しいカラムを作成
df['blood_tumor'] = df['prim_disease_hct'].apply(lambda x: disease_groups[x]['blood_tumor'])
df['acute_leukemia'] = df['prim_disease_hct'].apply(lambda x: disease_groups[x]['acute_leukemia'])
df['risk'] = df['prim_disease_hct'].apply(lambda x: disease_groups[x]['risk'])

df = df.drop(columns = ['prim_disease_hct'])


df['hla_high_res_6'].value_counts()


df['cmv_status'].value_counts()


# 血清状態はドナーと受け手で別変数に
df.loc[df['cmv_status'] == '+/+', 'donor_CMV'] = 1
df.loc[df['cmv_status'] == '+/+', 'recipient_CMV'] = 1
df.loc[df['cmv_status'] == '-/+', 'donor_CMV'] = 0
df.loc[df['cmv_status'] == '-/+', 'recipient_CMV'] = 1
df.loc[df['cmv_status'] == '+/-', 'donor_CMV'] = 1
df.loc[df['cmv_status'] == '+/-', 'recipient_CMV'] = 0
df.loc[df['cmv_status'] == '-/-', 'donor_CMV'] = 0
df.loc[df['cmv_status'] == '-/-', 'recipient_CMV'] = 0
df.loc[df['cmv_status'].isna(), 'donor_CMV'] = 0
df.loc[df['cmv_status'].isna(), 'recipient_CMV'] = 0
df.loc[df['cmv_status'].isna(), 'donor_CMV_missing'] = 1
df.loc[df['cmv_status'].isna(), 'recipient_CMV_missing'] = 1

df['donor_CMV_missing'] = df['donor_CMV_missing'].fillna(0)
df['recipient_CMV_missing'] = df['recipient_CMV_missing'].fillna(0)

df = df.drop(columns = ['cmv_status'])


print(df['hla_high_res_10'].value_counts())
print(df['hla_match_dqb1_high'].value_counts())


df['tce_imm_match'].value_counts()


# 1. tce_imm_matchについて
# 「P/P」「G/G」「H/H」は完全一致とみなし1、不一致は0、
# 欠損値はそのままnan（または別フラグで扱う）にする

def calc_tce_match_flag(val):
    if pd.isna(val):
        return np.nan  # 不明としてnanにする
    # '/'で分割して左右が同じなら完全一致
    parts = val.split('/')
    if len(parts) == 2 and parts[0] == parts[1]:
        return 1
    else:
        return 0

df['tce_imm_match_binary'] = df['tce_imm_match'].apply(calc_tce_match_flag)

# また、欠損（不明）を示すフラグも作成する場合
df['tce_imm_match_missing'] = df['tce_imm_match'].isna().astype(int)


# 2. donor_tceについて
# 2. ドナー側と受け手側の情報を分割する（nanの場合はそのままnanになる）
# 文字列が'X/Y'形式の場合、左側をdonor, 右側をrecipientとして抽出
df[['donor_tce', 'recipient_tce']] = df['tce_imm_match'].str.split('/', expand=True)
# P -> 1, G -> 2, H -> 3, B -> 4 とし、欠損はnan
donor_tce_map = {'P': 1, 'G': 2, 'H': 3, 'B': 4}

def convert_donor_tce(val):
    if pd.isna(val):
        return np.nan  # 欠損値はnan
    return donor_tce_map.get(val, np.nan)

df['donor_tce'] = df['donor_tce'].apply(convert_donor_tce)
df['recipient_tce'] = df['recipient_tce'].apply(convert_donor_tce)

# donor_tceの欠損（不明）を示すフラグ
df['donor_tce_missing'] = df['donor_tce'].isna().astype(int)
df['recipient_tce_missing'] = df['recipient_tce'].isna().astype(int)

df = df.drop(columns = ['tce_imm_match'])


df['rituximab'].value_counts()


# 人コンディショニングでのリツキシマブ投与について
# nullは欠損(0)とし、それ以外は01でフラグ付
df.loc[df['rituximab'] == 'Yes', 'rituximab'] = 1
df.loc[df['rituximab'] == 'No', 'rituximab'] = 0
df.loc[df['rituximab'].isna(), 'rituximab_missing'] = 1
df.loc[df['rituximab'].isna(), 'rituximab'] = 0
df['rituximab_missing'] = df['rituximab_missing'].fillna(0)


df['prod_type'].value_counts()


df.loc[df['prod_type'] == 'PB', 'prod_type'] = 1
df.loc[df['prod_type'] == 'BM', 'prod_type'] = 0


df['cyto_score_detail'].value_counts()


# まず、良好、中間、不良をオーダナルエンコーディングするためのマッピングを定義
mapping = {
    'Favorable': 1,
    'Intermediate': 2,
    'Poor': 3
}

# オーダナル変数として新たな列を作成。該当しない値（TBD, Not tested, nan）はそのままNaNにする
df['cyto_score_detail_ord'] = df['cyto_score_detail'].map(mapping)

# 欠損フラグを作成。値がTBDまたはNot testedまたはNaNの場合は1、それ以外は0
def missing_flag(val):
    if pd.isna(val) or val in ['TBD', 'Not tested']:
        return 1
    else:
        return 0

df['cyto_score_detail_missing'] = df['cyto_score_detail'].apply(missing_flag)

df = df.drop(columns = ['cyto_score_detail'])


df['conditioning_intensity'].value_counts()


# まず、良好、中間、不良をオーダナルエンコーディングするためのマッピングを定義
mapping = {
    'NMA': 1,
    'RIC': 2,
    'MAC': 3
}

# オーダナル変数として新たな列を作成。該当しない値（TBD, Not tested, nan）はそのままNaNにする
df['conditioning_intensity_ord'] = df['conditioning_intensity'].map(mapping)

# 欠損フラグを作成。値がTBDまたはNot testedまたはNaNの場合は1、それ以外は0
def missing_flag(val):
    if pd.isna(val) or val in ['TBD', 'No drugs reported', 'N/A, F(pre-TED) not submitted']:
        return 1
    else:
        return 0

df['conditioning_intensity_missing'] = df['conditioning_intensity'].apply(missing_flag)

df = df.drop(columns = ['conditioning_intensity'])


df['ethnicity'].value_counts()


# # 例: df['ethnicity']に['Not Hispanic or Latino', 'Hispanic or Latino', nan, 'Non-resident of the U.S.']が含まれているとする
# # ワンホットエンコーディング
# ethnicity_dummies = pd.get_dummies(df['ethnicity'], prefix='ethnicity', dummy_na=True)
# ethnicity_dummies = ethnicity_dummies.astype(int)
# df = pd.concat([df, ethnicity_dummies], axis=1)

# # もし"Hispanic"か否かのバイナリ変数を作るなら：
# df['is_hispanic'] = df['ethnicity'].apply(lambda x: 1 if x=='Hispanic or Latino' else (0 if pd.notnull(x) else None))

# # Non-residentの場合も別途フラグを作成するなら：
# df['non_resident'] = df['ethnicity'].apply(lambda x: 1 if x=='Non-resident of the U.S.' else (0 if pd.notnull(x) else None))

# df = df.drop(columns = ['ethnicity'])


df['obesity'].value_counts()


# Not doneは欠損(0)とし、それ以外は01でフラグ付
df.loc[df['obesity'] == 'Yes', 'obesity'] = 1
df.loc[df['obesity'] == 'No', 'obesity'] = 0
df.loc[df['obesity'] == 'Not done', 'obesity_missing'] = 1
df.loc[df['obesity'] == 'Not done', 'obesity'] = 0
df.loc[df['obesity'].isna(), 'obesity_missing'] = 1
df.loc[df['obesity'].isna(), 'obesity'] = 0
df['obesity_missing'] = df['obesity_missing'].fillna(0)


df['mrd_hct'].value_counts()


# 欠損をどう扱うか、2としておくけど
df.loc[df['mrd_hct'] == 'Positive', 'mrd_hct'] = 1
df.loc[df['mrd_hct'] == 'Negative', 'mrd_hct'] = 0
df.loc[df['mrd_hct'].isna(), 'mrd_hct_missing'] = 1
df.loc[df['mrd_hct'].isna(), 'mrd_hct'] = 0
df['mrd_hct_missing'] = df['mrd_hct_missing'].fillna(0)


df['in_vivo_tcd'].value_counts()


# Not doneは欠損(0)とし、それ以外は01でフラグ付
df.loc[df['in_vivo_tcd'] == 'Yes', 'in_vivo_tcd'] = 1
df.loc[df['in_vivo_tcd'] == 'No', 'in_vivo_tcd'] = 0
df.loc[df['in_vivo_tcd'].isna(), 'in_vivo_tcd_missing'] = 1
df.loc[df['in_vivo_tcd'].isna(), 'in_vivo_tcd'] = 0
df['in_vivo_tcd_missing'] = df['in_vivo_tcd_missing'].fillna(0)


df['tce_match'].value_counts()


# 方法2: オーダナルエンコーディング（臨床的順序が仮に下記のような場合）
# Fully matched (最良) < Permissive < GvH non-permissive < HvG non-permissive (最悪)
ordinal_mapping = {
    'Fully matched': 1,
    'Permissive': 2,
    'GvH non-permissive': 3,
    'HvG non-permissive': 4
}
df['tce_match_ord'] = df['tce_match'].map(ordinal_mapping)
# 欠損フラグを別途作成
df['tce_match_missing'] = df['tce_match'].isna().astype(int)

df = df.drop(columns = ['tce_match'])


df['hepatic_severe'].value_counts()


# Not doneは欠損(0)とし、それ以外は01でフラグ付
df.loc[df['hepatic_severe'] == 'Yes', 'hepatic_severe'] = 1
df.loc[df['hepatic_severe'] == 'No', 'hepatic_severe'] = 0
df.loc[df['hepatic_severe'] == 'Not done', 'hepatic_severe_missing'] = 1
df.loc[df['hepatic_severe'] == 'Not done', 'hepatic_severe'] = 0
df.loc[df['hepatic_severe'].isna(), 'hepatic_severe_missing'] = 1
df.loc[df['hepatic_severe'].isna(), 'hepatic_severe'] = 0
df['hepatic_severe_missing'] = df['hepatic_severe_missing'].fillna(0)


print(df['prior_tumor'].value_counts())
print(df['peptic_ulcer'].value_counts())


# Not doneは欠損(0)とし、それ以外は01でフラグ付
df.loc[df['prior_tumor'] == 'Yes', 'prior_tumor'] = 1
df.loc[df['prior_tumor'] == 'No', 'prior_tumor'] = 0
df.loc[df['prior_tumor'] == 'Not done', 'prior_tumor_missing'] = 1
df.loc[df['prior_tumor'] == 'Not done', 'prior_tumor'] = 0
df.loc[df['prior_tumor'].isna(), 'prior_tumor_missing'] = 1
df.loc[df['prior_tumor'].isna(), 'prior_tumor'] = 0
df['prior_tumor_missing'] = df['prior_tumor_missing'].fillna(0)


# Not doneは欠損(0)とし、それ以外は01でフラグ付
df.loc[df['peptic_ulcer'] == 'Yes', 'peptic_ulcer'] = 1
df.loc[df['peptic_ulcer'] == 'No', 'peptic_ulcer'] = 0
df.loc[df['peptic_ulcer'] == 'Not done', 'peptic_ulcer_missing'] = 1
df.loc[df['peptic_ulcer'] == 'Not done', 'peptic_ulcer'] = 0
df.loc[df['peptic_ulcer'].isna(), 'peptic_ulcer_missing'] = 1
df.loc[df['peptic_ulcer'].isna(), 'peptic_ulcer'] = 0
df['peptic_ulcer_missing'] = df['peptic_ulcer_missing'].fillna(0)


df['gvhd_proph'].value_counts()


from sklearn.preprocessing import MultiLabelBinarizer

# 例: df['gvhd_proph']に以下のような値が入っているとする
# ['FK+ MMF +- others', 'Parent Q = yes, but no agent', nan, 
#  'FK+ MTX +- others(not MMF)', 'FKalone', 'Cyclophosphamide alone', ... ]

def extract_agents(val):
    if pd.isna(val):
        return []
    agents = []
    # 各主要な薬剤や治療法の有無をチェック
    if 'FK' in val:
        agents.append('FK')
    if 'CSA' in val:
        agents.append('CSA')
    if 'MMF' in val:
        agents.append('MMF')
    if 'MTX' in val:
        agents.append('MTX')
    if 'Cyclophosphamide' in val:
        agents.append('Cyclophosphamide')
    if 'TDEPLETION' in val:
        agents.append('TDEPLETION')
    if 'CDselect' in val:
        agents.append('CDselect')
    # 予防が行われていない場合
    if 'No GvHD Prophylaxis' in val or 'Parent Q = yes, but no agent' in val:
        agents.append('NoProphylaxis')
    # その他の予防法（必要に応じて）
    if 'Other GVHD Prophylaxis' in val:
        agents.append('Other')
    return agents

# 各ケースから使用された薬剤をリスト化
df['gvhd_agents'] = df['gvhd_proph'].apply(extract_agents)

# MultiLabelBinarizerを使って、各薬剤のバイナリ変数に変換
mlb = MultiLabelBinarizer()
agent_dummies = pd.DataFrame(mlb.fit_transform(df['gvhd_agents']),
                               columns=mlb.classes_,
                               index=df.index)

# 元のデータフレームに結合
df = pd.concat([df, agent_dummies], axis=1)

df = df.drop(columns = 'gvhd_proph')
df = df.drop(columns = 'gvhd_agents')


df['rheum_issue'].value_counts()


# Not doneは欠損(0)とし、それ以外は01でフラグ付
df.loc[df['rheum_issue'] == 'Yes', 'rheum_issue'] = 1
df.loc[df['rheum_issue'] == 'No', 'rheum_issue'] = 0
df.loc[df['rheum_issue'] == 'Not done', 'rheum_issue_missing'] = 1
df.loc[df['rheum_issue'] == 'Not done', 'rheum_issue'] = 0
df.loc[df['rheum_issue'].isna(), 'rheum_issue_missing'] = 1
df.loc[df['rheum_issue'].isna(), 'rheum_issue'] = 0
df['rheum_issue_missing'] = df['rheum_issue_missing'].fillna(0)


df['sex_match'].value_counts()


# 方法1: 同一性のバイナリ変数作成
def encode_sex_match(val):
    if pd.isna(val):
        return np.nan
    if val in ['M-M', 'F-F']:
        return 1
    elif val in ['M-F', 'F-M']:
        return 0

df['sex_match_bin'] = df['sex_match'].apply(encode_sex_match)

# 方法2: ドナーと受け手の性別を分割
# df[['donor_sex', 'recipient_sex']] = df['sex_match'].str.split('-', expand=True)

# 方法3: 欠損フラグ作成
df['sex_match_missing'] = df['sex_match'].isna().astype(int)

df = df.drop(columns = ['sex_match'])


df['race_group'].value_counts()


# # 例: df['race_group']に ['White', 'Black or African-American', 'Native Hawaiian or other Pacific Islander', 'Asian', 'American Indian or Alaska Native', 'More than one race'] が入っている場合
# race_dummies = pd.get_dummies(df['race_group'], prefix='race')
# race_dummies = race_dummies.astype(int)
# df = pd.concat([df, race_dummies], axis=1)
# df = df.drop(columns = ['race_group'])


df['hepatic_mild'].value_counts()


# Not doneは欠損(0)とし、それ以外は01でフラグ付
df.loc[df['hepatic_mild'] == 'Yes', 'hepatic_mild'] = 1
df.loc[df['hepatic_mild'] == 'No', 'hepatic_mild'] = 0
df.loc[df['hepatic_mild'] == 'Not done', 'hepatic_mild_missing'] = 1
df.loc[df['hepatic_mild'] == 'Not done', 'hepatic_mild'] = 0
df.loc[df['hepatic_mild'].isna(), 'hepatic_mild_missing'] = 1
df.loc[df['hepatic_mild'].isna(), 'hepatic_mild'] = 0
df['hepatic_mild_missing'] = df['hepatic_mild_missing'].fillna(0)


df['tce_div_match'].value_counts()


# # シンプルにワンホットエンコーディングする
# tce_div_dummies = pd.get_dummies(df['tce_div_match'], prefix='tce_div', dummy_na=True)
# tce_div_dummies = tce_div_dummies.astype(int)
# df = pd.concat([df, tce_div_dummies], axis=1)

# df = df.drop(columns = ['tce_div_match'])


df['donor_related'].value_counts()


# donor_related_dummies = pd.get_dummies(df['donor_related'], prefix='donor_rel', dummy_na=True)
# donor_related_dummies = donor_related_dummies.astype(int)
# df = pd.concat([df, donor_related_dummies], axis=1)

# df = df.drop(columns = ['donor_related'])


df['melphalan_dose'].value_counts()


# 例: df['melphalan_dose'] に ['N/A, Mel not given', 'MEL', nan] が入っているとする

def encode_melphalan(val):
    if pd.isna(val):
        return np.nan
    elif val == 'MEL':
        return 1
    elif 'Mel not given' in val:
        return 0
    else:
        return np.nan

df['melphalan_given'] = df['melphalan_dose'].apply(encode_melphalan)
# 欠損フラグ
df['melphalan_missing'] = df['melphalan_dose'].isna().astype(int)

df = df.drop(columns = ['melphalan_dose'])


print(df['cardiac'].value_counts())
print(df['pulm_moderate'].value_counts())


# Not doneは欠損(0)とし、それ以外は01でフラグ付
df.loc[df['cardiac'] == 'Yes', 'cardiac'] = 1
df.loc[df['cardiac'] == 'No', 'cardiac'] = 0
df.loc[df['cardiac'] == 'Not done', 'cardiac_missing'] = 1
df.loc[df['cardiac'] == 'Not done', 'cardiac'] = 0
df.loc[df['cardiac'].isna(), 'cardiac_missing'] = 1
df.loc[df['cardiac'].isna(), 'cardiac'] = 0
df['cardiac_missing'] = df['cardiac_missing'].fillna(0)


# Not doneは欠損(0)とし、それ以外は01でフラグ付
df.loc[df['pulm_moderate'] == 'Yes', 'pulm_moderate'] = 1
df.loc[df['pulm_moderate'] == 'No', 'pulm_moderate'] = 0
df.loc[df['pulm_moderate'] == 'Not done', 'pulm_moderate_missing'] = 1
df.loc[df['pulm_moderate'] == 'Not done', 'pulm_moderate'] = 0
df.loc[df['pulm_moderate'].isna(), 'pulm_moderate_missing'] = 1
df.loc[df['pulm_moderate'].isna(), 'pulm_moderate'] = 0
df['pulm_moderate_missing'] = df['pulm_moderate_missing'].fillna(0)


df.head()


print(df.shape)


train = df.loc[df['flag'] == 'train']
train = train.drop(columns = ['flag'])
test = df.loc[df['flag'] == 'test']
test = test.drop(columns = ['flag'])


from lifelines import KaplanMeierFitter
def transform_survival_probability(df, time_col='efs_time', event_col='efs'):
    kmf = KaplanMeierFitter()
    kmf.fit(df[time_col], df[event_col])
    y = kmf.survival_function_at_times(df[time_col]).values
    return y
train["y"] = transform_survival_probability(train, time_col='efs_time', event_col='efs')

plt.hist(train.loc[train.efs==1,"y"],bins=100,label="efs=1, Yes Event")
plt.hist(train.loc[train.efs==0,"y"],bins=100,label="efs=0, Maybe Event")
plt.xlabel("Transformed Target y")
plt.ylabel("Density")
plt.title("KaplanMeier Transformed Target y using both efs and efs_time.")
plt.legend()
plt.show()


RMV = ["ID","efs","efs_time","y"]
FEATURES = [c for c in train.columns if not c in RMV]
print(f"There are {len(FEATURES)} FEATURES: {FEATURES}")


CATS = []
for c in FEATURES:
    if train[c].dtype=="object":
        CATS.append(c)
        train[c] = train[c].fillna("NAN")
        test[c] = test[c].fillna("NAN")
print(f"In these features, there are {len(CATS)} CATEGORICAL FEATURES: {CATS}")


combined = pd.concat([train,test],axis=0,ignore_index=True)
#print("Combined data shape:", combined.shape )

# LABEL ENCODE CATEGORICAL FEATURES
print("We LABEL ENCODE the CATEGORICAL FEATURES: ",end="")
for c in FEATURES:

    # LABEL ENCODE CATEGORICAL AND CONVERT TO INT32 CATEGORY
    if c in CATS:
        print(f"{c}, ",end="")
        combined[c],_ = combined[c].factorize()
        combined[c] -= combined[c].min()
        combined[c] = combined[c].astype("int32")
        combined[c] = combined[c].astype("category")
        
    # REDUCE PRECISION OF NUMERICAL TO 32BIT TO SAVE MEMORY
    else:
        if combined[c].dtype=="float64":
            combined[c] = combined[c].astype("float32")
        if combined[c].dtype=="int64":
            combined[c] = combined[c].astype("int32")
    
train = combined.iloc[:len(train)].copy()
test = combined.iloc[len(train):].reset_index(drop=True).copy()


from sklearn.model_selection import KFold
from xgboost import XGBRegressor, XGBClassifier
import xgboost as xgb
print("Using XGBoost version",xgb.__version__)


%%time
FOLDS = 10
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    
oof_xgb = np.zeros(len(train))
pred_xgb = np.zeros(len(test))

for i, (train_index, test_index) in enumerate(kf.split(train)):

    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)
    
    x_train = train.loc[train_index,FEATURES].copy()
    y_train = train.loc[train_index,"y"]
    x_valid = train.loc[test_index,FEATURES].copy()
    y_valid = train.loc[test_index,"y"]
    x_test = test[FEATURES].copy()

    model_xgb = XGBRegressor(
        device="cuda",
        max_depth=3,  
        colsample_bytree=0.5,  
        subsample=0.8,  
        n_estimators=2000,  
        learning_rate=0.02,  
        enable_categorical=True,
        min_child_weight=80,
        #early_stopping_rounds=25,
    )
    model_xgb.fit(
        x_train, y_train,
        eval_set=[(x_valid, y_valid)],  
        verbose=500 
    )

    # INFER OOF
    oof_xgb[test_index] = model_xgb.predict(x_valid)
    # INFER TEST
    pred_xgb += model_xgb.predict(x_test)

# COMPUTE AVERAGE TEST PREDS
pred_xgb /= FOLDS


from metric import score

y_true = train[["ID","efs","efs_time","race_group"]].copy()
y_pred = train[["ID"]].copy()
y_pred["prediction"] = oof_xgb
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for XGBoost KaplanMeier =",m)


from catboost import CatBoostRegressor, CatBoostClassifier
import catboost as cb
print("Using CatBoost version",cb.__version__)


%%time
FOLDS = 10
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    
oof_cat = np.zeros(len(train))
pred_cat = np.zeros(len(test))

for i, (train_index, test_index) in enumerate(kf.split(train)):

    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)
    
    x_train = train.loc[train_index,FEATURES].copy()
    y_train = train.loc[train_index,"y"]
    x_valid = train.loc[test_index,FEATURES].copy()
    y_valid = train.loc[test_index,"y"]
    x_test = test[FEATURES].copy()

    model_cat = CatBoostRegressor(
        task_type="GPU",  
        learning_rate=0.1,    
        grow_policy='Lossguide',
        #early_stopping_rounds=25,
    )
    model_cat.fit(x_train,y_train,
              eval_set=(x_valid, y_valid),
              cat_features=CATS,
              verbose=250)

    # INFER OOF
    oof_cat[test_index] = model_cat.predict(x_valid)
    # INFER TEST
    pred_cat += model_cat.predict(x_test)

# COMPUTE AVERAGE TEST PREDS
pred_cat /= FOLDS


y_true = train[["ID","efs","efs_time","race_group"]].copy()
y_pred = train[["ID"]].copy()
y_pred["prediction"] = oof_cat
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for CatBoost KaplanMeier =",m)


from lightgbm import LGBMRegressor
import lightgbm as lgb
print("Using LightGBM version",lgb.__version__)


FOLDS = 10
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    
oof_lgb = np.zeros(len(train))
pred_lgb = np.zeros(len(test))

for i, (train_index, test_index) in enumerate(kf.split(train)):

    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)
    
    x_train = train.loc[train_index,FEATURES].copy()
    y_train = train.loc[train_index,"y"]    
    x_valid = train.loc[test_index,FEATURES].copy()
    y_valid = train.loc[test_index,"y"]
    x_test = test[FEATURES].copy()

    model_lgb = LGBMRegressor(
        device="gpu", 
        max_depth=3, 
        colsample_bytree=0.4,  
        #subsample=0.9, 
        n_estimators=2500, 
        learning_rate=0.02, 
        objective="regression", 
        verbose=-1, 
        #early_stopping_rounds=25,
    )
    model_lgb.fit(
        x_train, y_train,
        eval_set=[(x_valid, y_valid)],
    )
    
    # INFER OOF
    oof_lgb[test_index] = model_lgb.predict(x_valid)
    # INFER TEST
    pred_lgb += model_lgb.predict(x_test)

# COMPUTE AVERAGE TEST PREDS
pred_lgb /= FOLDS


y_true = train[["ID","efs","efs_time","race_group"]].copy()
y_pred = train[["ID"]].copy()
y_pred["prediction"] = oof_lgb
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for LightGBM KaplanMeier =",m)


# SURVIVAL COX NEEDS THIS TARGET (TO DIGEST EFS AND EFS_TIME)
train["efs_time2"] = train.efs_time.copy()
train.loc[train.efs==0,"efs_time2"] *= -1


FOLDS = 10
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    
oof_xgb_cox = np.zeros(len(train))
pred_xgb_cox = np.zeros(len(test))

for i, (train_index, test_index) in enumerate(kf.split(train)):

    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)
    
    x_train = train.loc[train_index,FEATURES].copy()
    y_train = train.loc[train_index,"efs_time2"]    
    x_valid = train.loc[test_index,FEATURES].copy()
    y_valid = train.loc[test_index,"efs_time2"]
    x_test = test[FEATURES].copy()

    model_xgb_cox = XGBRegressor(
        device="cuda",
        max_depth=3,  
        colsample_bytree=0.5,  
        subsample=0.8,  
        n_estimators=2000,  
        learning_rate=0.02,  
        enable_categorical=True,
        min_child_weight=80,
        objective='survival:cox',
        eval_metric='cox-nloglik',
    )
    model_xgb_cox.fit(
        x_train, y_train,
        eval_set=[(x_valid, y_valid)],  
        verbose=500  
    )
    
    # INFER OOF
    oof_xgb_cox[test_index] = model_xgb_cox.predict(x_valid)
    # INFER TEST
    pred_xgb_cox += model_xgb_cox.predict(x_test)

# COMPUTE AVERAGE TEST PREDS
pred_xgb_cox /= FOLDS


y_true = train[["ID","efs","efs_time","race_group"]].copy()
y_pred = train[["ID"]].copy()
y_pred["prediction"] = oof_xgb_cox
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for XGBoost Survival:Cox =",m)


FOLDS = 10
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    
oof_cat_cox = np.zeros(len(train))
pred_cat_cox = np.zeros(len(test))

for i, (train_index, test_index) in enumerate(kf.split(train)):

    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)
    
    x_train = train.loc[train_index,FEATURES].copy()
    y_train = train.loc[train_index,"efs_time2"]    
    x_valid = train.loc[test_index,FEATURES].copy()
    y_valid = train.loc[test_index,"efs_time2"]
    x_test = test[FEATURES].copy()

    model_cat_cox = CatBoostRegressor(
        loss_function="Cox",
        #task_type="GPU",   
        iterations=400,     
        learning_rate=0.1,  
        grow_policy='Lossguide',
        use_best_model=False,
    )
    model_cat_cox.fit(x_train,y_train,
              eval_set=(x_valid, y_valid),
              cat_features=CATS,
              verbose=100)
    
    # INFER OOF
    oof_cat_cox[test_index] = model_cat_cox.predict(x_valid)
    # INFER TEST
    pred_cat_cox += model_cat_cox.predict(x_test)

# COMPUTE AVERAGE TEST PREDS
pred_cat_cox /= FOLDS


y_true = train[["ID","efs","efs_time","race_group"]].copy()
y_pred = train[["ID"]].copy()
y_pred["prediction"] = oof_cat_cox
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for CatBoost Survival:Cox =",m)


from scipy.stats import rankdata 

y_true = train[["ID","efs","efs_time","race_group"]].copy()
y_pred = train[["ID"]].copy()
y_pred["prediction"] = rankdata(oof_xgb) + rankdata(oof_cat) + rankdata(oof_lgb)\
                     + rankdata(oof_xgb_cox) + rankdata(oof_cat_cox)
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for Ensemble =",m)


sub = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv")
sub.prediction = rankdata(pred_xgb) + rankdata(pred_cat) + rankdata(pred_lgb)\
                     + rankdata(pred_xgb_cox) + rankdata(pred_cat_cox)
sub.to_csv("submission.csv",index=False)
print("Sub shape:",sub.shape)
sub.head()




