!unzip /kaggle/input/santander-product-recommendation/train_ver2.csv.zip


import pandas as pd
df = pd.read_csv("/kaggle/working/train_ver2.csv")


df.shape, len(set(df["ncodpers"]))


df['ncodpers'].value_counts()


# cleaning
import numpy as np

df["fecha_dato"] = pd.to_datetime(df["fecha_dato"],format="%Y-%m-%d")
df["fecha_alta"] = pd.to_datetime(df["fecha_alta"],format="%Y-%m-%d")
df["fecha_dato"].unique()
df["month"] = pd.DatetimeIndex(df["fecha_dato"]).month
df["age"]   = pd.to_numeric(df["age"], errors="coerce")
df.loc[df.age < 18,"age"]  = df.loc[(df.age >= 18) & (df.age <= 30),"age"].mean(skipna=True)
df.loc[df.age > 100,"age"] = df.loc[(df.age >= 30) & (df.age <= 100),"age"].mean(skipna=True)
df["age"].fillna(df["age"].mean(),inplace=True)
df["age"]      = df["age"].astype(int)
months_active = df.loc[df["ind_nuevo"].isnull(),:].groupby("ncodpers", sort=False).size()

df.loc[df["ind_nuevo"].isnull(),"ind_nuevo"] = 1
df.antiguedad = pd.to_numeric(df.antiguedad,errors="coerce")

df.loc[df.antiguedad.isnull(),"antiguedad"] = df.antiguedad.min()
df.loc[df.antiguedad <0, "antiguedad"]      = 0 # Thanks @StephenSmith for bug-find
dates=df.loc[:,"fecha_alta"].sort_values().reset_index()
median_date = int(np.median(dates.index.values))
df.loc[df.fecha_alta.isnull(),"fecha_alta"] = dates.loc[median_date,"fecha_alta"]

df.loc[df.indrel.isnull(),"indrel"] = 1
df.drop(["tipodom","cod_prov"],axis=1,inplace=True)
df.isnull().any()


df.loc[df.ind_actividad_cliente.isnull(),"ind_actividad_cliente"] = \
df["ind_actividad_cliente"].median()
df.loc[df.nomprov=="CORU\xc3\x91A, A","nomprov"] = "CORUNA, A"
df.loc[df.nomprov.isnull(),"nomprov"] = "UNKNOWN"
df.loc[df.ind_nomina_ult1.isnull(), "ind_nomina_ult1"] = 0
df.loc[df.ind_nom_pens_ult1.isnull(), "ind_nom_pens_ult1"] = 0


feature_cols = df.iloc[:1,].filter(regex="ind_+.*ult.*").columns.values.tolist()

for col in feature_cols:
    #as float fot the flop in gpu
    df[col] = df[col].astype(float)


## for i in range(1,4):
sample_size = 200000
keep_cols = feature_cols + ['ncodpers', 'month']

#sample
df = df.sample(sample_size, random_state=42)
df = df[keep_cols]

df = pd.melt(df, id_vars   = [col for col in df.columns if col not in feature_cols],
        value_vars= [col for col in feature_cols])


df.to_csv(f"train_v2_melt.csv", index=False)


df.shape, df.keys()


# import pandas as pd
# import numpy as np

# # Assume df is your original DataFrame
# df['key'] = df['ncodpers'].astype(str) + '_' + df['fecha_dato'].astype(str)

# # Get unique keys
# unique_keys = df['key'].drop_duplicates()
# shuffled_keys = unique_keys.sample(frac=1, random_state=42)

# # Split the keys
# split_size = 100_000
# split1_keys = set(shuffled_keys.iloc[:split_size])
# split2_keys = set(shuffled_keys.iloc[split_size:2*split_size])
# split3_keys = set(shuffled_keys.iloc[2*split_size:3*split_size])

# # Filter the original DataFrame
# split1 = df[df['key'].isin(split1_keys)].copy()
# split2 = df[df['key'].isin(split2_keys)].copy()
# split3 = df[df['key'].isin(split3_keys)].copy()

# # Optionally drop the helper column
# for split in [split1, split2, split3]:
#     split.drop(columns='key', inplace=True)














