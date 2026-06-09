x = 2*2 + 3**5 / 243


print(x)


print(x - 3)


from pathlib import Path


DATA_PATH = Path('/kaggle/input/2025-stellar-temperature-challenge')

# Pythonは型を持っており、先ほどの「x」は数値型、この「DATA_PATH」はpathlibによって定義された'pathlib.PosixPath'型であることが分かります
print('DATA_PATHの型', type(DATA_PATH))

# DATA_PATHの中身を表示（Pathオブジェクトの内容そのもの）
print('DATA_PATHの中身:', DATA_PATH)

# stemは、パスの末尾（ファイル名またはディレクトリ名）の拡張子なしの部分
# この場合は "stellar-temperature-challenge-2025" が表示される
print('DATA_PATHのstem（末尾の名前部分）:', DATA_PATH.stem)

# parentは、DATA_PATHの1つ上の階層のパス（親ディレクトリ）を返す
print('DATA_PATHのparent（親ディレクトリ）:', DATA_PATH.parent)


# pandasの場合
import pandas as pd

# pathlibは「DATA_PATH　/ 'ファイル名'」とするだけで、DATA_PATH下にあるファイルを認識してくれて読みやすいぞ！
df_train_pd = pd.read_csv(DATA_PATH / 'train.csv')


# 'head'はpandasのDataFrameの便利な関数で、データフレームの先頭5行を表示できる
df_train_pd.head()


# polarsの場合
import polars as pl

df_train_pl = pl.read_csv(DATA_PATH / 'train.csv')


# 'describe'はpolarsのDataFrameの便利な関数で、データフレームの詳細を表示できる
df_train_pl.describe()


# 取り出す列名を指定
FEATURES = ['kepmag', 'logg', 'mass']
target = 'teff'


X = df_train_pd[FEATURES] # 列名がFEATURESの列のみを抽出
y = df_train_pd[target] # 列名がtargetの列のみを抽出


# 'shape'はpandasのDataFrameオブジェクトにおいてテーブルデータの形状を表す

print('df_train_pdの形', df_train_pd.shape) # 100010行20列であるとわかる
print('Xの形', X.shape) # 100010行3列であるとわかる


print(y)


# 'to_numpy'でnumpyライブラリの配列に変換することができる
y_numpy_arr = y.to_numpy()

print('y_numpy_arrの中身',y_numpy_arr)
print('y_numpy_arrの形',y_numpy_arr.shape) # 100010個
print('y_numpy_arrの型',type(y_numpy_arr))


# pandasライブラリは一度上のセルで読み込んでいるので、もう一度importする必要はない

df_sample_submission = pd.read_csv(DATA_PATH / 'sample_submission.csv')


df_sample_submission


# 予測対象である列'teff'の値を全て5020.0に変更する
df_sample_submission[target] = 5020.0 


# 変更できたか確認
df_sample_submission


# CSVファイルとして書き出す
df_sample_submission.to_csv("submission.csv", index=False)

