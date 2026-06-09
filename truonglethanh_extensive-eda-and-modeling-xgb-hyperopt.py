import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy as sp
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Standard plotly imports
#import plotly.plotly as py
import plotly.graph_objs as go
import plotly.tools as tls
from plotly.offline import iplot, init_notebook_mode
#import cufflinks
#import cufflinks as cf
import plotly.figure_factory as ff

# Using plotly + cufflinks in offline mode
init_notebook_mode(connected=True)
#cufflinks.go_offline(connected=True)

# Preprocessing, modelling and evaluating
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold
from xgboost import XGBClassifier
import xgboost as xgb

## Hyperopt modules
from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK, STATUS_RUNNING
from functools import partial

import os
import gc
print(os.listdir("../input"))


df_id = pd.read_csv("../input/train_identity.csv")
df_trans = pd.read_csv("../input/train_transaction.csv")


def resumetable(df):
    """
    Tạo bảng tóm tắt các đặc điểm của DataFrame.
    """
    print(f"Kích thước tập dữ liệu: {df.shape}")
    
    # Lấy kiểu dữ liệu của các cột
    summary = pd.DataFrame(df.dtypes, columns=['KieuDuLieu'])
    summary = summary.reset_index()
    summary['TenCot'] = summary['index']
    summary = summary[['TenCot', 'KieuDuLieu']]
    
    # Đếm số giá trị bị thiếu
    summary['Thieu'] = df.isnull().sum().values
    
    # Đếm số giá trị duy nhất
    summary['DuyNhat'] = df.nunique().values
    
    # Lấy 3 giá trị đầu tiên để xem qua
    summary['GiaTriDau'] = df.loc[0].values
    summary['GiaTriHai'] = df.loc[1].values
    summary['GiaTriBa'] = df.loc[2].values

    # Tính toán Entropy (độ "ngẫu nhiên" của cột)
    for name in summary['TenCot'].value_counts().index:
        summary.loc[summary['TenCot'] == name, 'Entropy'] = round(stats.entropy(df[name].value_counts(normalize=True), base=2), 2) 

    return summary


## Hàm để giảm dung lượng DataFrame
def reduce_mem_usage(df, verbose=True):
    """
    Duyệt qua các cột và giảm kiểu dữ liệu số để tiết kiệm bộ nhớ.
    """
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    
    # Tính dung lượng bộ nhớ ban đầu
    start_mem = df.memory_usage().sum() / 1024**2 
    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            
            # Kiểm tra và ép kiểu cho số nguyên (Integer)
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            
            # Kiểm tra và ép kiểu cho số thực (Float)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
                    
    # Tính dung lượng bộ nhớ sau khi giảm
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: 
        print('Dung lượng bộ nhớ giảm xuống {:5.2f} Mb (giảm {:.1f}%)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
        
    return df


def CalcOutliers(df_num): 
    """
    Tính toán và in ra thông tin về các giá trị ngoại lai (outliers)
    dựa trên quy tắc 3 lần độ lệch chuẩn.
    """
    
    # tính giá trị trung bình và độ lệch chuẩn
    data_mean, data_std = np.mean(df_num), np.std(df_num)

    # thiết lập ngưỡng cắt (cut-off) là 3 lần độ lệch chuẩn
    # Bạn có thể thay đổi số 3 này nếu muốn
    cut = data_std * 3

    # Tính toán các giá trị ngưỡng trên và ngưỡng dưới
    lower, upper = data_mean - cut, data_mean + cut

    # tạo mảng chứa các giá trị ngoại lai (thấp, cao, và tổng cộng)
    outliers_lower = [x for x in df_num if x < lower]
    outliers_higher = [x for x in df_num if x > upper]
    outliers_total = [x for x in df_num if x < lower or x > upper]

    # mảng không chứa giá trị ngoại lai
    outliers_removed = [x for x in df_num if x > lower and x < upper]
    
    print('Số lượng ngoại lai thấp: %d' % len(outliers_lower))
    print('Số lượng ngoại lai cao: %d' % len(outliers_higher))
    print('Tổng số quan sát ngoại lai: %d' % len(outliers_total))
    print('Số quan sát không ngoại lai: %d' % len(outliers_removed))
    
    # Tính phần trăm ngoại lai
    if len(outliers_removed) > 0:
        percent_outliers = round((len(outliers_total) / len(outliers_removed)) * 100, 4)
        print("Tổng phần trăm ngoại lai (so với phần không ngoại lai): ", percent_outliers)
    else:
        print("Không có quan sát nào không ngoại lai để so sánh.")
    
    return


df_trans = reduce_mem_usage(df_trans)
df_id = reduce_mem_usage(df_id)


resumetable(df_trans)[:25]


df_trans['TransactionAmt'] = df_trans['TransactionAmt'].astype(float)
total = len(df_trans)
total_amt = df_trans.groupby(['isFraud'])['TransactionAmt'].sum().sum()
plt.figure(figsize=(16,6))

# --- Biểu đồ 1: Đếm số lượng giao dịch ---
plt.subplot(121)
g = sns.countplot(x='isFraud', data=df_trans)
g.set_title("Phân bổ Giao dịch Gian lận \n# 0: Không Gian lận | 1: Gian lận #", fontsize=22)
g.set_xlabel("Có phải gian lận?", fontsize=18)
g.set_ylabel('Số lượng', fontsize=18)
for p in g.patches:
    height = p.get_height()
    g.text(p.get_x() + p.get_width()/2.,
           height + 3,
           '{:1.2f}%'.format(height/total*100),
           ha="center", fontsize=15) 

# --- Biểu đồ 2: Tổng giá trị giao dịch ---
perc_amt = (df_trans.groupby(['isFraud'])['TransactionAmt'].sum())
perc_amt = perc_amt.reset_index()
plt.subplot(122)
g1 = sns.barplot(x='isFraud', y='TransactionAmt', dodge=True, data=perc_amt)
g1.set_title("% Tổng Giá trị Giao dịch \n# 0: Không Gian lận | 1: Gian lận #", fontsize=22)
g1.set_xlabel("Có phải gian lận?", fontsize=18)
g1.set_ylabel('Tổng Giá trị Giao dịch', fontsize=18)
for p in g1.patches:
    height = p.get_height()
    g1.text(p.get_x() + p.get_width()/2.,
           height + 3,
           '{:1.2f}%'.format(height/total_amt * 100),
           ha="center", fontsize=15) 
    
plt.show()


df_trans['TransactionAmt'] = df_trans['TransactionAmt'].astype(float)
print("Các Phân vị của Số tiền Giao dịch:")
print(df_trans['TransactionAmt'].quantile([.01, .025, .1, .25, .5, .75, .9, .975, .99]))


# --- Hình 1: Hai biểu đồ Phân bổ (Histogram) ---
plt.figure(figsize=(16,12))
plt.suptitle('Phân bổ Giá trị Giao dịch', fontsize=22)

# Biểu đồ 1.1: Phân bổ với giá trị <= 1000
plt.subplot(221)
# Lưu ý: sns.distplot đã cũ, bạn có thể thay bằng sns.histplot(..., kde=True)
g = sns.distplot(df_trans[df_trans['TransactionAmt'] <= 1000]['TransactionAmt'])
g.set_title("Phân bổ Số tiền Giao dịch <= 1000", fontsize=18)
g.set_xlabel("")
g.set_ylabel("Xác suất", fontsize=15)

# Biểu đồ 1.2: Phân bổ theo Logarit
plt.subplot(222)
# Lưu ý: sns.distplot đã cũ, bạn có thể thay bằng sns.histplot(..., kde=True)
g1 = sns.distplot(np.log(df_trans['TransactionAmt']))
g1.set_title("Phân bổ Số tiền Giao dịch (Log)", fontsize=18)
g1.set_xlabel("")
g1.set_ylabel("Xác suất", fontsize=15)

# --- Hình 2: Biểu đồ ECDF Gộp ---
# Lưu ý: Code gốc của bạn tạo một hình mới ở đây
plt.figure(figsize=(16,12))

plt.subplot(212)
g4 = plt.scatter(range(df_trans[df_trans['isFraud'] == 0].shape[0]),
                 np.sort(df_trans[df_trans['isFraud'] == 0]['TransactionAmt'].values), 
                 label='Không Gian lận', alpha=.2)
g4 = plt.scatter(range(df_trans[df_trans['isFraud'] == 1].shape[0]),
                 np.sort(df_trans[df_trans['isFraud'] == 1]['TransactionAmt'].values), 
                 label='Gian lận', alpha=.2)
g4 = plt.title("ECDF \nPhân bổ Số tiền GIAN LẬN và KHÔNG GIAN LẬN", fontsize=18)
g4 = plt.xlabel("Chỉ số (đã sắp xếp)")
g4 = plt.ylabel("Phân bổ Số tiền", fontsize=15)
g4 = plt.legend()

# --- Hình 3: Biểu đồ ECDF Riêng lẻ ---
# Lưu ý: Code gốc của bạn tạo một hình mới nữa ở đây
plt.figure(figsize=(16,12))

# Biểu đồ 3.1: ECDF cho GIAN LẬN
plt.subplot(321)
g = plt.scatter(range(df_trans[df_trans['isFraud'] == 1].shape[0]), 
                 np.sort(df_trans[df_trans['isFraud'] == 1]['TransactionAmt'].values), 
                 label='Gian lận', alpha=.4)
plt.title("GIAN LẬN - ECDF Số tiền Giao dịch", fontsize=18)
plt.xlabel("Chỉ số (đã sắp xếp)")
plt.ylabel("Phân bổ Số tiền", fontsize=12)

# Biểu đồ 3.2: ECDF cho KHÔNG GIAN LẬN
plt.subplot(322)
g1 = plt.scatter(range(df_trans[df_trans['isFraud'] == 0].shape[0]),
                 np.sort(df_trans[df_trans['isFraud'] == 0]['TransactionAmt'].values), 
                 label='Không Gian lận', alpha=.2)
g1 = plt.title("KHÔNG GIAN LẬN - ECDF Số tiền Giao dịch", fontsize=18)
g1 = plt.xlabel("Chỉ số (đã sắp xếp)")
g1 = plt.ylabel("Phân bổ Số tiền", fontsize=15)

plt.suptitle('Phân bổ ECDF riêng lẻ', fontsize=22)

plt.show()


print(pd.concat([df_trans[df_trans['isFraud'] == 1]['TransactionAmt']\
                 .quantile([.01, .1, .25, .5, .75, .9, .99])\
                 .reset_index(), 
                 df_trans[df_trans['isFraud'] == 0]['TransactionAmt']\
                 .quantile([.01, .1, .25, .5, .75, .9, .99])\
                 .reset_index()],
                 axis=1, keys=['Gian lận', 'Không Gian lận']))


CalcOutliers(df_trans['TransactionAmt'])


# Chuẩn bị dữ liệu cho biểu đồ 2 (tỷ lệ % gian lận)
tmp = pd.crosstab(df_trans['ProductCD'], df_trans['isFraud'], normalize='index') * 100
tmp = tmp.reset_index()
# Đổi tên cột 0 và 1 sang tiếng Việt để dùng ở biểu đồ 2
tmp.rename(columns={0:'KhongGianLan', 1:'GianLan'}, inplace=True)

plt.figure(figsize=(14,10))
plt.suptitle('Phân bổ theo ProductCD (Mã Sản phẩm)', fontsize=22)

# --- Biểu đồ 1 (Trên, trái): Đếm số lượng ProductCD ---
plt.subplot(221)
g = sns.countplot(x='ProductCD', data=df_trans)

g.set_title("Phân bổ ProductCD", fontsize=19)
g.set_xlabel("Mã ProductCD", fontsize=17)
g.set_ylabel("Số lượng", fontsize=17)
g.set_ylim(0,500000)
for p in g.patches:
    height = p.get_height()
    g.text(p.get_x()+p.get_width()/2.,
           height + 3,
           '{:1.2f}%'.format(height/total*100),
           ha="center", fontsize=14) 

# --- Biểu đồ 2 (Trên, phải): Số lượng VÀ Tỷ lệ % gian lận ---
plt.subplot(222)
g1 = sns.countplot(x='ProductCD', hue='isFraud', data=df_trans)
# Đặt chú thích (legend) cho g1
plt.legend(title='Gian lận', loc='best', labels=['Không', 'Có'])

# Tạo trục y thứ hai (twinx) để vẽ tỷ lệ %
gt = g1.twinx()
# Vẽ tỷ lệ % gian lận lên trục y thứ hai
gt = sns.pointplot(x='ProductCD', y='GianLan', data=tmp, color='black', order=['W', 'H',"C", "S", "R"], legend=False)
gt.set_ylabel("% Giao dịch Gian lận", fontsize=16)

g1.set_title("ProductCD theo Mục tiêu (isFraud)", fontsize=19)
g1.set_xlabel("Mã ProductCD", fontsize=17)
g1.set_ylabel("Số lượng", fontsize=17)

# --- Biểu đồ 3 (Dưới): Phân bổ số tiền theo ProductCD ---
plt.subplot(212)
# Lọc các giao dịch <= 2000 để biểu đồ dễ nhìn hơn
g3 = sns.boxenplot(x='ProductCD', y='TransactionAmt', hue='isFraud', 
                   data=df_trans[df_trans['TransactionAmt'] <= 2000] )
g3.set_title("Phân bổ Số tiền Giao dịch theo ProductCD và Mục tiêu", fontsize=20)
g3.set_xlabel("Mã ProductCD", fontsize=17)
g3.set_ylabel("Giá trị Giao dịch", fontsize=17)

# Thêm chú thích (legend) cho biểu đồ 3
handles, labels = g3.get_legend_handles_labels()
g3.legend(handles, ['Không', 'Có'], title='Gian lận', loc='best')


plt.subplots_adjust(hspace = 0.6, top = 0.85)

plt.show()


## Knowning the Card Features
resumetable(df_trans[['card1', 'card2', 'card3','card4', 'card5', 'card6']])


print("Các Phân vị của Đặc trưng Thẻ: ")
print(df_trans[['card1', 'card2', 'card3', 'card5']].quantile([0.01, .025, .1, .25, .5, .75, .975, .99]))



df_trans.loc[df_trans.card3.isin(df_trans.card3.value_counts()[df_trans.card3.value_counts() < 200].index), 'card3'] = "Others"
df_trans.loc[df_trans.card5.isin(df_trans.card5.value_counts()[df_trans.card5.value_counts() < 300].index), 'card5'] = "Others"


# Chuẩn bị dữ liệu cho card3
tmp = pd.crosstab(df_trans['card3'], df_trans['isFraud'], normalize='index') * 100
tmp = tmp.reset_index()
# Đổi tên cột 0 và 1 sang tiếng Việt (không dấu) để dễ tham chiếu
tmp.rename(columns={0:'KhongGianLan', 1:'GianLan'}, inplace=True)

# Chuẩn bị dữ liệu cho card5
tmp2 = pd.crosstab(df_trans['card5'], df_trans['isFraud'], normalize='index') * 100
tmp2 = tmp2.reset_index()
# Đổi tên cột 0 và 1 sang tiếng Việt (không dấu) để dễ tham chiếu
tmp2.rename(columns={0:'KhongGianLan', 1:'GianLan'}, inplace=True)

plt.figure(figsize=(14,22))

# --- Biểu đồ 1 (card1) ---
plt.subplot(411)
# Lưu ý: sns.distplot đã cũ, bạn có thể thay bằng sns.histplot(..., kde=True)
g = sns.distplot(df_trans[df_trans['isFraud'] == 1]['card1'], label='Gian lận')
g = sns.distplot(df_trans[df_trans['isFraud'] == 0]['card1'], label='Không Gian lận')
g.legend()
g.set_title("Phân bổ Giá trị Card 1 theo Mục tiêu", fontsize=20)
g.set_xlabel("Giá trị Card 1", fontsize=18)
g.set_ylabel("Xác suất", fontsize=18)

# --- Biểu đồ 2 (card2) ---
plt.subplot(412)
# Lưu ý: sns.distplot đã cũ, bạn có thể thay bằng sns.histplot(..., kde=True)
g1 = sns.distplot(df_trans[df_trans['isFraud'] == 1]['card2'].dropna(), label='Gian lận')
g1 = sns.distplot(df_trans[df_trans['isFraud'] == 0]['card2'].dropna(), label='Không Gian lận')
g1.legend()
g1.set_title("Phân bổ Giá trị Card 2 theo Mục tiêu", fontsize=20)
g1.set_xlabel("Giá trị Card 2", fontsize=18)
g1.set_ylabel("Xác suất", fontsize=18)

# --- Biểu đồ 3 (card3) ---
plt.subplot(413)
g2 = sns.countplot(x='card3', data=df_trans, order=list(tmp.card3.values))
g22 = g2.twinx()
# Sử dụng tên cột 'GianLan' đã đổi ở trên
gg2 = sns.pointplot(x='card3', y='GianLan', data=tmp, 
                    color='black', order=list(tmp.card3.values))
gg2.set_ylabel("% Giao dịch Gian lận", fontsize=16)
g2.set_title("Phân bổ Giá trị Card 3 và % Giao dịch Gian lận", fontsize=20)
g2.set_xlabel("Giá trị Card 3", fontsize=18)
g2.set_ylabel("Số lượng", fontsize=18)
for p in g2.patches:
    height = p.get_height()
    g2.text(p.get_x()+p.get_width()/2.,
            height + 25,
            '{:1.2f}%'.format(height/total*100),
            ha="center") 

# --- Biểu đồ 4 (card5) ---
plt.subplot(414)
g3 = sns.countplot(x='card5', data=df_trans, order=list(tmp2.card5.values))
g3t = g3.twinx()
# Sử dụng tên cột 'GianLan' đã đổi ở trên
g3t = sns.pointplot(x='card5', y='GianLan', data=tmp2, 
                    color='black', order=list(tmp2.card5.values))
g3t.set_ylabel("% Giao dịch Gian lận", fontsize=16)
g3.set_title("Phân bổ Giá trị Card 5 và % Giao dịch Gian lận", fontsize=20)
g3.set_xticklabels(g3.get_xticklabels(),rotation=90)
g3.set_xlabel("Giá trị Card 5", fontsize=18)
g3.set_ylabel("Số lượng", fontsize=18)
for p in g3.patches:
    height = p.get_height()
    g3.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}%'.format(height/total*100),
            ha="center",fontsize=11) 
    
plt.subplots_adjust(hspace = 0.6, top = 0.85)


plt.show()


# Chuẩn bị dữ liệu cho card4
tmp = pd.crosstab(df_trans['card4'], df_trans['isFraud'], normalize='index') * 100
tmp = tmp.reset_index()
# Đổi tên cột 0 và 1 sang tiếng Việt (không dấu) để dễ tham chiếu
tmp.rename(columns={0:'KhongGianLan', 1:'GianLan'}, inplace=True)

plt.figure(figsize=(14,10))
plt.suptitle('Phân bổ Card 4', fontsize=22)

# --- Biểu đồ 1 (Trên, trái): Đếm số lượng card4 ---
plt.subplot(221)
g = sns.countplot(x='card4', data=df_trans)

g.set_title("Phân bổ Card 4", fontsize=19)
g.set_ylim(0,420000)
g.set_xlabel("Các loại Card 4", fontsize=17)
g.set_ylabel("Số lượng", fontsize=17)
for p in g.patches:
    height = p.get_height()
    g.text(p.get_x()+p.get_width()/2.,
           height + 3,
           '{:1.2f}%'.format(height/total*100),
           ha="center",fontsize=14) 


# --- Biểu đồ 2 (Trên, phải): Số lượng VÀ Tỷ lệ % gian lận ---
plt.subplot(222)
g1 = sns.countplot(x='card4', hue='isFraud', data=df_trans)
# Đặt chú thích (legend) cho g1
plt.legend(title='Gian lận', loc='best', labels=['Không', 'Có'])
gt = g1.twinx()
# Vẽ tỷ lệ % gian lận lên trục y thứ hai, sử dụng cột 'GianLan' đã đổi tên
gt = sns.pointplot(x='card4', y='GianLan', data=tmp, 
                   color='black', legend=False, 
                   order=['discover', 'mastercard', 'visa', 'american express'])
gt.set_ylabel("% Giao dịch Gian lận", fontsize=16)
g1.set_title("Card 4 theo Mục tiêu (isFraud)", fontsize=19)
g1.set_xlabel("Các loại Card 4", fontsize=17)
g1.set_ylabel("Số lượng", fontsize=17)

# --- Biểu đồ 3 (Dưới): Phân bổ số tiền theo card4 ---
plt.subplot(212)
# Lọc các giao dịch <= 2000 để biểu đồ dễ nhìn hơn
g3 = sns.boxenplot(x='card4', y='TransactionAmt', hue='isFraud', 
                   data=df_trans[df_trans['TransactionAmt'] <= 2000] )
# Sửa lại tiêu đề cho đúng (code gốc bị nhầm "ProductCD")
g3.set_title("Phân bổ Số tiền Giao dịch theo Card 4 và Mục tiêu", fontsize=20)
g3.set_xlabel("Các loại Card 4", fontsize=17)
g3.set_ylabel("Giá trị Giao dịch", fontsize=17)

# Thêm chú thích (legend) cho biểu đồ 3
handles, labels = g3.get_legend_handles_labels()
g3.legend(handles, ['Không', 'Có'], title='Gian lận', loc='best')


plt.subplots_adjust(hspace = 0.6, top = 0.85)

plt.show()


# Chuẩn bị dữ liệu cho card6
tmp = pd.crosstab(df_trans['card6'], df_trans['isFraud'], normalize='index') * 100
tmp = tmp.reset_index()
# Đổi tên cột 0 và 1 sang tiếng Việt (không dấu) để dễ tham chiếu
tmp.rename(columns={0:'KhongGianLan', 1:'GianLan'}, inplace=True)

plt.figure(figsize=(14,10))
plt.suptitle('Phân bổ Card 6', fontsize=22)

# --- Biểu đồ 1 (Trên, trái): Đếm số lượng card6 ---
plt.subplot(221)
g = sns.countplot(x='card6', data=df_trans, order=list(tmp.card6.values))
g.set_title("Phân bổ Card 6", fontsize=19)
g.set_ylim(0,480000)
g.set_xlabel("Các loại Card 6", fontsize=17)
g.set_ylabel("Số lượng", fontsize=17)
for p in g.patches:
    height = p.get_height()
    g.text(p.get_x()+p.get_width()/2.,
           height + 3,
           '{:1.2f}%'.format(height/total*100),
           ha="center",fontsize=14) 

# --- Biểu đồ 2 (Trên, phải): Số lượng VÀ Tỷ lệ % gian lận ---
plt.subplot(222)
g1 = sns.countplot(x='card6', hue='isFraud', data=df_trans, order=list(tmp.card6.values))
# Đặt chú thích (legend) cho g1
plt.legend(title='Gian lận', loc='best', labels=['Không', 'Có'])
gt = g1.twinx()
# Vẽ tỷ lệ % gian lận lên trục y thứ hai, sử dụng cột 'GianLan' đã đổi tên
gt = sns.pointplot(x='card6', y='GianLan', data=tmp, order=list(tmp.card6.values),
                   color='black', legend=False, )
gt.set_ylim(0,20)
gt.set_ylabel("% Giao dịch Gian lận", fontsize=16)
g1.set_title("Card 6 theo Mục tiêu (isFraud)", fontsize=19)
g1.set_xlabel("Các loại Card 6", fontsize=17)
g1.set_ylabel("Số lượng", fontsize=17)

# --- Biểu đồ 3 (Dưới): Phân bổ số tiền theo card6 ---
plt.subplot(212)
# Lọc các giao dịch <= 2000 để biểu đồ dễ nhìn hơn
g3 = sns.boxenplot(x='card6', y='TransactionAmt', hue='isFraud', order=list(tmp.card6.values),
                   data=df_trans[df_trans['TransactionAmt'] <= 2000] )
# Sửa lại tiêu đề cho đúng (code gốc bị nhầm "ProductCD")
g3.set_title("Phân bổ Số tiền Giao dịch theo Card 6 và Mục tiêu", fontsize=20)
g3.set_xlabel("Các loại Card 6", fontsize=17)
g3.set_ylabel("Giá trị Giao dịch", fontsize=17)

# Thêm chú thích (legend) cho biểu đồ 3
handles, labels = g3.get_legend_handles_labels()
g3.legend(handles, ['Không', 'Có'], title='Gian lận', loc='best')


plt.subplots_adjust(hspace = 0.6, top = 0.85)

plt.show()


# --- Bước 1: Điền giá trị bị thiếu (NaN) bằng "Miss" ---
for col in ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9']:
    df_trans[col] = df_trans[col].fillna("Miss") # "Miss" nghĩa là "Thiếu"
    
# --- Bước 2: Định nghĩa hàm vẽ biểu đồ ---
def ploting_dist_ratio(df, col, lim=2000):
    # Chuẩn bị dữ liệu 1: Tính % gian lận theo *số lượng*
    # tmp = Bảng tạm
    tmp = pd.crosstab(df[col], df['isFraud'], normalize='index') * 100
    tmp = tmp.reset_index()
    # Đổi tên cột 0 -> KhongGianLan, 1 -> GianLan
    tmp.rename(columns={0:'KhongGianLan', 1:'GianLan'}, inplace=True)

    plt.figure(figsize=(20,5))
    plt.suptitle(f'Phân bổ {col} ', fontsize=22)

    # --- Biểu đồ 1 (Trái): Số lượng VÀ % Gian lận ---
    plt.subplot(121)
    g = sns.countplot(x=col, data=df, order=list(tmp[col].values))
    g.set_title(f"Phân bổ {col}\nSố lượng và % Gian lận theo từng loại", fontsize=18)
    g.set_ylim(0,400000)
    
    # Trục Y thứ 2: Vẽ % Gian lận
    gt = g.twinx()
    gt = sns.pointplot(x=col, y='GianLan', data=tmp, order=list(tmp[col].values),
                       color='black', legend=False, )
    gt.set_ylim(0,20)
    gt.set_ylabel("% Giao dịch Gian lận", fontsize=16)
    g.set_xlabel(f"Các loại {col}", fontsize=16)
    g.set_ylabel("Số lượng", fontsize=17)
    
    # Thêm nhãn % vào các cột (ĐÃ SỬA: lặp qua g.patches thay vì gt.patches)
    for p in g.patches:
        height = p.get_height()
        g.text(p.get_x()+p.get_width()/2.,
                height + 3, # Thêm 3 để nhãn ở trên cột
                '{:1.2f}%'.format(height/total*100),
                ha="center",fontsize=14) 
        
    # Chuẩn bị dữ liệu 2: Tính % gian lận theo *tổng số tiền*
    # perc_amt = Phần trăm số tiền
    perc_amt = (df_trans.groupby(['isFraud',col])['TransactionAmt'].sum() / total_amt * 100).unstack('isFraud')
    perc_amt = perc_amt.reset_index()
    perc_amt.rename(columns={0:'KhongGianLan', 1:'GianLan'}, inplace=True)

    # --- Biểu đồ 2 (Phải): Phân bổ số tiền VÀ % Tổng tiền gian lận ---
    plt.subplot(122)
    g1 = sns.boxplot(x=col, y='TransactionAmt', hue='isFraud', 
                     data=df[df['TransactionAmt'] <= lim], order=list(tmp[col].values))
    
    # Thêm chú thích (legend) cho boxplot
    handles, labels = g1.get_legend_handles_labels()
    g1.legend(handles, ['Không', 'Có'], title='Gian lận', loc='best')

    # Trục Y thứ 2: Vẽ % Tổng tiền gian lận
    g1t = g1.twinx()
    g1t = sns.pointplot(x=col, y='GianLan', data=perc_amt, order=list(tmp[col].values),
                        color='black', legend=False, )
    g1t.set_ylim(0,5)
    g1t.set_ylabel("% Tổng tiền Gian lận", fontsize=16)
    g1.set_title(f"Phân bổ {col} theo Giao dịch", fontsize=18)
    g1.set_xlabel(f"Các loại {col}", fontsize=16)
    g1.set_ylabel("Số tiền Giao dịch (U$)", fontsize=16)
        
    plt.subplots_adjust(hspace=.4, wspace = 0.35, top = 0.80)
    
    plt.show()

# --- Bước 3: Gọi hàm để vẽ biểu đồ cho từng cột M ---
# for col in ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9']:
#     ploting_dist_ratio(df_trans, col)


for col in ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9']:
    ploting_dist_ratio(df_trans, col, lim=2500)


print("Các Phân vị của Đặc trưng (addr1, addr2): ")
print(df_trans[['addr1', 'addr2']].quantile([0.01, .025, .1, .25, .5, .75, .90, .975, .99]))


df_trans.loc[df_trans.addr1.isin(df_trans.addr1.value_counts()[df_trans.addr1.value_counts() <= 5000 ].index), 'addr1'] = "Others"
df_trans.loc[df_trans.addr2.isin(df_trans.addr2.value_counts()[df_trans.addr2.value_counts() <= 50 ].index), 'addr2'] = "Others"


def ploting_cnt_amt(df, col, lim=2000):
    """
    Vẽ 2 biểu đồ:
    1. Số lượng giao dịch VÀ % giao dịch gian lận.
    2. Tổng số tiền giao dịch VÀ % tổng tiền gian lận.
    """
    
    # --- Chuẩn bị dữ liệu 1: Tỷ lệ gian lận theo SỐ LƯỢNG ---
    tmp = pd.crosstab(df[col], df['isFraud'], normalize='index') * 100
    tmp = tmp.reset_index()
    # Đổi tên cột 0 -> KhongGianLan, 1 -> GianLan
    tmp.rename(columns={0:'KhongGianLan', 1:'GianLan'}, inplace=True)
    
    plt.figure(figsize=(16,14))    
    plt.suptitle(f'Phân bổ {col} ', fontsize=24)
    
    # --- Biểu đồ 1 (Trên): Số lượng VÀ % Giao dịch Gian lận ---
    plt.subplot(211)
    g = sns.countplot( x=col,  data=df, order=list(tmp[col].values))
    gt = g.twinx()
    # Vẽ % gian lận (theo số lượng) lên trục y thứ 2
    gt = sns.pointplot(x=col, y='GianLan', data=tmp, order=list(tmp[col].values),
                       color='black', legend=False, )
    gt.set_ylim(0,tmp['GianLan'].max()*1.1)
    gt.set_ylabel("% Giao dịch Gian lận", fontsize=16)
    
    g.set_title(f"Các giá trị {col} phổ biến nhất và % Giao dịch Gian lận", fontsize=20)
    g.set_xlabel(f"Tên loại {col}", fontsize=16)
    g.set_ylabel("Số lượng", fontsize=17)
    g.set_xticklabels(g.get_xticklabels(),rotation=45) # Xoay nhãn trục x
    
    sizes = []
    for p in g.patches:
        height = p.get_height()
        sizes.append(height)
        g.text(p.get_x()+p.get_width()/2.,
               height + 3,
               '{:1.2f}%'.format(height/total*100),
               ha="center",fontsize=12) 
        
    g.set_ylim(0,max(sizes)*1.15)
    
    # --- Chuẩn bị dữ liệu 2: Tỷ lệ gian lận theo SỐ TIỀN ---
    # perc_amt = Phần trăm số tiền
    perc_amt = (df.groupby(['isFraud',col])['TransactionAmt'].sum() \
                / df.groupby([col])['TransactionAmt'].sum() * 100).unstack('isFraud')
    perc_amt = perc_amt.reset_index()
    perc_amt.rename(columns={0:'KhongGianLan', 1:'GianLan'}, inplace=True)
    
    # amt = Tổng số tiền
    amt = df.groupby([col])['TransactionAmt'].sum().reset_index()
    perc_amt = perc_amt.fillna(0)
    
    # --- Biểu đồ 2 (Dưới): Tổng số tiền VÀ % Tổng tiền Gian lận ---
    plt.subplot(212)
    g1 = sns.barplot(x=col, y='TransactionAmt', 
                   data=amt, 
                   order=list(tmp[col].values))
    g1t = g1.twinx()
    # Vẽ % gian lận (theo số tiền) lên trục y thứ 2
    g1t = sns.pointplot(x=col, y='GianLan', data=perc_amt, 
                        order=list(tmp[col].values),
                        color='black', legend=False, )
    g1t.set_ylim(0,perc_amt['GianLan'].max()*1.1)
    g1t.set_ylabel("% Tổng tiền Gian lận", fontsize=16)
    
    g1.set_title(f"Tổng Giao dịch {col} + % Tổng và % Giao dịch Gian lận", fontsize=20)
    g1.set_xlabel(f"Tên loại {col}", fontsize=16)
    g1.set_ylabel("Tổng Số tiền Giao dịch (U$)", fontsize=16)
    g1.set_xticklabels(g1.get_xticklabels(),rotation=45) # Xoay nhãn trục x
    
    for p in g1.patches:
        height = p.get_height()
        g1.text(p.get_x()+p.get_width()/2.,
                height + 3,
                '{:1.2f}%'.format(height/total_amt*100),
                ha="center",fontsize=12) 
        
    plt.subplots_adjust(hspace=.4, top = 0.9)
    plt.show()
    
# --- Gọi hàm để vẽ cho cột 'addr1' ---
# Lọc 20 giá trị 'addr1' phổ biến nhất để vẽ
top_20_addr1 = list(df_trans['addr1'].value_counts().head(20).index)
ploting_cnt_amt(df_trans[df_trans['addr1'].isin(top_20_addr1)], 'addr1')


ploting_cnt_amt(df_trans, 'addr2')


df_trans.loc[df_trans['P_emaildomain'].isin(['gmail.com', 'gmail']),'P_emaildomain'] = 'Google'

df_trans.loc[df_trans['P_emaildomain'].isin(['yahoo.com', 'yahoo.com.mx',  'yahoo.co.uk',
                                         'yahoo.co.jp', 'yahoo.de', 'yahoo.fr',
                                         'yahoo.es']), 'P_emaildomain'] = 'Yahoo Mail'
df_trans.loc[df_trans['P_emaildomain'].isin(['hotmail.com','outlook.com','msn.com', 'live.com.mx', 
                                         'hotmail.es','hotmail.co.uk', 'hotmail.de',
                                         'outlook.es', 'live.com', 'live.fr',
                                         'hotmail.fr']), 'P_emaildomain'] = 'Microsoft'
df_trans.loc[df_trans.P_emaildomain.isin(df_trans.P_emaildomain\
                                         .value_counts()[df_trans.P_emaildomain.value_counts() <= 500 ]\
                                         .index), 'P_emaildomain'] = "Others"
df_trans.P_emaildomain.fillna("NoInf", inplace=True)


ploting_cnt_amt(df_trans, 'P_emaildomain')


df_trans.loc[df_trans['R_emaildomain'].isin(['gmail.com', 'gmail']),'R_emaildomain'] = 'Google'

df_trans.loc[df_trans['R_emaildomain'].isin(['yahoo.com', 'yahoo.com.mx',  'yahoo.co.uk',
                                             'yahoo.co.jp', 'yahoo.de', 'yahoo.fr',
                                             'yahoo.es']), 'R_emaildomain'] = 'Yahoo Mail'
df_trans.loc[df_trans['R_emaildomain'].isin(['hotmail.com','outlook.com','msn.com', 'live.com.mx', 
                                             'hotmail.es','hotmail.co.uk', 'hotmail.de',
                                             'outlook.es', 'live.com', 'live.fr',
                                             'hotmail.fr']), 'R_emaildomain'] = 'Microsoft'
df_trans.loc[df_trans.R_emaildomain.isin(df_trans.R_emaildomain\
                                         .value_counts()[df_trans.R_emaildomain.value_counts() <= 300 ]\
                                         .index), 'R_emaildomain'] = "Others"
df_trans.R_emaildomain.fillna("NoInf", inplace=True)


ploting_cnt_amt(df_trans, 'R_emaildomain')


resumetable(df_trans[['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8',
                      'C9', 'C10', 'C11', 'C12', 'C13', 'C14']])


df_trans[['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8',
                      'C9', 'C10', 'C11', 'C12', 'C13', 'C14']].describe()


df_trans.loc[df_trans.C1.isin(df_trans.C1\
                              .value_counts()[df_trans.C1.value_counts() <= 400 ]\
                              .index), 'C1'] = "Others"


ploting_cnt_amt(df_trans, 'C1')


df_trans.loc[df_trans.C2.isin(df_trans.C2\
                              .value_counts()[df_trans.C2.value_counts() <= 350 ]\
                              .index), 'C2'] = "Others"


ploting_cnt_amt(df_trans, 'C2')


# https://www.kaggle.com/c/ieee-fraud-detection/discussion/100400#latest-579480
import datetime

START_DATE = '2017-12-01'
startdate = datetime.datetime.strptime(START_DATE, "%Y-%m-%d")
df_trans["Date"] = df_trans['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds=x)))

df_trans['_Weekdays'] = df_trans['Date'].dt.dayofweek
df_trans['_Hours'] = df_trans['Date'].dt.hour
df_trans['_Days'] = df_trans['Date'].dt.day


ploting_cnt_amt(df_trans, '_Days')


ploting_cnt_amt(df_trans, '_Weekdays')


ploting_cnt_amt(df_trans, '_Hours')


# Giả sử df_trans đã có cột 'Date'
# import plotly.graph_objs as go
# from plotly.offline import iplot

# Đặt một số tùy chọn màu tĩnh
color_op = ['#5527A0', '#BB93D7', '#834CF7', '#6C941E', '#93EAEA', '#7425FF', '#F2098A', '#7E87AC', 
            '#EBE36F', '#7FD394', '#49C35D', '#3058EE', '#44FDCF', '#A38F85', '#C4CEE0', '#B63A05', 
            '#4856BF', '#F0DB1B', '#9FDBD9', '#B123AC']

# --- Biểu đồ 1: Tổng SỐ LƯỢNG giao dịch ---
# Gom nhóm theo ngày và đếm số lượng giao dịch
dates_temp = df_trans.groupby(df_trans.Date.dt.date)['TransactionAmt'].count().reset_index()

# Tạo 'trace' (đường biểu diễn) đầu tiên với các tham số cần thiết
trace = go.Scatter(x=dates_temp['Date'], y=dates_temp.TransactionAmt,
                   opacity = 0.8, line = dict(color = color_op[7]), name= 'Tổng số lượng Giao dịch')

# --- Biểu đồ 2: Tổng GIÁ TRỊ giao dịch ---
# Lấy tổng số tiền giao dịch theo ngày
dates_temp_sum = df_trans.groupby(df_trans.Date.dt.date)['TransactionAmt'].sum().reset_index()

# Sử dụng dates_temp_sum để tạo 'trace' (đường biểu diễn) thứ hai
trace1 = go.Scatter(x=dates_temp_sum.Date, line = dict(color = color_op[1]), name="Tổng Giá trị Giao dịch",
                    y=dates_temp_sum['TransactionAmt'], opacity = 0.8, yaxis='y2') # Chỉ định trục y thứ 2

# Tạo 'layout' (bố cục) cho phép chúng ta đặt tiêu đề 
# và cung cấp các tùy chọn thú vị để xử lý biểu đồ
layout = dict(
    title= "Tổng Giao dịch và Thông tin Gian lận theo Ngày",
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label='1 tháng', step='month', stepmode='backward'),
                dict(count=3, label='3 tháng', step='month', stepmode='backward'),
                dict(count=6, label='6 tháng', step='month', stepmode='backward'),
                dict(step='all', label='Tất cả')
            ])
        ),
        rangeslider=dict(visible = True), # Thanh trượt chọn khoảng thời gian
        type='date' ),
    yaxis=dict(title='Tổng số lượng Giao dịch'), # Tiêu đề trục y 1 (bên trái)
    yaxis2=dict(overlaying='y',
                anchor='x', side='right', # Đặt trục y 2 ở bên phải
                zeroline=False, showgrid=False,
                title='Tổng Giá trị Giao dịch') # Tiêu đề trục y 2
)

# Tạo hình (figure) với cả hai 'trace' và 'layout'
fig = dict(data= [trace, trace1,], layout=layout)

# Vẽ biểu đồ (tương đương với plt.show() nhưng dành cho Plotly)
iplot(fig)


# Đặt một số tùy chọn màu tĩnh
color_op = ['#5527A0', '#BB93D7', '#834CF7', '#6C941E', '#93EAEA', '#7425FF', '#F2098A', '#7E87AC', 
            '#EBE3F', '#7FD394', '#49C35D', '#3058EE', '#44FDCF', '#A38F85', '#C4CEE0', '#B63A05', 
            '#4856BF', '#F0DB1B', '#9FDBD9', '#B123AC']

# --- Chuẩn bị Dữ liệu ---

# Tính TỔNG GIÁ TRỊ giao dịch theo Ngày và theo (Không/Có) Gian lận
tmp_amt = df_trans.groupby([df_trans.Date.dt.date, 'isFraud'])['TransactionAmt'].sum().reset_index()
# Đếm TỔNG SỐ LƯỢNG giao dịch theo Ngày và theo (Không/Có) Gian lận
tmp_trans = df_trans.groupby([df_trans.Date.dt.date, 'isFraud'])['TransactionAmt'].count().reset_index()

# Lọc ra chỉ dữ liệu GIAN LẬN (isFraud == 1)
tmp_trans_fraud = tmp_trans[tmp_trans['isFraud'] == 1]
tmp_amt_fraud = tmp_amt[tmp_amt['isFraud'] == 1]

# (Đoạn này có vẻ không được sử dụng trong biểu đồ bên dưới)
# dates_temp = df_trans.groupby(df_trans.Date.dt.date)['TransactionAmt'].count().reset_index()

# --- Biểu đồ 1: SỐ LƯỢNG giao dịch GIAN LẬN ---
# Tạo 'trace' (đường biểu diễn) đầu tiên: SỐ LƯỢNG giao dịch gian lận
trace = go.Scatter(x=tmp_trans_fraud['Date'], y=tmp_trans_fraud.TransactionAmt,
                   opacity = 0.8, line = dict(color = color_op[1]), name= 'Số lượng Giao dịch Gian lận')

# --- Biểu đồ 2: GIÁ TRỊ giao dịch GIAN LẬN ---
# Tạo 'trace' (đường biểu diễn) thứ hai: TỔNG GIÁ TRỊ giao dịch gian lận
trace1 = go.Scatter(x=tmp_amt_fraud.Date, line = dict(color = color_op[7]), name="Giá trị Giao dịch Gian lận",
                    y=tmp_amt_fraud['TransactionAmt'], opacity = 0.8, yaxis='y2') # Chỉ định trục y thứ 2

# Tạo 'layout' (bố cục)
layout = dict(
    title= "GIAO DỊCH GIAN LẬN - Tổng số lượng và Giá trị theo Ngày",
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label='1 tháng', step='month', stepmode='backward'),
                dict(count=3, label='3 tháng', step='month', stepmode='backward'),
                dict(count=6, label='6 tháng', step='month', stepmode='backward'),
                dict(step='all', label='Tất cả')
            ])
        ),
        rangeslider=dict(visible = True), # Thanh trượt chọn khoảng thời gian
        type='date' ),
    yaxis=dict(title='Tổng số lượng Giao dịch'), # Tiêu đề trục y 1 (bên trái)
    yaxis2=dict(overlaying='y',
                anchor='x', side='right', # Đặt trục y 2 ở bên phải
                zeroline=False, showgrid=False,
                title='Tổng Giá trị Giao dịch') # Tiêu đề trục y 2
)

# Tạo hình (figure) với cả hai 'trace' và 'layout'
fig = dict(data= [trace, trace1], layout=layout)

# Vẽ biểu đồ (tương đương với plt.show()
iplot(fig)


df_id[['id_12', 'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_18',
       'id_19', 'id_20', 'id_21', 'id_22', 'id_23', 'id_24', 'id_25',
       'id_26', 'id_27', 'id_28', 'id_29', 'id_30', 'id_31', 'id_32',
       'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38']].describe(include='all')


df_train = df_trans.merge(df_id, how='left', left_index=True, right_index=True)


def cat_feat_ploting(df, col):
    tmp = pd.crosstab(df[col], df['isFraud'], normalize='index') * 100
    tmp = tmp.reset_index()
    tmp.rename(columns={0:'NoFraud', 1:'Fraud'}, inplace=True)

    plt.figure(figsize=(14,10))
    plt.suptitle(f'{col} Distributions', fontsize=22)

    plt.subplot(221)
    g = sns.countplot(x=col, data=df, order=tmp[col].values)
    # plt.legend(title='Fraud', loc='upper center', labels=['No', 'Yes'])

    g.set_title(f"{col} Distribution", fontsize=19)
    g.set_xlabel(f"{col} Name", fontsize=17)
    g.set_ylabel("Count", fontsize=17)
    # g.set_ylim(0,500000)
    for p in g.patches:
        height = p.get_height()
        g.text(p.get_x()+p.get_width()/2.,
                height + 3,
                '{:1.2f}%'.format(height/total*100),
                ha="center", fontsize=14) 

    plt.subplot(222)
    g1 = sns.countplot(x=col, hue='isFraud', data=df, order=tmp[col].values)
    plt.legend(title='Fraud', loc='best', labels=['No', 'Yes'])
    gt = g1.twinx()
    gt = sns.pointplot(x=col, y='Fraud', data=tmp, color='black', order=tmp[col].values, legend=False)
    gt.set_ylabel("% of Fraud Transactions", fontsize=16)

    g1.set_title(f"{col} by Target(isFraud)", fontsize=19)
    g1.set_xlabel(f"{col} Name", fontsize=17)
    g1.set_ylabel("Count", fontsize=17)

    plt.subplot(212)
    g3 = sns.boxenplot(x=col, y='TransactionAmt', hue='isFraud', 
                       data=df[df['TransactionAmt'] <= 2000], order=tmp[col].values )
    g3.set_title("Transaction Amount Distribuition by ProductCD and Target", fontsize=20)
    g3.set_xlabel("ProductCD Name", fontsize=17)
    g3.set_ylabel("Transaction Values", fontsize=17)

    plt.subplots_adjust(hspace = 0.4, top = 0.85)

    plt.show()


for col in ['id_12', 'id_15', 'id_16', 'id_23', 'id_27', 'id_28', 'id_29']:
    df_train[col] = df_train[col].fillna('NaN')
    cat_feat_ploting(df_train, col)


df_train.loc[df_train['id_30'].str.contains('Windows', na=False), 'id_30'] = 'Windows'
df_train.loc[df_train['id_30'].str.contains('iOS', na=False), 'id_30'] = 'iOS'
df_train.loc[df_train['id_30'].str.contains('Mac OS', na=False), 'id_30'] = 'Mac'
df_train.loc[df_train['id_30'].str.contains('Android', na=False), 'id_30'] = 'Android'
df_train['id_30'].fillna("NAN", inplace=True)


ploting_cnt_amt(df_train, 'id_30')


df_train.loc[df_train['id_31'].str.contains('chrome', na=False), 'id_31'] = 'Chrome'
df_train.loc[df_train['id_31'].str.contains('firefox', na=False), 'id_31'] = 'Firefox'
df_train.loc[df_train['id_31'].str.contains('safari', na=False), 'id_31'] = 'Safari'
df_train.loc[df_train['id_31'].str.contains('edge', na=False), 'id_31'] = 'Edge'
df_train.loc[df_train['id_31'].str.contains('ie', na=False), 'id_31'] = 'IE'
df_train.loc[df_train['id_31'].str.contains('samsung', na=False), 'id_31'] = 'Samsung'
df_train.loc[df_train['id_31'].str.contains('opera', na=False), 'id_31'] = 'Opera'
df_train['id_31'].fillna("NAN", inplace=True)
df_train.loc[df_train.id_31.isin(df_train.id_31.value_counts()[df_train.id_31.value_counts() < 200].index), 'id_31'] = "Others"


ploting_cnt_amt(df_train, 'id_31')


df_trans = pd.read_csv('../input/train_transaction.csv')
df_test_trans = pd.read_csv('../input/test_transaction.csv')

df_id = pd.read_csv('../input/train_identity.csv')
df_test_id = pd.read_csv('../input/test_identity.csv')

sample_submission = pd.read_csv('../input/sample_submission.csv', index_col='TransactionID')

df_train = df_trans.merge(df_id, how='left', left_index=True, right_index=True, on='TransactionID')
df_test = df_test_trans.merge(df_test_id, how='left', left_index=True, right_index=True, on='TransactionID')
# Bỏ 'left_index=True' và 'right_index=True'
# df_train = df_trans.merge(df_id, on='TransactionID', how='left')
# df_test = df_test_trans.merge(df_test_id, on='TransactionID', how='left')

print(df_train.shape)
print(df_test.shape)

# y_train = df_train['isFraud'].copy()
del df_trans, df_id, df_test_trans, df_test_id



df_train = reduce_mem_usage(df_train)
df_test = reduce_mem_usage(df_test)



emails = {'gmail': 'google', 'att.net': 'att', 'twc.com': 'spectrum', 
          'scranton.edu': 'other', 'optonline.net': 'other', 'hotmail.co.uk': 'microsoft',
          'comcast.net': 'other', 'yahoo.com.mx': 'yahoo', 'yahoo.fr': 'yahoo',
          'yahoo.es': 'yahoo', 'charter.net': 'spectrum', 'live.com': 'microsoft', 
          'aim.com': 'aol', 'hotmail.de': 'microsoft', 'centurylink.net': 'centurylink',
          'gmail.com': 'google', 'me.com': 'apple', 'earthlink.net': 'other', 'gmx.de': 'other',
          'web.de': 'other', 'cfl.rr.com': 'other', 'hotmail.com': 'microsoft', 
          'protonmail.com': 'other', 'hotmail.fr': 'microsoft', 'windstream.net': 'other', 
          'outlook.es': 'microsoft', 'yahoo.co.jp': 'yahoo', 'yahoo.de': 'yahoo',
          'servicios-ta.com': 'other', 'netzero.net': 'other', 'suddenlink.net': 'other',
          'roadrunner.com': 'other', 'sc.rr.com': 'other', 'live.fr': 'microsoft',
          'verizon.net': 'yahoo', 'msn.com': 'microsoft', 'q.com': 'centurylink', 
          'prodigy.net.mx': 'att', 'frontier.com': 'yahoo', 'anonymous.com': 'other', 
          'rocketmail.com': 'yahoo', 'sbcglobal.net': 'att', 'frontiernet.net': 'yahoo', 
          'ymail.com': 'yahoo', 'outlook.com': 'microsoft', 'mail.com': 'other', 
          'bellsouth.net': 'other', 'embarqmail.com': 'centurylink', 'cableone.net': 'other', 
          'hotmail.es': 'microsoft', 'mac.com': 'apple', 'yahoo.co.uk': 'yahoo', 'netzero.com': 'other', 
          'yahoo.com': 'yahoo', 'live.com.mx': 'microsoft', 'ptd.net': 'other', 'cox.net': 'other',
          'aol.com': 'aol', 'juno.com': 'other', 'icloud.com': 'apple'}

us_emails = ['gmail', 'net', 'edu']

# https://www.kaggle.com/c/ieee-fraud-detection/discussion/100499#latest-579654
for c in ['P_emaildomain', 'R_emaildomain']:
    df_train[c + '_bin'] = df_train[c].map(emails)
    df_test[c + '_bin'] = df_test[c].map(emails)
    
    df_train[c + '_suffix'] = df_train[c].map(lambda x: str(x).split('.')[-1])
    df_test[c + '_suffix'] = df_test[c].map(lambda x: str(x).split('.')[-1])
    
    df_train[c + '_suffix'] = df_train[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')
    df_test[c + '_suffix'] = df_test[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')


# # Label Encoding
# for f in df_train.drop('isFraud', axis=1).columns:
#     if df_train[f].dtype=='object' or df_test[f].dtype=='object': 
#         lbl = preprocessing.LabelEncoder()
#         lbl.fit(list(df_train[f].values) + list(df_test[f].values))
#         df_train[f] = lbl.transform(list(df_train[f].values))
#         df_test[f] = lbl.transform(list(df_test[f].values))   


from sklearn import preprocessing

# 1. Lấy danh sách các cột 'object' từ CẢ HAI bộ dữ liệu
cols_train = [f for f in df_train.columns if df_train[f].dtype=='object']
cols_test = [f for f in df_test.columns if df_test[f].dtype=='object']

# 2. Tìm các cột 'object' chung (quan trọng: dùng set để tìm giao)
common_object_cols = list(set(cols_train).intersection(set(cols_test)))

print(f"Tìm thấy {len(common_object_cols)} cột 'object' chung để mã hóa.")

# 3. Lặp qua CHỈ các cột chung
for f in common_object_cols:
    lbl = preprocessing.LabelEncoder()
    
    # 4. Fit trên TẤT CẢ giá trị (từ cả train và test) để đảm bảo an toàn
    # (Sử dụng .astype(str) để xử lý trường hợp có cả số và chữ lẫn lộn)
    all_values = list(df_train[f].astype(str).values) + list(df_test[f].astype(str).values)
    lbl.fit(all_values)
    
    # 5. Transform (biến đổi) cả hai cột
    df_train[f] = lbl.transform(list(df_train[f].astype(str).values))
    df_test[f] = lbl.transform(list(df_test[f].astype(str).values))

print("Mã hóa Label Encoding hoàn tất.")


df_train['Trans_min_mean'] = df_train['TransactionAmt'] - df_train['TransactionAmt'].mean()
df_train['Trans_min_std'] = df_train['Trans_min_mean'] / df_train['TransactionAmt'].std()
df_test['Trans_min_mean'] = df_test['TransactionAmt'] - df_test['TransactionAmt'].mean()
df_test['Trans_min_std'] = df_test['Trans_min_mean'] / df_test['TransactionAmt'].std()


df_train['TransactionAmt_to_mean_card1'] = df_train['TransactionAmt'] / df_train.groupby(['card1'])['TransactionAmt'].transform('mean')
df_train['TransactionAmt_to_mean_card4'] = df_train['TransactionAmt'] / df_train.groupby(['card4'])['TransactionAmt'].transform('mean')
df_train['TransactionAmt_to_std_card1'] = df_train['TransactionAmt'] / df_train.groupby(['card1'])['TransactionAmt'].transform('std')
df_train['TransactionAmt_to_std_card4'] = df_train['TransactionAmt'] / df_train.groupby(['card4'])['TransactionAmt'].transform('std')

df_test['TransactionAmt_to_mean_card1'] = df_test['TransactionAmt'] / df_test.groupby(['card1'])['TransactionAmt'].transform('mean')
df_test['TransactionAmt_to_mean_card4'] = df_test['TransactionAmt'] / df_test.groupby(['card4'])['TransactionAmt'].transform('mean')
df_test['TransactionAmt_to_std_card1'] = df_test['TransactionAmt'] / df_test.groupby(['card1'])['TransactionAmt'].transform('std')
df_test['TransactionAmt_to_std_card4'] = df_test['TransactionAmt'] / df_test.groupby(['card4'])['TransactionAmt'].transform('std')


df_train['TransactionAmt'] = np.log(df_train['TransactionAmt'])
df_test['TransactionAmt'] = np.log(df_test['TransactionAmt'])


df_test['isFraud'] = 'test'
df = pd.concat([df_train, df_test], axis=0, sort=False )
df = df.reset_index()
df = df.drop('index', axis=1)


def PCA_change(df, cols, n_components, prefix='PCA_', rand_seed=4):
    pca = PCA(n_components=n_components, random_state=rand_seed)

    principalComponents = pca.fit_transform(df[cols])

    principalDf = pd.DataFrame(principalComponents)

    df.drop(cols, axis=1, inplace=True)

    principalDf.rename(columns=lambda x: str(prefix)+str(x), inplace=True)

    df = pd.concat([df, principalDf], axis=1)
    
    return df


mas_v = df_train.columns[55:394]


from sklearn.preprocessing import minmax_scale
from sklearn.decomposition import PCA
# from sklearn.cluster import KMeans

for col in mas_v:
    df[col] = df[col].fillna((df[col].min() - 2))
    df[col] = (minmax_scale(df[col], feature_range=(0,1)))

    
df = PCA_change(df, mas_v, prefix='PCA_V_', n_components=30)


df = reduce_mem_usage(df)


df_train, df_test = df[df['isFraud'] != 'test'], df[df['isFraud'] == 'test'].drop('isFraud', axis=1)


df_train.shape


X_train = df_train.sort_values('TransactionDT').drop(['isFraud', 
                                                      'TransactionDT', 
                                                      #'Card_ID'
                                                     ],
                                                     axis=1)
y_train = df_train.sort_values('TransactionDT')['isFraud'].astype(bool)

X_test = df_test.sort_values('TransactionDT').drop(['TransactionDT',
                                                    #'Card_ID'
                                                   ], 
                                                   axis=1)
del df_train
df_test = df_test[["TransactionDT"]]


import os

# 1. Định nghĩa thư mục đầu ra
output_dir = "/kaggle/working/"

print(f"Bắt đầu lưu dữ liệu đã xử lý vào: {output_dir}")

# 2. Lưu các file CSV
# (index=False là quan trọng để không lưu cột index của DataFrame)
X_train.to_csv(os.path.join(output_dir, "X_train_processed.csv"), index=False)
X_test.to_csv(os.path.join(output_dir, "X_test_processed.csv"), index=False)
y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)

print("Đã lưu xong 3 file CSV:")

# In ra để xác nhận các file đã nằm trong /kaggle/working/
print(os.listdir(output_dir))


# from sklearn.model_selection import KFold,TimeSeriesSplit
# from sklearn.metrics import roc_auc_score
# from xgboost import plot_importance
# from sklearn.metrics import make_scorer

# import time
# def objective(params):
#     time1 = time.time()
#     params = {
#         'max_depth': int(params['max_depth']),
#         'gamma': "{:.3f}".format(params['gamma']),
#         'subsample': "{:.2f}".format(params['subsample']),
#         'reg_alpha': "{:.3f}".format(params['reg_alpha']),
#         'reg_lambda': "{:.3f}".format(params['reg_lambda']),
#         'learning_rate': "{:.3f}".format(params['learning_rate']),
#         'num_leaves': '{:.3f}'.format(params['num_leaves']),
#         'colsample_bytree': '{:.3f}'.format(params['colsample_bytree']),
#         'min_child_samples': '{:.3f}'.format(params['min_child_samples']),
#         'feature_fraction': '{:.3f}'.format(params['feature_fraction']),
#         'bagging_fraction': '{:.3f}'.format(params['bagging_fraction'])
#     }

#     print("\n############## New Run ################")
#     print(f"params = {params}")
#     FOLDS = 7
#     count=1
#     skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

#     tss = TimeSeriesSplit(n_splits=FOLDS)
#     y_preds = np.zeros(sample_submission.shape[0])
#     y_oof = np.zeros(X_train.shape[0])
#     score_mean = 0
#     for tr_idx, val_idx in tss.split(X_train, y_train):
#         clf = xgb.XGBClassifier(
#             n_estimators=600, random_state=4, verbose=True, 
#             tree_method='gpu_hist', 
#             **params
#         )

#         X_tr, X_vl = X_train.iloc[tr_idx, :], X_train.iloc[val_idx, :]
#         y_tr, y_vl = y_train.iloc[tr_idx], y_train.iloc[val_idx]
        
#         clf.fit(X_tr, y_tr)
#         #y_pred_train = clf.predict_proba(X_vl)[:,1]
#         #print(y_pred_train)
#         score = make_scorer(roc_auc_score, needs_proba=True)(clf, X_vl, y_vl)
#         # plt.show()
#         score_mean += score
#         print(f'{count} CV - score: {round(score, 4)}')
#         count += 1
#     time2 = time.time() - time1
#     print(f"Total Time Run: {round(time2 / 60,2)}")
#     gc.collect()
#     print(f'Mean ROC_AUC: {score_mean / FOLDS}')
#     del X_tr, X_vl, y_tr, y_vl, clf, score
#     return -(score_mean / FOLDS)


# space = {
#     # The maximum depth of a tree, same as GBM.
#     # Used to control over-fitting as higher depth will allow model 
#     # to learn relations very specific to a particular sample.
#     # Should be tuned using CV.
#     # Typical values: 3-10
#     'max_depth': hp.quniform('max_depth', 7, 23, 1),
    
#     # reg_alpha: L1 regularization term. L1 regularization encourages sparsity 
#     # (meaning pulling weights to 0). It can be more useful when the objective
#     # is logistic regression since you might need help with feature selection.
#     'reg_alpha':  hp.uniform('reg_alpha', 0.01, 0.4),
    
#     # reg_lambda: L2 regularization term. L2 encourages smaller weights, this
#     # approach can be more useful in tree-models where zeroing 
#     # features might not make much sense.
#     'reg_lambda': hp.uniform('reg_lambda', 0.01, .4),
    
#     # eta: Analogous to learning rate in GBM
#     # Makes the model more robust by shrinking the weights on each step
#     # Typical final values to be used: 0.01-0.2
#     'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
    
#     # colsample_bytree: Similar to max_features in GBM. Denotes the 
#     # fraction of columns to be randomly samples for each tree.
#     # Typical values: 0.5-1
#     'colsample_bytree': hp.uniform('colsample_bytree', 0.3, .9),
    
#     # A node is split only when the resulting split gives a positive
#     # reduction in the loss function. Gamma specifies the 
#     # minimum loss reduction required to make a split.
#     # Makes the algorithm conservative. The values can vary depending on the loss function and should be tuned.
#     'gamma': hp.uniform('gamma', 0.01, .7),
    
#     # more increases accuracy, but may lead to overfitting.
#     # num_leaves: the number of leaf nodes to use. Having a large number 
#     # of leaves will improve accuracy, but will also lead to overfitting.
#     'num_leaves': hp.choice('num_leaves', list(range(20, 250, 10))),
    
#     # specifies the minimum samples per leaf node.
#     # the minimum number of samples (data) to group into a leaf. 
#     # The parameter can greatly assist with overfitting: larger sample
#     # sizes per leaf will reduce overfitting (but may lead to under-fitting).
#     'min_child_samples': hp.choice('min_child_samples', list(range(100, 250, 10))),
    
#     # subsample: represents a fraction of the rows (observations) to be 
#     # considered when building each subtree. Tianqi Chen and Carlos Guestrin
#     # in their paper A Scalable Tree Boosting System recommend 
#     'subsample': hp.choice('subsample', [0.2, 0.4, 0.5, 0.6, 0.7, .8, .9]),
    
#     # randomly select a fraction of the features.
#     # feature_fraction: controls the subsampling of features used
#     # for training (as opposed to subsampling the actual training data in 
#     # the case of bagging). Smaller fractions reduce overfitting.
#     'feature_fraction': hp.uniform('feature_fraction', 0.4, .8),
    
#     # randomly bag or subsample training data.
#     'bagging_fraction': hp.uniform('bagging_fraction', 0.4, .9)
    
#     # bagging_fraction and bagging_freq: enables bagging (subsampling) 
#     # of the training data. Both values need to be set for bagging to be used.
#     # The frequency controls how often (iteration) bagging is used. Smaller
#     # fractions and frequencies reduce overfitting.
# }



# # Set algoritm parameters
# best = fmin(fn=objective,
#             space=space,
#             algo=tpe.suggest,
#             max_evals=27)

# # Print best parameters
# best_params = space_eval(space, best)


# print("BEST PARAMS: ", best_params)

# best_params['max_depth'] = int(best_params['max_depth'])


# clf = xgb.XGBClassifier(
#     n_estimators=300,
#     **best_params,
#     tree_method='gpu_hist'
# )

# clf.fit(X_train, y_train)

# y_preds = clf.predict_proba(X_test)[:,1] 


# feature_important = clf.get_booster().get_score(importance_type="weight")
# keys = list(feature_important.keys())
# values = list(feature_important.values())

# data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score", ascending=False)

# # Top 10 features
# data.head(20)



# sample_submission['isFraud'] = y_preds
# sample_submission.to_csv('XGB_hypopt_model.csv')




