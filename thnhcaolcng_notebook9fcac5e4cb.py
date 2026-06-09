
# # # **Đồ án Lập trình Python cho máy học**
!pip install pandas numpy seaborn matplotlib scipy scikit-learn
!pip install fuzzywuzzy python-Levenshtein # Levenshtein để tăng tốc fuzzywuzzy
!pip install nltk
!pip install xgboost
!pip install lightgbm
!pip install mlflow

# Đừng quên tải dữ liệu stopwords sau khi cài nltk và trước khi dùng
import nltk
nltk.download('stopwords')

# # %% [markdown] {"id":"39HOe0PYU_99"}
# # **Thành viên nhóm:**
# # 
# # Đỗ Phương Duy - 23520362
# # 
# # Đặng Quang Vinh - 23521786
# # 
# # Cao Lê Công Thành - 23521437


import pandas as pd
import mlflow
import mlflow.xgboost
import numpy as np
import xgboost

# Đọc dữ liệu đã xử lý (dạng .pkl nhanh và tiết kiệm bộ nhớ hơn .csv)
df = pd.read_pickle("/kaggle/input/check-point2/checkpoint_final_0.84.pkl")

# Giới hạn giá trị doanh số bán ra từ 0 đến 20
df['item_cnt_month'] = df['item_cnt_month'].clip(0, 20)
df = df.rename(columns={"item_cnt_month": "item_cnt"})
df = df[df != np.inf]

# %% [markdown] {"id":"Mp8SMQd8YrHX","jupyter":{"outputs_hidden":false}}
# Load file feature dataframe lên và lọc ra các hàng có item_cnt_month trong khoảng (0-20) để đảm bảo độ lỗi thấp

# %% [markdown] {"id":"t8TA6ASpQJZo","jupyter":{"outputs_hidden":false}}
# **Hàm huấn luyện mô hình LightGBM (sklearn API)**

# %% [code] {"id":"PHaS4YnJaVXA","jupyter":{"outputs_hidden":false}}
import warnings
warnings.filterwarnings("ignore", module="lightgbm")

import lightgbm as lgbm

# Định nghĩa hàm huấn luyện mô hình LightGBM với API của sklearn
# Định nghĩa hàm huấn luyện mô hình LightGBM với API của sklearn
def fit_booster(
    X_train,
    y_train,
    X_test=None,
    y_test=None,
    params=None,
    test_run=False,
    categoricals=[],
    dropcols=[],
    early_stopping=True,
):
    # Nếu không truyền tham số thì dùng mặc định
    if params is None:
        params = {"learning_rate": 0.1, "subsample_for_bin": 300000, "n_estimators": 50}

    # Số vòng dừng sớm nếu cần
    early_stopping_rounds = None
    if early_stopping:
        early_stopping_rounds = 30 # Đây là số vòng, không phải giá trị boolean

    # Tập dùng để đánh giá trong huấn luyện
    if test_run:
        eval_set = [(X_train, y_train)]
    else:
        eval_set = [(X_train, y_train), (X_test, y_test)]

    booster = lgbm.LGBMRegressor(**params)

    # Chỉ giữ lại các cột phân loại có tồn tại trong X_train
    categoricals = [c for c in categoricals if c in X_train.columns]

    # Huấn luyện mô hình
    booster.fit(
        X_train,
        y_train,
        eval_set=eval_set,
        eval_metric="rmse",  # eval_metric có thể là chuỗi hoặc list trong fit(), dùng chuỗi cũng được
        
        categorical_feature=categoricals,
        callbacks=[lgbm.early_stopping(early_stopping_rounds, verbose=100)] if early_stopping_rounds else None # CÁCH ĐÚNG ĐỂ DÙNG EARLY STOPPING VÀ VERBOSE TRONG SCIKIT-LEARN API MỚI
        # Hoặc chỉ đơn giản là early_stopping_rounds=early_stopping_rounds nếu không cần verbose in mỗi N vòng
    )

    return booster
# %% [markdown] {"id":"4V2PgTUwcsKU","jupyter":{"outputs_hidden":false}}
# Mô hình LightGBM (lgbm.LGBMRegressor) được  chọn cho hàm fit_booster vì nó là một thuật toán Gradient Boosting cực kỳ hiệu quả và chính xác đối với dữ liệu dạng bảng như trong cuộc thi này. Nó nổi bật nhờ tốc độ huấn luyện nhanh và sử dụng ít bộ nhớ hơn nhiều thuật toán khác, điều này rất quan trọng khi xử lý dữ liệu lớn. Hơn nữa, LightGBM có khả năng xử lý các đặc trưng hạng mục (categorical features) một cách tự nhiên và hiệu quả, giúp đơn giản hóa quá trình tiền xử lý và thường mang lại kết quả tốt hơn cho các cột như shop_id hay item_category_id

# %% [markdown] {"id":"Q0S_7Jtbc-DG","jupyter":{"outputs_hidden":false}}
# Chia tập train và valdation từ feature matrix , dữ liệu tháng 33 dùng để validation

# %% [markdown] {"id":"iOPITLvpRpkD","jupyter":{"outputs_hidden":false}}
# **Hyperparameter tối ưu**

# %% [code] {"id":"axtCe45DdMdn","jupyter":{"outputs_hidden":false}}
params = {
    "num_leaves": 966,
    "cat_smooth": 45.01680827234465,
    "min_child_samples": 27,
    "min_child_weight": 0.021144950289224463,
    "max_bin": 214,
    "learning_rate": 0.01,
    "subsample_for_bin": 300000,
    "min_data_in_bin": 7,
    "colsample_bytree": 0.8,
    "subsample": 0.6,
    "subsample_freq": 5,
    "n_estimators": 750,
}

# %% [markdown] {"id":"dNuPJeGwalre","jupyter":{"outputs_hidden":false}}
# Bộ tham số này cấu hình một mô hình LightGBM có khả năng học các mẫu phức tạp (do num_leaves cao). Nó sử dụng tốc độ học rất nhỏ (learning_rate: 0.01) và số lượng cây lớn (n_estimators: 750), kết hợp với các kỹ thuật regularization mạnh mẽ như chỉ dùng một phần dữ liệu (subsample) và đặc trưng (colsample_bytree) cho mỗi cây, yêu cầu số mẫu tối thiểu trong lá (min_child_samples), và làm mịn xử lý đặc trưng hạng mục (cat_smooth). Ngoài ra, các tham số liên quan đến gom nhóm đặc trưng (max_bin, subsample_for_bin) giúp tối ưu tốc độ và bộ nhớ. Mô hình được tinh chỉnh để hướng tới độ chính xác cao trong khi vẫn kiểm soát tốt overfitting.

# %% [markdown] {"id":"dik7zGMiSAGR","jupyter":{"outputs_hidden":false}}
# **Hàm huấn luyện bằng native API của LightGBM (lgb.train)**

# %% [code] {"id":"WOIyOO3scrIU","jupyter":{"outputs_hidden":false}}


import pandas as pd
import numpy as np
import lightgbm as lgb


import warnings
warnings.filterwarnings("ignore")

# Hàm huấn luyện sử dụng lgb.train (native API của LightGBM)
def build_lgb_model(X_train, y_train):
    params = {
        'objective': 'rmse',
        'metric': 'rmse',
        'num_leaves': 1023,
        'min_data_in_leaf': 10,
        'feature_fraction': 0.7,
        'learning_rate': 0.01,
        'num_rounds': 500,
        'seed': 1
    }

    # Các cột phân loại giúp mô hình học tốt hơn các đặc trưng rời rạc
    cat_features = ['item_category_id', 'month', 'shop_id']

    # Định nghĩa tập huấn luyện
    lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=cat_features)

    # Huấn luyện mô hình (ở đây dùng lại lgb_train làm tập đánh giá - nên cải thiện)
    model = lgb.train(
        params=params,
        train_set=lgb_train,
        valid_sets=lgb_train
    )

    return model

# %% [markdown] {"id":"4vYV-AMOgm4n","jupyter":{"outputs_hidden":false}}
# Định nghĩa hàm build_lgb_model dùng để huấn luyện một mô hình LightGBM khác,dùng trong việc kết hợp (stacking/ensembling) với mô hình từ fit_booster. Bên trong hàm, nó thiết lập một bộ siêu tham số riêng cho mô hình này (với số lá rất cao là 1023, tốc độ học nhỏ 0.01, và 500 cây) và chỉ định rõ các cột hạng mục (item_category_id, month, shop_id). Điểm đáng chú ý là nó  đánh giá hiệu suất ngay trên chính tập huấn luyện (valid_sets=lgb_train) thay vì một tập kiểm tra riêng biệt. Sẽ cho ra một mô hình overfit .Trong Stacking, mục tiêu không nhất thiết là tạo ra từng mô hình cơ sở hoàn hảo một cách độc lập. Thay vào đó, việc có một tập hợp các mô hình cơ sở "đủ tốt" và "đủ khác biệt" là quan trọng hơn. Một mô hình hơi overfit (như build_lgb_model có thể trở thành) đóng góp vào sự khác biệt đó, cung cấp một tín hiệu dự đoán bổ sung mà mô hình meta có thể tận dụng để cải thiện hiệu suất tổng thể, miễn là các mô hình khác trong ensemble có thể bù đắp cho điểm yếu của nó. Nếu các mô hình còn lại trong ensemble mang tính regular hóa cao (bias cao), thì mô hình overfit này giúp bổ sung phương sai (variance) vào hệ thống.

# %% [code] {"id":"gi8PofnNijVM","jupyter":{"outputs_hidden":false}}
# Loại bỏ 2 tháng đầu do dữ liệu có thể bị nhiễu
keep_from_month = 2

# Xác định tháng dự đoán (tháng 33 là tháng test)
test_month = 33

# Loại bỏ một số cột không cần thiết
dropcols = [
    "shop_id",
    "item_id",
    "new_item",
]

# %% [markdown] {"id":"pXo1Pe66m79o","jupyter":{"outputs_hidden":false}}
# Các đặc trưng được tạo ra trong quá trình feature engineering thường dựa trên dữ liệu lịch sử (ví dụ: lag, rolling mean, mean encoding, tuổi item/shop).
# Đối với tháng 0, không có dữ liệu lịch sử nào trước đó, nên nhiều đặc trưng sẽ có giá trị mặc định, NaN, hoặc không có ý nghĩa.
# Đối với tháng 1, chỉ có dữ liệu của tháng 0 để tính toán, nên các đặc trưng (đặc biệt là các đặc trưng dựa trên cửa sổ dài hạn như rolling 12 tháng) vẫn có thể chưa ổn định hoặc chưa phản ánh đúng xu hướng.
# Đặc trưng item_age sẽ bằng 0 cho tất cả các item trong tháng 0, và có thể không phản ánh đúng tuổi thực nếu item đã tồn tại trước đó.
# Bằng cách loại bỏ dữ liệu của tháng 0 và 1 (date_block_num < 2) khỏi quá trình huấn luyện cuối cùng, tránh đưa vào mô hình những dữ liệu có đặc trưng bị sai lệch này, hy vọng giúp mô hình học được các mối quan hệ chính xác hơn từ dữ liệu ổn định hơn.

# %% [markdown] {"id":"4-LGMBXNafHm","jupyter":{"outputs_hidden":false}}
# ## **Huấn luyện các mô hình theo từng tháng để tạo đầu vào cho mô hình tổng hợp (stacking)**

# %% [markdown] {"id":"mm3gncNoejMp","jupyter":{"outputs_hidden":false}}
# Huấn luyện các mô hình XGBoost và LightGBM theo từng tháng.
# 
# ---
# 
# 
# 
# Mỗi mô hình sẽ dự đoán giá trị item_cnt cho tháng kế tiếp (tháng i).
# 
# Kết quả dự đoán của mỗi mô hình trên tập X_val được lưu lại để làm đặc trưng đầu vào cho mô hình tổng hợp (meta model) trong stacking.
# 
# Kết quả được lưu thành file .pkl.

# %% [code] {"id":"4vGuzawUbX5X","jupyter":{"outputs_hidden":false}}
# Các thư viện cần thiết
import xgboost as xgb
import pandas as pd
import numpy as np

# Khởi tạo các list lưu dự đoán và đặc trưng phục vụ cho stacking
preds_arr_lgb=[]              # Dự đoán từ mô hình LightGBM native API
vals_arr_lgb=[]               # Dự đoán từ mô hình LightGBM native API
vals_arr_lgb_84=[]            # Dự đoán từ LightGBM (fit_booster - sklearn API)
preds_arr_xgb=[]              # Dự đoán từ mô hình XGBoost
vals_arr_xgb=[]               # Dự đoán từ mô hình XGBoost
preds_arr_nn=[]               # Placeholder cho mô hình neural net (nếu có)
shop_id=[]
item_id=[]
cat_id=[]
month_arr=[]

# %% [code] {"id":"KV4KdCu4dAog","jupyter":{"outputs_hidden":false}}
"""
Tạo tập huấn luyện và tập kiểm tra

"""
# Lặp qua từng tháng từ 25 đến 34 để huấn luyện mô hình
# for i in range(25, 35):
#     # Chuẩn bị dữ liệu huấn luyện: lấy tất cả dữ liệu trước tháng i
#     X_train = df[df["date_block_num"] < i].drop(['item_cnt', "date_block_num"], axis=1)
#     y_train = df.loc[df["date_block_num"] < i, 'item_cnt']

#     # Dữ liệu kiểm tra (validation): là tháng hiện tại (i)
#     X_val = df[df["date_block_num"] == i].drop(['item_cnt', "date_block_num"], axis=1)
#     y_val = df.loc[df["date_block_num"] == i, 'item_cnt']

"""
    Huấn luyện và đánh giá mô hình theo từng tháng (date_block_num từ 25 đến 34), mô phỏng quá trình dự đoán thực tế.
    Đây là hình thức time series cross-validation, cực kỳ phù hợp với bài toán chuỗi thời gian như dự đoán doanh số.

    Huấn luyện các tháng từ 25 đến 34 vì:

    - Đang thực hiện việc huấn luyện theo từng tháng, trong đó:

      + X_train: gồm tất cả dữ liệu trước tháng i.

      + X_val: là dữ liệu của tháng i.

    - Nếu bắt đầu từ tháng 0 → không còn "tháng -1" để làm X_train.

    - Nếu bắt đầu từ tháng 1 → X_train sẽ rất ít (chỉ có tháng 0).

    - Từ tháng 0 đến 24 không đủ thông tin huấn luyện ổn định cho mô hình.

"""

"""
    -------------------Huấn luyện mô hình XGBoost-------------------------
"""
    # model_name="XGB_iterations_100"+str(i)
    # # mlflow.xgboost.autolog(registered_model_name=model_name) # Ghi lại mô hình tự động bằng MLflow
    # SEED=0 # Biến SEED được khai báo nhưng không sử dụng ở đây
    # xgb_model = xgb.XGBRegressor(
    #     objective='reg:squarederror', # Explicitly set the objective (default for regressor anyway)
    #     # You can potentially set eval_metric here if needed, but it defaults to rmse for reg:squarederror
    #     learning_rate=0.05,
    #     max_leaves=800,
    #     num_round=1000, # This parameter 'num_round' seems like native API, not common for sklearn API, maybe leftover? 'n_estimators' is used for sklearn API number of boosting rounds/trees
    #     n_estimators=100, # Number of trees - this is the parameter for sklearn API
    #     max_depth=10,
    #     early_stopping_rounds=10 # early_stopping_rounds is correct for sklearn API fit()
    # )
    # # Huấn luyện mô hình XGBoost
    # # REMOVE eval_metric argument here
    # xgb_model.fit(X_train, y_train, eval_set=[(X_train,y_train)], verbose=True) # Thêm verbose để thấy kết quả eval set nếu cần
    # # XGBoost Regressor với objective='reg:squarederror' sẽ tự động sử dụng RMSE làm metric cho eval_set

    # # Dự đoán và giới hạn kết quả trong khoảng [0, 20] như yêu cầu bài toán
    # val_pred = xgb_model.predict(X_val).clip(0, 20)
    # vals_arr_xgb.append(val_pred) # Lưu dự đoán

"""
    -------------------Huấn luyện mô hình LightGBM với sklearn API (fit_booster)-------------------------
"""
    # Các cột hạng mục sẽ được xử lý đặc biệt bởi LightGBM (categorical features)
    # categoricals = [
    #     "item_category_id",
        
    #     "month",
    # ]
    

    

    # Ghi LightGBM vào MLflow
    # mlflow.lightgbm.autolog(registered_model_name=model_name)

    #Huấn luyện mô hình LightGBM sử dụng sklearn API (fit_booster)
    # lgbooster = fit_booster(
    #     X_train,
    #     y_train,
    #     X_val,
    #     y_val,
    #     params=params,
    #     test_run=True,
    #     categoricals=categoricals,
    # )

    # # Dự đoán và giới hạn giá trị
    # val_pred = lgbooster.predict(X_val).clip(0, 20)
    # vals_arr_lgb_84.append(val_pred)  # Lưu dự đoán từ mô hình này
"""
    Dùng LightGBM dưới dạng LGBMRegressor thông qua hàm fit_booster .

    Sử dụng categoricals giúp LightGBM xử lý tốt hơn đặc trưng phân loại → tăng độ chính xác.

    Dự đoán được lưu lại phục vụ cho stacking.

"""


"""
    -------------------Huấn luyện mô hình LightGBM với native API (build_lgb_model)-----------------
"""
    # model_name="LGB_iteration_750"+str(i)
    # # mlflow.lightgbm.autolog(registered_model_name=model_name)

    # lgb_model = build_lgb_model(X_train, y_train)
    # val_pred=lgb_model.predict(X_val).clip(0,20)

    # vals_arr_lgb.append(val_pred)



# path='/kaggle/working/'
# vals_arr_xgb_series=pd.Series(vals_arr_xgb)
# vals_arr_xgb_series.to_pickle(path+'vals_arr_xgb_84.pkl')



# vals_arr_lgb_series_84s=pd.Series(vals_arr_lgb_84)
# vals_arr_lgb_series_84s.to_pickle(path+'vals_arr_lgb_84.pkl')


# vals_arr_lgb_series=pd.Series(vals_arr_lgb)
# vals_arr_lgb_series.to_pickle(path+'vals_arr_lgb.pkl')

# %% [markdown] {"id":"LDzEUviP0XOQ","jupyter":{"outputs_hidden":false}}
# ## **Huấn luyện và dự đoán với mô hình cuối cùng (hồi quy tuyến tính) trong mô hình stacking**

# %% [markdown] {"id":"hIFJAIVnNvpd","jupyter":{"outputs_hidden":false}}
# **Chuẩn bị dữ liệu huấn luyện cho mô hình cuối cùng**

# %% [code] {"id":"Yaq2sObn0yvu","jupyter":{"outputs_hidden":false}}
#Đọc lại 3 file pickle chứa kết quả dự đoán của các mô hình con
vals_arr_xgb = pd.read_pickle('/kaggle/input/datapkl1/vals_arr_xgb_84.pkl')
vals_arr_lgb = pd.read_pickle('/kaggle/input/datapkl1/vals_arr_lgb.pkl')
vals_arr_lgb_84s = pd.read_pickle('/kaggle/input/datapkl1/vals_arr_lgb_84.pkl')

# # %% [markdown] {"id":"SPmxQl6XkIV7","jupyter":{"outputs_hidden":false}}
# # Phân tách dữ liệu theo từng fold, sau đó ghép lại tạo nên chuỗi dự đoán đầy đủ theo thời gian (bỏ fold cuối vì fold cuối là tập test)

# # %% [code] {"id":"icay5Eh81k8-","jupyter":{"outputs_hidden":false}}
# # Giải nén từng mảng dự đoán thành danh sách Series, rồi gộp thành một Series lớn
# # Mỗi phần tử ban đầu là một mảng các giá trị dự đoán của từng fold trong cross-validation

for i in range(25,35):
    X_train=df[df["date_block_num"]<i].drop(['item_cnt',"date_block_num"],axis=1)
    y_train=df.loc[(df["date_block_num"]<i),'item_cnt']
    X_val=df[df["date_block_num"]==i].drop(['item_cnt',"date_block_num"],axis=1)
    y_val=df.loc[(df["date_block_num"]==i),'item_cnt']
    
    shop_id.append(X_val["shop_id"].values)
    item_id.append(X_val["item_id"].values)
    cat_id.append(X_val["item_category_id"].values)
    month_arr.append(X_val["month"].values)
    preds_arr_lgb.append([y_val])
    
s = []
for i in vals_arr_xgb[:-1]:
    a = pd.Series(i)
    s.append(a)
series_xgb = pd.concat(s)

s = []
for i in vals_arr_lgb[:-1]:
    a = pd.Series(i)
    s.append(a)
series_lgb = pd.concat(s)

s = []
for i in vals_arr_lgb_84s[:-1]:
    a = pd.Series(i)
    s.append(a)
series_lgbs = pd.concat(s)

# # %% [markdown] {"id":"yZuBQgY9kPr7","jupyter":{"outputs_hidden":false}}
# # Lấy lại nhãn thực tế (actual_all) từ các fold (bỏ fold cuối vì fold cuối là tập test)

# # %% [code] {"id":"yz0_j45hdTc0","jupyter":{"outputs_hidden":false}}
# # Lấy lại giá trị nhãn thực tế (y_val) từ kết quả dự đoán lưu trước đó
# # Mỗi phần tử là (y_val, y_pred), ta chỉ lấy y_val

s = []
for i in preds_arr_lgb[:-1]: 
    s.append(pd.Series(i[0].values))  
actual_all = pd.concat(s)

# # %% [markdown] {"id":"mpAYbEPrkZL1","jupyter":{"outputs_hidden":false}}
# # Chuyển các đặc trưng về shop_id, item_id, category_id, month thành Series (bỏ fold cuối vì fold cuối là tập test)

# # %% [code] {"id":"nyzCJsTNdV06","jupyter":{"outputs_hidden":false}}
# # Chuyển danh sách các đặc trưng shop_id, item_id, category_id, month thành các Series

s = []
for i in shop_id[:-1]:
    s.append(pd.Series(i))
series_shop = pd.concat(s)

s = []
for i in item_id[:-1]:
    s.append(pd.Series(i))
series_item = pd.concat(s)

s = []
for i in cat_id[:-1]:
    s.append(pd.Series(i))
series_cat = pd.concat(s)

s = []
for i in month_arr[:-1]:
    s.append(pd.Series(i))
series_month = pd.concat(s)

# # %% [markdown] {"id":"KBduJkq2kiZE","jupyter":{"outputs_hidden":false}}
# # Gộp toàn bộ dữ liệu thành một DataFrame

# # %% [code] {"id":"DzbN_KE4dnqe","jupyter":{"outputs_hidden":false}}
# # Gộp tất cả các Series lại thành một DataFrame duy nhất
# # Đây là dữ liệu đầu vào cho mô hình stacking

datframe = pd.concat([
    series_xgb,     # dự đoán từ mô hình xgb
    series_lgb,     # dự đoán từ mô hình LightGBM với native API
    series_shop,    # shop_id
    series_item,    # item_id
    actual_all,     # giá trị thực tế
    series_cat,     # category_id
    series_lgbs,    # dự đoán từ mô hình LightGBM với sklearn API
    series_month    # tháng (month)
], axis=1)

# # %% [markdown] {"id":"ZEicak8fkmw4","jupyter":{"outputs_hidden":false}}
# # Đổi tên các cột trong DataFrame

# # %% [code] {"id":"0pIfSIYDjWHm","jupyter":{"outputs_hidden":false}}
# # Đặt tên cột cho rõ ràng
datframe = datframe.rename(columns={
    0: "xgb",
    1: "lgb",
    2: "shop_id",
    3: "item_id",
    4: "actual",
    5: "category_id",
    6: "lgb_new",
    7: "month"
})


    
series_xgb=pd.Series(vals_arr_xgb[-1:].reset_index(drop=True)[0])
    
series_lgb=pd.Series(vals_arr_lgb[-1:].reset_index(drop=True)[0])
series_lgbs=pd.Series(vals_arr_lgb_84s[-1:].reset_index(drop=True)[0])
series_month_ser=pd.Series(month_arr[-1:][0])

series_shop_ser=pd.Series(shop_id[-1])   
series_item_ser=pd.Series(item_id[-1])
series_cat_ser=pd.Series(cat_id[-1])



actual_all=pd.Series(preds_arr_lgb[-1][0].values)



datframe_test=pd.concat([series_xgb.reset_index(drop=True),series_lgb.reset_index(drop=True),actual_all.reset_index(drop=True),\
                        series_shop_ser.reset_index(drop=True),series_item_ser.reset_index(drop=True),\
                        series_cat_ser.reset_index(drop=True),series_lgbs.reset_index(drop=True)\
                        ,series_month_ser.reset_index(drop=True)],axis=1)
datframe_test=datframe_test.rename(columns={0:"xgb",1:"lgb",2:"actual",3:"shop_id",4:"item_id",5:"category_id",6:"lgb_new",\
                                          7:"month" })

to_drop=["actual","month","shop_id","item_id"]

X=datframe.drop(to_drop,axis=1)
Y=datframe["actual"]
X_test=datframe_test.drop(to_drop,axis=1)


from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(X, Y)

pred_stack_reg=reg.predict(X_test)


submission = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sample_submission.csv')
submission['item_cnt_month'] = pred_stack_reg.clip(0,20)

submission[['ID', 'item_cnt_month']].to_csv('submission.csv', index=False)






