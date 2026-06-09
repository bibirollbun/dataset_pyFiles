!pip install tensorflow


!pip install sdv


import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.linear_model import PoissonRegressor, GammaRegressor
from sklearn.metrics import mean_squared_log_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import pandas as pd
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import Metadata


df = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
df.head(10)


df.info()


df.describe()


df.columns


def check_missing_value(df):
    mask = (
        df.isnull() | 
        df.apply(lambda x : isinstance(x, float) and np.isnan(x)) | 
        df.apply(lambda x : isinstance(x, str) and x.strip().lower() in ['nan', 'null', 'none', ''])
    )

    missing_summary = mask.sum()

    print("Total missing values of each column: ")
    print(missing_summary)
    print("\nTotal missing values of Dataframe: ", mask.sum().sum())

check_missing_value(df)


#Vẽ histogram
plt.figure(figsize = (8, 5))
plt.hist(df['Age'], bins = [20,30,40,50,60,70,80], color = 'skyblue', edgecolor = 'black')

#Thêm nhãn
plt.title("Age's distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")

plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()



for feature in df.columns:
    print(df[feature].dtype)


numerical_features = []
categorical_features = []

for feature in df.columns:
    if df[feature].dtype == "object":
        categorical_features.append(feature)
    else:
        numerical_features.append(feature)

numerical_features.remove("Calories")
numerical_features.remove("id")

print(f"Numerical_features: {numerical_features}")
print(f"\nCategorical_features: {categorical_features}")


n_cols = 2
n_rows = len(numerical_features) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize = (12, 4 * n_rows))

for idx, numerical_feature in enumerate(numerical_features):
    i = idx // n_cols  # hàng
    j = idx % n_cols   # cột
    ax_ij = axes[i, j]
    sns.scatterplot(x=df[numerical_feature], y=df["Calories"], ax=ax_ij)
    ax_ij.set_title(f"Calories follow by {numerical_feature}")

#Xóa các ô dư thừa (ô vượt qua số lượng feature)
total_subplots = n_rows * n_cols
for idx in range(len(numerical_features), total_subplots):
    i = idx // n_cols
    j = idx % n_cols
    fig.delaxes(axes[i, j])

#Tự động điều chỉnh các phần tử trong biểu đồ (trục, nhãn, tiêu đề)
#giúp không bị chồng chéo dữ liệu
plt.tight_layout() 
plt.show()


#Vẽ histogram
plt.figure(figsize = (8, 5))
plt.hist(df['Calories'], color = 'skyblue', edgecolor = 'black')

#Thêm nhãn
plt.title("Calories's distribution")
plt.xlabel("Calories")
plt.ylabel("Frequency")

plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()



#Vẽ histogram
plt.figure(figsize = (8, 5))
plt.hist(df['Body_Temp'], color = 'skyblue', edgecolor = 'black')

#Thêm nhãn
plt.title("Body_Temp's distribution")
plt.xlabel("Body_Temp")
plt.ylabel("Frequency")

plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()



# Tính correlation matrix
corr_matrix = df.corr(numeric_only=True)

# Vẽ heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=0.5)
plt.title('Heatmap of Correlation Matrix')
plt.tight_layout()
plt.show()



import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Body_Temp', y='Duration')
plt.title('Mối quan hệ giữa Body_temp và Duration')
plt.xlabel('Nhiệt độ cơ thể (°C)')
plt.ylabel('Thời gian vận động (Duration)')
plt.grid(True)
plt.show()



print(100//3)
print(100/3)


#Chuẩn hóa cột height thành mét
df['Height_m'] = df['Height'] / 100
df['BMI'] = df['Weight'] / (df['Height_m'] ** 2)
df.drop(columns = ['Height_m'], inplace = True)

df['MET_duration'] = df['Duration'] * df['Heart_Rate'] / 100

conditions = [
    (df['Body_Temp'] >= 37) & (df['Body_Temp'] < 38),
    (df['Body_Temp'] >= 38) & (df['Body_Temp'] < 39),
    (df['Body_Temp'] >= 39) & (df['Body_Temp'] < 40),
    (df['Body_Temp'] >= 40)
]
values = [0, 1, 2, 3]
df['Temp_level'] = np.select(conditions, values)

df['Temp_per_min'] = df['Body_Temp'] / df['Duration']

df['BodyTemp_Heart'] = df['Body_Temp'] * df['Heart_Rate']



df.columns


import numpy as np
df['Calories'] = df['Calories'].apply(lambda x: np.log1p(x))


onehot = OneHotEncoder()
encoded = onehot.fit_transform(df[['Sex']])
encoded = onehot.transform(df[['Sex']]).toarray()
columns = onehot.get_feature_names_out(['Sex'])

print(f"encoded: {encoded}")
print(f"columns: {columns}")

# Gộp với DataFrame gốc
import pandas as pd
df_encoded = pd.DataFrame(encoded, columns=columns)
df = pd.concat([df.drop(columns=['Sex']), df_encoded], axis=1)



df.head()


import sklearn.ensemble
print(dir(sklearn.ensemble))



X = df.drop(columns = ['Calories', 'id'])
y = df['Calories']


test_size = 0.2
random_state = 42

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=test_size, random_state=random_state
)


print(f"Len of training_set: {len(X_train)}")
print(f"Len of validation_set: {len(X_val)}")


def rmsle(y_true, y_pred):
    from sklearn.metrics import mean_squared_log_error
    return np.sqrt(mean_squared_log_error(y_true, y_pred))


RandomForest_model = RandomForestRegressor(random_state = 42, n_jobs = -1, verbose = 1)
RandomForest_model.fit(X_train, y_train)

y_preds_rf = RandomForest_model.predict(X_val)
y_preds_rf


rmsle_score_rf = rmsle(y_val, y_preds_rf)
print(f"RandomForest has {rmsle_score_rf} rmsle")


GradientBoosting_model = GradientBoostingRegressor()
GradientBoosting_model.fit(X_train, y_train)

y_preds_gb = GradientBoosting_model.predict(X_val)


rmsle_score_gb = rmsle(y_val, y_preds_gb)
print(f"GradientBoosting has {rmsle_score_gb} rmsle score") #cũ: 0.01875


LightGBM_model = LGBMRegressor()
LightGBM_model.fit(X_train, y_train)
 
y_preds_lgbm = LightGBM_model.predict(X_val) #cũ: 0.0177


rmsle_score_lgbm = rmsle(y_val, y_preds_lgbm)
print(f"LightGBM has {rmsle_score_lgbm} rmsle score")


XGBoost_model = XGBRegressor()
XGBoost_model.fit(X_train, y_train)

y_preds_xgb = XGBoost_model.predict(X_val)


rmsle_score_xgb = rmsle(y_val, y_preds_xgb)
print(f"XGboost has {rmsle_score_xgb} rmsle score")


# import tensorflow as tf
# from tensorflow.keras import Model, layers, regularizers

# class TransformerBlock(layers.Layer):
#     def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
#         super(TransformerBlock, self).__init__()
#         self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
#         self.ffn = tf.keras.Sequential([
#             layers.Dense(ff_dim, activation='relu'),
#             layers.Dense(embed_dim),
#         ])
#         self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
#         self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
#         self.dropout1 = layers.Dropout(rate)
#         self.dropout2 = layers.Dropout(rate)

#     def call(self, inputs, training=False):
#         attn_output = self.att(inputs, inputs)
#         attn_output = self.dropout1(attn_output, training=training)
#         out1 = self.layernorm1(inputs + attn_output)

#         ffn_output = self.ffn(out1)
#         ffn_output = self.dropout2(ffn_output, training=training)
#         return self.layernorm2(out1 + ffn_output)

# class TransformerCNNModel(Model):
#     def __init__(self, input_shape):
#         super(TransformerCNNModel, self).__init__()
#         self.conv1 = layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')
#         self.conv2 = layers.Conv1D(128, kernel_size=3, activation='relu', padding='same')
#         self.pool = layers.MaxPooling1D(pool_size=2)
#         self.dropout = layers.Dropout(0.3)

#         self.projection = layers.Dense(128)  # để chuẩn hóa chiều input cho Transformer

#         self.transformer = TransformerBlock(embed_dim=128, num_heads=4, ff_dim=256, rate=0.3)

#         self.global_pool = layers.GlobalAveragePooling1D()
#         self.dense = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-4))
#         self.out = layers.Dense(1)

#     def call(self, inputs, training=False):
#         x = self.conv1(inputs)
#         x = self.conv2(x)
#         x = self.pool(x)
#         x = self.dropout(x, training=training)

#         x = self.projection(x)
#         x = self.transformer(x, training=training)

#         x = self.global_pool(x)
#         x = self.dense(x)
#         return self.out(x)


# #Hàm loss tensorflow
# def rmsle(y_true, y_pred):
#     y_pred = tf.clip_by_value(y_pred, 1e-7, tf.reduce_max(y_pred))  # tránh log(0)
#     y_true = tf.clip_by_value(y_true, 1e-7, tf.reduce_max(y_true))
    
#     log_pred = tf.math.log(y_pred + 1.0)
#     log_true = tf.math.log(y_true + 1.0)
#     return tf.sqrt(tf.reduce_mean(tf.square(log_pred - log_true)))


# # ----- 3. Tạo sliding window -----
# def create_sliding_window(data, labels, window_size=100): #time_step = 1
#     X, y = [], []
#     for i in range(len(data) - window_size):
#         X.append(data[i:i+window_size])
#         y.append(labels[i+window_size-1])
#     return np.array(X), np.array(y)


# from sklearn.utils import shuffle
# X = df.drop(columns = ['Calories', 'id'])
# y = df['Calories']

# X,y = create_sliding_window(X, y, window_size = 50)
# X,y = shuffle(X, y, random_state = 42)


# print(y.shape)


# model = TransformerCNNModel(input_shape=(X_train.shape[1], X_train.shape[2]))



# model.compile(optimizer='adam', loss=rmsle, metrics=[rmsle])


# # Huấn luyện
# model.fit(X, y, validation_split=0.2, epochs=50, batch_size=32)


# model.fit(X, y, validation_split=0.2, epochs=20, batch_size=32)



AdaBoost_model = AdaBoostRegressor()
AdaBoost_model.fit(X_train, y_train)

y_preds_ada = AdaBoost_model.predict(X_val)
rmsle_ada = rmsle(y_val, y_preds_ada)
print(f"score: {rmsle_ada}")


Bagging_model = BaggingRegressor()
Bagging_model.fit(X_train, y_train)

y_preds_bagging = Bagging_model.predict(X_val)
rmsle_bagging = rmsle(y_val, y_preds_bagging)

print(f"score: {rmsle_bagging}")


# y_preds_transformer = model.predict(X_val)
# y_preds_transformer = np.expm1(y_preds_transformer)


# y_preds_transformer = np.expm1(y_preds_transformer)


# fig, axes = plt.subplots(2, 2, figsize = (12, 4 * 2))

# y_pred_dict = {
#     "RandomForest" : y_preds_rf,
#     "GradientBoosting" : y_preds_gb,
#     "LightGBM" : y_preds_lgbm,
#     "XGBoost" : y_preds_xgb
# }

# # 3. Vẽ residual plot
# for i, model in enumerate(y_pred_dict.keys()):
#     residuals = y_val - y_pred_dict[model]
    
#     col = i % 2
#     row = i // 2
#     ax_ij = axes[col, row]
    
#     sns.scatterplot(x=y_pred_dict[model], y=residuals, ax = ax_ij)
#     ax_ij.axhline(y=0, color='r', linestyle='--')
#     ax_ij.set_xlabel('Predicted values')
#     ax_ij.set_ylabel('Residuals (y_true - y_pred)')
#     ax_ij.set_title(f'Residual Plot of {model} model')
#     ax_ij.grid(True)

# plt.tight_layout()
# plt.show()


from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression

base_models = [
    ('rf', RandomForestRegressor(n_jobs = -1)),
    ('xgb', XGBRegressor()),
    ('lgbm', LGBMRegressor()),
    ('gb', GradientBoostingRegressor())
]

meta_model = LinearRegression()

stacking_model = StackingRegressor(
    estimators = base_models,
    final_estimator = meta_model,
    cv = 5
)

stacking_model.fit(X_train, y_train)
y_preds_sm = stacking_model.predict(X_val)
rmsle_score_sm = rmsle(y_val, y_preds_sm)
print(f"Stacking model has {rmsle_score_sm} rmsle score")


df_test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
df_test.head(10)


X_test = df_test.drop(columns = ['id'])

#Chuẩn hóa cột height thành mét
X_test['Height_m'] = X_test['Height'] / 100
X_test['BMI'] = X_test['Weight'] / (X_test['Height_m'] ** 2)
X_test.drop(columns = ['Height_m'], inplace = True)

X_test['MET_duration'] = X_test['Duration'] * X_test['Heart_Rate'] / 100

conditions = [
    (X_test['Body_Temp'] >= 37) & (X_test['Body_Temp'] < 38),
    (X_test['Body_Temp'] >= 38) & (X_test['Body_Temp'] < 39),
    (X_test['Body_Temp'] >= 39) & (X_test['Body_Temp'] < 40),
    (X_test['Body_Temp'] >= 40)
]
values = [0, 1, 2, 3]
X_test['Temp_level'] = np.select(conditions, values)

X_test['Temp_per_min'] = X_test['Body_Temp'] / X_test['Duration']

X_test['BodyTemp_Heart'] = X_test['Body_Temp'] * X_test['Heart_Rate']

# One-hot encode cột 'Sex'
onehot = OneHotEncoder(sparse=False, handle_unknown='ignore')
encoded = onehot.fit_transform(X_test[['Sex']])  # trả về numpy array
columns = onehot.get_feature_names_out(['Sex'])  # tên cột như: ['Sex_Female', 'Sex_Male']

# Chuyển encoded thành DataFrame
encoded_df = pd.DataFrame(encoded, columns=columns, index=X_test.index)

# Xóa cột 'Sex' gốc và gộp với encoded
X_test = pd.concat([X_test.drop(columns=['Sex']), encoded_df], axis=1)


X_test.head()


df.head()


y_preds_submission = stacking_model.predict(X_test)
y_preds_submission


y_preds_submission_inversed = np.expm1(y_preds_submission)
y_preds_submission_inversed


df_test_sm = df_test.copy()
df_test_sm['Calories'] = y_preds_submission_inversed
df_test_sm.head()


final_result = df_test_sm[['id', 'Calories']]
final_result.head()


final_result.to_csv("/kaggle/working/calories_predicting_2.csv", encoding = 'utf-8-sig', index = False)



from sklearn.decomposition import PCA
pca = PCA(n_components=6)
X_train_pca = pca.fit_transform(X_train)

# stacking_model = StackingRegressor(
#     estimators = base_models,
#     final_estimator = meta_model,
#     cv = 5
# )

stacking_model.fit(X_train_pca, y_train)


X_val = pca.transform(X_val)

y_preds_sm_pca = stacking_model.predict(X_val)

rmsle_sm_pca = rmsle(y_val, y_preds_sm_pca)
rmsle_sm_pca

