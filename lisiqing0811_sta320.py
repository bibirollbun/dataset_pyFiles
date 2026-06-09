# Suppress Warnings
import warnings
warnings.filterwarnings("ignore")

# Core Libraries
import numpy as np
import pandas as pd

# Visualization Libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning Utilities
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import make_scorer, mean_squared_log_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split

# Gradient Boosting Models
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor


train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test  = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
origin = pd.read_csv("/kaggle/input/calories-burnt-prediction/calories.csv")


print("Train 数据集前5行数据:")
print(train.head())
print("\nTrain 数据集的信息:")
print(train.info())


print("\n训练集缺失值统计:")
print(train.isnull().sum())


# （1）如果存在重复数据，可以先去重
train = train.drop_duplicates()

# （2）对类别数据进行编码，示例中 'Sex' 字段为 "male"/"female"
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train['Sex'] = le.fit_transform(train['Sex'])
test['Sex'] = le.transform(test['Sex'])  # 使用相同的转化


print("Train 数据集前5行数据:")
print(train.head())
print("\nTrain 数据集的信息:")
print(train.info())


def feature_engineering(df):
    # 检查必要的列是否存在
    required_columns = ['Weight', 'Height', 'Heart_Rate', 'Duration', 'Age', 'Sex']
    if not all(col in df.columns for col in required_columns):
        raise ValueError("数据框中缺少必要的列。")

    # 检查数据类型
    if not all(df[col].dtype in ['int64', 'float64'] for col in required_columns if col != 'Sex'):
        raise ValueError("数据框中的数值列数据类型不正确。")

    # 计算 BMI：体重（kg）除以身高（m）的平方（注意：身高单位从 cm 转换为 m）
    df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)

    # 计算 Activity Intensity：心率 * 活动时长，表征运动强度
    df['Activity_Intensity'] = df['Heart_Rate'] * df['Duration']

    # 计算 BMR (基础代谢率)：
    # 根据 Mifflin-St Jeor Equation，不同性别采用不同的公式
    df['BMR'] = np.where(
        df['Sex'] == '1',  # 男性 BMR
        (10 * df['Weight']) + (6.25 * df['Height']) - (5 * df['Age']) + 5,
        (10 * df['Weight']) + (6.25 * df['Height']) - (5 * df['Age']) - 161  # 女性 BMR
    )

    return df


train = feature_engineering(train)
test = feature_engineering(test)


print("训练集新增特征示例:")
print(train[['BMI', 'Activity_Intensity', 'BMR']].head())

print("\n测试集新增特征示例:")
print(test[['BMI', 'Activity_Intensity', 'BMR']].head())


plt.figure(figsize=(10, 6))
sns.histplot(train['Calories'], kde=True, bins=30, color='skyblue')
plt.title("Distribution of Calories in Training Set", fontsize=14)
plt.xlabel("Calories")
plt.ylabel("Frequency")
plt.show()


plt.figure(figsize=(6, 4))
sns.countplot(x=train['Sex'], palette='Set2')
plt.title("Count of Samples by Sex (Encoded)", fontsize=14)
plt.xlabel("Sex (0=female, 1=male)")
plt.ylabel("Count")
plt.show()


plt.figure(figsize=(10, 8))
corr = train.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap", fontsize=16)
plt.show()


print("\nTrain 数据集的信息:")
print(train.info())


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 定义要探究的数值列
num_cols = ['Heart_Rate', 'Body_Temp', 'Calories', 'BMI', 'BMR']

# 遍历每一列，并绘制对应的图表
for col in num_cols:
    plt.figure(figsize=(15, 5))

    # 创建一个副本数据集
    dd = train.copy()
    
    # 如果当前列是 'Calories'，只保留训练集部分
    if col == 'Calories':
        dd = train

    # 子图1：分布图（按性别区分）
    plt.subplot(1, 3, 1)
    sns.histplot(data=dd, x=col, hue='Sex', kde=True, bins=30, multiple='layer')
    plt.title(f'Distribution of {col} by Sex', fontsize=14)
    plt.xlabel(col, fontsize=12)
    plt.ylabel('Frequency', fontsize=12)

    # 子图2：箱线图（按性别）
    plt.subplot(1, 3, 2)
    sns.boxplot(data=dd, x='Sex', y=col)
    plt.title(f'Boxplot of {col} by Sex', fontsize=14)
    plt.xlabel('Sex', fontsize=12)
    plt.ylabel(col, fontsize=12)

    # 子图3：折线图（Duration vs 数值列，按性别区分）
    plt.subplot(1, 3, 3)
    data = dd.groupby(['Duration', 'Sex'])[[col]].mean().reset_index()
    sns.lineplot(data=data, x='Duration', y=col, hue='Sex')
    plt.title(f'Duration Vs {col} by Sex', fontsize=14)
    plt.xlabel('Duration', fontsize=12)
    plt.ylabel(col, fontsize=12)

    # 调整布局并显示
    plt.tight_layout()
    plt.show()


import numpy as np

# 添加 Calories_log 列
train['Calories_log'] = np.log1p(train['Calories'])  # 使用 log1p 处理，避免 log(0) 的问题


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 定义要探究的数值列
num_cols = ['Heart_Rate', 'Body_Temp', 'Age', 'BMR']

# 遍历每一列并绘制对应的图表
for col in num_cols:
    plt.figure(figsize=(20, 5))
    
    # 创建一个副本数据集
    dd = train.copy()
    
    # 子图：散点图（Calories_log vs 数值列，按性别区分）
    plt.subplot(1, 3, 3)
    data = dd.groupby(['Calories_log', 'Sex'])[[col]].mean().reset_index()
    sns.scatterplot(x='Calories_log', y=col, data=data, hue='Sex', palette='viridis')
    plt.title(f'Calories_log Vs {col} by Sex', fontsize=14)
    plt.xlabel('Calories_log', fontsize=12)
    plt.ylabel(col, fontsize=12)
    
    # 调整布局并显示
    plt.tight_layout()
    plt.show()


# --- 5. 数据预处理与特征工程 ---
# 本例选取除 id 和 Calories 之外的特征
X = train.drop(columns=['id', 'Calories'])
y = train['Calories']

# 划分训练集与验证集
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 针对部分模型（例如线性回归）比较敏感，进行特征缩放
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# 对数值型特征进行缩放（可对 X_train 全部特征做缩放）
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)


def rmsle(y_true, y_pred):
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))


from sklearn.metrics import mean_squared_error, r2_score

# 6.1. 线性回归
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
pred_lr = lr.predict(X_val_scaled)
mse_lr = mean_squared_error(y_val, pred_lr)
r2_lr = r2_score(y_val, pred_lr)
rmsle_lr = rmsle(y_val, pred_lr)

print("Linear Regression:")
print(f"  MSE: {mse_lr:.2f}")
print(f"  R²: {r2_lr:.4f}\n")
print(f" RMSLE: {rmsle_lr:.4f}\n")


# 弃用

# 6.2. 随机森林回归
from sklearn.ensemble import RandomForestRegressor
# 对于树模型，通常无需缩放，因此使用原始数据
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_val)
mse_rf = mean_squared_error(y_val, pred_rf)
r2_rf = r2_score(y_val, pred_rf)
# rmsle_lr = rmsle(y_val, pred_lr)
rmsle_rf = rmsle(y_val, pred_rf)

print("Random Forest Regression:")
print(f"  MSE: {mse_rf:.2f}")
print(f"  R²: {r2_rf:.4f}\n")
print(f" RMSLE: {rmsle_rf:.4f}\n")


# 6.3. XGBoost 回归
import xgboost as xgb
xgbr = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
xgbr.fit(X_train, y_train)
pred_xgb = xgbr.predict(X_val)
mse_xgb = mean_squared_error(y_val, pred_xgb)
r2_xgb = r2_score(y_val, pred_xgb)
# rmsle_lr = rmsle(y_val, pred_lr)
rmsle_xgb = rmsle(y_val, pred_xgb)

print("XGBoost Regression:")
print(f"  MSE: {mse_xgb:.2f}")
print(f"  R²: {r2_xgb:.4f}\n")
print(f" RMSLE: {rmsle_xgb:.4f}\n")


# test
print("hello")


from sklearn.model_selection import KFold, cross_val_score
from catboost import CatBoostRegressor

# 2. 定义特征列和标签
features = [
    'Sex', 'Age', 'Height', 'Weight', 'Duration',
    'Heart_Rate', 'Body_Temp', 'BMI', 'Activity_Intensity', 'BMR'
]
X = train[features]
y = train['Calories']
X_test = test[features]

# 3. 设定 5 折交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 4. 定义评估函数，返回 RMSLE, MSE 和 R2 三项指标（平均 ± 标准差）
def evaluate_metrics(model, X, y):
    # RMSLE
    neg_msle = cross_val_score(
        model, X, y, cv=kf,
        scoring='neg_mean_squared_log_error',
        n_jobs=-1
    )
    rmsle_scores = np.sqrt(-neg_msle)

    # MSE
    neg_mse = cross_val_score(
        model, X, y, cv=kf,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    mse_scores = -neg_mse

    # R2
    r2_scores = cross_val_score(
        model, X, y, cv=kf,
        scoring='r2',
        n_jobs=-1
    )

    return {
        'RMSLE': (rmsle_scores.mean(), rmsle_scores.std()),
        'MSE':   (mse_scores.mean(),   mse_scores.std()),
        'R2':    (r2_scores.mean(),    r2_scores.std())
    }

# 5. 初始化 CatBoost 回归器
model = CatBoostRegressor(
    iterations=300,
    learning_rate=0.05,
    depth=6,
    loss_function='RMSE',
    random_seed=42,
    verbose=False
)

# 6. 交叉验证评估并打印结果
results = evaluate_metrics(model, X, y)
print("CatBoost 5 折 CV 评估：")
print(f"  RMSLE = {results['RMSLE'][0]:.5f} ± {results['RMSLE'][1]:.5f}")
print(f"   MSE  = {results['MSE'][0]:.5f} ± {results['MSE'][1]:.5f}")
print(f"   R²   = {results['R2'][0]:.5f} ± {results['R2'][1]:.5f}\n")

# 7. 在全量训练集上训练模型
model.fit(X, y)

# 8. 在测试集上进行预测，并保存结果
preds = model.predict(X_test)
test['Predicted_Calories'] = preds
# 可选：保存预测到 CSV
# test[['Predicted_Calories']].to_csv('submission.csv', index=False)

# 9. 输出特征重要性
importances = model.get_feature_importance(prettified=True)
print("特征重要性（从高到低）：")
print(importances)



# 4. 定义特征列和标签
features = [
    'Sex', 'Age', 'Height', 'Weight', 'Duration',
    'Heart_Rate', 'Body_Temp', 'BMI', 'Activity_Intensity', 'BMR'
]
X      = train[features]
y      = train['Calories']
X_test = test[features]

# 5. 设定 5 折交叉验证
from sklearn.model_selection import KFold, cross_val_score
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 6. 定义评估函数，返回 RMSLE、MSE 和 R2 三项指标（平均 ± 标准差）
def evaluate_metrics(model, X, y):
    # RMSLE
    neg_msle = cross_val_score(
        model, X, y, cv=kf,
        scoring='neg_mean_squared_log_error',
        n_jobs=-1
    )
    rmsle_scores = np.sqrt(-neg_msle)

    # MSE
    neg_mse = cross_val_score(
        model, X, y, cv=kf,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    mse_scores = -neg_mse

    # R2
    r2_scores = cross_val_score(
        model, X, y, cv=kf,
        scoring='r2',
        n_jobs=-1
    )

    return {
        'RMSLE': (rmsle_scores.mean(), rmsle_scores.std()),
        'MSE':   (mse_scores.mean(),   mse_scores.std()),
        'R2':    (r2_scores.mean(),    r2_scores.std())
    }

# 7. 初始化 LightGBM 回归器
from lightgbm import LGBMRegressor

model = LGBMRegressor(
    n_estimators=300,        # 树的数量
    learning_rate=0.05,      # 学习率
    num_leaves=31,           # 叶子节点数
    subsample=0.8,           # 行采样比例
    colsample_bytree=0.8,    # 列采样比例
    random_state=42,
    n_jobs=-1
)

# 8. 交叉验证评估并打印结果
results = evaluate_metrics(model, X, y)
print("LightGBM 5 折 CV 评估：")
print(f"  RMSLE = {results['RMSLE'][0]:.5f} ± {results['RMSLE'][1]:.5f}")
print(f"   MSE  = {results['MSE'][0]:.5f} ± {results['MSE'][1]:.5f}")
print(f"   R²   = {results['R2'][0]:.5f} ± {results['R2'][1]:.5f}\n")

# 9. 在全量训练集上训练模型
model.fit(X, y)

# 10. 在测试集上进行预测，并将结果加入 test DataFrame
# preds = model.predict(X_test)
# test['Predicted_Calories_LGBM'] = preds
# 可选：保存预测到 CSV
# test[['id','Predicted_Calories_LGBM']].to_csv('submission_lgbm.csv', index=False)

# 11. 输出特征重要性（按从高到低）
importances = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False).reset_index(drop=True)

print("特征重要性（从高到低）：")
print(importances)


#MLP
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
X_test_tensor = torch.FloatTensor(X_val_scaled)
y_test_tensor = torch.FloatTensor(y_val.to_numpy()).reshape(-1, 1)

# ----------------------------
# 2. Define MLP Architecture
# ----------------------------
class MLPRegressor(nn.Module):
    def __init__(self, input_size):
        super(MLPRegressor, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),  # Regularization
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)   # Output layer (regression)
        )
    
    def forward(self, x):
        return self.layers(x)

# Initialize model
model = MLPRegressor(input_size=X_train_scaled.shape[1])

# ----------------------------
# 3. Train the Model
# ----------------------------
# Hyperparameters
criterion = nn.MSELoss()  # Mean Squared Error loss
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 200

# Training loop
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    # Print progress every 20 epochs
    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# ----------------------------
# 4. Evaluate the Model
# ----------------------------
model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor)
    test_loss = criterion(y_pred, y_test_tensor)
    mae = mean_absolute_error(y_test_tensor.numpy(), y_pred.numpy())
    r2 = r2_score(y_test_tensor.numpy(), y_pred.numpy())
    rmsle_mlp=rmsle(y_test_tensor.numpy(), y_pred.numpy())

print(f'\nTest MSE: {test_loss.item():.4f}')
print(f'MAE: {mae:.4f}')
print(f'R² Score: {r2:.4f}')
print(f" RMSLE: {rmsle_mlp:.4f}\n")

# ----------------------------
# 5. Save the Model (Optional)
# ----------------------------
torch.save(model.state_dict(), 'mlp_regressor.pth')


#MLP
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
X_test_tensor = torch.FloatTensor(X_val_scaled)
y_test_tensor = torch.FloatTensor(y_val.to_numpy()).reshape(-1, 1)

# ----------------------------
# 2. Define MLP Architecture
# ----------------------------
class MLPRegressor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),  # 加宽
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),          # 新增一层
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.layers(x)

# Initialize model
model = MLPRegressor(input_size=X_train_scaled.shape[1])

# ----------------------------
# 3. Train the Model
# ----------------------------
# Hyperparameters
criterion = nn.MSELoss()  # Mean Squared Error loss
optimizer = optim.Adam(model.parameters(), lr=0.005)
epochs = 200
loss_history = []

# Training loop
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss_history.append(loss.item())
    loss.backward()
    optimizer.step()
    
    # Print progress every 20 epochs
    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# ----------------------------
# 4. Evaluate the Model
# ----------------------------
model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor)
    test_loss = criterion(y_pred, y_test_tensor)
    mae = mean_absolute_error(y_test_tensor.numpy(), y_pred.numpy())
    r2 = r2_score(y_test_tensor.numpy(), y_pred.numpy())
    rmsle_mlp=rmsle(y_test_tensor.numpy(), y_pred.numpy())

print(f'\nTest MSE: {test_loss.item():.4f}')
print(f'MAE: {mae:.4f}')
print(f'R² Score: {r2:.4f}')
print(f" RMSLE: {rmsle_mlp:.4f}\n")

plt.plot(loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()

# ----------------------------
# 5. Save the Model (Optional)
# ----------------------------
torch.save(model.state_dict(), 'mlp_regressor.pth')


# Suppress Warnings
import warnings
warnings.filterwarnings("ignore")

# Core Libraries
import numpy as np
import pandas as pd

# Visualization Libraries
# import matplotlib.pyplot as plt # 在此脚本的随机森林部分未直接使用
# import seaborn as sns # 在此脚本的随机森林部分未直接使用

# Machine Learning Utilities
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


# --- 数据加载 ---
# 尝试加载数据，如果文件不存在，则创建模拟数据以便代码可运行
try:
    train_df = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
    test_df  = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
    # origin_df = pd.read_csv("/kaggle/input/calories-burnt-prediction/calories.csv") # 'origin' 未在后续代码中使用
except FileNotFoundError:
    print("CSV 文件未找到。将使用模拟数据进行演示。")
    # 创建符合结构的模拟数据
    data_size = 1000
    train_data = {
        'id': range(data_size),
        'Sex': np.random.choice(['male', 'female'], size=data_size),
        'Age': np.random.randint(18, 70, size=data_size),
        'Height': np.random.randint(150, 200, size=data_size),
        'Weight': np.random.randint(50, 120, size=data_size),
        'Duration': np.random.randint(5, 60, size=data_size),
        'Heart_Rate': np.random.randint(60, 180, size=data_size),
        'Body_Temp': np.random.uniform(36.0, 40.0, size=data_size),
        'Calories': np.random.randint(50, 800, size=data_size)
    }
    train_df = pd.DataFrame(train_data)
    test_data = { # 确保测试集有相同的特征列（除了目标变量）
        'id': range(data_size, data_size + 200),
        'Sex': np.random.choice(['male', 'female'], size=200),
        'Age': np.random.randint(18, 70, size=200),
        'Height': np.random.randint(150, 200, size=200),
        'Weight': np.random.randint(50, 120, size=200),
        'Duration': np.random.randint(5, 60, size=200),
        'Heart_Rate': np.random.randint(60, 180, size=200),
        'Body_Temp': np.random.uniform(36.0, 40.0, size=200),
    }
    test_df = pd.DataFrame(test_data)

# （1）数据去重
train_df = train_df.drop_duplicates()

# （2）类别数据编码 ('Sex')
le = LabelEncoder()
train_df['Sex'] = le.fit_transform(train_df['Sex'])
test_df['Sex'] = le.transform(test_df['Sex'])

# 动态确定 'male' 的编码值
MALE_ENCODED_VALUE = 1 # 默认值
if 'male' in le.classes_:
    MALE_ENCODED_VALUE = int(le.transform(['male'])[0])
elif len(le.classes_) > 1: # 如果 'male' 不在类别中，但存在多个类别
    # 尝试基于其他类别的值来推断（这是一种启发式方法，可能不完美）
    if le.transform([le.classes_[0]])[0] == 0 and le.transform([le.classes_[1]])[0] == 1:
        MALE_ENCODED_VALUE = 1 # 假设第二个类别是男性且编码为1
    elif le.transform([le.classes_[0]])[0] == 1 and le.transform([le.classes_[1]])[0] == 0:
         MALE_ENCODED_VALUE = le.transform([le.classes_[0]])[0] # 假设第一个类别是男性且编码为1
# print(f"性别编码: {dict(zip(le.classes_, le.transform(le.classes_)))}. 男性编码值为: {MALE_ENCODED_VALUE}")

# --- 特征工程函数 ---
def feature_engineering(df):
    required_columns = ['Weight', 'Height', 'Heart_Rate', 'Duration', 'Age', 'Sex']
    if not all(col in df.columns for col in required_columns):
        missing_cols = [col for col in required_columns if col not in df.columns]
        raise ValueError(f"数据框中缺少必要的列: {missing_cols}")

    numeric_cols_to_check = ['Weight', 'Height', 'Heart_Rate', 'Duration', 'Age']
    for col in numeric_cols_to_check:
        if df[col].dtype not in ['int64', 'float64', 'int32']:
            raise ValueError(f"列 {col} 的数据类型不正确: {df[col].dtype}。应为数值型。")
    if df['Sex'].dtype not in ['int64', 'int32']: # LabelEncoder 输出整数类型
         raise ValueError(f"Sex 列在编码后应为整数类型, 当前是: {df['Sex'].dtype}")

    df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)
    df['Activity_Intensity'] = df['Heart_Rate'] * df['Duration']
    
    df['BMR'] = np.where(
        df['Sex'] == MALE_ENCODED_VALUE,  # 男性 BMR
        (10 * df['Weight']) + (6.25 * df['Height']) - (5 * df['Age']) + 5,
        (10 * df['Weight']) + (6.25 * df['Height']) - (5 * df['Age']) - 161  # 女性 BMR
    )
    return df

train_df = feature_engineering(train_df)
test_df = feature_engineering(test_df)

# （可选）创建 Calories_log 列。如果创建，确保它不作为特征泄漏给模型。
# train_df['Calories_log'] = np.log1p(train_df['Calories'])


# --- 数据预处理与特征选择 ---
columns_to_drop = ['id', 'Calories']
if 'Calories_log' in train_df.columns: # 如果创建了 Calories_log，则从特征中排除
    columns_to_drop.append('Calories_log')

X = train_df.drop(columns=columns_to_drop)
y = train_df['Calories']


# --- 划分训练集与验证集 ---
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42)

# --- 特征缩放 ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# --- RMSLE 评估函数 ---
def rmsle(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_pred = np.maximum(0, y_pred) # 裁剪预测值以避免负数或零的对数问题
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))

# --- 6.1. 线性回归模型 ---
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
pred_lr = lr.predict(X_val_scaled)

mse_lr = mean_squared_error(y_val, pred_lr)
r2_lr = r2_score(y_val, pred_lr)
rmsle_lr = rmsle(y_val, pred_lr) # rmsle 函数内部处理裁剪

print("Linear Regression:")
print(f"  MSE: {mse_lr:.2f}")
print(f"  R²: {r2_lr:.4f}")
print(f"  RMSLE: {rmsle_lr:.4f}\n")

# --- 6.2. 随机森林回归模型 (新增部分) ---
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1,
                           max_depth=None, min_samples_split=2, min_samples_leaf=1)

# 训练模型
# 随机森林对特征缩放不敏感，但为与之前步骤一致，此处使用缩放后的数据。
# 如果希望，也可以使用 X_train (未缩放数据)。
rf.fit(X_train_scaled, y_train)

# 进行预测
pred_rf = rf.predict(X_val_scaled)

# 评估模型
mse_rf = mean_squared_error(y_val, pred_rf)
r2_rf = r2_score(y_val, pred_rf)
# 随机森林通常不会预测负值（如果训练目标为正），但 rmsle 函数会处理。
rmsle_rf = rmsle(y_val, pred_rf) 

print("Random Forest Regressor:")
print(f"  MSE: {mse_rf:.2f}")
print(f"  R²: {r2_rf:.4f}")
print(f"  RMSLE: {rmsle_rf:.4f}\n")

