import dask.dataframe as dd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
import numpy as np
from imblearn.over_sampling import SMOTE  # 导入SMOTE用于处理类别不平衡

# Step 1: Reading and loading the plays data using Dask
# 第一步：使用Dask加载play数据
print("Step 1: Reading and loading the plays data using Dask...")  # 读取并加载plays数据
plays = dd.read_csv('/kaggle/input/nfl-big-data-bowl-2025/plays.csv', assume_missing=True)
print("Step 1 complete: Plays data loaded.")  # 完成：加载plays数据

# Step 2: Reading and concatenating tracking data files
# 第二步：读取并连接tracking数据文件
print("Step 2: Reading and concatenating tracking data files...")  # 读取并连接tracking数据
tracking_files = [f'/kaggle/input/nfl-big-data-bowl-2025/tracking_week_{i}.csv' for i in range(1, 10)]
tracking_data = dd.concat([dd.read_csv(file, assume_missing=True) for file in tracking_files])
print("Step 2 complete: Tracking data loaded.")  # 完成：加载tracking数据

# Step 3: Filtering data for quarterback (nflId = 35459)
# 第三步：过滤出四分卫的数据（nflId = 35459）
print("Step 3: Filtering data for quarterback (nflId = 35459)...")  # 过滤四分卫数据
qb_data = tracking_data[tracking_data['nflId'] == 35459]
print(f"Step 3 complete: Quarterback data filtered. Number of records: {qb_data.shape[0].compute()}")  # 完成：过滤四分卫数据，记录数

# Step 4: Filtering pass plays
# 第四步：过滤出传球事件
print("Step 4: Filtering pass plays...")  # 过滤传球事件
pass_plays = plays[plays['playDescription'].str.contains('pass', case=False, na=False)]
print(f"Step 4 complete: Pass plays filtered. Number of pass plays: {pass_plays.shape[0].compute()}")  # 完成：过滤传球事件，传球事件数量

# Step 5: Merging quarterback data with pass plays data
# 第五步：将四分卫数据与传球事件数据合并
print("Step 5: Merging quarterback data with pass plays data...")  # 合并四分卫数据与传球事件数据
qb_data = pass_plays.merge(qb_data[['gameId', 'playId', 'x', 'y', 's', 'a']], on=['gameId', 'playId'], how='inner')
print(f"Step 5 complete: Data merged. Shape of merged data: {qb_data.shape[0].compute()}")  # 完成：数据合并，合并后的数据形状

# Step 6: Feature engineering - calculating distance to target
# 第六步：特征工程 - 计算到目标的距离
print("Step 6: Feature engineering - calculating distance to target...")  # 计算到目标的距离
qb_data['distance_to_target'] = np.sqrt(qb_data['x']**2 + qb_data['y']**2)
print("Step 6 complete: Distance to target calculated.")  # 完成：计算到目标的距离

# Step 7: Preparing features and target variable for training
# 第七步：准备特征和目标变量用于训练
print("Step 7: Preparing features and target variable for training...")  # 准备特征和目标变量
X = qb_data[['x', 'y', 's', 'a', 'distance_to_target']].compute()  # 特征矩阵
y = qb_data['passResult'].compute()  # 目标变量
print(f"Step 7 complete: Features and target variable prepared. Shape of X: {X.shape}, Shape of y: {y.shape}")  # 完成：特征和目标变量准备好，特征矩阵X和目标变量y的形状

# Step 8: Handling class imbalance using SMOTE
# 第八步：使用SMOTE处理类别不平衡问题
print("Step 8: Handling class imbalance using SMOTE...")  # 处理类别不平衡问题
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
print("Step 8 complete: SMOTE applied to handle class imbalance.")  # 完成：使用SMOTE处理类别不平衡

# Step 9: Splitting data into training and testing sets
# 第九步：将数据拆分为训练集和测试集
print("Step 9: Splitting data into training and testing sets...")  # 拆分数据为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)
print("Step 9 complete: Data split into training and testing sets.")  # 完成：数据拆分

# Step 10: Training the ensemble model with soft voting
# 第十步：使用软投票训练集成模型
print("Step 10: Training the ensemble model with soft voting...")  # 使用软投票训练集成模型
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
sgd_model = SGDClassifier(loss='log', max_iter=1000, random_state=42)

# 使用软投票（soft voting）进行集成
ensemble_model = VotingClassifier(estimators=[
    ('rf', rf_model), 
    ('gb', gb_model), 
    ('sgd', sgd_model)], voting='soft')
ensemble_model.fit(X_train, y_train)
print("Step 10 complete: Ensemble model training finished.")  # 完成：集成模型训练完成

# Step 11: Evaluating the ensemble model
# 第十一步：评估集成模型
print("Step 11: Evaluating the ensemble model...")  # 评估集成模型
y_pred = ensemble_model.predict(X_test)

# 输出分类报告
print(classification_report(y_test, y_pred))  # 输出分类报告
print("Step 11 complete: Model evaluation finished.")  # 完成：模型评估结束


