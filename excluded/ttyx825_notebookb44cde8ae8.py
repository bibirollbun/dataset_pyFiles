import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, RandomizedSearchCV, KFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.feature_selection import SelectFromModel
import warnings
warnings.filterwarnings('ignore')

# 设置图形样式
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# 检查高级库可用性
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
    print("✅ XGBoost 可用")
except ImportError:
    XGB_AVAILABLE = False
    print("❌ XGBoost 不可用")

try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
    print("✅ LightGBM 可用")
except ImportError:
    LGBM_AVAILABLE = False
    print("❌ LightGBM 不可用")


# 1. 数据加载和基础处理
print("步骤1: 数据加载和基础处理...")

try:
    train_df = pd.read_csv('/kaggle/input/GiveMeSomeCredit/cs-training.csv')
    test_df = pd.read_csv('/kaggle/input/GiveMeSomeCredit/cs-test.csv')
except:
    train_df = pd.read_csv('cs-training.csv')
    test_df = pd.read_csv('cs-test.csv')

print(f"训练集形状: {train_df.shape}")
print(f"测试集形状: {test_df.shape}")

# 立即处理ID列
train_df = train_df.rename(columns={'Unnamed: 0': 'Id'})
test_df = test_df.rename(columns={'Unnamed: 0': 'Id'})
train_df['Id'] = train_df['Id'].astype('Int32')
test_df['Id'] = test_df['Id'].astype('Int32')

# 定义特征和目标
feature_columns = [
    'RevolvingUtilizationOfUnsecuredLines',
    'age',
    'NumberOfTime30-59DaysPastDueNotWorse',
    'DebtRatio',
    'MonthlyIncome',
    'NumberOfOpenCreditLinesAndLoans',
    'NumberOfTimes90DaysLate',
    'NumberRealEstateLoansOrLines',
    'NumberOfTime60-89DaysPastDueNotWorse',
    'NumberOfDependents'
]

target_column = 'SeriousDlqin2yrs'

print(f"特征数量: {len(feature_columns)}")
print(f"目标变量: {target_column}")


# 2. 第一版特征工程
print("步骤2: 第一版特征工程...")

def basic_feature_engineering(df):
    """基础特征工程"""
    df_basic = df.copy()
    
    # 确保数值类型
    numeric_cols = df_basic.select_dtypes(include=[np.number]).columns
    df_basic[numeric_cols] = df_basic[numeric_cols].astype(float)
    
    # 基础特征
    past_due_cols = ['NumberOfTime30-59DaysPastDueNotWorse', 
                    'NumberOfTime60-89DaysPastDueNotWorse', 
                    'NumberOfTimes90DaysLate']
    
    if all(col in df_basic.columns for col in past_due_cols):
        df_basic['TotalPastDue'] = (
            df_basic['NumberOfTime30-59DaysPastDueNotWorse'] + 
            df_basic['NumberOfTime60-89DaysPastDueNotWorse'] + 
            df_basic['NumberOfTimes90DaysLate']
        )
    
    # 是否有收入
    if 'MonthlyIncome' in df_basic.columns:
        df_basic['HasIncome'] = (df_basic['MonthlyIncome'] > 0).astype(float)
    
    # 债务收入比
    if 'DebtRatio' in df_basic.columns and 'MonthlyIncome' in df_basic.columns:
        df_basic['DebtToIncome'] = df_basic['DebtRatio'] / np.maximum(df_basic['MonthlyIncome'], 1)
        df_basic['DebtToIncome'] = np.minimum(df_basic['DebtToIncome'], 10)
    
    # 信用使用率分组
    if 'RevolvingUtilizationOfUnsecuredLines' in df_basic.columns:
        df_basic['UtilizationGroup'] = pd.cut(
            df_basic['RevolvingUtilizationOfUnsecuredLines'],
            bins=[-1, 0.1, 0.3, 0.7, 0.9, 10],
            labels=['VeryLow', 'Low', 'Medium', 'High', 'VeryHigh']
        )
    
    # 年龄分组
    if 'age' in df_basic.columns:
        df_basic['AgeGroup'] = pd.cut(
            df_basic['age'],
            bins=[0, 25, 35, 45, 55, 65, 100],
            labels=['Young', 'YoungAdult', 'Middle', 'Senior', 'Elderly', 'Retired']
        )
    
    # 收入分组
    if 'MonthlyIncome' in df_basic.columns:
        df_basic['IncomeGroup'] = pd.cut(
            df_basic['MonthlyIncome'],
            bins=[-1, 0, 3000, 6000, 10000, np.inf],
            labels=['NoIncome', 'Low', 'Medium', 'High', 'VeryHigh']
        )
    
    return df_basic

# 应用基础特征工程
train_basic = basic_feature_engineering(train_df)
test_basic = basic_feature_engineering(test_df)

print("基础特征工程完成!")


# 3. 第二版高级特征工程
print("步骤3: 第二版高级特征工程...")

def advanced_feature_engineering_v2(df):
    """第二版高级特征工程 - 针对信用风险的领域特定特征"""
    df_advanced = df.copy()
    
    # 逾期行为模式特征
    df_advanced['LatePaymentRatio'] = (
        df_advanced['NumberOfTimes90DaysLate'] / 
        np.maximum(df_advanced['TotalPastDue'], 1)
    )
    
    # 信用使用强度
    df_advanced['CreditUtilizationIntensity'] = (
        df_advanced['RevolvingUtilizationOfUnsecuredLines'] * 
        df_advanced['NumberOfOpenCreditLinesAndLoans']
    )
    
    # 债务负担与收入的关系
    df_advanced['DebtIncomeInteraction'] = (
        df_advanced['DebtRatio'] * np.log1p(df_advanced['MonthlyIncome'])
    )
    
    # 年龄与信用的非线性关系
    df_advanced['AgeCreditInteraction'] = (
        df_advanced['age'] * df_advanced['RevolvingUtilizationOfUnsecuredLines']
    )
    
    # 还款能力综合指标
    df_advanced['RepaymentAbilityScore'] = (
        np.log1p(df_advanced['MonthlyIncome']) / 
        np.maximum(df_advanced['DebtRatio'] + 0.1, 1)
    )
    
    # 信用历史复杂度
    df_advanced['CreditComplexity'] = (
        df_advanced['NumberOfOpenCreditLinesAndLoans'] + 
        df_advanced['NumberRealEstateLoansOrLines']
    ) / np.maximum(df_advanced['age'] - 18, 1)
    
    # 风险集中度指标
    df_advanced['RiskConcentration_v2'] = (
        (df_advanced['NumberOfTime60-89DaysPastDueNotWorse'] + 
         df_advanced['NumberOfTimes90DaysLate']) / 
        np.maximum(df_advanced['TotalPastDue'], 1)
    )
    
    # 收入稳定性指标
    df_advanced['IncomeStability'] = (
        df_advanced['MonthlyIncome'] / 
        np.maximum(df_advanced['NumberOfDependents'] + 1, 1)
    )
    
    # 债务结构指标
    df_advanced['DebtStructure'] = (
        df_advanced['NumberRealEstateLoansOrLines'] / 
        np.maximum(df_advanced['NumberOfOpenCreditLinesAndLoans'], 1)
    )
    
    # 创建分箱特征
    # 信用使用率分箱
    df_advanced['UtilizationBins'] = pd.cut(
        df_advanced['RevolvingUtilizationOfUnsecuredLines'],
        bins=[0, 0.1, 0.3, 0.6, 0.9, 1, 5, 10, np.inf],
        labels=['0-10%', '10-30%', '30-60%', '60-90%', '90-100%', '100-500%', '500-1000%', '>1000%']
    )
    
    # 债务比率分箱
    df_advanced['DebtRatioBins'] = pd.cut(
        df_advanced['DebtRatio'],
        bins=[0, 0.1, 0.3, 0.5, 0.8, 1, 2, 5, 10, np.inf],
        labels=['0-10%', '10-30%', '30-50%', '50-80%', '80-100%', '100-200%', '200-500%', '500-1000%', '>1000%']
    )
    
    return df_advanced

# 应用第二版特征工程
train_advanced = advanced_feature_engineering_v2(train_basic)
test_advanced = advanced_feature_engineering_v2(test_basic)

print("第二版高级特征工程完成!")
print(f"训练集特征数量: {train_advanced.shape[1]}")


# 4. 目标编码 (修复版本)
print("步骤4: 目标编码...")

def target_encode_fixed(df, columns, target, n_splits=5, smooth=20):
    """
    修复版目标编码函数，处理分类变量类型问题
    """
    df_encoded = df.copy()
    
    for col in columns:
        if col in df.columns:
            # 计算全局均值
            global_mean = df[target].mean()
            
            # 将分类列转换为字符串类型，避免类型冲突
            df_temp = df.copy()
            df_temp[col] = df_temp[col].astype(str)
            
            # 使用交叉验证进行目标编码
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            encoded_col = np.zeros(len(df))
            
            for train_idx, val_idx in kf.split(df_temp):
                train_df = df_temp.iloc[train_idx]
                
                # 计算每个类别的目标均值
                means = train_df.groupby(col)[target].mean()
                counts = train_df.groupby(col)[target].count()
                
                # 平滑处理
                encoded_values = (counts * means + smooth * global_mean) / (counts + smooth)
                
                # 应用到验证集
                val_df = df_temp.iloc[val_idx]
                val_mapped = val_df[col].map(encoded_values)
                
                # 用全局均值填充缺失值
                val_filled = val_mapped.fillna(global_mean)
                encoded_col[val_idx] = val_filled.values
            
            df_encoded[f'{col}_TargetEnc'] = encoded_col
    
    return df_encoded

# 选择分类特征进行目标编码
categorical_for_encoding = ['UtilizationGroup', 'AgeGroup', 'IncomeGroup', 'UtilizationBins', 'DebtRatioBins']
categorical_for_encoding = [f for f in categorical_for_encoding if f in train_advanced.columns]

if categorical_for_encoding:
    print(f"对 {len(categorical_for_encoding)} 个分类特征进行目标编码...")
    train_encoded = target_encode_fixed(train_advanced, categorical_for_encoding, target_column)
    test_encoded = target_encode_fixed(test_advanced, categorical_for_encoding, target_column)
    print("目标编码完成!")
else:
    train_encoded = train_advanced.copy()
    test_encoded = test_advanced.copy()
    print("没有可用的分类特征进行目标编码")


# 5. 稳健缺失值处理
print("步骤5: 稳健缺失值处理...")

def robust_missing_value_imputation(train_df, test_df):
    """使用稳健的方法处理缺失值，避免无穷大和异常值问题"""
    train_filled = train_df.copy()
    test_filled = test_df.copy()
    
    # 获取所有数值特征（排除ID和目标变量）
    numeric_features = train_filled.select_dtypes(include=[np.number]).columns.tolist()
    numeric_features = [f for f in numeric_features if f not in ['Id', target_column]]
    
    # 分类特征
    categorical_features = ['UtilizationGroup', 'AgeGroup', 'IncomeGroup', 'UtilizationBins', 'DebtRatioBins']
    categorical_features = [f for f in categorical_features if f in train_filled.columns]
    
    # 处理无穷大和异常值
    print("处理无穷大和异常值...")
    for feature in numeric_features:
        if feature in train_filled.columns:
            # 替换无穷大为NaN
            train_filled[feature] = train_filled[feature].replace([np.inf, -np.inf], np.nan)
            test_filled[feature] = test_filled[feature].replace([np.inf, -np.inf], np.nan)
            
            # 对极端值进行截断（基于分位数）
            if feature not in ['age', 'NumberOfDependents']:  # 这些特征不需要截断
                # 计算1%和99%分位数
                q_low = train_filled[feature].quantile(0.01)
                q_high = train_filled[feature].quantile(0.99)
                
                # 应用截断
                train_filled[feature] = np.clip(train_filled[feature], q_low, q_high)
                test_filled[feature] = np.clip(test_filled[feature], q_low, q_high)
    
    # 使用中位数填充数值特征（更稳定）
    print("使用中位数填充数值特征...")
    imputer_median = SimpleImputer(strategy='median')
    
    # 确保所有数值列都是float类型
    for feature in numeric_features:
        if feature in train_filled.columns:
            train_filled[feature] = train_filled[feature].astype(float)
            test_filled[feature] = test_filled[feature].astype(float)
    
    # 应用中位数填充
    train_filled[numeric_features] = imputer_median.fit_transform(train_filled[numeric_features])
    test_filled[numeric_features] = imputer_median.transform(test_filled[numeric_features])
    
    # 对分类特征使用众数填充
    for cat_feat in categorical_features:
        if cat_feat in train_filled.columns:
            most_frequent = train_filled[cat_feat].mode()
            if len(most_frequent) > 0:
                train_filled[cat_feat].fillna(most_frequent[0], inplace=True)
                test_filled[cat_feat].fillna(most_frequent[0], inplace=True)
    
    return train_filled, test_filled, numeric_features + categorical_features

# 应用稳健的缺失值处理
train_filled, test_filled, all_features = robust_missing_value_imputation(train_encoded, test_encoded)

print("稳健缺失值处理完成!")
print(f"处理后的特征数量: {len(all_features)}")


# 6. 特征编码和转换
print("步骤6: 特征编码和转换...")

# 对分类特征进行one-hot编码
categorical_features = ['UtilizationGroup', 'AgeGroup', 'IncomeGroup', 'UtilizationBins', 'DebtRatioBins']
categorical_features = [f for f in categorical_features if f in train_filled.columns]

train_processed = train_filled.copy()
test_processed = test_filled.copy()

for cat_feat in categorical_features:
    if cat_feat in train_processed.columns:
        train_dummies = pd.get_dummies(train_processed[cat_feat], prefix=cat_feat)
        test_dummies = pd.get_dummies(test_processed[cat_feat], prefix=cat_feat)
        
        # 确保测试集有训练集中的所有列
        for col in train_dummies.columns:
            if col not in test_dummies.columns:
                test_dummies[col] = 0
        
        # 重新排列测试集列的顺序以匹配训练集
        test_dummies = test_dummies[train_dummies.columns]
        
        # 合并回原数据框
        train_processed = pd.concat([train_processed.drop(cat_feat, axis=1), train_dummies], axis=1)
        test_processed = pd.concat([test_processed.drop(cat_feat, axis=1), test_dummies], axis=1)

# 获取所有特征列（排除ID和目标变量）
feature_columns_all = [col for col in train_processed.columns if col not in ['Id', target_column]]

# 高级特征缩放 - 对不同的特征使用不同的缩放方法
print("应用高级特征缩放...")

# 识别需要特殊处理的特征
robust_features = ['RevolvingUtilizationOfUnsecuredLines', 'DebtRatio', 'DebtToIncome', 'DebtBurden']
quantile_features = ['MonthlyIncome', 'IncomePerDependent', 'IncomePerLoan', 'RepaymentCapacity', 'RepaymentAbilityScore']
standard_features = [f for f in feature_columns_all if f not in robust_features + quantile_features and 
                    not f.startswith(tuple(categorical_features))]

# 应用不同的缩放器
if robust_features:
    robust_scaler = RobustScaler()
    # 只选择实际存在的特征
    robust_features_exist = [f for f in robust_features if f in train_processed.columns]
    if robust_features_exist:
        train_processed[robust_features_exist] = robust_scaler.fit_transform(train_processed[robust_features_exist])
        test_processed[robust_features_exist] = robust_scaler.transform(test_processed[robust_features_exist])

if quantile_features:
    quantile_scaler = QuantileTransformer(output_distribution='normal', random_state=42)
    # 只选择实际存在的特征
    quantile_features_exist = [f for f in quantile_features if f in train_processed.columns]
    if quantile_features_exist:
        train_processed[quantile_features_exist] = quantile_scaler.fit_transform(train_processed[quantile_features_exist])
        test_processed[quantile_features_exist] = quantile_scaler.transform(test_processed[quantile_features_exist])

if standard_features:
    standard_scaler = StandardScaler()
    # 只选择实际存在的特征
    standard_features_exist = [f for f in standard_features if f in train_processed.columns]
    if standard_features_exist:
        train_processed[standard_features_exist] = standard_scaler.fit_transform(train_processed[standard_features_exist])
        test_processed[standard_features_exist] = standard_scaler.transform(test_processed[standard_features_exist])

print("特征编码和转换完成!")
print(f"最终训练集形状: {train_processed.shape}")
print(f"最终测试集形状: {test_processed.shape}")


# 7. 准备建模数据
print("步骤7: 准备建模数据...")

# 准备特征和目标变量
X = train_processed[feature_columns_all]
y = train_processed[target_column].astype(float)

# 分割训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"训练集: {X_train.shape}, 验证集: {X_val.shape}")
print(f"训练集正样本比例: {float(y_train.mean()):.4f}")
print(f"验证集正样本比例: {float(y_val.mean()):.4f}")

# 计算类别权重
positive_weight = np.sum(y == 0) / np.sum(y == 1)
print(f"正样本权重: {positive_weight:.2f}")


# 8. 基础模型训练
print("步骤8: 基础模型训练...")

# 定义基础模型
base_models = {
    'LogisticRegression': LogisticRegression(
        C=0.1, 
        random_state=42, 
        max_iter=1000,
        class_weight='balanced'
    ),
    'RandomForest': RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=50,
        min_samples_leaf=20,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    ),
    'GradientBoosting': GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        min_samples_split=100,
        min_samples_leaf=50,
        random_state=42,
        subsample=0.8
    )
}

# 添加高级模型（如果可用）
if XGB_AVAILABLE:
    base_models['XGBoost'] = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=positive_weight
    )

if LGBM_AVAILABLE:
    base_models['LightGBM'] = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=7,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )

# 训练和评估基础模型
results = {}

for name, model in base_models.items():
    print(f"\n训练 {name}...")
    
    try:
        # 对于树模型使用早停
        if name in ['XGBoost', 'LightGBM']:
            if name == 'XGBoost':
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=50,
                    verbose=False
                )
            elif name == 'LightGBM':
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=50,
                    verbose=False
                )
        else:
            model.fit(X_train, y_train)
        
        # 预测
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        auc_score = roc_auc_score(y_val, y_pred_proba)
        
        # 交叉验证
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
        
        results[name] = {
            'model': model,
            'auc': auc_score,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        print(f"{name} - 验证集AUC: {auc_score:.5f}")
        print(f"{name} - 交叉验证AUC: {cv_scores.mean():.5f} (+/- {cv_scores.std() * 2:.5f})")
    
    except Exception as e:
        print(f"训练 {name} 时出错: {e}")
        results[name] = {
            'model': None,
            'auc': 0,
            'cv_mean': 0,
            'cv_std': 0
        }

# 显示最佳基础模型
valid_models = {name: result for name, result in results.items() if result['model'] is not None}
if valid_models:
    best_model_name = max(valid_models, key=lambda x: valid_models[x]['auc'])
    best_auc = valid_models[best_model_name]['auc']
    print(f"\n最佳基础模型: {best_model_name}, AUC: {best_auc:.5f}")
else:
    print("\n没有模型成功训练!")


# 9. 高级模型调优
print("步骤9: 高级模型调优...")

if LGBM_AVAILABLE:
    print("进行LightGBM高级调优...")
    
    # 定义参数空间
    param_dist = {
        'n_estimators': [300, 500, 700, 1000],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [5, 7, 9, 11],
        'num_leaves': [15, 31, 63, 127],
        'min_child_samples': [10, 20, 30, 50],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'reg_alpha': [0, 0.1, 0.5, 1],
        'reg_lambda': [0, 0.1, 0.5, 1]
    }
    
    # 创建LightGBM分类器
    lgb_tuner = LGBMClassifier(
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    # 使用随机搜索
    random_search = RandomizedSearchCV(
        lgb_tuner, 
        param_distributions=param_dist,
        n_iter=20,
        cv=3,
        scoring='roc_auc',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    print("开始随机搜索调优...")
    random_search.fit(X_train, y_train)
    
    print("最佳参数:", random_search.best_params_)
    print("最佳分数:", random_search.best_score_)
    
    # 使用最佳参数训练模型
    best_lgb = random_search.best_estimator_
    
    # 评估
    y_pred_best_lgb = best_lgb.predict_proba(X_val)[:, 1]
    best_lgb_auc = roc_auc_score(y_val, y_pred_best_lgb)
    print(f"调优后LightGBM AUC: {best_lgb_auc:.5f}")
    
    # 添加到结果中
    results['LGB_tuned'] = {
        'model': best_lgb,
        'auc': best_lgb_auc,
        'cv_mean': random_search.best_score_,
        'cv_std': 0
    }
    
else:
    print("LightGBM不可用，跳过高级调优")
    best_lgb_auc = 0


# 10. 模型集成和堆叠
print("步骤10: 模型集成和堆叠...")

# 准备可用的基模型
available_models = [(name, results[name]['model']) for name in results if results[name]['model'] is not None]

# 1. 简单投票集成
if len(available_models) >= 2:
    from sklearn.ensemble import VotingClassifier
    
    voting_clf = VotingClassifier(
        estimators=available_models,
        voting='soft',
        n_jobs=-1
    )
    
    print("训练投票集成模型...")
    voting_clf.fit(X_train, y_train)
    y_pred_voting = voting_clf.predict_proba(X_val)[:, 1]
    voting_auc = roc_auc_score(y_val, y_pred_voting)
    print(f"投票集成AUC: {voting_auc:.5f}")
    
    # 2. 堆叠集成
    print("训练堆叠集成模型...")
    
    # 尝试不同的元学习器
    meta_learners = [
        ('LogisticRegression', LogisticRegression(C=0.1, random_state=42, class_weight='balanced')),
        ('XGBoost', xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42, n_jobs=-1)) if XGB_AVAILABLE else None,
        ('LightGBM', LGBMClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42, n_jobs=-1)) if LGBM_AVAILABLE else None
    ]
    
    # 移除None值
    meta_learners = [ml for ml in meta_learners if ml is not None]
    
    best_stacking_auc = 0
    best_stacking_model = None
    
    for meta_name, meta_learner in meta_learners:
        print(f"尝试元学习器: {meta_name}")
        
        try:
            stacking_clf = StackingClassifier(
                estimators=available_models,
                final_estimator=meta_learner,
                cv=5,
                n_jobs=-1
            )
            
            stacking_clf.fit(X_train, y_train)
            y_pred_stacking = stacking_clf.predict_proba(X_val)[:, 1]
            stacking_auc = roc_auc_score(y_val, y_pred_stacking)
            
            print(f"{meta_name} 堆叠模型 AUC: {stacking_auc:.5f}")
            
            if stacking_auc > best_stacking_auc:
                best_stacking_auc = stacking_auc
                best_stacking_model = stacking_clf
                
        except Exception as e:
            print(f"{meta_name} 堆叠失败: {e}")
    
    print(f"最佳堆叠模型 AUC: {best_stacking_auc:.5f}")
    
    # 选择最佳集成方法
    ensemble_auc = max(voting_auc, best_stacking_auc)
    if ensemble_auc == voting_auc and voting_auc > best_lgb_auc:
        ensemble_model = voting_clf
        print("选择投票集成作为集成模型")
    elif ensemble_auc == best_stacking_auc and best_stacking_auc > best_lgb_auc:
        ensemble_model = best_stacking_model
        print("选择堆叠集成作为集成模型")
    else:
        # 使用最佳单个模型
        best_single_name = max(valid_models, key=lambda x: valid_models[x]['auc'])
        ensemble_model = valid_models[best_single_name]['model']
        ensemble_auc = valid_models[best_single_name]['auc']
        print(f"使用最佳单个模型: {best_single_name}")
    
else:
    print("基学习器数量不足，无法进行集成")
    # 使用最佳单个模型
    if valid_models:
        best_model_name = max(valid_models, key=lambda x: valid_models[x]['auc'])
        ensemble_model = valid_models[best_model_name]['model']
        ensemble_auc = valid_models[best_model_name]['auc']
        print(f"使用最佳单个模型: {best_model_name}")
    else:
        ensemble_model = None
        ensemble_auc = 0

print(f"集成模型AUC: {ensemble_auc:.5f}")


# 11. 概率校准
print("步骤11: 概率校准...")

if ensemble_model is not None:
    # 尝试不同的校准方法
    calibration_methods = ['sigmoid', 'isotonic']
    best_calibrated_auc = ensemble_auc
    best_calibrated_model = ensemble_model
    
    for method in calibration_methods:
        print(f"尝试 {method} 校准...")
        
        try:
            calibrated_clf = CalibratedClassifierCV(
                ensemble_model, 
                method=method, 
                cv=3
            )
            
            calibrated_clf.fit(X_train, y_train)
            y_pred_calibrated = calibrated_clf.predict_proba(X_val)[:, 1]
            calibrated_auc = roc_auc_score(y_val, y_pred_calibrated)
            
            print(f"{method} 校准后 AUC: {calibrated_auc:.5f}")
            
            if calibrated_auc > best_calibrated_auc:
                best_calibrated_auc = calibrated_auc
                best_calibrated_model = calibrated_clf
                
        except Exception as e:
            print(f"{method} 校准失败: {e}")
    
    # 更新最终模型
    if best_calibrated_auc > ensemble_auc:
        final_model = best_calibrated_model
        final_auc = best_calibrated_auc
        print(f"使用校准模型，AUC: {final_auc:.5f}")
    else:
        final_model = ensemble_model
        final_auc = ensemble_auc
        print("校准没有提升性能，使用未校准模型")
    
    # 显示校准曲线
    plt.figure(figsize=(10, 8))
    CalibrationDisplay.from_estimator(final_model, X_val, y_val, n_bins=10, name='Final Model')
    plt.title('概率校准曲线')
    plt.tight_layout()
    plt.show()
    
else:
    print("没有可用的集成模型")
    final_model = None
    final_auc = 0


# 12. 特征重要性分析
print("步骤12: 特征重要性分析...")

# 使用最佳模型分析特征重要性
importance_model = None
if LGBM_AVAILABLE and 'LGB_tuned' in results and results['LGB_tuned']['model'] is not None:
    importance_model = results['LGB_tuned']['model']
elif LGBM_AVAILABLE and 'LightGBM' in results and results['LightGBM']['model'] is not None:
    importance_model = results['LightGBM']['model']
elif XGB_AVAILABLE and 'XGBoost' in results and results['XGBoost']['model'] is not None:
    importance_model = results['XGBoost']['model']
elif 'RandomForest' in results and results['RandomForest']['model'] is not None:
    importance_model = results['RandomForest']['model']

if importance_model is not None and hasattr(importance_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': importance_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12, 10))
    sns.barplot(data=feature_importance.head(20), x='importance', y='feature')
    plt.title('Top 20 特征重要性')
    plt.tight_layout()
    plt.show()
    
    print("Top 15 最重要特征:")
    print(feature_importance.head(15))
    
    # 基于特征重要性选择重要特征
    important_features = feature_importance.head(len(feature_importance) // 2)['feature'].tolist()
    print(f"选择 {len(important_features)} 个最重要特征")
    
    # 在重要特征上训练一个新模型
    X_important = X[important_features]
    X_train_important = X_train[important_features]
    X_val_important = X_val[important_features]
    
    # 使用LightGBM在重要特征上训练
    if LGBM_AVAILABLE:
        important_model = LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=7,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
    else:
        important_model = GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            min_samples_split=100,
            min_samples_leaf=50,
            random_state=42,
            subsample=0.8
        )
    
    print("在重要特征上训练新模型...")
    important_model.fit(X_train_important, y_train)
    
    # 评估
    y_pred_important = important_model.predict_proba(X_val_important)[:, 1]
    important_auc = roc_auc_score(y_val, y_pred_important)
    print(f"重要特征模型AUC: {important_auc:.5f}")
    
    # 比较性能
    if important_auc > final_auc:
        final_model = important_model
        final_auc = important_auc
        important_features_list = important_features
        print("使用重要特征模型")
    else:
        important_features_list = None
        print("保持原始模型")
        
else:
    important_features_list = None
    print("无法进行特征重要性分析")


# 13. 生成最终提交文件
print("步骤13: 生成最终提交文件...")

if final_model is not None:
    # 准备测试集
    if important_features_list is not None:
        # 使用重要特征
        X_test_final = test_processed[important_features_list]
        print(f"使用 {len(important_features_list)} 个重要特征")
    else:
        # 使用所有特征
        X_test_final = test_processed[feature_columns_all]
        print(f"使用所有 {len(feature_columns_all)} 个特征")
    
    # 预测概率
    try:
        test_predictions = final_model.predict_proba(X_test_final)[:, 1]
        
        # 对预测概率进行后处理（轻微调整）
        test_predictions_processed = np.clip(test_predictions, 0.001, 0.999)
        
        print("预测完成，进行后处理...")
        
    except Exception as e:
        print(f"预测时出错: {e}")
        # 使用后备模型
        print("使用后备模型...")
        if LGBM_AVAILABLE:
            backup_model = LGBMClassifier(n_estimators=500, random_state=42, class_weight='balanced')
        else:
            backup_model = LogisticRegression(C=0.1, random_state=42, class_weight='balanced')
        
        if important_features_list is not None:
            X_train_backup = X_train[important_features_list]
        else:
            X_train_backup = X_train
            
        backup_model.fit(X_train_backup, y_train)
        test_predictions_processed = backup_model.predict_proba(X_test_final)[:, 1]
        print("后备模型预测完成")
    
    # 创建提交文件
    submission = pd.DataFrame({
        'Id': test_processed['Id'],
        'Probability': test_predictions_processed
    })
    
    # 验证提交文件格式
    print(f"提交文件形状: {submission.shape}")
    print(f"ID列数据类型: {submission['Id'].dtype}")
    print(f"Probability列数据类型: {submission['Probability'].dtype}")
    
    # 确保ID是整数
    submission['Id'] = submission['Id'].astype('Int32')
    
    print(f"ID列前5个值: {submission['Id'].head().tolist()}")
    print(f"Probability列统计:")
    print(f"  最小值: {test_predictions_processed.min():.6f}")
    print(f"  最大值: {test_predictions_processed.max():.6f}")
    print(f"  均值: {test_predictions_processed.mean():.6f}")
    print(f"  标准差: {test_predictions_processed.std():.6f}")
    
    # 保存提交文件
    submission_file = 'submission_final_optimized.csv'
    submission.to_csv(submission_file, index=False)
    
    print(f"提交文件已保存: {submission_file}")
    
    # 显示预测分布
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(test_predictions_processed, bins=50, alpha=0.7, color='lightcoral')
    plt.title('优化模型预测概率分布')
    plt.xlabel('预测概率')
    plt.ylabel('频数')
    
    plt.subplot(1, 2, 2)
    plt.hist(test_predictions_processed, bins=50, alpha=0.7, color='lightseagreen', cumulative=True, density=True)
    plt.title('预测概率累积分布')
    plt.xlabel('预测概率')
    plt.ylabel('累积比例')
    
    plt.tight_layout()
    plt.show()
    
else:
    print("没有可用的最终模型!")

