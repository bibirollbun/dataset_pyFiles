# 导入必要的库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression  # <-- 新增这一行
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report
import xgboost as xgb
import lightgbm as lgb
from lightgbm import early_stopping
import warnings
import time
warnings.filterwarnings('ignore')
# 设置随机种子以确保结果可复现
SEED = 42
np.random.seed(SEED)


# 注意：由于实际数据可能不可用，这里提供了数据加载的示例代码
# 请根据实际情况修改文件路径

def load_data():
    # 尝试加载数据，如果文件不存在则创建模拟数据
    try:
        # 尝试从多个可能的路径加载数据
        try:
            train_data = pd.read_csv('/kaggle/input/porto-seguro-safe-driver-prediction/train.csv')
            test_data = pd.read_csv('/kaggle/input/porto-seguro-safe-driver-prediction/test.csv')
        except FileNotFoundError:
            # 2. 如果找不到，尝试当前目录
            try:
                train_data = pd.read_csv('train.csv')
                test_data = pd.read_csv('test.csv')
                print("成功从当前目录加载数据集...")
            except FileNotFoundError:
                # 3. 如果都找不到，创建模拟数据用于演示
                print("未找到数据集文件，使用模拟数据进行演示...")
                # 模拟数据生成
                np.random.seed(42)
                n_samples = 10000
                
                # 创建特征：连续特征、类别特征和二元特征
                cont_features = [f'ps_reg_{i}' for i in range(1, 4)] + [f'ps_car_{i}' for i in range(1, 14) if i != 11]
                cat_features = [f'ps_ind_{i}' for i in range(1, 10)] + [f'ps_car_{i}' for i in range(11, 14)]
                bin_features = [f'ps_ind_{i}_bin' for i in range(1, 8)] + [f'ps_calc_{i}_bin' for i in range(1, 8)]
                
                features = cont_features + cat_features + bin_features
                
                # 创建训练数据
                train_data = pd.DataFrame(np.random.randn(n_samples, len(features)), columns=features)
                
                # 添加目标变量（不平衡数据）
                train_data['target'] = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])
                
                # 添加id列
                train_data['id'] = range(n_samples)
                
                # 创建测试数据
                test_data = pd.DataFrame(np.random.randn(n_samples//2, len(features)), columns=features)
                test_data['id'] = range(n_samples, n_samples + n_samples//2)
                
                # 添加一些缺失值来模拟实际数据
                for col in train_data.columns:
                    mask = np.random.rand(n_samples) < 0.05
                    train_data.loc[mask, col] = -1
                
                for col in test_data.columns:
                    mask = np.random.rand(n_samples//2) < 0.05
                    test_data.loc[mask, col] = -1
        
        return train_data, test_data
    except Exception as e:
        print(f"加载数据出错: {e}")
        return None, None

# 加载数据
train_data, test_data = load_data()

if train_data is not None:
    print(f"训练数据形状: {train_data.shape}")
    print(f"测试数据形状: {test_data.shape}")
    print("训练数据前5行:\n", train_data.head())
    print("目标变量分布:\n", train_data['target'].value_counts())
    print("缺失值情况:\n", train_data.isnull().sum().sum(), "个缺失值")


def preprocess_data(train_data, test_data):
    """
       数据预处理函数
       功能：处理缺失值、编码类别特征、标准化数值特征
       参数：
           train_data: 训练数据集
           test_data: 测试数据集
       返回：
           X_scaled: 预处理后的训练特征
           y: 目标变量
           X_test_scaled: 预处理后的测试特征
           feature_names: 特征名称列表
    """
    # 复制数据以避免修改原始数据
    train = train_data.copy()
    test = test_data.copy()
    # 分离特征和目标变量
    X = train.drop(['id', 'target'], axis=1)
    y = train['target']
    X_test = test.drop(['id'], axis=1)
    # 处理缺失值(用-1标记的缺失值)
    X = X.replace(-1, np.nan)
    X_test = X_test.replace(-1, np.nan)
    # 填充缺失值
    # 对于数值型特征，使用中位数填充
    for col in X.columns:
        if X[col].isnull().sum() > 0:
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val)
            X_test[col] = X_test[col].fillna(median_val)
    # 识别并处理类别特征
    cat_cols = [col for col in X.columns if col.endswith('_cat') or col.endswith('_bin')]
    for col in cat_cols:
        le = LabelEncoder()
        # 合并训练和测试数据进行编码，确保所有类别都被覆盖
        combined = pd.concat([X[col], X_test[col]], axis=0).astype(str)
        le.fit(combined)
        X[col] = le.transform(X[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))
    # 特征标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_test_scaled = scaler.transform(X_test)
    return X_scaled, y, X_test_scaled, X.columns
if train_data is not None:
    X_scaled, y, X_test_scaled, feature_names = preprocess_data(train_data, test_data)
    print(f"预处理后训练特征形状: {X_scaled.shape}")
    print(f"预处理后测试特征形状: {X_test_scaled.shape}")


def feature_engineering(X, X_test, feature_names):
    """
        特征工程函数
        功能：创建特征交互、统计特征等
        参数：
            X: 训练特征
            X_test: 测试特征
            feature_names: 特征名称列表
        返回：
            X_engineered: 特征工程后的训练特征
            X_test_engineered: 特征工程后的测试特征
    """
    # 将数组转换回DataFrame以便进行特征工程
    df_train = pd.DataFrame(X, columns=feature_names)
    df_test = pd.DataFrame(X_test, columns=feature_names)
    # 1. 特征交互
    # 选择几个重要特征进行交互
    if len(feature_names) >= 2:
         # 选择前两个特征进行乘法和加法交互
        feat1, feat2 = feature_names[0], feature_names[1]
        df_train[f'{feat1}_times_{feat2}'] = df_train[feat1] * df_train[feat2]
        df_train[f'{feat1}_plus_{feat2}'] = df_train[feat1] + df_train[feat2]
        df_test[f'{feat1}_times_{feat2}'] = df_test[feat1] * df_test[feat2]
        df_test[f'{feat1}_plus_{feat2}'] = df_test[feat1] + df_test[feat2]
    # 2. 创建统计特征
    # 对于连续特征，计算特征组的统计信息
    # 识别可能相关的特征组
    reg_features = [col for col in feature_names if col.startswith('ps_reg')]
    car_features = [col for col in feature_names if col.startswith('ps_car')]
    # 创建特征组的统计特征
    if len(reg_features) > 0:
        df_train['reg_mean'] = df_train[reg_features].mean(axis=1)
        df_train['reg_std'] = df_train[reg_features].std(axis=1)
        df_test['reg_mean'] = df_test[reg_features].mean(axis=1)
        df_test['reg_std'] = df_test[reg_features].std(axis=1)
    if len(car_features) > 0:
        df_train['car_mean'] = df_train[car_features].mean(axis=1)
        df_train['car_std'] = df_train[car_features].std(axis=1)
        df_test['car_mean'] = df_test[car_features].mean(axis=1)
        df_test['car_std'] = df_test[car_features].std(axis=1)
    return df_train.values, df_test.values
if 'X_scaled' in locals():
    X_engineered, X_test_engineered = feature_engineering(X_scaled, X_test_scaled, feature_names)
    print(f"特征工程后训练特征形状: {X_engineered.shape}")
    print(f"特征工程后测试特征形状: {X_test_engineered.shape}")


def train_and_evaluate_advanced_models(X, y, save_roc_path=None):
    """
        训练并评估高级机器学习模型
        功能：使用Random Forest、XGBoost和LightGBM进行模型训练、交叉验证，适配低内存环境
        参数:
            X: 特征数据（pandas DataFrame或numpy数组）
            y: 目标变量（pandas Series或numpy数组）
            save_roc_path: ROC曲线保存路径（如'./roc_curve.png'），默认不保存
        返回：
            best_model: 性能最好的模型（在完整训练集上重新训练）
            results: 各模型性能结果（含交叉验证AUC、验证集AUC）
            training_times: 各模型训练时间
            feature_importance: 各模型的特征重要性（DataFrame格式）
    """
    # 数据类型检查
    if not isinstance(X, (pd.DataFrame, np.ndarray)) or not isinstance(y, (pd.Series, np.ndarray)):
        raise TypeError("X必须是DataFrame/numpy数组，y必须是Series/numpy数组")
    
    # 分割训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    
    # 为numpy数组添加特征名（方便特征重要性输出）
    if isinstance(X, np.ndarray) and not isinstance(X, pd.DataFrame):
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        X_train = pd.DataFrame(X_train, columns=feature_names)
        X_val = pd.DataFrame(X_val, columns=feature_names)
    else:
        feature_names = X.columns.tolist()
    
    print("="*50)
    print("训练不同的模型并评估性能（已适配低内存环境）")
    print(f"训练集规模: {X_train.shape[0]} 样本 × {X_train.shape[1]} 特征")
    print("="*50)
    
    # 定义模型：LightGBM已设置force_col_wise=true，XGBoost添加内存优化
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100, random_state=SEED, n_jobs=-1, verbose=0,
            max_samples=0.8  # 随机森林内存优化：仅用80%样本训练，降低内存占用
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=100, random_state=SEED, n_jobs=-1, verbosity=0,
            use_label_encoder=False, eval_metric='auc',
            tree_method='hist',  # XGBoost内存优化：用直方图算法减少内存
            max_bin=256  # 控制直方图粒度，平衡内存与精度
        ),
        'LightGBM': lgb.LGBMClassifier(
            n_estimators=100, random_state=SEED, n_jobs=-1, verbose=0,
            force_col_wise=True,  # 核心：列优先存储，大幅降低LightGBM内存占用
            max_bin=256  # 减少特征分箱数量，进一步优化内存
        )
    }
    
    # 初始化结果存储字典
    results = {}
    training_times = {}
    feature_importance = pd.DataFrame(index=feature_names)
    best_auc = 0
    best_model = None
    
    for name, model in models.items():
        print(f"\n【{name}】训练开始")
        start_time = time.time()
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        cv_scores = []
        fold_importance = []
        
        try:
            for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
                X_fold_train = X_train.iloc[train_idx]
                X_fold_val = X_train.iloc[val_idx]
                y_fold_train = y_train.iloc[train_idx] if hasattr(y_train, 'iloc') else y_train[train_idx]
                y_fold_val = y_train.iloc[val_idx] if hasattr(y_train, 'iloc') else y_train[val_idx]
                
                print(f"  折{fold}/5: 训练中...")
                # 训练模型
                model.fit(X_fold_train, y_fold_train)
                # 计算AUC
                y_pred_proba = model.predict_proba(X_fold_val)[:, 1]
                cv_scores.append(roc_auc_score(y_fold_val, y_pred_proba))
                print(f"  折{fold}/5: AUC = {cv_scores[-1]:.6f}")
                
                # 记录特征重要性
                if name == 'Random Forest':
                    fold_importance.append(model.feature_importances_)
                elif name == 'XGBoost':
                    fold_importance.append(model.get_booster().get_score(importance_type='gain'))
                elif name == 'LightGBM':
                    fold_importance.append(model.feature_importances_)
            
            # 计算训练结果
            avg_auc = np.mean(cv_scores)
            std_auc = np.std(cv_scores)
            y_val_pred_proba = model.predict_proba(X_val)[:, 1]
            val_auc = roc_auc_score(y_val, y_val_pred_proba)
            training_time = time.time() - start_time
            
            # 存储结果
            results[name] = {
                'cv_mean_auc': avg_auc, 'cv_std_auc': std_auc,
                'cv_fold_scores': cv_scores, 'val_auc': val_auc
            }
            training_times[name] = training_time
            # 计算平均特征重要性
            if name in ['Random Forest', 'LightGBM']:
                feature_importance[name] = np.mean(fold_importance, axis=0)
            else:  # XGBoost特征重要性对齐
                xgb_imp = {f: 0 for f in feature_names}
                for fold_dict in fold_importance:
                    for f, imp in fold_dict.items():
                        xgb_imp[f] += imp / len(fold_importance)
                feature_importance[name] = feature_importance.index.map(xgb_imp)
            
            # 打印模型总结
            print(f"\n【{name}】训练完成")
            print(f"  交叉验证平均AUC: {avg_auc:.6f} (±{std_auc:.6f})")
            print(f"  验证集AUC: {val_auc:.6f}")
            print(f"  训练时间: {training_time:.2f} 秒")
            
            # 更新最佳模型
            if avg_auc > best_auc:
                best_auc = avg_auc
                best_model = models[name].__class__(**models[name].get_params())
        
        # 捕获内存不足错误，给出具体解决方案
        except MemoryError:
            print(f"\n⚠️ 【{name}】训练失败：内存不足")
            if name == 'Random Forest':
                print("  建议：1. 降低max_samples（如设为0.6）；2. 减少n_estimators（如设为50）")
            elif name == 'XGBoost':
                print("  建议：1. 将tree_method改为'gpu_hist'（需GPU）；2. 增加max_bin（如512）")
            elif name == 'LightGBM':
                print("  建议：1. 检查是否误设force_row_wise=true；2. 减少n_estimators（如设为50）")
            continue  # 跳过当前模型，继续训练其他模型
    
    # 评估并绘制最佳模型ROC曲线
    if best_model is not None:
        best_model.fit(X_train, y_train)
        best_model_name = [k for k, v in models.items() if v.__class__ == best_model.__class__][0]
        y_val_pred_proba = best_model.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, y_val_pred_proba)
        
        print(f"最佳模型: {best_model_name}")
        print(f"最佳模型验证集AUC: {val_auc:.6f}")
        
        # 绘制ROC曲线
        plt.figure(figsize=(10, 6))
        fpr, tpr, _ = roc_curve(y_val, y_val_pred_proba)
        plt.plot(fpr, tpr, label=f'{best_model_name} (AUC = {val_auc:.6f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Best Model')
        plt.legend()
        plt.grid(True)
        if save_roc_path:
            plt.savefig(save_roc_path, dpi=300, bbox_inches='tight')
            print(f"ROC曲线已保存至: {save_roc_path}")
        plt.show()
    
    return best_model, results, training_times, feature_importance

# 执行训练（检查数据是否存在）
if 'X_engineered' in locals() and 'y' in locals():
    best_model, model_results, training_times, feat_imp = train_and_evaluate_advanced_models(
        X_engineered, y, save_roc_path='./best_model_roc.png'  # 可修改ROC保存路径
    )



# import time
# import numpy as np
# import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import GridSearchCV, train_test_split
# from sklearn.metrics import roc_auc_score
# import xgboost as xgb
# import lightgbm as lgb
# from lightgbm import early_stopping

# --------------------------
# 1. 加载数据集（你的原始定义）
# --------------------------
# train_data = pd.read_csv('/kaggle/input/porto-seguro-safe-driver-prediction/train.csv')
# test_data = pd.read_csv('/kaggle/input/porto-seguro-safe-driver-prediction/test.csv')

# 提取特征和目标变量
X = train_data.drop(columns=['id', 'target'])
y = train_data['target']
X_test = test_data.drop(columns=['id'])


# --------------------------
# 2. 参数调优函数（修复早停验证集）
# --------------------------
def hyperparameter_tuning(X, y, SEED=42):
    print("\n开始使用训练集进行参数调优...")
    
    # 为LightGBM的早停机制单独划分验证集（从训练集中取10%）
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.1, random_state=SEED, stratify=y
    )
    print(f"为早停划分完成：训练集{X_train.shape[0]}样本，早停验证集{X_val.shape[0]}样本")
    
    tuned_models = {}
    tuning_results = {}
    
    # 定义基础模型
    base_models = {
        'Random Forest': RandomForestClassifier(
            random_state=SEED,
            n_jobs=-1,
            class_weight='balanced'
        ),
        'XGBoost': xgb.XGBClassifier(
            random_state=SEED,
            n_jobs=-1,
            use_label_encoder=False,
            eval_metric='auc',
            scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
            verbose=False
        ),
        'LightGBM': lgb.LGBMClassifier(
            random_state=SEED,
            n_jobs=-1,
            class_weight='balanced',
            eval_metric='auc',
            verbose=-1
        )
    }
    
    # 参数网格
    param_grids = {
        'Random Forest': {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'max_features': ['sqrt', 'log2']
        },
        'XGBoost': {
            'n_estimators': [100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        },
        'LightGBM': {
            'n_estimators': [100, 200],
            'max_depth': [3, 6, -1],
            'learning_rate': [0.01, 0.1],
            'num_leaves': [31, 63],
            'min_child_samples': [10, 20]
        }
    }
    
    # 模型调优
    for name in ['Random Forest', 'XGBoost', 'LightGBM']:
        print(f"\n===== 开始调优模型: {name} =====")
        start_time = time.time()
        
        try:
            grid_search = GridSearchCV(
                estimator=base_models[name],
                param_grid=param_grids[name],
                cv=3,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=0,
                return_train_score=True
            )
            
            # 拟合逻辑（为LightGBM指定早停验证集）
            if name == 'LightGBM':
                # 显式传递早停验证集给LightGBM
                grid_search.fit(
                    X_train, y_train,  # 用90%训练集拟合
                    eval_set=[(X_val, y_val)],  # 用10%验证集做早停评估
                    callbacks=[early_stopping(20)]
                )
            else:
                # 其他模型用完整训练集（X和y）
                grid_search.fit(X, y)
            
            # 记录结果
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_
            tuning_time = time.time() - start_time
            
            tuned_models[name] = grid_search.best_estimator_
            tuning_results[name] = {
                'best_params': best_params,
                'best_score': best_score,
                'tuning_time': tuning_time,
                'cv_results': grid_search.cv_results_
            }
            
            print(f"✅ {name}调优完成")
            print(f"最佳参数：{best_params}")
            print(f"训练集交叉验证AUC: {best_score:.6f}")
            print(f"调优耗时：{tuning_time:.2f}秒")
            
        except Exception as e:
            print(f"❌ {name}调优失败：{str(e)}")
            tuned_models[name] = None
            tuning_results[name] = {'error': str(e)}
    
    return tuned_models, tuning_results


# --------------------------
# 3. 执行调优并生成提交文件
# --------------------------
if X.shape[0] == y.shape[0] and X.shape[0] > 0:
    tuned_models, tuning_results = hyperparameter_tuning(X, y)
    
    # 生成测试集预测结果
    print("\n===== 生成测试集预测结果 =====")
    for name, model in tuned_models.items():
        if model is not None:
            test_pred_proba = model.predict_proba(X_test)[:, 1]
            submission = pd.DataFrame({
                'id': test_data['id'],
                'target': test_pred_proba
            })
            submission.to_csv(f'{name}_submission.csv', index=False)
            print(f"{name}的预测结果已保存")
else:
    print("⚠️ 数据格式错误：X和y的样本数不匹配或为空")


def stacking_ensemble(X, y, X_test):
    """
        模型融合（Stacking）函数
        功能：使用Random Forest、XGBoost和LightGBM作为基础模型进行Stacking集成
        参数：
            X: 特征数据
            y: 目标变量
            x_test: 特征测试
        返回：
            stacking_models: Stacking集成模型
            test_predictions: 测试集预测结果
            val_auc:验证集AUC分数
    """
    print("\n实施模型融合(Stacking)...")
    # 定义基础模型
    base_models = [
        ('rf', RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs = -1)),
        ('xgb', xgb.XGBClassifier(n_estimators=100, random_state=SEED, n_jobs = -1)),
        ('lgb', lgb.LGBMClassifier(n_estimators=100, random_state=SEED, n_jobs = -1))
    ]

    # 分割训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

    # 生成元特征（meta-features）
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    meta_features_train = np.zeros((X_train.shape[0], len(base_models)))
    meta_features_val = np.zeros((X_val.shape[0], len(base_models)))
    meta_features_test = np.zeros((X_test.shape[0], len(base_models)))

    # 为每个基础模型生成元特征
    for i, (name, model) in enumerate(base_models):
        print(f"训练基础模型 {name}...")
        # 交叉验证生成训练集的元特征
        for train_idx, val_idx in skf.split(X_train, y_train):
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train = y_train.iloc[train_idx]
            
            model.fit(X_fold_train, y_fold_train)
            meta_features_train[val_idx, i] = model.predict_proba(X_fold_val)[:, 1]

        # 在整个训练集上训练模型，然后预测验证集和测试集
        model.fit(X_train, y_train)
        meta_features_val[:, i] = model.predict_proba(X_val)[:, 1]
        meta_features_test[:, i] = model.predict_proba(X_test)[:, 1]
        
    # 训练元模型（meta-model）
    meta_model = LogisticRegression(random_state=SEED)
    meta_model.fit(meta_features_train, y_train)
    
    # 在验证集上评估集成模型
    y_val_pred_proba = meta_model.predict_proba(meta_features_val)[:, 1]
    val_auc = roc_auc_score(y_val, y_val_pred_proba)
    
    print(f"\nStacking 集成模型验证集AUC: {val_auc:.6f}")
    
    # 绘制ROC曲线
    plt.figure(figsize=(10, 6))
    fpr, tpr, _ = roc_curve(y_val, y_val_pred_proba)
    plt.plot(fpr, tpr, label=f'Stacking Ensemble (AUC = {val_auc:.6f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Stacking Ensemble')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # 生成测试集预测结果
    test_pred_proba = meta_model.predict_proba(meta_features_test)[:, 1]
    
    return meta_model, test_pred_proba, val_auc

if 'X_engineered' in locals() and 'X_test_engineered' in locals():
    stacking_model, test_predictions, stacking_auc = stacking_ensemble(X_engineered, y, X_test_engineered)


def analyze_feature_importance(model, feature_names, top_n=10):
    """
    特征重要性分析函数
    功能：分析并可视化模型的特征重要性
    参数：
        model: 训练好的模型
        feature_names: 特征名列表
        top_n: 显示前n个重要特征
    返回：
        top_features: 前n个重要特征的DataFrame
    """
    try:
        # 获取特征重要性（兼容树模型和线性模型）
        if hasattr(model, 'feature_importances_'):
            # 树模型（如RF、XGBoost、LightGBM）
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # 线性模型（如逻辑回归，取系数绝对值）
            importances = np.abs(model.coef_) if model.coef_.ndim == 1 else np.abs(model.coef_[0])
        else:
            print("所选模型不支持特征重要性分析。")
            return None

        # 校验特征名称与重要性长度是否匹配
        if len(feature_names) != len(importances):
            print(f"特征名称数量（{len(feature_names)}）与特征重要性数量（{len(importances)}）不匹配！")
            return None

        # 创建特征重要性DataFrame（修正列名拼写和赋值）
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances  # 修正：补充重要性数据列
        })

        # 排序并选择前N个重要特征（修正语法错误）
        top_features = feature_importance_df.sort_values(  # 修正：sortvalues→sort_values
            'Importance', ascending=False
        ).head(top_n)  # 修正：括号位置错误

        # 绘制特征重要性条形图
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=top_features)
        plt.title(f"TOP{top_n}特征重要性")
        plt.tight_layout()
        plt.show()

        return top_features  # 修正：拼写错误featuers→features

    except Exception as e:  # 修正：捕获异常时需要定义e
        print(f"特征重要性分析出错：{e}")
        return None


# --------------------------
# 执行特征重要性分析
# --------------------------
if 'best_model' in locals() and 'feature_names' in locals():
    # 生成完整的特征名称列表（兼容特征工程后的新特征）
    # 假设X_engineered是特征工程后的训练集
    if 'X_engineered' in locals():
        df_temp = pd.DataFrame(X_engineered)
        new_feature_names = [f"feature_{i}" for i in range(df_temp.shape[1])]
        
        # 尝试匹配原始特征名称+新特征（修正语法和变量拼写）
        if len(feature_names) + 4 == df_temp.shape[1]:  # 修正：缺少冒号，df_femp→df_temp
            complete_feature_names = list(feature_names)  # 修正：feature_name→feature_names
            
            # 补充特征工程生成的新特征（如交互特征、统计特征）
            if len(feature_names) >= 2:
                feat1, feat2 = feature_names[0], feature_names[1]
                complete_feature_names.extend([f'{feat1}_times_{feat2}', f'{feat1}_plus_{feat2}'])
            complete_feature_names.extend(['reg_mean', 'reg_std', 'car_mean', 'car_std'])
            new_feature_names = complete_feature_names
        
        # 调用特征重要性分析函数
        top_features = analyze_feature_importance(best_model, new_feature_names)
    else:
        print("⚠️ 未找到特征工程后的数据集X_engineered")
else:
    print("⚠️ 未找到最佳模型（best_model）或原始特征名称列表（feature_names）")


def generate_submission(test_data, predictions, filename='submission_1.csv'):
    """
    生成提交结果函数
    功能:创建符合竞赛要求格式的提交文件
    参数：
        test_data: 测试数据集（包含'id'列）
        predictions: 预测概率结果（与测试集样本数对应）
        filename: 输出文件名
    """
    # 校验输入格式
    if 'id' not in test_data.columns:
        print("⚠️ 测试数据集必须包含'id'列")
        return None
    if len(predictions) != len(test_data):
        print(f"⚠️ 预测结果数量（{len(predictions)}）与测试集样本数（{len(test_data)}）不匹配")
        return None
    
    # 创建提交DataFrame
    submission = pd.DataFrame({
        'id': test_data['id'],
        'target': predictions
    })
    
    # 保存到CSV文件
    submission.to_csv(filename, index=False)
    print(f"提交文件已生成: {filename}")
    print(f"提交文件形状：{submission.shape}")
    print("提交文件前5行:\n", submission.head())  # 修正：submission_1→submission
    
    return submission


# 执行提交文件生成
if 'test_predictions' in locals() and 'test_data' in locals():
    submission = generate_submission(
        test_data=test_data,
        predictions=test_predictions,
        filename='stacking_submission.csv'  # 可自定义文件名
    )
else:
    print("⚠️ 未找到测试集预测结果（test_predictions）或测试数据集（test_data）")


