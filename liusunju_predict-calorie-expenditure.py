# é€™å€‹ Python 3 ç’°å¢ƒé �è£�äº†è¨±å¤šå¯¦ç”¨çš„åˆ†æ��å‡½å¼�åº«
# å®ƒæ˜¯ä»¥ kaggle/python Docker æ˜ åƒ�æª”ç‚ºåŸºç¤�æ‰€å®šç¾©ï¼š[https://github.com/kaggle/docker-python](https://github.com/kaggle/docker-python)
# ä¾‹å¦‚ï¼Œä»¥ä¸‹æ˜¯ä¸€äº›è¼‰å…¥çš„å¯¦ç”¨å¥—ä»¶

import numpy as np # ç·šæ€§ä»£æ•¸
import pandas as pd # è³‡æ–™è™•ç�†ã€�CSV æª”æ¡ˆè¼¸å…¥/è¼¸å‡º (ä¾‹å¦‚ pd.read_csv)

# è¼¸å…¥è³‡æ–™æª”æ¡ˆä½�æ–¼å”¯è®€çš„ "../input/" ç›®éŒ„ä¸‹
# ä¾‹å¦‚ï¼ŒåŸ·è¡Œæ­¤è™• (é»�æ“Š "run" æˆ–æŒ‰ä¸‹ Shift+Enter) å°‡æœƒåˆ—å‡ºè¼¸å…¥ç›®éŒ„ä¸‹çš„æ‰€æœ‰æª”æ¡ˆ

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# ä½ å�¯ä»¥å°‡æœ€å¤š 20GB çš„è³‡æ–™å¯«å…¥ç›®å‰�ç›®éŒ„ (/kaggle/working/)ï¼Œç•¶ä½ ä½¿ç”¨ "Save & Run All" å»ºç«‹ç‰ˆæœ¬æ™‚ï¼Œé€™äº›è³‡æ–™æœƒè¢«å„²å­˜ä¸‹ä¾†
# ä½ ä¹Ÿå�¯ä»¥å°‡æš«å­˜æª”æ¡ˆå¯«å…¥ /kaggle/temp/ï¼Œä½†é€™äº›æª”æ¡ˆåœ¨ç›®å‰�å·¥ä½œéš�æ®µçµ�æ�Ÿå¾Œä¸�æœƒè¢«å„²å­˜


# ==============================================
# Kaggle å�¡è·¯é‡Œæ¶ˆè€—é �æ¸¬ç«¶è³½ - ç¬¬1æ®µï¼šç’°å¢ƒè¨­ç½®èˆ‡è³‡æ–™è¼‰å…¥
# ==============================================

import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# è¨­å®šåœ–å½¢é¡¯ç¤ºå�ƒæ•¸ï¼ˆä¸�ä½¿ç”¨ä¸­æ–‡å­—é«”ï¼‰
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
sns.set_style("whitegrid")

print("å¥—ä»¶è¼‰å…¥å®Œæˆ�")
print("=" * 50)

# è¼‰å…¥è³‡æ–™
try:
    # å�‡è¨­æ‚¨å·²ç¶“å¾�Kaggleä¸‹è¼‰äº†è³‡æ–™é›†
    train_df = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
    test_df = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
    sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')
    
    print("è³‡æ–™è¼‰å…¥æˆ�åŠŸï¼�")
    print(f"è¨“ç·´è³‡æ–™å½¢ç‹€: {train_df.shape}")
    print(f"æ¸¬è©¦è³‡æ–™å½¢ç‹€: {test_df.shape}")
    print(f"æ��äº¤ç¯„ä¾‹å½¢ç‹€: {sample_submission.shape}")
    
except FileNotFoundError:
    print("è«‹å…ˆå¾�Kaggleä¸‹è¼‰è³‡æ–™é›†ä¸¦ä¸Šå‚³åˆ°Colab")
    print("æ‚¨éœ€è¦�çš„æª”æ¡ˆ:")
    print("- train.csv")
    print("- test.csv") 
    print("- sample_submission.csv")

print("=" * 50)


# ==============================================
# Kaggle å�¡è·¯é‡Œæ¶ˆè€—é �æ¸¬ç«¶è³½ - ç¬¬2æ®µï¼šè³‡æ–™æ�¢ç´¢èˆ‡åˆ†æ��
# ==============================================

# æª¢è¦–è³‡æ–™åŸºæœ¬è³‡è¨Š
print("=== è¨“ç·´è³‡æ–™åŸºæœ¬è³‡è¨Š ===")
print(train_df.info())
print("\n" + "=" * 50)

print("=== è¨“ç·´è³‡æ–™æ��è¿°çµ±è¨ˆ ===")
print(train_df.describe())
print("\n" + "=" * 50)

print("=== å‰�5ç­†è¨“ç·´è³‡æ–™ ===")
print(train_df.head())
print("\n" + "=" * 50)

# æª¢æŸ¥ç¼ºå¤±å€¼
print("=== ç¼ºå¤±å€¼æª¢æŸ¥ ===")
missing_train = train_df.isnull().sum()
missing_test = test_df.isnull().sum()

print("è¨“ç·´è³‡æ–™ç¼ºå¤±å€¼:")
print(missing_train[missing_train > 0])
print("\næ¸¬è©¦è³‡æ–™ç¼ºå¤±å€¼:")
print(missing_test[missing_test > 0])

if missing_train.sum() == 0 and missing_test.sum() == 0:
    print("âœ“ æ²’æœ‰ç¼ºå¤±å€¼")
print("\n" + "=" * 50)

# æª¢è¦–æ•¸å€¼å�‹å’Œé¡�åˆ¥å�‹è®Šæ•¸
numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()

print("=== è®Šæ•¸é¡�å�‹åˆ†æ�� ===")
print(f"æ•¸å€¼å�‹è®Šæ•¸ ({len(numeric_cols)}å€‹): {numeric_cols}")
print(f"é¡�åˆ¥å�‹è®Šæ•¸ ({len(categorical_cols)}å€‹): {categorical_cols}")
print("\n" + "=" * 50)


# ==============================================
# Kaggle å�¡è·¯é‡Œæ¶ˆè€—é �æ¸¬ç«¶è³½ - ç¬¬3æ®µï¼šè³‡æ–™è¦–è¦ºåŒ–åˆ†æ��
# ==============================================

# ç›®æ¨™è®Šæ•¸åˆ†å¸ƒåˆ†æ��ï¼ˆå�‡è¨­ç›®æ¨™è®Šæ•¸ç‚º 'Calories'ï¼‰
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# ç›®æ¨™è®Šæ•¸åˆ†å¸ƒ
if 'Calories' in train_df.columns:
    target_col = 'Calories'
elif 'calories' in train_df.columns:
    target_col = 'calories'
else:
    # å�‡è¨­æœ€å¾Œä¸€æ¬„æ˜¯ç›®æ¨™è®Šæ•¸
    target_col = train_df.columns[-1]

print(f"ç›®æ¨™è®Šæ•¸: {target_col}")

# åˆ†å¸ƒåœ–
axes[0,0].hist(train_df[target_col], bins=50, alpha=0.7, edgecolor='black')
axes[0,0].set_title('Target Variable Distribution')
axes[0,0].set_xlabel('Calories')
axes[0,0].set_ylabel('Frequency')

# ç®±å�‹åœ–
axes[0,1].boxplot(train_df[target_col])
axes[0,1].set_title('Target Variable Boxplot')
axes[0,1].set_ylabel('Calories')

# Q-Qåœ–æª¢é©—å¸¸æ…‹æ€§
from scipy import stats
stats.probplot(train_df[target_col], dist="norm", plot=axes[1,0])
axes[1,0].set_title('Q-Q Plot')

# logè½‰æ�›å¾Œçš„åˆ†å¸ƒ
log_target = np.log1p(train_df[target_col])
axes[1,1].hist(log_target, bins=50, alpha=0.7, edgecolor='black')
axes[1,1].set_title('Log-transformed Target Distribution')
axes[1,1].set_xlabel('Log(Calories + 1)')
axes[1,1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

print(f"ç›®æ¨™è®Šæ•¸çµ±è¨ˆ:")
print(f"å¹³å�‡å€¼: {train_df[target_col].mean():.2f}")
print(f"ä¸­ä½�æ•¸: {train_df[target_col].median():.2f}")
print(f"æ¨™æº–å·®: {train_df[target_col].std():.2f}")
print(f"å��åº¦: {train_df[target_col].skew():.2f}")
print(f"å³°åº¦: {train_df[target_col].kurtosis():.2f}")
print("=" * 50)


# ==============================================
# Kaggle å�¡è·¯é‡Œæ¶ˆè€—é �æ¸¬ç«¶è³½ - ç¬¬4æ®µï¼šç‰¹å¾µåˆ†æ��èˆ‡ç›¸é—œæ€§
# ==============================================

# æ•¸å€¼å�‹è®Šæ•¸ç›¸é—œæ€§ç†±åŠ›åœ–
numeric_features = train_df.select_dtypes(include=[np.number]).columns.tolist()
if 'id' in numeric_features:
    numeric_features.remove('id')

correlation_matrix = train_df[numeric_features].corr()

plt.figure(figsize=(14, 10))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, 
            mask=mask,
            annot=True, 
            cmap='coolwarm', 
            center=0,
            square=True,
            fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.show()

# èˆ‡ç›®æ¨™è®Šæ•¸ç›¸é—œæ€§æœ€é«˜çš„ç‰¹å¾µ
target_corr = correlation_matrix[target_col].abs().sort_values(ascending=False)
print("=== èˆ‡ç›®æ¨™è®Šæ•¸ç›¸é—œæ€§æ�’åº� ===")
print(target_corr.drop(target_col))
print("=" * 50)

# é‡�è¦�ç‰¹å¾µçš„æ•£é»�åœ–
top_features = target_corr.drop(target_col).head(6).index.tolist()

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

for i, feature in enumerate(top_features):
    axes[i].scatter(train_df[feature], train_df[target_col], alpha=0.5)
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel(target_col)
    axes[i].set_title(f'{feature} vs {target_col}')
    
    # è¨ˆç®—ç›¸é—œä¿‚æ•¸
    corr = train_df[feature].corr(train_df[target_col])
    axes[i].text(0.05, 0.95, f'Corr: {corr:.3f}', 
                transform=axes[i].transAxes, 
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.show()

print("=" * 50)


# ==============================================
# Kaggle å�¡è·¯é‡Œæ¶ˆè€—é �æ¸¬ç«¶è³½ - ç¬¬5æ®µï¼šè³‡æ–™å‰�è™•ç�†
# ==============================================

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# è¤‡è£½è³‡æ–™ä»¥é�¿å…�ä¿®æ”¹å�Ÿå§‹è³‡æ–™
train_processed = train_df.copy()
test_processed = test_df.copy()

print("=== é–‹å§‹è³‡æ–™å‰�è™•ç�† ===")

# è™•ç�†é¡�åˆ¥å�‹è®Šæ•¸
categorical_features = train_processed.select_dtypes(include=['object']).columns.tolist()
if 'id' in categorical_features:
    categorical_features.remove('id')

print(f"éœ€è¦�è™•ç�†çš„é¡�åˆ¥å�‹è®Šæ•¸: {categorical_features}")

# æ¨™ç±¤ç·¨ç¢¼
label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    # å�ˆä½µè¨“ç·´å’Œæ¸¬è©¦è³‡æ–™çš„é¡�åˆ¥å€¼ä¾†æ“¬å�ˆç·¨ç¢¼å™¨
    combined_values = pd.concat([train_processed[col], test_processed[col]]).astype(str)
    le.fit(combined_values)
    
    train_processed[col] = le.transform(train_processed[col].astype(str))
    test_processed[col] = le.transform(test_processed[col].astype(str))
    
    label_encoders[col] = le
    print(f"âœ“ {col} å·²å®Œæˆ�æ¨™ç±¤ç·¨ç¢¼")

# ç‰¹å¾µå·¥ç¨‹ - å‰µå»ºæ–°ç‰¹å¾µ
print("\n=== ç‰¹å¾µå·¥ç¨‹ ===")

# å�‡è¨­é€™æ˜¯å�¥èº«ç›¸é—œè³‡æ–™ï¼Œå‰µå»ºä¸€äº›å�¯èƒ½æœ‰ç”¨çš„ç‰¹å¾µ
if 'Age' in train_processed.columns and 'Weight' in train_processed.columns:
    # BMI (å�‡è¨­æœ‰èº«é«˜è³‡æ–™)
    if 'Height' in train_processed.columns:
        train_processed['BMI'] = train_processed['Weight'] / (train_processed['Height'] / 100) ** 2
        test_processed['BMI'] = test_processed['Weight'] / (test_processed['Height'] / 100) ** 2
        print("âœ“ å‰µå»ºBMIç‰¹å¾µ")

# å¦‚æ�œæœ‰é�‹å‹•å¼·åº¦å’Œæ™‚é–“ï¼Œå‰µå»ºç¸½é�‹å‹•é‡�ç‰¹å¾µ
if 'Duration' in train_processed.columns and 'Heart_Rate' in train_processed.columns:
    train_processed['Exercise_Intensity'] = train_processed['Duration'] * train_processed['Heart_Rate']
    test_processed['Exercise_Intensity'] = test_processed['Duration'] * test_processed['Heart_Rate']
    print("âœ“ å‰µå»ºé�‹å‹•å¼·åº¦ç‰¹å¾µ")

# ç§»é™¤IDæ¬„ä½�
features_to_drop = ['id']
if target_col in features_to_drop:
    features_to_drop.remove(target_col)

X = train_processed.drop(columns=features_to_drop + [target_col])
y = train_processed[target_col]
X_test = test_processed.drop(columns=features_to_drop)

print(f"\nè™•ç�†å¾Œç‰¹å¾µæ•¸é‡�: {X.shape[1]}")
print(f"ç‰¹å¾µå��ç¨±: {list(X.columns)}")

# åˆ†å‰²è¨“ç·´å’Œé©—è­‰è³‡æ–™
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nè³‡æ–™åˆ†å‰²çµ�æ�œ:")
print(f"è¨“ç·´é›†: {X_train.shape}")
print(f"é©—è­‰é›†: {X_val.shape}")
print(f"æ¸¬è©¦é›†: {X_test.shape}")

print("=" * 50)


# ==============================================
# Kaggle å�¡è·¯é‡Œæ¶ˆè€—é �æ¸¬ç«¶è³½ - ç¬¬6æ®µï¼šæ¨¡å�‹è¨“ç·´èˆ‡è©•ä¼°
# ==============================================

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb

# å®‰è£�XGBoostå’ŒLightGBMï¼ˆå¦‚æ�œå°šæœªå®‰è£�ï¼‰
try:
    import xgboost as xgb
    import lightgbm as lgb
    print("âœ“ XGBoost å’Œ LightGBM å·²è¼‰å…¥")
except ImportError:
    print("å®‰è£� XGBoost å’Œ LightGBM...")
    !pip install xgboost lightgbm
    import xgboost as xgb
    import lightgbm as lgb

print("=== æ¨¡å�‹è¨“ç·´èˆ‡è©•ä¼° ===")

# å®šç¾©è©•ä¼°å‡½æ•¸
def evaluate_model(model, X_train, y_train, X_val, y_val, model_name):
    """è©•ä¼°æ¨¡å�‹æ€§èƒ½"""
    # è¨“ç·´æ¨¡å�‹
    model.fit(X_train, y_train)
    
    # é �æ¸¬
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    
    # è¨ˆç®—è©•ä¼°æŒ‡æ¨™
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    train_mae = mean_absolute_error(y_train, train_pred)
    val_mae = mean_absolute_error(y_val, val_pred)
    train_r2 = r2_score(y_train, train_pred)
    val_r2 = r2_score(y_val, val_pred)
    
    print(f"\n{model_name} çµ�æ�œ:")
    print(f"  è¨“ç·´ RMSE: {train_rmse:.4f}")
    print(f"  é©—è­‰ RMSE: {val_rmse:.4f}")
    print(f"  è¨“ç·´ MAE:  {train_mae:.4f}")
    print(f"  é©—è­‰ MAE:  {val_mae:.4f}")
    print(f"  è¨“ç·´ RÂ²:   {train_r2:.4f}")
    print(f"  é©—è­‰ RÂ²:   {val_r2:.4f}")
    
    return model, val_rmse

# åˆ�å§‹åŒ–æ¨¡å�‹
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0, random_state=42),
    'Lasso Regression': Lasso(alpha=1.0, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42),
    'LightGBM': lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
}

# è¨“ç·´å’Œè©•ä¼°æ‰€æœ‰æ¨¡å�‹
results = {}
trained_models = {}

for model_name, model in models.items():
    try:
        trained_model, val_rmse = evaluate_model(model, X_train, y_train, X_val, y_val, model_name)
        results[model_name] = val_rmse
        trained_models[model_name] = trained_model
    except Exception as e:
        print(f"æ¨¡å�‹ {model_name} è¨“ç·´å¤±æ•—: {str(e)}")

# æ‰¾å‡ºæœ€ä½³æ¨¡å�‹
best_model_name = min(results, key=results.get)
best_model = trained_models[best_model_name]
best_rmse = results[best_model_name]

print(f"\n=== æœ€ä½³æ¨¡å�‹: {best_model_name} ===")
print(f"é©—è­‰ RMSE: {best_rmse:.4f}")

print("=" * 50)


# ==============================================
# Kaggle å�¡è·¯é‡Œæ¶ˆè€—é �æ¸¬ç«¶è³½ - ç¬¬7æ®µï¼šæ¨¡å�‹å„ªåŒ–èˆ‡ç‰¹å¾µé‡�è¦�æ€§
# ==============================================

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import VotingRegressor

print("=== æ¨¡å�‹è¶…å�ƒæ•¸å„ªåŒ– ===")

# é‡�å°�æœ€ä½³æ¨¡å�‹é€²è¡Œè¶…å�ƒæ•¸èª¿å„ª
if best_model_name == 'XGBoost':
    # XGBoost è¶…å�ƒæ•¸èª¿å„ª
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0]
    }
    
    grid_search = GridSearchCV(
        xgb.XGBRegressor(random_state=42),
        param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    
elif best_model_name == 'LightGBM':
    # LightGBM è¶…å�ƒæ•¸èª¿å„ª
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.1, 0.2],
        'num_leaves': [31, 50, 100]
    }
    
    grid_search = GridSearchCV(
        lgb.LGBMRegressor(random_state=42, verbose=-1),
        param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    
elif best_model_name == 'Random Forest':
    # Random Forest è¶…å�ƒæ•¸èª¿å„ª
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    grid_search = GridSearchCV(
        RandomForestRegressor(random_state=42),
        param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )

else:
    grid_search = None
    print(f"è·³é�� {best_model_name} çš„è¶…å�ƒæ•¸èª¿å„ª")

# åŸ·è¡Œè¶…å�ƒæ•¸èª¿å„ª
if grid_search is not None:
    print(f"æ­£åœ¨é€²è¡Œ {best_model_name} è¶…å�ƒæ•¸èª¿å„ª...")
    grid_search.fit(X_train, y_train)
    
    optimized_model = grid_search.best_estimator_
    print(f"æœ€ä½³å�ƒæ•¸: {grid_search.best_params_}")
    
    # è©•ä¼°å„ªåŒ–å¾Œçš„æ¨¡å�‹
    optimized_val_pred = optimized_model.predict(X_val)
    optimized_val_rmse = np.sqrt(mean_squared_error(y_val, optimized_val_pred))
    
    print(f"å„ªåŒ–å‰� RMSE: {best_rmse:.4f}")
    print(f"å„ªåŒ–å¾Œ RMSE: {optimized_val_rmse:.4f}")
    
    if optimized_val_rmse < best_rmse:
        best_model = optimized_model
        best_rmse = optimized_val_rmse
        print("âœ“ æ¨¡å�‹æ€§èƒ½å·²æ”¹å–„")
    else:
        print("å„ªåŒ–å¾Œæ€§èƒ½æœªæ”¹å–„ï¼Œä¿�æŒ�å�Ÿæ¨¡å�‹")

# ç‰¹å¾µé‡�è¦�æ€§åˆ†æ��
print("\n=== ç‰¹å¾µé‡�è¦�æ€§åˆ†æ�� ===")

if hasattr(best_model, 'feature_importances_'):
    # ç�²å�–ç‰¹å¾µé‡�è¦�æ€§
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("å‰�10å€‹é‡�è¦�ç‰¹å¾µ:")
    print(feature_importance.head(10))
    
    # ç¹ªè£½ç‰¹å¾µé‡�è¦�æ€§åœ–
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(15)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance')
    plt.title('Top 15 Feature Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

else:
    print("æ­¤æ¨¡å�‹ä¸�æ”¯æ�´ç‰¹å¾µé‡�è¦�æ€§åˆ†æ��")

print("=" * 50)


# ==============================================
# Kaggle å�¡è·¯é‡Œæ¶ˆè€—é �æ¸¬ç«¶è³½ - ç¬¬8æ®µï¼šé �æ¸¬èˆ‡æ��äº¤
# ==============================================

print("=== æœ€çµ‚é �æ¸¬èˆ‡æ��äº¤æª”æ¡ˆç”Ÿæˆ� ===")

# ä½¿ç”¨æœ€ä½³æ¨¡å�‹å°�æ¸¬è©¦è³‡æ–™é€²è¡Œé �æ¸¬
final_predictions = best_model.predict(X_test)

print(f"æ¸¬è©¦è³‡æ–™é �æ¸¬å®Œæˆ�")
print(f"é �æ¸¬å€¼ç¯„åœ�: {final_predictions.min():.2f} ~ {final_predictions.max():.2f}")
print(f"é �æ¸¬å€¼å¹³å�‡: {final_predictions.mean():.2f}")
print(f"é �æ¸¬å€¼æ¨™æº–å·®: {final_predictions.std():.2f}")

# æª¢æŸ¥é �æ¸¬å€¼æ˜¯å�¦å�ˆç�†
print(f"\nèˆ‡è¨“ç·´è³‡æ–™ç›®æ¨™è®Šæ•¸æ¯”è¼ƒ:")
print(f"è¨“ç·´è³‡æ–™ç¯„åœ�: {y.min():.2f} ~ {y.max():.2f}")
print(f"è¨“ç·´è³‡æ–™å¹³å�‡: {y.mean():.2f}")

# å‰µå»ºæ��äº¤æª”æ¡ˆ
submission = sample_submission.copy()
submission[target_col] = final_predictions

# é¡¯ç¤ºæ��äº¤æª”æ¡ˆå‰�å¹¾ç­†
print(f"\næ��äº¤æª”æ¡ˆé �è¦½:")
print(submission.head(10))

# å„²å­˜æ��äº¤æª”æ¡ˆ
submission_filename = 'submission.csv'
submission.to_csv(submission_filename, index=False)
print(f"\nâœ“ æ��äº¤æª”æ¡ˆå·²å„²å­˜ç‚º: {submission_filename}")

# å‰µå»ºæ¨¡å�‹ensembleï¼ˆå�¯é�¸ï¼‰
print("\n=== æ¨¡å�‹é›†æˆ�é �æ¸¬ ===")

# é�¸æ“‡è¡¨ç�¾è¼ƒå¥½çš„å‰�3å€‹æ¨¡å�‹é€²è¡Œé›†æˆ�
sorted_results = sorted(results.items(), key=lambda x: x[1])[:3]
ensemble_models = []
ensemble_weights = []

for model_name, rmse in sorted_results:
    ensemble_models.append(trained_models[model_name])
    # ä½¿ç”¨å€’æ•¸ä½œç‚ºæ¬Šé‡�ï¼ˆRMSEè¶Šå°�ï¼Œæ¬Šé‡�è¶Šå¤§ï¼‰
    weight = 1 / rmse
    ensemble_weights.append(weight)

# æ­£è¦�åŒ–æ¬Šé‡�
total_weight = sum(ensemble_weights)
ensemble_weights = [w / total_weight for w in ensemble_weights]

print(f"é›†æˆ�æ¨¡å�‹:")
for i, (model_name, rmse) in enumerate(sorted_results):
    print(f"  {model_name}: RMSE={rmse:.4f}, æ¬Šé‡�={ensemble_weights[i]:.3f}")

# é€²è¡Œé›†æˆ�é �æ¸¬
ensemble_predictions = np.zeros(len(X_test))
for model, weight in zip(ensemble_models, ensemble_weights):
    pred = model.predict(X_test)
    ensemble_predictions += weight * pred

# å‰µå»ºé›†æˆ�æ¨¡å�‹æ��äº¤æª”æ¡ˆ
ensemble_submission = sample_submission.copy()
ensemble_submission[target_col] = ensemble_predictions

print(f"\né›†æˆ�é �æ¸¬å€¼ç¯„åœ�: {ensemble_predictions.min():.2f} ~ {ensemble_predictions.max():.2f}")
print(f"é›†æˆ�é �æ¸¬å€¼å¹³å�‡: {ensemble_predictions.mean():.2f}")

# å„²å­˜é›†æˆ�æ��äº¤æª”æ¡ˆ
ensemble_filename = 'ensemble_submission.csv'
ensemble_submission.to_csv(ensemble_filename, index=False)
print(f"âœ“ é›†æˆ�æ��äº¤æª”æ¡ˆå·²å„²å­˜ç‚º: {ensemble_filename}")

# é©—è­‰é›†æˆ�æ¨¡å�‹æ•ˆæ�œ
ensemble_val_pred = np.zeros(len(X_val))
for model, weight in zip(ensemble_models, ensemble_weights):
    pred = model.predict(X_val)
    ensemble_val_pred += weight * pred

ensemble_val_rmse = np.sqrt(mean_squared_error(y_val, ensemble_val_pred))
print(f"\né›†æˆ�æ¨¡å�‹é©—è­‰ RMSE: {ensemble_val_rmse:.4f}")
print(f"æœ€ä½³å–®æ¨¡å�‹é©—è­‰ RMSE: {best_rmse:.4f}")

if ensemble_val_rmse < best_rmse:
    print("âœ“ é›†æˆ�æ¨¡å�‹è¡¨ç�¾æ›´ä½³ï¼Œå»ºè­°ä½¿ç”¨é›†æˆ�é �æ¸¬")
    final_submission_file = ensemble_filename
else:
    print("å–®æ¨¡å�‹è¡¨ç�¾æ›´ä½³ï¼Œå»ºè­°ä½¿ç”¨å–®æ¨¡å�‹é �æ¸¬")
    final_submission_file = submission_filename

print(f"\n=== æœ€çµ‚å»ºè­°æ��äº¤æª”æ¡ˆ: {final_submission_file} ===")
print("=" * 50)


# ==============================================
# Kaggle å�¡è·¯é‡Œæ¶ˆè€—é �æ¸¬ç«¶è³½ - ç¬¬9æ®µï¼šçµ�æ�œè¦–è¦ºåŒ–èˆ‡æ¨¡å�‹è¨ºæ–·
# ==============================================

print("=== æ¨¡å�‹çµ�æ�œè¦–è¦ºåŒ–èˆ‡è¨ºæ–· ===")

# 1. é �æ¸¬å€¼ vs å¯¦éš›å€¼æ•£é»�åœ–
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# è¨“ç·´é›†é �æ¸¬ vs å¯¦éš›
train_pred = best_model.predict(X_train)
axes[0,0].scatter(y_train, train_pred, alpha=0.5)
axes[0,0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
axes[0,0].set_xlabel('Actual Values (Training)')
axes[0,0].set_ylabel('Predicted Values')
axes[0,0].set_title('Training Set: Actual vs Predicted')
axes[0,0].text(0.05, 0.95, f'RÂ² = {r2_score(y_train, train_pred):.3f}', 
               transform=axes[0,0].transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# é©—è­‰é›†é �æ¸¬ vs å¯¦éš›
val_pred = best_model.predict(X_val)
axes[0,1].scatter(y_val, val_pred, alpha=0.5)
axes[0,1].plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
axes[0,1].set_xlabel('Actual Values (Validation)')
axes[0,1].set_ylabel('Predicted Values')
axes[0,1].set_title('Validation Set: Actual vs Predicted')
axes[0,1].text(0.05, 0.95, f'RÂ² = {r2_score(y_val, val_pred):.3f}', 
               transform=axes[0,1].transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# 2. æ®˜å·®åˆ†æ��
train_residuals = y_train - train_pred
val_residuals = y_val - val_pred

# è¨“ç·´é›†æ®˜å·®åœ–
axes[1,0].scatter(train_pred, train_residuals, alpha=0.5)
axes[1,0].axhline(y=0, color='r', linestyle='--')
axes[1,0].set_xlabel('Predicted Values (Training)')
axes[1,0].set_ylabel('Residuals')
axes[1,0].set_title('Training Residuals')

# é©—è­‰é›†æ®˜å·®åœ–
axes[1,1].scatter(val_pred, val_residuals, alpha=0.5)
axes[1,1].axhline(y=0, color='r', linestyle='--')
axes[1,1].set_xlabel('Predicted Values (Validation)')
axes[1,1].set_ylabel('Residuals')
axes[1,1].set_title('Validation Residuals')

plt.tight_layout()
plt.show()

# 3. æ®˜å·®åˆ†å¸ƒåˆ†æ��
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# æ®˜å·®ç›´æ–¹åœ–
axes[0].hist(val_residuals, bins=30, alpha=0.7, edgecolor='black')
axes[0].set_xlabel('Residuals')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Distribution of Validation Residuals')
axes[0].axvline(x=0, color='r', linestyle='--')

# Q-Q åœ–æª¢æŸ¥æ®˜å·®å¸¸æ…‹æ€§
from scipy import stats
stats.probplot(val_residuals, dist="norm", plot=axes[1])
axes[1].set_title('Q-Q Plot of Validation Residuals')

plt.tight_layout()
plt.show()

# 4. æ¨¡å�‹æ€§èƒ½æ¯”è¼ƒåœ–
model_names = list(results.keys())
model_rmse = list(results.values())

plt.figure(figsize=(12, 6))
bars = plt.bar(model_names, model_rmse)
plt.xlabel('Models')
plt.ylabel('Validation RMSE')
plt.title('Model Performance Comparison')
plt.xticks(rotation=45)

# æ¨™è¨˜æœ€ä½³æ¨¡å�‹
best_idx = model_rmse.index(min(model_rmse))
bars[best_idx].set_color('red')

# åœ¨æŸ±ç‹€åœ–ä¸Šé¡¯ç¤ºæ•¸å€¼
for i, v in enumerate(model_rmse):
    plt.text(i, v + max(model_rmse) * 0.01, f'{v:.4f}', 
             ha='center', va='bottom')

plt.tight_layout()
plt.show()

print(f"\n=== æ¨¡å�‹è¨ºæ–·çµ±è¨ˆ ===")
print(f"é©—è­‰é›†æ®˜å·®çµ±è¨ˆ:")
print(f"  å¹³å�‡å€¼: {val_residuals.mean():.4f}")
print(f"  æ¨™æº–å·®: {val_residuals.std():.4f}")
print(f"  å��åº¦: {val_residuals.skew():.4f}")
print(f"  å³°åº¦: {val_residuals.kurtosis():.4f}")

# è¨ˆç®—é �æ¸¬æº–ç¢ºåº¦ï¼ˆåœ¨å®¹å¿�ç¯„åœ�å…§çš„é �æ¸¬ç™¾åˆ†æ¯”ï¼‰
tolerance_5pct = np.abs(val_residuals) / y_val <= 0.05
tolerance_10pct = np.abs(val_residuals) / y_val <= 0.10
tolerance_20pct = np.abs(val_residuals) / y_val <= 0.20

print(f"\né �æ¸¬æº–ç¢ºåº¦åˆ†æ��:")
print(f"  5%å®¹å¿�ç¯„åœ�å…§é �æ¸¬æ­£ç¢ºç�‡: {tolerance_5pct.mean():.2%}")
print(f"  10%å®¹å¿�ç¯„åœ�å…§é �æ¸¬æ­£ç¢ºç�‡: {tolerance_10pct.mean():.2%}")
print(f"  20%å®¹å¿�ç¯„åœ�å…§é �æ¸¬æ­£ç¢ºç�‡: {tolerance_20pct.mean():.2%}")

print("=" * 50)


# ==============================================
# Kaggle å�¡è·¯é‡Œæ¶ˆè€—é �æ¸¬ç«¶è³½ - ç¬¬11æ®µï¼šæœ€çµ‚ç¸½çµ�èˆ‡å»ºè­°
# ==============================================

print("=== ç«¶è³½è§£æ±ºæ–¹æ¡ˆç¸½çµ� ===")

# æœ€çµ‚çµ�æ�œå½™ç¸½
print(f"ğŸ“Š è³‡æ–™é›†è³‡è¨Š:")
print(f"   â€¢ è¨“ç·´è³‡æ–™: {train_df.shape}")
print(f"   â€¢ æ¸¬è©¦è³‡æ–™: {test_df.shape}")
print(f"   â€¢ ç‰¹å¾µæ•¸é‡�: {X.shape[1]}")
print(f"   â€¢ ç›®æ¨™è®Šæ•¸: {target_col}")

print(f"\nğŸ�† æ¨¡å�‹æ€§èƒ½æ�’å��:")
sorted_results = sorted(results.items(), key=lambda x: x[1])
for i, (model_name, rmse) in enumerate(sorted_results):
    print(f"   {i+1}. {model_name}: RMSE = {rmse:.4f}")

print(f"\nğŸ’¡ æœ€ä½³ç­–ç•¥:")
if 'final_submission_file' in locals():
    print(f"   â€¢ å»ºè­°æ��äº¤æª”æ¡ˆ: {final_submission_file}")
else:
    print(f"   â€¢ å»ºè­°æ��äº¤æª”æ¡ˆ: {submission_filename}")

print(f"   â€¢ æœ€ä½³é©—è­‰ RMSE: {best_rmse:.4f}")

# ç‰¹å¾µé‡�è¦�æ€§ç¸½çµ�ï¼ˆå¦‚æ�œå�¯ç”¨ï¼‰
if 'feature_importance' in locals():
    print(f"\nğŸ”� é—œé�µç‰¹å¾µï¼ˆå‰�5å��ï¼‰:")
    for i, row in feature_importance.head(5).iterrows():
        print(f"   â€¢ {row['feature']}: {row['importance']:.4f}")

# æ¨¡å�‹æ”¹é€²å»ºè­°
print(f"\nğŸš€ é€²ä¸€æ­¥æ”¹é€²å»ºè­°:")
print(f"   1. ç‰¹å¾µå·¥ç¨‹:")
print(f"      â€¢ å‰µå»ºå¤šé …å¼�ç‰¹å¾µ")
print(f"      â€¢ ç‰¹å¾µäº¤äº’ä½œç”¨")
print(f"      â€¢ ç‰¹å¾µé�¸æ“‡/é™�ç¶­")
print(f"   2. æ¨¡å�‹èª¿å„ª:")
print(f"      â€¢ æ›´è©³ç´°çš„è¶…å�ƒæ•¸æ�œç´¢")
print(f"      â€¢ äº¤å�‰é©—è­‰ç­–ç•¥å„ªåŒ–")
print(f"      â€¢ æ›´å¤šæ¨¡å�‹é›†æˆ�")
print(f"   3. æ•¸æ“šå“�è³ª:")
print(f"      â€¢ ç•°å¸¸å€¼æª¢æ¸¬èˆ‡è™•ç�†")
print(f"      â€¢ ç‰¹å¾µæ¨™æº–åŒ–/æ­£è¦�åŒ–")
print(f"      â€¢ è³‡æ–™å¢�å¼·æŠ€è¡“")

# ç«¶è³½ç­–ç•¥å»ºè­°
print(f"\nğŸ“ˆ Kaggle ç«¶è³½ç­–ç•¥:")
print(f"   â€¢ ç›£æ�§ Public Leaderboard åˆ†æ•¸")
print(f"   â€¢ ä¿�ç•™é©—è­‰ç­–ç•¥çš„ä¸€è‡´æ€§")
print(f"   â€¢ å˜—è©¦å¤šç¨®æ¨¡å�‹çµ„å�ˆ")
print(f"   â€¢ æ³¨æ„�é��æ“¬å�ˆé¢¨éšª")
print(f"   â€¢ æº–å‚™å¤šå€‹æ��äº¤ç‰ˆæœ¬")

# ç¨‹å¼�ç¢¼ä½¿ç”¨èªªæ˜�
print(f"\nğŸ“� ç¨‹å¼�ç¢¼åŸ·è¡Œèªªæ˜�:")
print(f"   1. ç¢ºä¿�å·²ä¸‹è¼‰ç«¶è³½è³‡æ–™é›†åˆ° Colab")
print(f"   2. æŒ‰é †åº�åŸ·è¡Œå�„æ®µç¨‹å¼�ç¢¼")
print(f"   3. æ ¹æ“šè³‡æ–™ç‰¹æ€§èª¿æ•´ç‰¹å¾µå·¥ç¨‹éƒ¨åˆ†")
print(f"   4. è§€å¯Ÿæ¨¡å�‹æ€§èƒ½ä¸¦é�¸æ“‡æœ€ä½³ç­–ç•¥")
print(f"   5. ä¸‹è¼‰ä¸¦æ��äº¤æœ€çµ‚é �æ¸¬æª”æ¡ˆ")

# å»ºç«‹å®Œæ•´åŸ·è¡Œæ¸…å–®
print(f"\nâœ… æª¢æŸ¥æ¸…å–®:")
tasks = [
    "è³‡æ–™è¼‰å…¥èˆ‡åŸºæœ¬æª¢æŸ¥",
    "æ�¢ç´¢æ€§è³‡æ–™åˆ†æ��",
    "è³‡æ–™è¦–è¦ºåŒ–",
    "ç‰¹å¾µå·¥ç¨‹èˆ‡å‰�è™•ç�†",
    "æ¨¡å�‹è¨“ç·´èˆ‡æ¯”è¼ƒ",
    "è¶…å�ƒæ•¸èª¿å„ª",
    "æ¨¡å�‹è¨ºæ–·èˆ‡è¦–è¦ºåŒ–",
    "æœ€çµ‚é �æ¸¬èˆ‡æ��äº¤æª”æ¡ˆç”Ÿæˆ�"
]

for i, task in enumerate(tasks, 1):
    print(f"   {i}. {task} âœ“")

print(f"\nğŸ�¯ é �æœŸæˆ�æ�œ:")
print(f"   â€¢ å®Œæ•´çš„è³‡æ–™ç§‘å­¸å·¥ä½œæµ�ç¨‹")
print(f"   â€¢ å¤šç¨®æ©Ÿå™¨å­¸ç¿’æ¨¡å�‹æ¯”è¼ƒ")
print(f"   â€¢ è©³ç´°çš„æ¨¡å�‹æ€§èƒ½åˆ†æ��")
print(f"   â€¢ å�¯ç›´æ�¥æ��äº¤çš„é �æ¸¬æª”æ¡ˆ")
print(f"   â€¢ è¦–è¦ºåŒ–çµ�æ�œèˆ‡æ´�å¯Ÿ")

print(f"\n" + "=" * 50)
print(f"ğŸ�� ç«¶è³½è§£æ±ºæ–¹æ¡ˆå®Œæˆ�ï¼�")
print(f"ğŸ’ª è¨˜å¾—æ ¹æ“šå¯¦éš›è³‡æ–™èª¿æ•´å�ƒæ•¸ï¼Œä¸¦å˜—è©¦ä¸�å�Œçš„ç‰¹å¾µå·¥ç¨‹æ–¹æ³•")
print(f"ğŸ¤� Good luck with your competition!")
print(f"=" * 50)

