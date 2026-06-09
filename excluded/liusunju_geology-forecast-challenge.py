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


# é€™å€‹ Python 3 ç’°å¢ƒé �è£�äº†è¨±å¤šå¯¦ç”¨çš„åˆ†æ��å‡½å¼�åº«
# å®ƒæ˜¯ä»¥ kaggle/python Docker æ˜ åƒ�æª”ç‚ºåŸºç¤�æ‰€å®šç¾©
import os

# è¦–è¦ºåŒ–å’Œæ©Ÿå™¨å­¸ç¿’å¥—ä»¶
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
import warnings
warnings.filterwarnings('ignore')

# è¨­å®šåœ–å½¢å�ƒæ•¸ï¼ˆé�¿å…�ä¸­æ–‡é¡¯ç¤ºå•�é¡Œï¼‰
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['figure.figsize'] = (12, 6)
sns.set_style("whitegrid")

print("âœ… å¥—ä»¶åŒ¯å…¥å®Œæˆ�ï¼�")


print("\n=== è¼‰å…¥è³‡æ–™ ===")
try:
    train_data = pd.read_csv('/kaggle/input/geology-forecast-challenge-open/data/train.csv', index_col="geology_id")
    test_data = pd.read_csv('/kaggle/input/geology-forecast-challenge-open/data/test.csv', index_col="geology_id")
    sample_sub = pd.read_csv('/kaggle/input/geology-forecast-challenge-open/data/sample_submission.csv', index_col="geology_id")
    print("âœ… çœŸå¯¦è³‡æ–™è¼‰å…¥æˆ�åŠŸï¼�")
    real_data = True
    
except Exception as e:
    print(f"âš ï¸� è³‡æ–™è¼‰å…¥å¤±æ•—: {e}")
    print("ä½¿ç”¨æ¨¡æ“¬è³‡æ–™é€²è¡Œæ¼”ç¤º...")
    real_data = False
    
    # å‰µå»ºæ¨¡æ“¬è³‡æ–™
    np.random.seed(42)
    n_train = 1000
    n_test = 200
    
    # å‰µå»º-299åˆ°299çš„ä½�ç½®åˆ—ï¼ˆ600å€‹ç‰¹å¾µï¼‰
    position_cols = list(range(-299, 301))
    
    # ç”Ÿæˆ�è¨“ç·´è³‡æ–™
    train_ids = [f"train_{i:06d}" for i in range(n_train)]
    train_data = pd.DataFrame(index=train_ids)
    
    for pos in position_cols:
        depth_values = 100 + pos * 0.05 + np.random.normal(0, 8, n_train)
        train_data[pos] = depth_values
    
    # ç”Ÿæˆ�æ¸¬è©¦è³‡æ–™
    test_ids = [f"test_{i:06d}" for i in range(n_test)]
    test_data = pd.DataFrame(index=test_ids)
    
    for pos in position_cols:
        depth_values = 100 + pos * 0.05 + np.random.normal(0, 8, n_test)
        test_data[pos] = depth_values
    
    # å‰µå»ºæ¨£æœ¬æ��äº¤æª”æ¡ˆæ ¼å¼�
    sample_sub = pd.DataFrame(index=test_ids)
    target_positions = list(range(301, 601))  # é �æ¸¬ä½�ç½®301åˆ°600
    for pos in target_positions:
        sample_sub[pos] = 0

# é¡¯ç¤ºè³‡æ–™åŸºæœ¬è³‡è¨Š
print(f"\n=== è³‡æ–™æ‘˜è¦� ===")
print(f"è¨“ç·´è³‡æ–™å½¢ç‹€: {train_data.shape}")
print(f"æ¸¬è©¦è³‡æ–™å½¢ç‹€: {test_data.shape}")
print(f"æ¨£æœ¬æ��äº¤å½¢ç‹€: {sample_sub.shape}")
print(f"è¨“ç·´è³‡æ–™æ¬„ä½�ç¯„åœ�: {train_data.columns[0]} åˆ° {train_data.columns[-1]}")
print(f"ç¼ºå¤±å€¼ - è¨“ç·´: {train_data.isnull().sum().sum()}, æ¸¬è©¦: {test_data.isnull().sum().sum()}")


print("=== è³‡æ–™å‰�è™•ç�† ===")

# è¨ˆç®—æ•´é«”å¹³å�‡å€¼ï¼ˆç”¨æ–¼å¡«è£œç¼ºå¤±å€¼ï¼‰
overall_average = train_data.mean().mean()
print(f"æ•´é«”å¹³å�‡å€¼: {overall_average:.2f}")

# åˆ†å‰²ç‰¹å¾µå’Œç›®æ¨™è®Šæ•¸
# å‰�300å€‹ä½�ç½®ä½œç‚ºç‰¹å¾µï¼Œå¾Œ300å€‹ä½�ç½®ä½œç‚ºç›®æ¨™
feature_cols = train_data.columns[:300]
target_cols = train_data.columns[300:]

print(f"ç‰¹å¾µæ¬„ä½�ç¯„åœ�: {feature_cols[0]} åˆ° {feature_cols[-1]}")
print(f"ç›®æ¨™æ¬„ä½�ç¯„åœ�: {target_cols[0]} åˆ° {target_cols[-1]}")

# è™•ç�†è¨“ç·´è³‡æ–™
X_train = train_data[feature_cols].fillna(overall_average)
y_train = train_data[target_cols].fillna(overall_average)

# è™•ç�†æ¸¬è©¦è³‡æ–™
if test_data.shape[1] == train_data.shape[1]:
    X_test = test_data[feature_cols].fillna(overall_average)
else:
    X_test = test_data.fillna(overall_average)

print(f"\n=== è™•ç�†å¾Œçš„è³‡æ–™å½¢ç‹€ ===")
print(f"X_train: {X_train.shape}")
print(f"y_train: {y_train.shape}")
print(f"X_test: {X_test.shape}")

# æª¢æŸ¥ç¼ºå¤±å€¼è™•ç�†çµ�æ�œ
print(f"\nç¼ºå¤±å€¼æª¢æŸ¥ - X_train: {X_train.isnull().sum().sum()}, y_train: {y_train.isnull().sum().sum()}")

# ç‰¹å¾µæ¨™æº–åŒ–
print("\n=== ç‰¹å¾µæ¨™æº–åŒ– ===")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"æ¨™æº–åŒ–å®Œæˆ�ï¼�X_train_scaled å½¢ç‹€: {X_train_scaled.shape}")
print(f"æ¨™æº–åŒ–å¾Œç¯„åœ�: {X_train_scaled.min():.3f} åˆ° {X_train_scaled.max():.3f}")


print("=== æ¨¡å�‹æ¯”è¼ƒæ¸¬è©¦ ===")
from sklearn.model_selection import cross_val_score
from sklearn.multioutput import MultiOutputRegressor
import time

# æº–å‚™æ¯”è¼ƒçš„æ¨¡å�‹
models = {
    'KNN_3': KNeighborsRegressor(p=1, n_neighbors=3),
    'KNN_5': KNeighborsRegressor(p=1, n_neighbors=5),
    'KNN_weighted': KNeighborsRegressor(p=1, n_neighbors=5, weights='distance'),
    'RandomForest_50': RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
    'RandomForest_100': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'Ridge_1.0': Ridge(alpha=1.0),
    'Ridge_10.0': Ridge(alpha=10.0),
}

# å„²å­˜çµ�æ�œ
results = {}
print(f"æº–å‚™æ¸¬è©¦ {len(models)} å€‹æ¨¡å�‹...")


# ä½¿ç”¨è¼ƒå°�çš„æ¨£æœ¬é€²è¡Œå¿«é€Ÿæ¯”è¼ƒï¼ˆé�¿å…�è¨˜æ†¶é«”å•�é¡Œï¼‰
sample_size = min(500, X_train_scaled.shape[0])
sample_indices = np.random.choice(X_train_scaled.shape[0], sample_size, replace=False)
X_sample = X_train_scaled[sample_indices]
y_sample = y_train.iloc[sample_indices]

print(f"ä½¿ç”¨ {sample_size} å€‹æ¨£æœ¬é€²è¡Œæ¨¡å�‹æ¯”è¼ƒ")
print("=" * 60)

for name, model in models.items():
    print(f"\næ­£åœ¨æ¸¬è©¦: {name}")
    
    try:
        # è¨˜éŒ„è¨“ç·´æ™‚é–“
        start_time = time.time()
        model.fit(X_sample, y_sample)
        fit_time = time.time() - start_time
        
        # é �æ¸¬æ™‚é–“
        start_time = time.time()
        y_pred = model.predict(X_sample)
        pred_time = time.time() - start_time
        
        # è¨ˆç®—æŒ‡æ¨™
        mse = mean_squared_error(y_sample, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_sample, y_pred)
        
        # å„²å­˜çµ�æ�œ
        results[name] = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'Fit_Time': fit_time,
            'Pred_Time': pred_time,
            'Total_Time': fit_time + pred_time
        }
        
        print(f"  âœ… RMSE: {rmse:.4f} | MAE: {mae:.4f} | æ™‚é–“: {fit_time:.2f}s")
        
    except Exception as e:
        print(f"  â�Œ éŒ¯èª¤: {e}")
        results[name] = None

print("\n" + "=" * 60)
print("æ‰€æœ‰æ¨¡å�‹æ¸¬è©¦å®Œæˆ�ï¼�")


print("=== æ¨¡å�‹æ¯”è¼ƒçµ�æ�œ ===")

# é��æ¿¾æˆ�åŠŸçš„çµ�æ�œ
valid_results = {k: v for k, v in results.items() if v is not None}

if not valid_results:
    print("æ²’æœ‰æˆ�åŠŸçš„æ¨¡å�‹çµ�æ�œ")
else:
    # è½‰æ�›ç‚º DataFrame ä¾¿æ–¼åˆ†æ��
    results_df = pd.DataFrame(valid_results).T
    results_df = results_df.sort_values('RMSE')
    
    print("\nğŸ“Š æŒ‰ RMSE æ�’å�� (è¶Šå°�è¶Šå¥½):")
    print("-" * 80)
    print(f"{'æ¨¡å�‹':<20} {'RMSE':<10} {'MAE':<10} {'è¨“ç·´æ™‚é–“':<10} {'é �æ¸¬æ™‚é–“':<10}")
    print("-" * 80)
    
    for idx, (model_name, row) in enumerate(results_df.iterrows(), 1):
        print(f"{idx}. {model_name:<15} {row['RMSE']:<10.4f} {row['MAE']:<10.4f} "
              f"{row['Fit_Time']:<10.2f} {row['Pred_Time']:<10.2f}")
    
    # æ‰¾å‡ºæœ€ä½³æ¨¡å�‹
    best_model_name = results_df.index[0]
    best_model = models[best_model_name]
    
    print(f"\nğŸ�† æœ€ä½³æ¨¡å�‹: {best_model_name}")
    print(f"   RMSE: {results_df.loc[best_model_name, 'RMSE']:.4f}")
    print(f"   MAE: {results_df.loc[best_model_name, 'MAE']:.4f}")
    print(f"   ç¸½æ™‚é–“: {results_df.loc[best_model_name, 'Total_Time']:.2f}ç§’")
    
    # æ€§èƒ½ vs é€Ÿåº¦åˆ†æ��
    print(f"\nâš¡ æœ€å¿«æ¨¡å�‹: {results_df.sort_values('Total_Time').index[0]}")
    print(f"   æ™‚é–“: {results_df.sort_values('Total_Time').iloc[0]['Total_Time']:.2f}ç§’")
    
    # è¦–è¦ºåŒ–æ¯”è¼ƒ (å�¯é�¸)
    try:
        plt.figure(figsize=(12, 8))
        
        # RMSE vs æ™‚é–“æ•£é»�åœ–
        plt.subplot(2, 2, 1)
        plt.scatter(results_df['Total_Time'], results_df['RMSE'])
        for i, model in enumerate(results_df.index):
            plt.annotate(model, (results_df.iloc[i]['Total_Time'], results_df.iloc[i]['RMSE']))
        plt.xlabel('Total time (seconds)')
        plt.ylabel('RMSE')
        plt.title('Model performance vs speed')
        
        # RMSE æ�’å��æŸ±ç‹€åœ–
        plt.subplot(2, 2, 2)
        results_df['RMSE'].plot(kind='bar')
        plt.title('RMSE compare')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
        
    except:
        print("åœ–è¡¨é¡¯ç¤ºç•¥é��...")


print("=== ä½¿ç”¨æœ€ä½³æ¨¡å�‹é€²è¡Œå®Œæ•´è¨“ç·´ ===")

if 'best_model_name' in locals() and 'best_model' in locals():
    print(f"é�¸æ“‡çš„æœ€ä½³æ¨¡å�‹: {best_model_name}")
    
    # ä½¿ç”¨å®Œæ•´è³‡æ–™é›†è¨“ç·´æœ€ä½³æ¨¡å�‹
    print("é–‹å§‹ä½¿ç”¨å®Œæ•´è³‡æ–™é›†è¨“ç·´...")
    start_time = time.time()
    
    final_model = models[best_model_name]
    final_model.fit(X_train_scaled, y_train)
    
    total_training_time = time.time() - start_time
    print(f"âœ… å®Œæ•´è¨“ç·´å®Œæˆ�ï¼�æ™‚é–“: {total_training_time:.2f}ç§’")
    
    # ç”Ÿæˆ�æœ€çµ‚é �æ¸¬
    print("ç”Ÿæˆ�æœ€çµ‚é �æ¸¬...")
    final_predictions = final_model.predict(X_test_scaled)
    
    # å‰µå»ºæ��äº¤æª”æ¡ˆ
    final_output = pd.DataFrame(
        final_predictions,
        columns=sample_sub.columns,
        index=sample_sub.index
    )
    
    # æª¢æŸ¥ä¸¦è™•ç�†ç•°å¸¸å€¼
    nan_count = final_output.isnull().sum().sum()
    inf_count = np.isinf(final_output).sum().sum()
    
    if nan_count > 0 or inf_count > 0:
        print(f"è™•ç�†ç•°å¸¸å€¼: NaN={nan_count}, Inf={inf_count}")
        final_output = final_output.fillna(overall_average)
        final_output = final_output.replace([np.inf, -np.inf], overall_average)
    
    # å„²å­˜æœ€ä½³æ¨¡å�‹çš„çµ�æ�œ
    final_output.to_csv("submission.csv")
    
    print(f"\nğŸ�‰ æœ€ä½³æ¨¡å�‹é �æ¸¬å®Œæˆ�ï¼�")
    print(f"æ¨¡å�‹: {best_model_name}")
    print(f"é �æ¸¬ç¯„åœ�: {final_output.min().min():.3f} åˆ° {final_output.max().max():.3f}")
    print(f"é �æ¸¬å¹³å�‡: {final_output.mean().mean():.3f}")
    print(f"æª”æ¡ˆå·²å„²å­˜: submission.csv")
    
else:
    print("â�Œ æ²’æœ‰å�¯ç”¨çš„æœ€ä½³æ¨¡å�‹ï¼Œè«‹æª¢æŸ¥å‰�é�¢çš„çµ�æ�œ")

