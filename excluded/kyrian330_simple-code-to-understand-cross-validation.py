# ğŸ“Œ 1. å¯¼å…¥å¿…è¦�çš„åº“
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

print("âœ… ")



# è¯»å�–è®­ç»ƒå’Œæµ‹è¯•æ•°æ�®
train_df = pd.read_csv("/kaggle/input/stat462-862project1/train.csv")
test_df = pd.read_csv("/kaggle/input/stat462-862project1/test.csv")

print("è®­ç»ƒé›†ç»´åº¦ï¼š", train_df.shape)
print("æµ‹è¯•é›†ç»´åº¦ï¼š", test_df.shape)

print("âœ… ")



train_df.info()


train_df


test_df


train_df.columns


train_df['quality'].value_counts()


# â­� 1. è¯»å�–æ•°æ�®
train_df = pd.read_csv("/kaggle/input/stat462-862project1/train.csv")
test_df = pd.read_csv("/kaggle/input/stat462-862project1/test.csv")

print("âœ… ")



# â­� 2. åŸºæœ¬æ•°æ�®åˆ†æ��
print("\nè®­ç»ƒé›†æ•°æ�®è¡Œåˆ—ç»Ÿè®¡:")
print(train_df.shape)
print("\nè¿”å›�å€¼ quality åˆ†å¸ƒ:")
print(train_df['quality'].value_counts().sort_index())



# â­� 3. ç›®æ ‡åˆ†å¸ƒå�¯è§†åŒ–
plt.figure(figsize=(8, 4))
sns.countplot(x="quality", data=train_df)
plt.xlabel("quality")
plt.ylabel("simple value")
plt.show()



# â­� 4. æ•°æ�®åˆ�å§‹åŒ–
X = train_df.drop(columns=["quality"])
Y = train_df["quality"]

print("âœ… ")



models = {
    "Random Forest": RandomForestRegressor(n_estimators=110, max_depth=18, random_state=42),
    "Decision Tree": DecisionTreeRegressor(max_depth=7, random_state=42),
    "Linear Regression": LinearRegression()
}

for name, model in models.items():
    # äº¤å�‰éªŒè¯�è¿”å›�æ¯�æŠ˜çš„è´Ÿ MSE
    neg_mse_scores = cross_val_score(model, X, Y, cv=5, scoring='neg_mean_squared_error')
    
    # MSEï¼ˆå�–è´Ÿå�·ï¼‰
    mse_scores = -neg_mse_scores
    mse_mean = mse_scores.mean()
    mse_std = mse_scores.std()
    
    # RMSEï¼ˆå¯¹ MSE å¼€æ–¹ï¼‰
    rmse_scores = np.sqrt(mse_scores)
    rmse_mean = rmse_scores.mean()
    rmse_std = rmse_scores.std()

    print(f"{name}:")
    print(f"  ğŸ“‰ MSE  å¹³å�‡: {mse_mean:.4f}, æ ‡å‡†å·®: {mse_std:.4f}, æ¯�æŠ˜: {np.round(mse_scores, 3)}")
    print(f"  âœ… RMSE å¹³å�‡: {rmse_mean:.4f}, æ ‡å‡†å·®: {rmse_std:.4f}, æ¯�æŠ˜: {np.round(rmse_scores, 3)}\n")



# ç”¨å…¨é‡�è®­ç»ƒé›†åˆ†åˆ«è®­ç»ƒæ¯�ä¸ªæ¨¡å�‹ï¼Œå¹¶ç”Ÿæˆ�æ��äº¤æ–‡ä»¶

# å‡†å¤‡æµ‹è¯•æ•°æ�®
test_data = test_df.drop(columns=["id"])

print("ğŸš€ ã€�å¼€å§‹å…¨é‡�è®­ç»ƒå¹¶ç”Ÿæˆ�æ��äº¤æ–‡ä»¶ã€‘")
for name, model in models.items():
    model.fit(X, Y)
    preds = model.predict(test_data)
    submission = pd.DataFrame({
        "id": test_df["id"],
        "quality": preds
    })
    filename = f"submission_{name.lower().replace(' ', '_')}.csv"
    submission.to_csv(filename, index=False)
    print(f"âœ… å·²ä¿�å­˜: {filename}")

print("\nğŸ�� æ‰€æœ‰æ¨¡å�‹å·²å®Œæˆ�è®­ç»ƒå’Œæ��äº¤æ–‡ä»¶ç”Ÿæˆ�ã€‚å�¯å‰�å¾€ Kaggle æ��äº¤æŸ¥çœ‹å¾—åˆ†å·®å¼‚ã€‚")


# æ��å�– Random Forest æ¨¡å�‹
rf_model = models["Random Forest"]

# ç”¨å…¨é‡�è®­ç»ƒé›†è®­ç»ƒæ¨¡å�‹
rf_model.fit(X, Y)

# é¢„æµ‹æµ‹è¯•é›†
preds = rf_model.predict(test_data)

# ä¿�ç•™2ä¸ªå°�æ•°ç‚¹
preds = np.round(preds, 2)

print(f"é¢„æµ‹æœ€å¤§å€¼ï¼š{preds.max():.4f}")
print(f"é¢„æµ‹æœ€å°�å€¼ï¼š{preds.min():.4f}")
print(f"é¢„æµ‹å�‡å€¼ï¼š{preds.mean():.4f}")
plt.hist(preds, bins=100)
plt.title("Test Prediction Distribution")
plt.show()



preds = np.clip(preds, 4.2, 7)

print(f"é¢„æµ‹æœ€å¤§å€¼ï¼š{preds.max():.4f}")
print(f"é¢„æµ‹æœ€å°�å€¼ï¼š{preds.min():.4f}")
print(f"é¢„æµ‹å�‡å€¼ï¼š{preds.mean():.4f}")
plt.hist(preds, bins=100)
plt.title("Test Prediction Distribution")
plt.show()


# ç”Ÿæˆ�æ��äº¤æ–‡ä»¶ï¼ˆæ¯”èµ›æ��äº¤è§„èŒƒï¼‰
submission = pd.DataFrame({
    "id": test_df["id"],
    "quality": preds
})

filename = "submission.csv"
submission.to_csv(filename, index=False)
print(f"âœ… å·²ä¿�å­˜æ¯”èµ›å�¯æ��äº¤æ–‡ä»¶: {filename}")


submission.head(10)

