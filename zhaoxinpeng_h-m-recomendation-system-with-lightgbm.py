# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


transections = "/kaggle/input/h-and-m-personalized-fashion-recommendations/transactions_train.csv"
articals = "/kaggle/input/h-and-m-personalized-fashion-recommendations/articles.csv"
sample_submission = "/kaggle/input/h-and-m-personalized-fashion-recommendations/sample_submission.csv"
customers = "/kaggle/input/h-and-m-personalized-fashion-recommendations/customers.csv"

transactions_df = pd.read_csv(transections)
articles_df = pd.read_csv(articals)
customers_df = pd.read_csv(customers)


print(transactions_df.head())
print(articles_df.head())
print(customers_df.head())


# Convert `t_dat` to a datetime type for easy date manipulation
transactions_df['t_dat'] = pd.to_datetime(transactions_df['t_dat'])

# Get a high-level overview of the data
print("Transactions DataFrame Info:")
print(transactions_df.info())

# Count the number of unique customers and articles
num_unique_customers = transactions_df['customer_id'].nunique()
num_unique_articles = transactions_df['article_id'].nunique()

# Find the date range of the transactions
start_date = transactions_df['t_dat'].min()
end_date = transactions_df['t_dat'].max()

# Print the key statistics
print("\nBasic Statistics")
print(f"Number of unique customers: {num_unique_customers}")
print(f"Number of unique articles: {num_unique_articles}")
print(f"Date range of transactions: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")


print("--- Customers Dataframe EDA ---")
print("\nMissing values in customers data:")
print(customers_df.isnull().sum())
print("\nDistribution of club_member_status:")
print(customers_df['club_member_status'].value_counts())
print("\nDistribution of fashion_news_frequency:")
print(customers_df['fashion_news_frequency'].value_counts())


print("\n--- Articles Dataframe EDA ---")
print("\nMissing values in articles data:")
print(articles_df.isnull().sum())
print("\nDistribution of product_group_name:")
print(articles_df['product_group_name'].value_counts())
print("\nDistribution of garment_group_name:")
print(articles_df['garment_group_name'].value_counts())


# Convert `t_dat` to datetime
transactions_df['t_dat'] = pd.to_datetime(transactions_df['t_dat'])

# Filter transactions for the last 3 months
end_date = transactions_df['t_dat'].max()
start_date_filtered = end_date - pd.DateOffset(months=3)
recent_transactions = transactions_df[transactions_df['t_dat'] >= start_date_filtered]

# Merge the dataframes
merged_df = pd.merge(recent_transactions, customers_df, on='customer_id', how='left')
merged_df = pd.merge(merged_df, articles_df, on='article_id', how='left')

# Display the information of the new, merged dataframe
print("Merged DataFrame Info (last 3 months):")
print(merged_df.info())

print("\nMerged DataFrame Head:")
merged_df.head()


# Let's fill null with the median age of the customers
median_age = merged_df['age'].median()
merged_df['age'].fillna(median_age, inplace=True)

# Impute missing categorical values with a placeholder
merged_df['club_member_status'] = merged_df['club_member_status'].fillna('Unknown')
merged_df['fashion_news_frequency'] = merged_df['fashion_news_frequency'].fillna('Unknown')
merged_df['FN'] = merged_df['FN'].fillna(0)
merged_df['Active'] = merged_df['Active'].fillna(0)

print(merged_df[['club_member_status', 'fashion_news_frequency', 'FN', 'Active']].isna().sum())

# Create temporal features
merged_df['week'] = merged_df['t_dat'].dt.isocalendar().week.astype(int)
merged_df['day_of_week'] = merged_df['t_dat'].dt.dayofweek.astype(int)

merged_df[['t_dat', 'age', 'FN', 'Active', 'club_member_status', 'week', 'day_of_week']].head()



# calculate recency, frequency, and monetary value for each customer
customer_features = merged_df.groupby('customer_id').agg(
    total_purchase = ('article_id', 'count'),
    last_purchase_date = ('t_dat', 'max')
)
customer_features["recency_days"] = (merged_df['t_dat'].max() - customer_features['last_purchase_date']).dt.days
customer_features.head()


# calculate popularity and average price for each article
artical_features = merged_df.groupby('article_id').agg(
    purchase_count = ('customer_id', 'count'),
    average_price = ('price', 'mean')
)
artical_features.head()


# To make the process faster, let's work with a small sample of the merged data
sample_merged_df = merged_df.sample(n=50000, random_state=42).reset_index(drop=True)


# Create positive samples with a label of 1
positive_samples = sample_merged_df[['customer_id', 'article_id']].copy()
positive_samples['label'] = 1


# Negative Sampling: Get all unique article IDs
all_article_ids = sample_merged_df['article_id'].unique()


# generate negetive sample list
negative_samples_list = []
for customer in positive_samples['customer_id'].unique():
    customer_purchases = set(positive_samples[positive_samples['customer_id'] == customer]['article_id'])
    
    # Get articles not purchased by the customer
    non_purchased_articles = np.setdiff1d(all_article_ids, list(customer_purchases))
    
    # Randomly sample a few non-purchased items for each purchase
    num_neg_samples = min(len(non_purchased_articles), 4) # Take 4 negative samples for each positive
    if num_neg_samples > 0:
        neg_articles = np.random.choice(non_purchased_articles, num_neg_samples, replace=False)
        for neg_article in neg_articles:
            negative_samples_list.append([customer, neg_article, 0])

negative_samples = pd.DataFrame(negative_samples_list, columns=['customer_id', 'article_id', 'label'])

negative_samples.head()


# Now combine positive and negative samples
final_data = pd.concat([positive_samples, negative_samples], ignore_index=True)

# Merge our aggregated features
final_data = pd.merge(final_data, customer_features, on='customer_id', how='left')
final_data = pd.merge(final_data, artical_features, on='article_id', how='left')

# Show the final dataset structure
print("\nFinal Dataset Shape:")
print(final_data.shape)
print("\nDistribution of Labels:")
print(final_data['label'].value_counts())

print("Final Dataset for Modeling Head:")
final_data.head()









import lightgbm as lgb
from sklearn.model_selection import train_test_split



import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 特征与目标变量
features = ['total_purchase', 'recency_days', 'purchase_count', 'average_price']
target = 'label'

# 处理缺失值
final_data.dropna(subset=features, inplace=True)
X = final_data[features]
y = final_data[target]

# 数据拆分
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

lgb_model = lgb.LGBMClassifier(
    objective='binary',
    boosting_type='gbdt',
    num_leaves=127,
    learning_rate=0.03,
    n_estimators=2000,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=1.0,
    reg_lambda=2.0,
    random_state=42,
    device='gpu',
    gpu_platform_id=0,
    gpu_device_id=0,
)
lgb_model.fit(X_train, y_train)
print("LightGBM (GPU) 完成")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备:", device)

X_train_t = torch.tensor(X_train.values, dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train.values, dtype=torch.float32).view(-1,1).to(device)
X_val_t = torch.tensor(X_val.values, dtype=torch.float32).to(device)
y_val_t = torch.tensor(y_val.values, dtype=torch.float32).view(-1,1).to(device)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=512, shuffle=True)

class TabTransformer(nn.Module):
    def __init__(self, input_dim, emb_dim=64, heads=4, depth=2):
        super().__init__()
        self.emb = nn.Linear(1, emb_dim)   # 1 feature token -> embedding
        layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.out = nn.Sequential(
            nn.Linear(emb_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.unsqueeze(-1)           # B,4,1 4个feature = 4 token
        x = self.emb(x)               # B,4,emb
        x = self.encoder(x)           # transformer
        x = x.mean(dim=1)             # pooling
        return self.out(x)

tab_model = TabTransformer(input_dim=X_train.shape[1]).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(tab_model.parameters(), lr=3e-4)

epochs = 15
tab_model.train()
for epoch in range(epochs):
    loss_t = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        p = tab_model(xb)
        loss = criterion(p, yb)
        loss.backward()
        optimizer.step()
        loss_t += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, loss={loss_t/len(train_loader):.4f}")

lgb_val = lgb_model.predict_proba(X_val)[:,1]
tab_model.eval()
with torch.no_grad():
    tab_val = tab_model(X_val_t).cpu().numpy().flatten()

ensemble_val = (lgb_val + tab_val) / 2
print("LGB AUC:", roc_auc_score(y_val,lgb_val))
print("TAB AUC:", roc_auc_score(y_val,tab_val))
print("Ensemble AUC:", roc_auc_score(y_val,ensemble_val))
print("使用双模型融合 GPU")

test_df = final_data.copy()
X_test = test_df[features]

lgb_test = lgb_model.predict_proba(X_test)[:,1]
X_test_t = torch.tensor(X_test.values, dtype=torch.float32).to(device)
with torch.no_grad():
    tab_test = tab_model(X_test_t).cpu().numpy().flatten()

test_pred_prob = (lgb_test + tab_test) / 2  # 双模型融合

pred_df = pd.DataFrame({
    'customer_id': test_df['customer_id'],
    'article_id': test_df['article_id'].astype(str).str.zfill(10),
    'pred_prob': test_pred_prob
}).sort_values(['customer_id','pred_prob'], ascending=[True,False])

top12_df = pred_df.groupby("customer_id")['article_id'].apply(lambda x: ' '.join(x.head(12))).reset_index()
sample_sub = pd.read_csv("/kaggle/input/h-and-m-personalized-fashion-recommendations/sample_submission.csv")
submission = sample_sub[['customer_id']].merge(top12_df,on='customer_id',how='left')
submission['prediction']=submission['prediction'].fillna('')
submission.to_csv("submission.csv",index=False)
print("submission.csv已生成")



#  分析特征重要性 
import matplotlib.pyplot as plt
feature_importance = (
    pd.DataFrame({
        'feature': features,
        'importance': lgb_model.feature_importances_
    })
    .sort_values(by='importance', ascending=False)
)

print("\n 特征重要性：")
print(feature_importance)

# 可视化
plt.figure(figsize=(6,4))
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.gca().invert_yaxis()
plt.title('Feature Importance (LightGBM)')
plt.show()



from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, roc_curve
import matplotlib.pyplot as plt

#  预测概率与分类
y_pred_prob = lgb_model.predict_proba(X_val)[:, 1]
y_pred = (y_pred_prob > 0.5).astype(int)

#  主要评估指标
auc = roc_auc_score(y_val, y_pred_prob)
acc = accuracy_score(y_val, y_pred)

print("\n 模型评估结果：")
print(f"AUC: {auc:.4f}")
print(f"Accuracy: {acc:.4f}")
print("\n分类报告:")
print(classification_report(y_val, y_pred))

#  ROC曲线可视化
fpr, tpr, _ = roc_curve(y_val, y_pred_prob)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f'LightGBM (AUC={auc:.4f})')
plt.plot([0, 1], [0, 1], '--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()


