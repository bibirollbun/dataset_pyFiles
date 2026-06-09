import os
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss



# Dosya dizini
DIR = '/kaggle/input/pump-fun-graduation-february-2025'

# chunk dosyalarını oku ve birleştir
chunk_files = [os.path.join(DIR, f) for f in os.listdir(DIR) if f.startswith("chunk") and f.endswith(".csv")]
chunks = pd.concat([pd.read_csv(f) for f in chunk_files], ignore_index=True)



# Yeni özellikleri çıkaralım
chunks['relative_block'] = chunks.groupby('base_coin')['slot'].transform(lambda x: x - x.min())

stat_features = chunks.groupby("base_coin").agg(
    tx_count=('tx_idx', 'count'),
    wallet_count=('signing_wallet', pd.Series.nunique),
    avg_quote_per_tx=('quote_coin_amount', 'mean'),
    volatility_token=('virtual_token_balance_after', 'std'),
    volatility_sol=('virtual_sol_balance_after', 'std'),
    early_tx_ratio=('relative_block', lambda x: (x <= 5).sum() / len(x))
).reset_index()

buy_sell = chunks.groupby(['base_coin', 'direction'])['tx_idx'].count().unstack(fill_value=0)
buy_sell['buy_sell_ratio'] = buy_sell['buy'] / (buy_sell['sell'] + 1)
buy_sell = buy_sell[['buy_sell_ratio']].reset_index()



# Tüm özellikleri birleştiriyoruz
features_df = stat_features.merge(buy_sell, on='base_coin', how='left')
features_df.fillna(0, inplace=True)



# `train.csv` ile birleştiriyoruz
train = pd.read_csv('/kaggle/input/solana-skill-sprint-memcoin-graduation/train.csv')
df_merged = train.merge(features_df, left_on='mint', right_on='base_coin', how='left')
df_merged.fillna(0, inplace=True)



# Özellikleri seçiyoruz
features = [
    'tx_count',
    'wallet_count',
    'buy_sell_ratio',
    'avg_quote_per_tx',
    'volatility_token',
    'volatility_sol',
    'early_tx_ratio'
]

X = df_merged[features]
y = df_merged['has_graduated']



# Veriyi eğitim ve doğrulama setlerine ayıralım
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)



# LightGBM için veri setlerini hazırlayalım
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val)

# Model parametrelerini tanımlayalım
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'learning_rate': 0.01,
    'num_leaves': 64,
    'max_depth': 7,
    'seed': 42
}

callbacks = [
    lgb.early_stopping(stopping_rounds=20),
    lgb.log_evaluation(period=50)
]

# Modeli eğitelim
model = lgb.train(
    params,
    train_data,
    valid_sets=[val_data],
    valid_names=['valid'],
    num_boost_round=500,
    callbacks=callbacks
)



# Model tahmini yapalım
y_pred_proba = model.predict(X_val)

# Log Loss değerini hesaplayalım
print("Validation Log Loss:", log_loss(y_val, y_pred_proba))



# Test verisini oku
test = pd.read_csv('/kaggle/input/solana-skill-sprint-memcoin-graduation/test_unlabeled.csv')

# Test verisinde aynı özellikleri çıkaralım
test_df = test.merge(features_df, left_on='mint', right_on='base_coin', how='left')
test_df.fillna(0, inplace=True)

# Test verisi için özellikleri seçelim
X_test = test_df[features]

# Model ile tahmin yapalım
test_pred_proba = model.predict(X_test)

# Sonuçları submission formatında hazırlayalım
submission = pd.DataFrame({
    'mint': test['mint'],  # Test veri setindeki mint değerini alıyoruz
    'has_graduated': test_pred_proba  # Modelin tahmin ettiği değerleri kullanıyoruz
})

# submission dosyasını kaydedelim
submission.to_csv('/kaggle/working/submission.csv', index=False)
print("✅ submission.csv başarıyla oluşturuldu.")



df_merged.head()

