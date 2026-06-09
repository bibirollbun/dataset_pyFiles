import warnings
warnings.simplefilter('ignore')
import os
import gc
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
from tqdm import tqdm
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
from gensim.models import Word2Vec
import lightgbm as lgb
from lightgbm.callback import early_stopping


%%time

train = pd.read_csv('/kaggle/input/user-retention-prediction/train.csv')

train['DateTime'] = pd.to_datetime(train['Timestamp'], unit='s')

## 时区矫正 +8h (!!非常重要!!)
train['DateTime'] = train['DateTime'] + pd.Timedelta(hours=8)
# train['MonthDay'] = train['DateTime'].dt.month.astype(str) + '-' + train['DateTime'].dt.day.astype(str)
# 比上面的方式减少点内存占用，还能排序
train['MonthDay'] = train['DateTime'].dt.month * 100 + train['DateTime'].dt.day

display(train)


# 按时间划分训练集
train_df = train[(train['DateTime'] >= '2018-09-21 00:00:00')&\
                 (train['DateTime'] <= '2018-10-13 23:59:59')].reset_index(drop=True)
# 训练集最后一天的活跃用户作为验证集 id
train_ids = train_df[(train_df['DateTime'] >= '2018-10-13 00:00:00')&\
                     (train_df['DateTime'] <= '2018-10-13 23:59:59')]['ID'].unique().tolist()
# 构建验证集标签
train_label = train[(train['DateTime'] >= '2018-10-14 00:00:00')&\
                    (train['DateTime'] <= '2018-10-20 23:59:59')].reset_index(drop=True)
# 每天只要有一条记录即为有效活跃登录
label_df = pd.DataFrame({'ID': train_ids})
mapping = train_label.groupby('ID')['MonthDay'].nunique().to_dict()
label_df['label'] = label_df['ID'].map(mapping)
label_df['label'].fillna(0, inplace=True)
label_df['label'] = label_df['label'].astype(int)


display(train_df)
display(label_df)
display(label_df['label'].value_counts(normalize=True, dropna=False))


def make_base_feature(train_df, train_label):
    
    feats = train_label.copy()
    mapping = train_df.groupby(['ID'])['ID'].count().to_dict()
    feats['total_actions'] = feats['ID'].map(mapping)
    mapping = train_df.groupby(['ID'])['MonthDay'].nunique().to_dict()
    feats['active_days'] = feats['ID'].map(mapping)
    feats['actions_per_day'] = feats['total_actions'] / feats['active_days']
    mapping = train_df.groupby(['ID'])['ActionId'].nunique().to_dict()
    feats['actionid_nunique'] = feats['ID'].map(mapping)
    tmp_df = train_df.groupby(['ID', 'MonthDay'])['Timestamp'].max() - train_df.groupby(['ID', 'MonthDay'])['Timestamp'].min()
    tmp_df = tmp_df.to_frame(name='time_span_daily').reset_index()
    mapping = tmp_df.groupby('ID')['time_span_daily'].mean()
    feats['mean_time_span_daily'] = feats['ID'].map(mapping)
    mapping = tmp_df.groupby('ID')['time_span_daily'].max()
    feats['max_time_span_daily'] = feats['ID'].map(mapping)
    mapping = tmp_df.groupby('ID')['time_span_daily'].min()
    feats['min_time_span_daily'] = feats['ID'].map(mapping)
    mapping = tmp_df.groupby('ID')['time_span_daily'].std()
    feats['std_time_span_daily'] = feats['ID'].map(mapping)
    tmp_df = train_df[['ID', 'MonthDay', 'Timestamp']].copy()
    tmp_df['Timestamp_diff'] = tmp_df.groupby(['ID', 'MonthDay'])['Timestamp'].diff()
    mapping = tmp_df.groupby(['ID'])['Timestamp_diff'].mean().to_dict()
    feats['mean_timestamp_diff'] = feats['ID'].map(mapping)
    mapping = tmp_df.groupby(['ID'])['Timestamp_diff'].max().to_dict()
    feats['max_timestamp_diff'] = feats['ID'].map(mapping)
    mapping = tmp_df.groupby(['ID'])['Timestamp_diff'].std().to_dict()
    feats['std_timestamp_diff'] = feats['ID'].map(mapping)
    mapping = train_df.groupby(['ID'])['ActionId'].last().to_dict()
    feats['last_actionid'] = feats['ID'].map(mapping)

    # 最后七天的登录天数
    end_date = str(train_df['DateTime'].max()).split()[0]
    end_date = f'{end_date} 23:59:59'
    start_date = train_df['DateTime'].max() - pd.Timedelta(days=6)
    start_date = str(start_date).split()[0]
    start_date = f'{start_date} 00:00:00'
    tmp_df = train_df[(train_df['DateTime'] >= start_date)&\
                      (train_df['DateTime'] <= end_date)].reset_index(drop=True)
    mapping = tmp_df.groupby(['ID'])['MonthDay'].nunique().to_dict()
    feats['lastweek_active_days'] = feats['ID'].map(mapping)
    feats['lastweek_active_days'].fillna(0, inplace=True)

    # 用户在数据集里的最后一天的操作
    mapping = train_df.groupby('ID')['MonthDay'].max().to_dict()
    train_df['MaxMonthDay'] = train_df['ID'].map(mapping)
    lastday_df = train_df[train_df['MaxMonthDay'] == train_df['MonthDay']].reset_index(drop=True)
    mapping = lastday_df.groupby(['ID'])['ID'].count().to_dict()
    feats['lastday_actions'] = feats['ID'].map(mapping)
    mapping = lastday_df.groupby(['ID'])['ActionId'].nunique().to_dict()
    feats['lastday_actionid_nunique'] = feats['ID'].map(mapping)
    mapping = (lastday_df.groupby('ID')['Timestamp'].max() - lastday_df.groupby('ID')['Timestamp'].min()).to_dict()
    feats['lastday_time_span'] = feats['ID'].map(mapping)
    tmp_df = lastday_df[['ID', 'Timestamp']].copy()
    tmp_df['Timestamp_diff'] = tmp_df.groupby('ID')['Timestamp'].diff()
    mapping = tmp_df.groupby(['ID'])['Timestamp_diff'].mean().to_dict()
    feats['lastday_mean_timestamp_diff'] = feats['ID'].map(mapping)
    mapping = tmp_df.groupby(['ID'])['Timestamp_diff'].max().to_dict()
    feats['lastday_max_timestamp_diff'] = feats['ID'].map(mapping)
    mapping = tmp_df.groupby(['ID'])['Timestamp_diff'].std().to_dict()
    feats['lastday_std_timestamp_diff'] = feats['ID'].map(mapping)
    
    return feats


def make_w2v_feature(train_df, maxlen=100, vector_size=32, window=5):
    df_aids = train_df.groupby('ID')['ActionId'].agg(list).to_frame(name='id_list').reset_index()
    df_aids['id_list'] = df_aids['id_list'].apply(lambda x: x[::-1][:maxlen])
    texts = []
    for sent in df_aids[df_aids['ID'].isin(train['ID'].values)]['id_list'].values:
        texts.append([str(w) for w in sent])
    model = Word2Vec(texts,
                     vector_size=vector_size, 
                     window=window, 
                     min_count=1, 
                     sg=1,
                     hs=0,
                     seed=42)
    w2v_embs = []
    for _, row in tqdm(df_aids.iterrows(), total=len(df_aids)):
        emb = np.zeros((vector_size,))
        seq = row['id_list']
        for w in seq:
            emb += model.wv.get_vector(str(w)) / len(seq)
        w2v_embs.append(emb)
    w2v_embs = np.array(w2v_embs)
    for i in range(vector_size):
        df_aids[f'aid_w2v_v{vector_size}w{window}_{i}'] = w2v_embs[:, i]
    return df_aids, model


def infer_w2v_feature(w2v_model, df, maxlen=100):
    vector_size = w2v_model.vector_size
    window = w2v_model.window
    df_aids = df.groupby('ID')['ActionId'].agg(list).to_frame(name='id_list').reset_index()
    df_aids['id_list'] = df_aids['id_list'].apply(lambda x: x[::-1][:maxlen])
    texts = []
    for sent in df_aids[df_aids['ID'].isin(train['ID'].values)]['id_list'].values:
        texts.append([str(w) for w in sent])
    w2v_embs = []
    for _, row in tqdm(df_aids.iterrows(), total=len(df_aids)):
        emb = np.zeros((vector_size,))
        seq = row['id_list']
        for w in seq:
            if str(w) in w2v_model.wv.index_to_key:
                emb += w2v_model.wv.get_vector(str(w)) / len(seq)
        w2v_embs.append(emb)
    w2v_embs = np.array(w2v_embs)
    for i in range(vector_size):
        df_aids[f'aid_w2v_v{vector_size}w{window}_{i}'] = w2v_embs[:, i]
    return df_aids


%%time

# 生成特征
df_train = make_base_feature(train_df, label_df)

df_aids, w2v_model = make_w2v_feature(train_df, maxlen=100, vector_size=32, window=5)
df_train = df_train.merge(df_aids[['ID'] + [col for col in df_aids if col.startswith('aid_w2v_')]], 
                          on='ID', how='left')

display(df_train)
display(df_train['label'].value_counts(normalize=True, dropna=False))


%%time

feature_names = [c for c in df_train.columns if c not in ['ID', 'label']]

kf = KFold(n_splits=5, random_state=42, shuffle=True)
models = []
oof_pred = np.zeros(len(df_train))
for i, (train_index, valid_index) in enumerate(kf.split(df_train, df_train['label'])):
    print(f'Fold {i} ...')
    x_valid = df_train.loc[valid_index, feature_names]
    y_valid = df_train.loc[valid_index, 'label']
    x_train = df_train.loc[train_index, feature_names]
    y_train = df_train.loc[train_index, 'label']
    
    model = lgb.LGBMRegressor(
        objective='regression',
        max_depth=8, 
        num_leaves=64,
        min_child_samples=64,
        n_estimators=1000,
        learning_rate=0.05, 
        verbose=-1,
        importance_type='gain'
    )
    model.fit(
        x_train, y_train,
        eval_set=[(x_valid, y_valid)],
        callbacks=[early_stopping(stopping_rounds=100)]
    )
    oof_pred[valid_index] = model.predict(x_valid)
    models.append(model)
    del model; gc.collect()


feature_importances = np.mean([m.feature_importances_ for m in models], axis=0)
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": feature_importances
}).sort_values(by="Importance", ascending=False)

display(importance_df)


def smape(y_true, y_pred):
    scores = []
    for t, p in zip(y_true, y_pred):
        if t == 0 and p == 0:
            scores.append(0)
        else:
            scores.append(2 * abs(t - p) / (abs(t) + abs(p)))
    return 100 * np.mean(scores)


print('LastWeek:', smape(df_train['label'], df_train['lastweek_active_days']))
print('LGBMPred:', smape(df_train['label'], oof_pred))


df_train['model_pred'] = oof_pred

plt.figure(figsize=(10, 6))
sns.kdeplot(data=df_train, x='label', label='label', fill=True, alpha=0.3)
sns.kdeplot(data=df_train, x='model_pred', label='model_pred', fill=True, alpha=0.3)
plt.legend()
plt.show()


ratios = np.array([0.165077, 0.071544, 0.061548, 0.055787, 0.053056, 0.058299, 0.083344, 0.451345])
cumulative_ratios = np.cumsum(ratios[:-1])
quantiles = np.quantile(oof_pred, cumulative_ratios)
bins = np.concatenate([[oof_pred.min()], quantiles, [oof_pred.max()]])
oof_pred_pp = np.digitize(oof_pred, bins, right=True) - 1

df_train['pred'] = oof_pred_pp
print('LGBMPred:', smape(df_train['label'], df_train['pred']))


%%time

# 测试集

test_df = train[(train['DateTime'] >= '2018-09-28 00:00:00')&\
                (train['DateTime'] <= '2018-10-20 23:59:59')].reset_index(drop=True)
test_ids = test_df[(test_df['DateTime'] >= '2018-10-20 00:00:00')&\
                   (test_df['DateTime'] <= '2018-10-20 23:59:59')]['ID'].unique().tolist()

df_test = make_base_feature(test_df, pd.DataFrame({'ID': test_ids}))

df_aids = infer_w2v_feature(w2v_model, test_df, maxlen=100)
df_test = df_test.merge(df_aids[['ID'] + [col for col in df_aids if col.startswith('aid_w2v_')]], 
                        on='ID', how='left')

display(df_test)


%%time

pred_test = np.zeros(len(df_test))
for model in models:
    pred_test += model.predict(df_test[feature_names]) / kf.n_splits


ratios = np.array([0.113103, 0.065773, 0.047192, 0.055159, 0.054760, 0.076450, 0.121039, 0.466478])
cumulative_ratios = np.cumsum(ratios[:-1])
quantiles = np.quantile(pred_test, cumulative_ratios)
bins = np.concatenate([[pred_test.min()], quantiles, [pred_test.max()]])
pred_test_pp = np.digitize(pred_test, bins, right=True) - 1

df_test['pred'] = pred_test_pp
df_test[['ID', 'pred']].to_csv('baseline.csv', index=False)




