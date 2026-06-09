import pandas as pd

# 加载数据
train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')

# 计算每个ID的Fertilizer频率和全局Fertilizer频率（用于处理训练中未出现的ID）
train_grouped = train.groupby('id')['Fertilizer Name'].value_counts().reset_index(name='count')
global_freq = train['Fertilizer Name'].value_counts().sort_values(ascending=False).index.tolist()

pred_dict = {}
for id_ in test['id'].unique():
    # 获取当前ID的训练数据
    temp = train_grouped[train_grouped['id'] == id_].sort_values('count', ascending=False)
    top3 = temp['Fertilizer Name'].tolist()[:3]
    
    # 补充全局频率最高的Fertilizer（如果不足3个）
    global_idx = 0
    while len(top3) < 3 and global_idx < len(global_freq):
        fert = global_freq[global_idx]
        if fert not in top3:
            top3.append(fert)
        global_idx += 1
    
    pred_dict[id_] = top3

# 生成提交文件
submission = test[['id']].copy()
submission['Fertilizer Name'] = submission['id'].apply(lambda x: ' '.join(pred_dict[x]))

# 打印前10个预测结果
print("前10个测试样本的预测结果：")
for i, (id_, fert) in enumerate(submission.values[:10]):
    print(f"ID: {id_}, predict Fertilizer: {fert}")

# 保存结果
submission.to_csv('submission.csv', index=False)
print("\n提交文件已保存为 submission.csv")

