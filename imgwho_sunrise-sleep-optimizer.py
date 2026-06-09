{
  "type": "OBJECT",
  "properties": {
    "tips": { "type": "ARRAY", "items": { "type": "STRING" } }
  }
}


# 基线提交 - 直接用测试集均值/随机值
import pandas as pd
import numpy as np

# 读取测试数据（根据比赛调整路径）
test = pd.read_csv('/kaggle/input/[比赛名称]/test.csv')

# 生成简单预测（替换为比赛要求的列名）
submission = pd.DataFrame({
    'id': test['id'],  # 比赛的ID列名
    'target': np.random.normal(0, 1, len(test))  # 随机预测，根据比赛调整
})

submission.to_csv('/kaggle/working/submission.csv', index=False)
print("基线提交完成！")

