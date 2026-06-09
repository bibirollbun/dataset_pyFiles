
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



train_df =pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test_df =pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
sam_df =pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')


#train_df.head()
test_df.head()



# 区分连续变量和分类变量
# fastai 提供了一个辅助函数自动识别，但手动指定更准确
# cont_names, cat_names = cont_cat_split(df, dep_var='accident_risk')
cat_names=['road_type','lighting','weather','road_signs_present',
           'public_road','time_of_day','holiday','school_season']
cont_names=['num_lanes','curvature','speed_limit','num_reported_accidents']
dep_var='accident_risk'


sam_df.head()


from fastai.tabular.all import *
import pandas as pd
procs=[Categorify,FillMissing,Normalize]
path = Path('/kaggle/input/playground-series-s5e10')
splits=RandomSplitter(seed=42)(range_of(train_df))
dls = TabularDataLoaders.from_df(
    train_df,
    path,
    procs=procs,
    cat_names=cat_names,
    cont_names=cont_names,
    y_names=dep_var,
    valid_pct=0.2,
    bs=64
)


#第三步：训练模型（关键技巧：y_range ）
#回归问题，accident_risk通常在一定范围内（比如0到1），告诉模型这个范围能显著提高精度。
min_y = train_df['accident_risk'].min()
max_y = train_df['accident_risk'].max()
y_range =(min_y*0.9,max_y*1.1)


learn=tabular_learner(
    dls,
    layers=[200,100],
    y_range=y_range,
    metrics=rmse
)
import os
from pathlib import Path

# 确保使用可写目录
#if os.path.exists('/kaggle/working/'):
working_dir = Path('/kaggle/working/')
working_dir.mkdir(exist_ok=True)
    
# 设置learn的路径
learn.path = working_dir
    
# 现在可以安全运行
lr_res=learn.lr_find()
best_lr=lr_res.valley
print(f"建议的学习率是: {best_lr}")
# 简单的写法，但不会输出图像
#suggest_funcs 默认就是 (valley, slide)
#best_lr = learn.lr_find(suggest_funcs=(valley, slide)).valley


learn.fit_one_cycle(5,best_lr)


test_dl=learn.dls.test_dl(test_df)
preds,_=learn.get_preds(dl=test_dl)
print(preds.numpy())


# 2. 填入预测值
# 注意：preds 是一个二维张量 (N行, 1列)，而 DataFrame 需要一维数组
# 所以要用 .ravel() 把它“拉平”
sam_df[dep_var]=preds.numpy().ravel()
print(sam_df.head())
sam_df.to_csv("submission.csv",index=False)

