# 在代码里面安装
!pip install anndata > /dev/null


import anndata as ad
import warnings    # 用于过滤警告
warnings.filterwarnings('ignore')

adata = ad.read_h5ad('/kaggle/input/data-mining-hw-2/final_dataset.h5ad/final_dataset.h5ad')
adata


import pandas as pd

submission = pd.read_csv('/kaggle/input/data-mining-hw-2/sample_submission.csv')
submission.head(5)


# 输出提交文件，可以在右侧直接 submit
submission.to_csv('./submission.csv',index=False)

