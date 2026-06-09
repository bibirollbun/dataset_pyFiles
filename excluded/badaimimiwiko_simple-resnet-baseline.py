# 运行训练
# 准备环境
!wget https://raw.githubusercontent.com/lhiqwj173/dl_helper/master/envs/rl.py > /dev/null 2>&1
!python rl.py > /dev/null 2>&1

import os
os.environ['ALIST_USER'] = 'admin'
os.environ['ALIST_PWD'] = 'LHss6632673'

!wget -O run.py https://raw.githubusercontent.com/lhiqwj173/dl_helper/master/dl_helper/tests/other/test_leaf.py > /dev/null 2>&1
# !wget -O run.py https://raw.githubusercontent.com/lhiqwj173/dl_helper/master/dl_helper/tests/other/test_mnist.py > /dev/null 2>&1
!python run.py idx=0




