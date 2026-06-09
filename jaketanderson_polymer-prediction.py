!pwd
!ls
!ls /kaggle
!ls /kaggle/working
!ls /kaggle/inputs


import os
import sys


# os.system("sh Miniconda3-latest-Linux-x86_64.sh -b -p /kaggle/working/miniconda3")
os.environ["PYTHONPATH"] = "/kaggle/working/miniconda3/lib:/kaggle/lib/kagglegym:/kaggle/lib"
os.environ["PATH"] = "/kaggle/working/miniconda3/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/tools/node/bin:/tools/google-cloud-sdk/bin"
# os.system("/kaggle/working/miniconda3/bin/conda install -c conda-forge mamba -y")
# os.system("/kaggle/working/miniconda3/bin/mamba create -p /kaggle/working/miniconda3/envs/polymer-prediction libstdcxx-ng=13 python=3.10 -c conda-forge -c defaults -c anaconda --yes")
# lines = """-f https://data.pyg.org/whl/torch-2.4.0+cu121.html

# torch==2.4.0
# torchvision
# torchaudio
# torch-scatter==2.1.2
# torch-cluster==1.6.3
# decorator
# numpy<2
# rdkit-pypi>=2020.9
# matplotlib
# tqdm
# networkx
# ninja
# jinja2
# lmdb
# fair-esm
# pandas
# jupyterlab
# torchdrug
# PyQt6"""

# with open("/kaggle/working/requirements.txt", "w") as f:
#     f.write(lines)

os.system("chmod +x /kaggle/working/miniconda3/envs/polymer-prediction/bin/*")
# os.system("/kaggle/working/miniconda3/envs/polymer-prediction/bin/pip3 install -r requirements.txt")
sys.path.insert(0, "/kaggle/working/miniconda3/envs/polymer-prediction/lib/python3.10/site-packages")
os.system("cp property_prediction.py /kaggle/working/miniconda3/envs/polymer-prediction/lib/python3.10/site-packages/torchdrug/tasks/property_prediction.py")


!/kaggle/working/miniconda3/envs/polymer-prediction/bin/python main.py


!tar -cvzf all.tar.gz *


os.system("pip3 install magic-wormhole")


os.system("wormhole send /kaggle/working/all.tar.gz")

