# copy retrieval weights
!mkdir -p /root/.cache/torch/hub/netvlad
!cp /kaggle/input/imc-23-weigths/Pitts30K_struct.mat /root/.cache/torch/hub/netvlad/VGG16-NetVLAD-Pitts30K.mat
!cp -r /kaggle/input/imc-23-weigths/cosplace/gmberton_cosplace_main /root/.cache/torch/hub/gmberton_CosPlace_main
!cp -r /kaggle/input/imc-23-weigths/cosplace/checkpoints /root/.cache/torch/hub/checkpoints

# copy rotation prediction
!mkdir -p /root/.cache/huggingface/
!cp -r /kaggle/input/orientation-prediction-model/hub /root/.cache/huggingface/

# copy run_pixsfm.py script to working directory and make it executable 
!cp /kaggle/input/imc-23-repo/IMC-2023/run_pixsfm.py /kaggle/working/run_pixsfm.py
!chmod 777 run_pixsfm.py

# copy hloc_reconstruction.py script to working directory and make it executable
!cp /kaggle/input/imc-23-repo/IMC-2023/hloc_reconstruction.py /kaggle/working/hloc_reconstruction.py
!chmod 777 hloc_reconstruction.py


!python -m venv venv


!cp -r /kaggle/input/imc-23-repo/venv .


# copy pixsfm d2net
!mkdir -p /kaggle/working/venv/lib/python3.10/site-packages/pixsfm/features/models/checkpoints
!cp /kaggle/input/imc-23-weigths/s2dnet_weights.pth /kaggle/working/venv/lib/python3.10/site-packages/pixsfm/features/models/checkpoints/s2dnet_weights.pth


!ls venv/bin


# install hloc
# %pip install -q /kaggle/input/imc-23-repo/IMC-2023/wheels/hloc-1.3-py3-none-any.whl --no-deps

# install disk deps
!pip install -q /kaggle/input/imc-23-repo/IMC-2023/wheels/torch_dimcheck-0.0.1-py3-none-any.whl
!pip install -q /kaggle/input/imc-23-repo/IMC-2023/wheels/unets-0.2.0-py3-none-any.whl
!pip install -q /kaggle/input/imc-23-repo/IMC-2023/wheels/torch_localize-0.1.0-py3-none-any.whl

# install lightglue deps
!pip install -q /kaggle/input/imc-23-repo/IMC-2023/wheels/einops-0.6.1-py3-none-any.whl

# install rotation predition
!pip install -q /kaggle/input/imc-23-repo/IMC-2023/wheels/dioad-1.0.0-py3-none-any.whl

# install ALIKED deps
!pip install -q /kaggle/input/imc-23-repo/IMC-2023/wheels/thop-0.1.1.post2209072238-py3-none-any.whl --no-deps


# tf version of original env
# !pip install -q /kaggle/input/imc-23-repo/IMC-2023/wheels/protobuf-3.19.6-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl --no-deps
# !pip install -q /kaggle/input/imc-23-repo/IMC-2023/wheels/google_auth-2.19.1-py2.py3-none-any.whl --no-deps
# !pip install -q /kaggle/input/imc-23-repo/IMC-2023/wheels/google_auth_oauthlib-0.4.6-py2.py3-none-any.whl --no-deps
# !pip install -q /kaggle/input/imc-23-repo/IMC-2023/wheels/tensorboard_data_server-0.6.1-py3-none-manylinux2010_x86_64.whl --no-deps
# !pip install -q /kaggle/input/imc-23-repo/IMC-2023/wheels/tensorboard_plugin_wit-1.8.1-py3-none-any.whl --no-deps
# !pip install -q /kaggle/input/imc-23-repo/IMC-2023/wheels/tensorboard-2.11.2-py3-none-any.whl --no-deps
# !pip install -q /kaggle/input/imc-23-repo/IMC-2023/wheels/tensorflow_estimator-2.11.0-py2.py3-none-any.whl --no-deps
# !pip install -q /kaggle/input/imc-23-repo/IMC-2023/wheels/tensorflow_io_gcs_filesystem-0.32.0-cp310-cp310-manylinux_2_12_x86_64.manylinux2010_x86_64.whl --no-deps
# !pip install -q /kaggle/input/imc-23-repo/IMC-2023/wheels/keras-2.11.0-py2.py3-none-any.whl --no-deps
# !pip install -q /kaggle/input/imc-23-repo/IMC-2023/wheels/tensorflow-2.11.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl


!pwd


!ls


# Adjust pixsfm config
import yaml

conf_path = "/kaggle/input/imc-23-repo/IMC-2023/ext_deps/pixel-perfect-sfm/pixsfm/configs/low_memory.yaml"

with open(conf_path, 'r') as f:
    conf = yaml.load(f, Loader=yaml.FullLoader)

conf["mapping"]["BA"]["optimizer"]["refine_focal_length"] = True
conf["mapping"]["BA"]["optimizer"]["refine_extra_params"] = True
conf["mapping"]["BA"]["optimizer"]["refine_extrinsics"] = True

with open("pixsfm.yaml", 'w') as f:
    yaml.dump(conf, f)


import sys

sys.path.append("/kaggle/input/imc-23-repo/IMC-2023")
sys.path.append("/kaggle/input/imc-23-repo/IMC-2023/ext_deps/Hierarchical-Localization")
sys.path.append("/kaggle/input/imc-23-repo/IMC-2023/ext_deps/dioad")

import os
import argparse

from main import main

args = {
    "data": "/kaggle/input/image-matching-challenge-2023",
    "configs": ["ALIKED2Kh", "DISKh", "SIFT"],
    "retrieval": "netvlad",
    "n_retrieval": 30,
    "mode": "test",
    "output": "/kaggle/temp",
    "pixsfm": True,
    "pixsfm_max_imgs": 9999,
    "pixsfm_low_mem_threshold": 9999,
    "pixsfm_config": "/kaggle/working/pixsfm.yaml",
    "pixsfm_script_path": "/kaggle/input/imc-23-repo/IMC-2023/run_pixsfm.py",
    "rotation_matching": True,
    "rotation_wrapper": True,
    "resize": None,
    "shared_camera": True,
    "overwrite": True,
    "kaggle": True,
    "cropping": False,
    "max_rel_crop_size": 1,
    "min_rel_crop_size": 0,
    "localize_unregistered": True,
    "skip_scenes": None,
}

args = argparse.Namespace(**args)
os.makedirs(args.output, exist_ok=True)

main(args)


!rm -rf venv


!ls -l


!ls ../temp




