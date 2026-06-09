from os import path as osp

PATH_PREFIX = '/kaggle/input/nvidia-tensorrt-install'

PATH_NVIDIA = 'nvidia'

PATH_SOURCE = osp.join(PATH_PREFIX if osp.exists(PATH_PREFIX) else '', PATH_NVIDIA)

PATH_SOURCE


VERSION_TRT = '10.11.0.33'


%pip download --prefer-binary --dest {PATH_NVIDIA} tensorrt=={VERSION_TRT}


!pip install build


!pip install ./nvidia/nvidia_cuda_runtime_cu12-12.9.37-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl


!pip install ./nvidia/tensorrt_cu12_bindings-10.11.0.33-cp311-none-manylinux_2_28_x86_64.whl


!tar -xzf ./nvidia/tensorrt_cu12_libs-10.11.0.33.tar.gz
!python -m build --wheel ./tensorrt_cu12_libs-10.11.0.33
!pip install ./tensorrt_cu12_libs-10.11.0.33/dist/tensorrt_cu12_libs-10.11.0.33-py2.py3-none-manylinux_2_28_x86_64.whl


!tar -xzf ./nvidia/tensorrt_cu12-10.11.0.33.tar.gz
!python -m build --wheel ./tensorrt_cu12-10.11.0.33
!pip install ./tensorrt_cu12-10.11.0.33/dist/tensorrt_cu12-10.11.0.33-py2.py3-none-any.whl


!tar -xzf ./nvidia/tensorrt-10.11.0.33.tar.gz
!python -m build --wheel ./tensorrt-10.11.0.33
!pip install ./tensorrt-10.11.0.33/dist/tensorrt-10.11.0.33-py2.py3-none-any.whl


import tensorrt as trt

trt.__version__





