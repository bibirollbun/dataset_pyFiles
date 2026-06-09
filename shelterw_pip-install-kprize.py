!pip install --target=/kaggle/working vllm==0.7.2 bitsandbytes fsspec==2024.9.0 pre_commit==4.0.1 cfgv==3.4.0 dill==0.3.8


!rm -rf /kaggle/working/ray*


!pip install /kaggle/input/konwinski-prize/kaggle_evaluation/../kprize_setup/kprize-1.1.1-py3-none-any.whl --no-index --find-links /kaggle/input/konwinski-prize/kaggle_evaluation/../kprize_setup/pip_packages/kprize

