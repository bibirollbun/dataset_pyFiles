!pip install jsonlines


!pip wheel jsonlines -w wheelhouse


!zip -r wheelhouse.zip /kaggle/working/wheelhouse


!pip install /kaggle/input/jsonlines/jsonlines-4.0.0-py3-none-any.whl

