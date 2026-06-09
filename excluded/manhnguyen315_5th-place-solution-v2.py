!pip install /kaggle/input/konwinski-prize/kaggle_evaluation/../kprize_setup/kprize-1.1.1-py3-none-any.whl --no-index --find-links /kaggle/input/konwinski-prize/kaggle_evaluation/../kprize_setup/pip_packages/kprize


!pip install --no-index --find-links=/kaggle/input/vllm-ins vllm -U
!pip install --no-index --find-links=/kaggle/input/vllm-ins triton==3.1.0 -q
!pip install --no-index --find-links=/kaggle/input/vllm-ins bitsandbytes -q
!pip install --no-index --find-links=/kaggle/input/vllm-ins pynvml==12.0.0


!mkdir /kaggle/working/src

!cp /kaggle/input/kprize-5th-place-source-code/* /kaggle/working/src/
!cp -R /kaggle/input/pytest-xdist /kaggle/working
!mkdir /kaggle/working/submissions


%%writefile /kaggle/working/src/SETTING.json
{
  "TMP_DATA_DIR": "/tmp/kaggle/tmp/konwinski-prize-alt/",
  "DATA_DIR": "/kaggle/input/konwinski-prize/",
  "LLM_MODEL_PTH": "/kaggle/input/deepseek-r1/transformers/deepseek-r1-distill-qwen-14b-awq/1",
  "WORKING_DIR": "/kaggle/working/",
  "SUBMISSION_DIR": "./submissions/",
  "PYTEST_XDIST_DIR": "/kaggle/working/pytest-xdist",
  "PIP_PACKAGES_PATH": "/path/to/pip_packages"
}


# unpack data.a_zip from DATA_DIR to TMP_DATA_DIR
!python /kaggle/working/src/unpack_data.py


!python /kaggle/working/src/main.py


!ls /kaggle/working/submissions




