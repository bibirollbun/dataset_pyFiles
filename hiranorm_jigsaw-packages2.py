!mkdir whls

!pip download -d '/kaggle/working/whls' trl==0.21.0 optimum==1.27.0 auto-gptq==0.7.1 bitsandbytes==0.46.1 deepspeed==0.17.4 logits-processor-zoo==0.2.1 vllm==0.10.0
!pip download -d '/kaggle/working/whls' triton==3.2.0
!pip download -d '/kaggle/working/whls' clean-text
!pip download -d '/kaggle/working/whls' --no-deps peft accelerate datasets


!uv pip install --system --no-index --find-links='/kaggle/working/whls/' 'trl==0.21.0' 'optimum==1.27.0' 'auto-gptq==0.7.1' 'bitsandbytes==0.46.1' 'deepspeed==0.17.4' 'logits-processor-zoo==0.2.1' 'vllm==0.10.0'
!uv pip install --system --no-index --find-links='/kaggle/working/whls/' 'triton==3.2.0'
!uv pip install --system --no-index --find-links='/kaggle/working/whls/' 'clean-text'
!uv pip install --system --no-index -U --no-deps --find-links='/kaggle/working/whls/' 'peft' 'accelerate' 'datasets'




