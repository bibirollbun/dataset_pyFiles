# https://zenn.dev/porphyrio/articles/39005e2ba86057


%%writefile requirements.txt
accelerate>=1.10.0
bitsandbytes>=0.47.0
peft>=0.17.0
autoawq>=0.2.9
auto-gptq>=0.7.1

# !pip install -U bitsandbytes accelerate peft
# !pip download -d /kaggle/working/ bitsandbytes accelerate peft


! python -m pip download --destination-directory . -r requirements.txt




