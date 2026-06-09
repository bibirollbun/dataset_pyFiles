# https://zenn.dev/porphyrio/articles/39005e2ba86057
# https://pytorch.org/get-started/previous-versions/


%%writefile requirements.txt
--index-url https://download.pytorch.org/whl/cu121
torch
torchvision
torchaudio


! python -m pip download --destination-directory . -r requirements.txt




