# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session





!pip install -U "huggingface_hub[cli]"


!pwd


%ls


!git clone https://github.com/Pham-Xuan-Thanh/PSLD.git
! echo current dir $( pwd )
%cd PSLD



!huggingface-cli download thanhphxu/psld_cifar10_55m 'checkpoints/psld--epoch=2089-loss=0.0042.ckpt' --local-dir . --token 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
# !huggingface-cli download thanhphxu/psld_cifar10_55m 'outputs/cifar10_test_images' --local-dir /kaggle/working/PSLD --token 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


!head checkpoints/psld--epoch=2089-loss=0.0042.ckpt


# !curl 'https://personalmicrosoftsoftware-my.sharepoint.com/personal/pandeyk1_personalmicrosoftsoftware_uci_edu/_layouts/15/download.aspx?UniqueId=91772ca7%2Db4e3%2D436c%2D9673%2D2f77650e1406' \
#   -H 'accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7' \
#   -H 'accept-language: en-US,en;q=0.9' \
#   -H 'cache-control: no-cache' \
#   -b 'FeatureOverrides_experiments=[]; MSFPC=GUID=2601c284eb5d41c594a397d3c2dcd077&HASH=2601&LV=202409&V=4&LU=1726662138142; msal.cache.encryption=%7B%22id%22%3A%22019547e0-1024-7cf9-8ca7-140117d400cb%22%2C%22key%22%3A%22X0deg8AQE0mr_VJpyH-ISA6rAviI3A3G8SYHglZMm7k%22%7D; rtFa=dVlQFYH87han1F/vUBlrwWUvlLe2pxVAGyVkifu9ysgmNDAxMjdjZDQtNDVmMy00OWEzLWIwNWQtMzE1YTQzYTlmMDMzIzEzMzg2OTQwOTQyNjA1NjY3MiMyNzU1OGNhMS1mMDhjLTQwMDAtYzljYy1jOTYxYzQyYTQ4MTEjMjRDMTEwMjYlNDBzdHVkZW50LmhjbXVzLmVkdS52biMxOTY2MDkjYjNJUWg0NWt5YWZ4dmZKVE5TUDNSc3VvT0ZNI19FYnQ1c3NMalFNTnU0b1ZrRU4zVW1KYlVBcw2kgsoPURh/lDNLFHFICO92bs9Ap3HEzNkZDK5Wnsc9BBEIOdRXqAHUH5NIob4Uwu3rx6TG+JQpDd7i2TQ9SrKtKQLlC2LXq3+3IGQSYJaoa+viA/2BgZMuZ/Cjw0zzRLzZ8IrYyhqKX6JYuivEcoZ7pj9i3KBp3q8as3jJlWbxnYtCpc7+oiZl8lLUqMPjkOhWkeP/nzy8llxMkiRrSUXUu7ee21Yctse412uktyQ+QfpprN0k2CdxCPOj8hOk9Xu3BvLE4S2V5rM1n3UWFf+Yfre7v1ZDwf3CD9ZOAal87JQq+IZp01OWysQirukHq52LVLqhh8Iyc+jH+ZwMRtXcAAAA; SIMI=eyJzdCI6MH0=; MicrosoftApplicationsTelemetryDeviceId=4d0127c6-da94-4302-b9c3-cae851a4befd; ai_session=DHBrN1sEOfj4GsoOrfZNM/|1742982082883|1742982082883; nSGt-AD23666C9389064FC8B0D6B70C54B8A500A843977DA649C1=gYEwM0ZGMkVDQTA2NEU2OEVBQzU5QzRBRENDNTk2OUMxRjRDQTlCQjZEQjM1NEQ1M0U0MEFEMjM2NjZDOTM4OTA2NEZDOEIwRDZCNzBDNTRCOEE1MDBBODQzOTc3REE2NDlDMRIxMzM4NzQ1NTkyNDgwNDQyNzYrcGVyc29uYWxtaWNyb3NvZnRzb2Z0d2FyZS1teS5zaGFyZXBvaW50LmNvbVlCHhZ3mg7wnMuL6e7wwrDy52Hlojp/KubOKm3j1fpcYIppfBNiumTovk5BIgq7gH9rF3gxQGaisL1o5wlkD26Sivagc0Cw4eZyJ69rK37/jwm39OS0jSs5RI9x5prcKAG4/8K1FWLiEVevKtap01P50IlTrQ3ri9eQ6t7OWi2Hi7U3ekumyz44I5aHHxc1O9P232/xQeEwMT7YSy1uy2ULPs54xvHw60IA0f8QgGME8ElujEoXqbYkwCt41X2sFUvDB+/h/HmZ5hnd/GvfZOIuP0jEO5yi/UQmr1JmhNsWiz2UAaMdpjCPj6ZtrmEt0s99cZibG/mUih9Ty7i2z9CjAAAA; nSGt-480B957892B1D97D4518AE7153B9EE56C98885AFAB14742C=gYEwQzk1QkVCRERDQ0RBQzVCREI2QkY5RUJBQjRBODUyNTlERjhDNkE4N0ZGM0Y3REQzMDQ4MEI5NTc4OTJCMUQ5N0Q0NTE4QUU3MTUzQjlFRTU2Qzk4ODg1QUZBQjE0NzQyQxIxMzM4NzQ1NTkyNzM5NzgyNjUrcGVyc29uYWxtaWNyb3NvZnRzb2Z0d2FyZS1teS5zaGFyZXBvaW50LmNvbXJKkJUyONdKd/BWkINCXymT4TKS8oHK12NLhKpbcVk4JyopzedMLhmMbiOv1IHDzvmckULKZ1ebeVVOouQ4iZh47zkJ/y9Pdh0QcUcNWYsjjDnyNKpuyF/S98p/c489aQEmZs738YtxvHZcUZiyZDy2cW0gZH8LJNnuLBLra7uriJLIZXVVTCneODv2tBItDookrgoR8LEB4t46J97Ii75s0W3jH/iosEX4CzzDvOMX6hY4aRiOu97UllI5fmJ+Livv0+0OF4VcsWvvA0zqYCqTkzNh52IUeyBeALr28O6xUrI7DZn+v8edzzlG3ykPoHitSzNNGWZkbzWC5YWTUEGjAAAA; nSGt-BC2B439AC83E20E3E664C36BA943D8ACE8CBC311A1BDDFFC=gYEwMzJFMEJFRTZDMzUzM0IxOEYwQTMyMkNCRDc3OERERDJCMEJGODYxQkU5MDQ1MEYzMEJDMkI0MzlBQzgzRTIwRTNFNjY0QzM2QkE5NDNEOEFDRThDQkMzMTFBMUJEREZGQxIxMzM4NzQ1NTkyODUxODgyNjkrcGVyc29uYWxtaWNyb3NvZnRzb2Z0d2FyZS1teS5zaGFyZXBvaW50LmNvbTk9AxTjOrdfH3AHN/alLtOybRBZb4C0CrtKH0t/Wbn8GIAiKNH7NBTqfpGuuz/CXmUDfOTStsfwW5+5zgSCV0UHt4Ekmucsfc1P2eXGu+18Pf0M2wsJ2uAhisdEFTa/Yb/voyZlh6UEZyFSOwE9wJNV0W8fCmjnmSJzPUJo5gT+Sx+lXgiNgXc14TKO94E/vhHY+Ow1AZmJNr2tlzjDRNrhPe4W6b/SeYCQurpS7lsIVORQw9B70bZ3Eye5AkQzlwO6IyTf93d/939eknL8kEuqDIH6JZ0Rlx5rvb7r8E67VrDc7ms++8U4DYg6NvhFdJSGtbCCh4TsarXECSkm+BujAAAA; nSGt-0905A3E275AD58857EEA61D2DACB1B8C3D0C1984466FDEB1=gYEwOEY4RkI1QTZBNzM1MzU1NUEzRDdERDY1NEM3QUE4NThFRDdFMTlCRDQ1MzBDREM2MDA5MDVBM0UyNzVBRDU4ODU3RUVBNjFEMkRBQ0IxQjhDM0QwQzE5ODQ0NjZGREVCMRIxMzM4NzQ1NTkzMDE4OTU5NzQrcGVyc29uYWxtaWNyb3NvZnRzb2Z0d2FyZS1teS5zaGFyZXBvaW50LmNvbWbfAO56ri4l53bjiDRWG+7V0PaHXSE1Dm7bUnxiP+HgZcyLS5V7UVXB8ge9LUY2egRvuoN4juRfY84hZ1izU38xo50o+0HoCGY+JbAa8iuSO7tVqSM8sdkdksp7IGpMehYlM0CfQ1fjD05ipBcmrYZBx+lMA4BDKwwg7czB5qEPYTwc1RCgGTZ0EQF8SR5O8F3z6NUa6eSdNbso/jNrUx2kPKksJRVNbosVrI7yeQybuhO7iMpqtAj8NuM1VACxdiiaHphlfKvcybyu+rQFOEd1w4I9tOqQcuSEzR5N4KqrrDpJJnIsi6Oq5aSbqdjWn3oj9DfUTeoThLZIjUuf4cGjAAAA; nSGt-AB8F05D24D09EBD75705FE254AFF585C020D917F9DE6B644=gYEwM0EwOUQxQTc3NUEyMkZFMjJBQ0Q4OUQ2RjFDRTJCNUY4REQ3MzIxNEQ4MjdCMEM1MEFCOEYwNUQyNEQwOUVCRDc1NzA1RkUyNTRBRkY1ODVDMDIwRDkxN0Y5REU2QjY0NBIxMzM4NzQ1NTkzMTQ4Mjk4NTkrcGVyc29uYWxtaWNyb3NvZnRzb2Z0d2FyZS1teS5zaGFyZXBvaW50LmNvbawQvr8W6c6OgSHexTf1wEtJEePUCWjYLgozUAJRjtlmOX51re543eWEL1u73WFIF1/N2Pw6UOEQv8fWXCG3lakM4GBylh5ZpQWgkjOpDxkSUPYOFfQbSFWPuMeH6WGYHV49PclXQA3reA+xcpPMLtoQpLK/IOJv7tgkSw1Ij2CJN9iGImQq8anRUod1Up03qIvfNFGRlew1Yuxg8F8oo9zr6AEoUABqSrey+7s7hbACHag12Et/u3bAMklhDFrjDWrBuZWDvfi6EzDyW3zWewk7hNuOYcH6Ws6X8HUEXF6ThWz+lYszRz4SwFNQfQIzi9RkP/GEWONvuOEs/KfSC8SjAAAA; nSGt-EA6FB8CC168942B4D8F94B25BA453EA9714C45F92DBB8478=gYEwRkY2RDU5OThGQkE1ODYyRURCOEUwNERGMEI1RkI1RjIxRUFGNzYyOUMyOENDRjE0MEVBNkZCOENDMTY4OTQyQjREOEY5NEIyNUJBNDUzRUE5NzE0QzQ1RjkyREJCODQ3OBIxMzM4NzQ1NTkzMzIzODUxMjMrcGVyc29uYWxtaWNyb3NvZnRzb2Z0d2FyZS1teS5zaGFyZXBvaW50LmNvbZGEyrxRikUWn+7ATeSTbkm/aFGXCs1a5mJlzonoKKOqjj30iTIcsvqLimnr7iRm9B1U2Xc2QerHp1ooiT460awxW9wOV8X2jV3V3WJMOeeHH9LQi2MmlXUfGJoXW5vqPyRl0kc70ZgSx1u7yo1nZudn+08BNhEZRuQxu3jLYyX4mRW1/21ME4eO5aGrR/lkX1l9S8hqJIEFC7H1sM4rpJ6cZeiQ7lXciLogTN7i46Wh9m5RBEIf7F2FLTlmQ4oWRsTSG/qP18wdwJPByGynLMGsBfrNOWsnaG+COnKoCAxvCvDx5EZBtzi8S6Maf+8Cyz4QDxKW/3ZNau/+GqSvwpajAAAA; RpsContextCookie=UHJldmlvdXNSZXF1ZXN0Q29ycmVsYXRpb25JZD0wZjQwOGVhMSUyRDcwMTklMkQ4MDAwJTJEN2Y4ZCUyRDE5NzcyYzc0OWI1NSZSZXR1cm5Vcmw9aHR0cHMlM0ElMkYlMkZwZXJzb25hbG1pY3Jvc29mdHNvZnR3YXJlJTJEbXklMkVzaGFyZXBvaW50JTJFY29tJTJGcGVyc29uYWwlMkZwYW5kZXlrMSU1RnBlcnNvbmFsbWljcm9zb2Z0c29mdHdhcmUlNUZ1Y2klNUZlZHUlMkYlNUZsYXlvdXRzJTJGMTUlMkZvbmVkcml2ZSUyRWFzcHglM0ZpZCUzRCUyRnBlcnNvbmFsJTJGcGFuZGV5azElNUZwZXJzb25hbG1pY3Jvc29mdHNvZnR3YXJlJTVGdWNpJTVGZWR1JTJGRG9jdW1lbnRzJTJGUFNMRCUyNmdhJTNEMSUyNm5vQXV0aFJlZGlyZWN0JTNEMQ==; FedAuth=77u/PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0idXRmLTgiPz48U1A+VjEzLDBoLmZ8bWVtYmVyc2hpcHx1cm4lM2FzcG8lM2Fhbm9uI2ExYmZhMDNkZTM0NmViYmJiYzRkODBiMTE5OGVjZTUzN2M5OTliZmMxZDAwM2FiMmRiMWE0YjRlOTI2Yzc1MjUsMCMuZnxtZW1iZXJzaGlwfHVybiUzYXNwbyUzYWFub24jYTFiZmEwM2RlMzQ2ZWJiYmJjNGQ4MGIxMTk4ZWNlNTM3Yzk5OWJmYzFkMDAzYWIyZGIxYTRiNGU5MjZjNzUyNSwxMzM4NzQ1NjAwMTAwMDAwMDAsMCwxMzM4NzU0MjEwMjA3Mzg0MDMsMC4wLjAuMCwyNTgsM2U5NjZiNWQtYTZjNC00YTQ0LTgxMzMtODIxZWM0NzY5N2EzLCwsZDRiMmI4Y2QtZGViZi00NGZiLTgzOWUtYzE3Y2IwYjkzMmE3LGQ0YjJiOGNkLWRlYmYtNDRmYi04MzllLWMxN2NiMGI5MzJhNywwWnZrSEtRSDBFaU96Q1ZPbzFSTm53LDAsMCwwLCwsLDI2NTA0Njc3NDM5OTk5OTk5OTksMCwsLCwsLCwwLCwxOTI2NTQsTkxITm9QekJlYzNqTGxRc1NZb0w1UTFFZ2ZFLE1kNmRwR1Iwdmp2WGxjRnhwRVdmU01uaVV3V1BpUHRIeVY0bllkL2lqWExUVUFpN3NIUU9ISldweWZWdUlxWk5jb3d3ajN6V0w4UGVoUnBVMURtU0tLN1dYazNLVGl1RURMOFJOcXpxclB6RlNRK0J6WFkzaUJnMnB5Ti84cEFrWGVHUG9VYkpha053S0sreUp6aGJTVUVjNER4YWJwekxoeThwR2J6ckdGcUpaMHQ2N2lWNlpoUTdSczRMbnRpQVNUcXBFNkZzR3EzalBhOWNOdzM5MmRyS2Z4Z0ZaVE9NNEVkZWY4M043UXpXaHFtTEp6TEdEWEF1cC9aNzF1RWFrYW1LTTNXb3lxSVpQRjJ3VXVneGwwUjdYSkg1UjBOM0ptSVFlU01ESENKcjUvdU5IVFRFNjJEalFnK1FjSGxsSmRVUVpGeHFJQ3JwYmY0cGRnaC9qQT09PC9TUD4=' \
#   -H 'pragma: no-cache' \
#   -H 'priority: u=0, i' \
#   -H 'referer: https://personalmicrosoftsoftware-my.sharepoint.com/personal/pandeyk1_personalmicrosoftsoftware_uci_edu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fpandeyk1%5Fpersonalmicrosoftsoftware%5Fuci%5Fedu%2FDocuments%2FPSLD%2Fsota%2FCIFAR%2D10&ga=1' \
#   -H 'sec-ch-ua: "Not(A:Brand";v="99", "Google Chrome";v="133", "Chromium";v="133"' \
#   -H 'sec-ch-ua-mobile: ?0' \
#   -H 'sec-ch-ua-platform: "macOS"' \
#   -H 'sec-fetch-dest: iframe' \
#   -H 'sec-fetch-mode: navigate' \
#   -H 'sec-fetch-site: same-origin' \
#   -H 'sec-fetch-user: ?1' \
#   -H 'service-worker-navigation-preload: {"supportsFeatures":[1855,61313]}' \
#   -H 'upgrade-insecure-requests: 1' \
#   -H 'user-agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36' \
#   -o cifar10_55m.ckpt 


# !conda create -n newCondaEnvironment python=3.8 -y
# !source /opt/conda/bin/activate newCondaEnvironment && conda install python -y
# !sudo rm /opt/conda/bin/python3
# !sudo ln -sf /opt/conda/envs/newCondaEnvironment/bin/python3 /opt/conda/bin/python3




# !pip install hydra-core --upgrade  pytorch-lightning==1.8.4   torchdiffeq   pipenv "numpy<1.24"   #torchvision  
!pip install torch==1.13.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html
!pip install torchvision==0.14.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html
!pip install  torchdiffeq==0.2.3 torch-fidelity==0.3.0 torchmetrics==0.11.0  tqdm==4.64.1 wandb==0.13.5 scipy==1.9.3 scikit-learn==1.1.3 Pillow==9.3.0 numpy==1.23.5 matplotlib==3.6.2 hydra-core==1.2.0 pytorch-lightning==1.9.0 ninja==1.11.1
# print("Installed packages. Continuing to load dependencies:")
# !pipenv install 


!python main/train_sde.py +dataset=cifar100/cifar100_psld \
     dataset.diffusion.data.root='/kaggle/working/PSLD/outputs' \
     dataset.diffusion.data.norm=True  \
     dataset.diffusion.data.hflip=True  \
     dataset.diffusion.model.score_fn.in_ch=6 \
     dataset.diffusion.model.score_fn.out_ch=6 \
     dataset.diffusion.model.score_fn.nf=128 \
     dataset.diffusion.model.score_fn.ch_mult=[2,2,2] \
     dataset.diffusion.model.score_fn.num_res_blocks=4 \
     dataset.diffusion.model.score_fn.attn_resolutions=[16] \
     dataset.diffusion.model.score_fn.dropout=0.15 \
     dataset.diffusion.model.score_fn.progressive_input='residual' \
     dataset.diffusion.model.score_fn.fir=True \
     dataset.diffusion.model.score_fn.embedding_type='fourier' \
     dataset.diffusion.model.sde.beta_min=8.0 \
     dataset.diffusion.model.sde.beta_max=8.0 \
     dataset.diffusion.model.sde.decomp_mode='lower' \
     dataset.diffusion.model.sde.nu=4.01 \
     dataset.diffusion.model.sde.gamma=0.01 \
     dataset.diffusion.model.sde.kappa=0.04 \
     dataset.diffusion.training.seed=0 \
     dataset.diffusion.training.mode='hsm' \
     dataset.diffusion.training.fp16=False \
     dataset.diffusion.training.use_ema=True \
     dataset.diffusion.training.batch_size=64 \
     dataset.diffusion.training.accelerator='gpu' \
     dataset.diffusion.training.workers=1 \
     dataset.diffusion.training.restore_path="/kaggle/working/PSLD/checkpoints/psld--epoch\=2089-loss\=0.0042.ckpt" \
     dataset.diffusion.training.chkpt_interval=5 \
     dataset.diffusion.training.epochs=2101 \
     dataset.diffusion.training.devices=2 \
     dataset.diffusion.training.results_dir='./logs' 



# !python main/eval/sample.py +dataset=cifar10/cifar10_psld \
#                      dataset.diffusion.model.score_fn.in_ch=6 \
#                      dataset.diffusion.model.score_fn.out_ch=6 \
#                      dataset.diffusion.model.score_fn.nf=128 \
#                      dataset.diffusion.model.score_fn.ch_mult=[2,2,2] \
#                      dataset.diffusion.model.score_fn.num_res_blocks=8 \
#                      dataset.diffusion.model.score_fn.attn_resolutions=[16] \
#                      dataset.diffusion.model.score_fn.dropout=0.15 \
#                      dataset.diffusion.model.score_fn.progressive_input='residual' \
#                      dataset.diffusion.model.score_fn.fir=True \
#                      dataset.diffusion.model.score_fn.embedding_type='fourier' \
#                      dataset.diffusion.model.sde.beta_min=8.0 \
#                      dataset.diffusion.model.sde.beta_max=8.0 \
#                      dataset.diffusion.model.sde.nu=4.01 \
#                      dataset.diffusion.model.sde.gamma=0.01 \
#                      dataset.diffusion.model.sde.kappa=0.04 \
#                      dataset.diffusion.model.sde.decomp_mode='lower' \
#                      dataset.diffusion.evaluation.seed=0 \
#                      dataset.diffusion.evaluation.sample_prefix='gpu' \
#                      dataset.diffusion.evaluation.devices=2 \
#                      dataset.diffusion.evaluation.save_path='../outputs/' \
#                      dataset.diffusion.evaluation.batch_size=16 \
#                      dataset.diffusion.evaluation.stride_type='uniform' \
#                      dataset.diffusion.evaluation.sample_from='target' \
#                      dataset.diffusion.evaluation.workers=1 \
#                      dataset.diffusion.evaluation.chkpt_path='../cifar10_55m.ckpt' \
#                      dataset.diffusion.evaluation.sampler.name="em_sde" \
#                      dataset.diffusion.evaluation.n_samples=50000 \
#                      dataset.diffusion.evaluation.n_discrete_steps=50 \
#                      dataset.diffusion.evaluation.path_prefix="50"



# !python main/eval/sample.py +dataset=cifar100/cifar100_psld \
#                      dataset.diffusion.model.score_fn.in_ch=6 \
#                      dataset.diffusion.model.score_fn.out_ch=6 \
#                      dataset.diffusion.model.score_fn.nf=128 \
#                      dataset.diffusion.model.score_fn.ch_mult=[2,2,2] \
#                      dataset.diffusion.model.score_fn.num_res_blocks=4 \
#                      dataset.diffusion.model.score_fn.attn_resolutions=[16] \
#                      dataset.diffusion.model.score_fn.dropout=0.15 \
#                      dataset.diffusion.model.score_fn.progressive_input='Residual' \
#                      dataset.diffusion.model.score_fn.fir=True \
#                      dataset.diffusion.model.score_fn.embedding_type='Fourier' \
#                      dataset.diffusion.model.sde.beta_min=8.0 \
#                      dataset.diffusion.model.sde.beta_max=8.0 \
#                      dataset.diffusion.model.sde.nu=4.01 \
#                      dataset.diffusion.model.sde.gamma=0.01 \
#                      dataset.diffusion.model.sde.kappa=0.04 \
#                      dataset.diffusion.model.sde.decomp_mode='lower' \
#                      dataset.diffusion.evaluation.seed=0 \
#                      dataset.diffusion.evaluation.sample_prefix='test' \
#                      dataset.diffusion.evaluation.devices=2 \
#                      dataset.diffusion.evaluation.save_path="./outputs" \
#                      dataset.diffusion.evaluation.batch_size=512 \
#                      dataset.diffusion.evaluation.stride_type='uniform' \
#                      dataset.diffusion.evaluation.sample_from='target' \
#                      dataset.diffusion.evaluation.workers=1 \
#                      dataset.diffusion.evaluation.chkpt_path="/kaggle/working/PSLD/psld_hsm_gamma\=0.01_nu\=4.01_cifar10_continuous_sfn\=ncsnpp_nparams\=55M.ckpt" \
#                      dataset.diffusion.evaluation.sampler.name="em_sde" \
#                      dataset.diffusion.evaluation.n_samples=50000 \
#                      dataset.diffusion.evaluation.n_discrete_steps=50 \
#                      dataset.diffusion.evaluation.path_prefix="50"


# !python main/eval/sample.py +dataset=cifar10/cifar10_psld \
#                      dataset.diffusion.data.root='/kaggle/working/PSLD/outputs' \
#                      dataset.diffusion.data.name='cifar10' \
#                      dataset.diffusion.data.norm=True \
#                      dataset.diffusion.data.hflip=True \
#                      dataset.diffusion.data.return_target=True \
#                      dataset.diffusion.model.score_fn.in_ch=6 \
#                      dataset.diffusion.model.score_fn.out_ch=6 \
#                      dataset.diffusion.model.score_fn.nf=128 \
#                      dataset.diffusion.model.score_fn.ch_mult=[2,2,2] \
#                      dataset.diffusion.model.score_fn.num_res_blocks=4 \
#                      dataset.diffusion.model.score_fn.attn_resolutions=[16] \
#                      dataset.diffusion.model.score_fn.dropout=0.15 \
#                      dataset.diffusion.model.score_fn.progressive_input='Residual' \
#                      dataset.diffusion.model.score_fn.fir=True \
#                      dataset.diffusion.model.score_fn.embedding_type='Fourier' \
#                      dataset.diffusion.model.sde.beta_min=8.0 \
#                      dataset.diffusion.model.sde.beta_max=8.0 \
#                      dataset.diffusion.model.sde.nu=4.01 \
#                      dataset.diffusion.model.sde.gamma=0.01 \
#                      dataset.diffusion.model.sde.kappa=0.04 \
#                      dataset.diffusion.model.sde.decomp_mode='lower' \
#                      dataset.diffusion.evaluation.seed=0 \
#                      dataset.diffusion.evaluation.sample_prefix='test' \
#                      dataset.diffusion.evaluation.devices=2 \
#                      dataset.diffusion.evaluation.save_path="./outputs" \
#                      dataset.diffusion.evaluation.batch_size=512 \
#                      dataset.diffusion.evaluation.stride_type='uniform' \
#                      dataset.diffusion.evaluation.sample_from='target' \
#                      dataset.diffusion.evaluation.workers=1 \
#                      dataset.diffusion.evaluation.chkpt_path="/kaggle/working/PSLD/psld_hsm_gamma\=0.01_nu\=4.01_cifar10_continuous_sfn\=ncsnpp_nparams\=55M.ckpt" \
#                      dataset.diffusion.evaluation.sampler.name="em_sde" \
#                      dataset.diffusion.evaluation.n_samples=5000 \
#                      dataset.diffusion.evaluation.n_discrete_steps=50 \
#                      dataset.diffusion.evaluation.path_prefix="50"


%cd /kaggle/working/PSLD
!pwd





!pip install torch-fidelity


# !fidelity --gpu 0 --isc --input1 /kaggle/working/PSLD/outputs/2025-03-26/10-52-58/outputs/50/images


# !fidelity --gpu 0 --isc --input1 /kaggle/working/PSLD/outputs/2025-03-26/10-12-49/outputs/50/images


# !fidelity --gpu 0 --fid --input1 /kaggle/working/PSLD/outputs/cifar10_test_images --input2 /kaggle/working/PSLD/outputs/2025-03-26/10-52-58/outputs/50/images


# import os
# import numpy as np
# from torchvision import datasets
# from PIL import Image
# from tqdm import tqdm

# # Set directory to save test images
# save_dir = "/kaggle/working/PSLD/outputs/cifar10_test_images"
# os.makedirs(save_dir, exist_ok=True)

# # Download only the test dataset (train=False)
# dataset = datasets.CIFAR10(root="./data", train=False, download=True)
# data, labels = dataset.data, dataset.targets  # Extract images and labels

# # Save each test image as PNG
# for idx, img_array in tqdm(enumerate(data), total=len(data), desc="Saving Test Images"):
#     img = Image.fromarray(img_array)  # Convert NumPy array to PIL image
#     img.save(os.path.join(save_dir, f"test_image_{idx}.png"), format="PNG")

# print(f"All test images saved in: {save_dir}")






# ! fidelity --gpu 0 --fid \

