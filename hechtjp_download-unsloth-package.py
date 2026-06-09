# Install unsloth
!pip install unsloth


# Check imported unsloth
from unsloth import FastLanguageModel  # FastVisionModel for LLMs
import torch


%%writefile requirements.txt
unsloth


# Download library
! python -m pip download --destination-directory . -r requirements.txt




