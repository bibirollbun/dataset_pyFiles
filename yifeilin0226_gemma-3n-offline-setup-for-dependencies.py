# Save the existing dependencies
!pip freeze > requirements_before.txt


# Install the required dependencies
!pip install -Uq timm 
!pip install -q accelerate
!pip install -q git+https://github.com/huggingface/transformers.git


# Save the new dependencies
!pip freeze > requirements_after.txt


# Save the added and updated dependencies only
before = set(open("requirements_before.txt").readlines())
after = set(open("requirements_after.txt").readlines())

new_or_updated = after - before
print("New or updated packages:\n")
print("".join(sorted(new_or_updated)))

with open("requirements.txt", 'w') as f:
    f.write("".join(sorted(new_or_updated)))


# Download wheels accordingly
!pip download -q -r requirements.txt -d offline_packages


# unzip the github prebuilt huggingface transformers package
!unzip -q offline_packages/transformers-4.54.0.dev0.zip -d transformers


# Build the .whl for the tranformers package
!pip install build
%cd transformers/transformers
!python -m build --wheel --outdir ../../offline_packages


%cd ../..


# remove the zip file
!rm offline_packages/transformers-4.54.0.dev0.zip


# change the transformers Git address to the wheel version name
requirements = open('requirements.txt').readlines()
prev_name = requirements[-1]
requirements[-1] = 'transformers==4.54.0.dev0'

with open('requirements.txt', 'w') as f:
    f.write(''.join(requirements))


print(f'{prev_name} changed to {requirements[-1]}')


# zip the offline packages folder
!zip -q -r offline_packages.zip offline_packages


# Install the dependencies offline
!pip install --no-index -f /kaggle/input/gemma3n-offline-packages/offline_packages/offline_packages -r /kaggle/input/gemma3n-offline-packages/requirements.txt


# load model path
import kagglehub

GEMMA_PATH = kagglehub.model_download("google/gemma-3n/transformers/gemma-3n-e2b-it")


from transformers import AutoModelForImageTextToText, AutoProcessor


# test loading the model
processor = AutoProcessor.from_pretrained(GEMMA_PATH)
model = AutoModelForImageTextToText.from_pretrained(GEMMA_PATH, torch_dtype="auto", device_map="auto")

