# If the models are added as the input, Kaggle notebook will download them automatically before running user code
# walk the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# This model is quantized and needs bitsandbytes as well as a supported nvidia GPU (cuda)
# even for inference.

!pip install bitsandbytes


from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


# Path to the directory where the checkpoint is saved or the model is downloaded

# model_name_or_path='/kaggle/input/versedgemma-v0.2-unsloth-10ep/transformers/v1/1/'    # based on "unsloth/gemma-2b-bnb-4bit"
model_name_or_path='/kaggle/input/versedgemma-v0.3-unsloth-10ep/transformers/v1/1/'    # based on "unsloth/gemma-2-2b-bnb-4bit"
# The versedgemma models need cuda since it is quantized with bitsandbytes 4bit.
# Both Kaggle T4 and olde P100 are fine
device='cuda'

# Load the model and tokenizer from the checkpoint
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path = model_name_or_path,
).to(device)


config = AutoConfig.from_pretrained(
    pretrained_model_name_or_path = model_name_or_path,
)


# For these experiments we didn't train or save the tokenizer again, they are readily available from the base

pretrained_model_base = None
if "gemma2" == config.model_type \
    and 2304 == config.hidden_size \
    and 9216 == config.intermediate_size \
    and 26 == config.num_hidden_layers:
    pretrained_model_base = "unsloth/gemma-2-2b-bnb-4bit"
if "gemma" == config.model_type \
    and 2048 == config.hidden_size \
    and 16384 == config.intermediate_size \
    and 18 == config.num_hidden_layers:
    pretrained_model_base = "unsloth/gemma-2b-bnb-4bit"

pretrained_model_base


tokenizer = None
if pretrained_model_base:
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_base)
else:
    print(f'no pretrained_model_base found')


# try following 李白 別中都明府兄 
# line 7 of https://ctext.org/wiki.pl?if=gb&chapter=622006), pin yin can be found by clicking at #7 header

input_tokens = tokenizer(
[
    "吾wu2- 兄xiong1- 詩shi1^ 酒jiu3^ 繼ji4^ 陶tao2- 君jun1-，"
]*1, return_tensors = "pt").to(device)

output_tokens = model.generate(**input_tokens, max_new_tokens = 256, use_cache = True)
tokenizer.batch_decode(output_tokens)


# We can be creative here, for example
# 1. generate any text segment, preferrable 7 chinese glyph, 
# 2. use pypinyin to find the pin yin
# 3. assign the flat-oblique style use pin yin
# 4. watch the poems the model likes to generate :-)
# 5. verify the format and score it 
#    based on matching of pin yin, in-rhymeness, matching of flat-oblique
#
# "If you thoroughly studied just 300 Tang poems. It cannot make you a poet. 
# However, you'll at least be able to sing out poems (doggerel) naturally."

# this work is licensed under ASPLv2 for Kaggle community
# read model and base licenses before use of them and publish articles. :-)


# This is a simplied one without actually specifying the flat-oblique style rule
# instead it is using existing pin yin aof flat-oblique notation
poem_start = '故gu4^ 人ren2- 西xi- 辭ci- 黃huang2- 鶴he4^ 樓lou2-'

input_tokens = tokenizer(
[
    poem_start
]*1, return_tensors = "pt").to(device)

output_tokens = model.generate(**input_tokens, max_new_tokens = 256, use_cache = True)
tokenizer.batch_decode(output_tokens)

