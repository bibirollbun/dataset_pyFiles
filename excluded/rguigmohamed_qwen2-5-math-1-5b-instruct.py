import os
import gc
import time
import warnings

import pandas as pd
import polars as pl

import torch
import kaggle_evaluation.aimo_2_inference_server

pd.set_option('display.max_colwidth', None)
cutoff_time = time.time() + (4 * 60 + 30) * 60


# 1) CrÃ©e le dossier de config
!mkdir -p ~/.config/kaggle

# 2) Copie le token depuis l'input en lecture seule
!cp /kaggle/input/kaggle-token-1/kaggle.json ~/.config/kaggle/kaggle.json

# 3) SÃ©curise les permissions
!chmod 600 ~/.config/kaggle/kaggle.json


%%bash
cat > /kaggle/working/dataset-metadata.json << 'EOF'
{
  "title":        "Parsed CV Results",
  "id":           "rguigmohamed/save-parsing",
  "licenses": [
    {
      "name": "CC0-1.0"
    }
  ]
}
EOF



# 1. Installe et configure ton token (si ce n'est dÃ©jÃ  fait)
# !pip install kaggle --quiet

# 2. Test de connexion Ã  l'API Kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
import sys

api = KaggleApi()
try:
    api.authenticate()
except Exception as e:
    sys.exit(f"â�Œ Ã‰chec d'authentification Kaggle API : {e!s}")

# 3. Fais un appel simple, ici on liste tes datasets (max 3)
try:
    ds_list = api.datasets_list(user="YOUR_KAGGLE_USERNAME", page_size=3)
    print("âœ… Kaggle API OK, voici tes 3 premiers datasets :")
    for ds in ds_list:
        print(f" - {ds.ref}")
except Exception as e:
    sys.exit(f"â�Œ Ã‰chec de la requÃªte Kaggle API : {e!s}")



import vllm
print("vLLM version:", vllm.__version__)


from vllm import LLM, SamplingParams

warnings.simplefilter('ignore')

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def clean_memory(deep=False):
    gc.collect()
    if deep:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    torch.cuda.empty_cache()


llm_model_pth = '/kaggle/input/qwen2.5/transformers/14b-instruct-awq/1'

llm = LLM(
    llm_model_pth,
    max_num_seqs=16,
    max_model_len=8192,          # Model context length
    trust_remote_code=True,      # Trust remote code (e.g., from HuggingFace) when downloading the model and tokenizer
    tensor_parallel_size=4,      # The number of GPUs to use for distributed execution with tensor parallelism
    gpu_memory_utilization=0.95, # The ratio (between 0 and 1) of GPU memory to reserve for the model
    seed=2024,
)


# ------------------ CONFIG ------------------ #
CSV_PATH   = "/kaggle/input/parsing-cv-split-2/cv_split_part2.csv"
MODEL_PATH = "/kaggle/input/qwen2.5/transformers/14b-instruct-awq/1"
MAX_CTX    = 7000      # tokens du CV (4096-~500 marge)

# -------------- LIBS & MODEL ---------------- #
import pandas as pd, json, time
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from pydantic import BaseModel, ValidationError
from typing import List, Optional

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)



tokenizer = llm.get_tokenizer()


# -------------- SCHEMA ---------------------- #
class Skill(BaseModel):
    name: str
    years_experience: Optional[float] = None

class Experience(BaseModel):
    company: str
    title: str
    very_short_summary_of_experience: str
    duration_of_experience: Optional[float] = None
    

class CVParsed(BaseModel):
    summary_of_CV_generated_by_the_ai_model: str
    current_title: Optional[str]
    total_years_experience: Optional[float]
    location: Optional[str]
    skills: List[Skill]
    experiences: List[Experience]
    industries: Optional[List[str]]
    highest_degree: Optional[str]
    degrees: Optional[List[str]]
    certifications: Optional[List[str]]
    languages: Optional[List[str]]

schema_json = json.dumps(CVParsed.model_json_schema(), indent=2)

# -------------- LOAD 10 CV ------------------ #
df = pd.read_csv(CSV_PATH, nrows=20)
cvs = df["cv_text"].astype(str).tolist()

def truncate_to_ctx(text, max_tokens=7000):
    ids = tokenizer.encode(text)
    if len(ids) > max_tokens:
        ids = ids[:max_tokens]
    return tokenizer.decode(ids)

def make_prompt(cv_text: str) -> str:
    cv_short = truncate_to_ctx(cv_text)
    return (
        "You are a helpful assistant.\n"
        "Your task is to extract structured data from the resume below.\n"
        "Return ONLY a valid JSON object strictly matching the schema below.\n"
        "Do not add explanations, comments, or markdown.\n\n"
        f"Schema:\n{schema_json}\n\n"
        "Resume:\n"
        f"{cv_short}\n\n"
        "JSON:\n"
    )


prompts = [{"prompt": make_prompt(t)} for t in cvs]


CHUNK_SIZE    = 500

OUTPUT_CSV    = "/kaggle/working/parsed_cv_results.csv"
# ------------ CHARGEMENT / CHECKPOINT ----------- #
df = pd.read_csv(CSV_PATH)
DATASET_INPUT_CSV = "/kaggle/input/save-parsing/parsed_cv_results.csv"
if os.path.exists(DATASET_INPUT_CSV):
    print("ğŸ”„ Reprise depuis le Dataset Kaggle existant")
    out_df = pd.read_csv(DATASET_INPUT_CSV)
else:
    print("ğŸš€ Nouveau traitement â€“ aucun CSV trouvÃ© dans le Dataset")
    out_df = pd.DataFrame()

processed_ids = set(out_df.get("fd_ct_id", []))
to_do = df[~df["fd_ct_id"].isin(processed_ids)].reset_index(drop=True)

print(f"âš¡ {len(processed_ids)} CV dÃ©jÃ  faits, {len(to_do)} restants.")
sampling = SamplingParams(temperature=0.0, max_tokens=2048, stop=["\n\n"])
# --------------- TRAITEMENT BATCHÃ‰ -------------- #
for i in range(0, len(to_do), CHUNK_SIZE):
    batch = to_do.iloc[i : i + CHUNK_SIZE]
    cvs     = batch["cv_text"].astype(str).tolist()
    ids      = batch["fd_ct_id"].tolist()
    prompts = [{"prompt": make_prompt(cv)} for cv in cvs]

    t0 = time.time()
    outs = llm.generate(prompts, sampling)
    print(f"  â€“ chunk {i}-{i+len(cvs)} gÃ©nÃ©rÃ© en {time.time()-t0:.1f}s")

    results = []
    for cid, out in zip(ids, outs):
        parsed = {}
        parsed['parsed'] = out.outputs[0].text
        parsed["fd_ct_id"] = cid
        results.append(parsed)

    out_df = pd.concat([out_df, pd.DataFrame(results)], ignore_index=True)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"  -> sauv. intermÃ©diaire ({len(out_df)} CV totaux)\n")


    # 3) Save â€œpermanentâ€� sur Kaggle
    print(f"  -> Uploading checkpoint {i}-{i+len(batch)}â€¦")
    !kaggle datasets version \
        -p /kaggle/working \
        -m "Checkpoint CVs {i}-{i+len(batch)}" 
                                                                      

print("Tous les CV traitÃ©s â€“ rÃ©sultats dispo dans ton Dataset Kaggle !")
print("ğŸ�‰ Traitement terminÃ©. RÃ©sultats dans", OUTPUT_CSV)




