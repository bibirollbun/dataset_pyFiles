# Kaggle-ready EDA + submission pipeline for CURE-Bench
# - Pins scientific stack to avoid NumPy/SciPy ABI issues
# - Optional: auto-download competition files via Kaggle CLI (Phase 2 preferred)
# - Safe: does NOT auto-install vLLM to prevent upgrading NumPy/SciPy again
# Usage:
# 1) Start a fresh Kaggle Notebook (GPU optional). If you already installed conflicting packages, restart session.
# 2) Run this cell once. If you need vLLM later, install it in a SEPARATE cell and then RE-pin numpy/scipy and restart.

import os
import sys
import subprocess
import logging
from pathlib import Path

def _pip_install(packages, extra_args=None):
    cmd = [sys.executable, "-m", "pip", "install", "-qU"] + packages
    if extra_args:
        cmd += extra_args
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        print(f"[pip] Failed to install {packages}: {e}")

# 0) Install stable scientific stack BEFORE imports
# Notes:
# - numpy==1.26.4 + scipy==1.11.4 is a well-known stable pair on Kaggle
# - pyarrow==19.0.0 plays nicer with existing RAPIDS/cuDF in Kaggle images
CORE_PKGS = [
    "numpy==1.26.4",
    "scipy==1.11.4",
    "pandas==2.2.2",
    "seaborn==0.13.2",
    "pydantic==2.11.4",
    "pyarrow==19.0.0",
    "numba==0.60.0",
    "wordcloud",
    "jsonlines",
    "tqdm",
    "nest-asyncio",
    "matplotlib==3.8.4",  # compatible with seaborn 0.13.x
]
_pip_install(CORE_PKGS)

# IMPORTANT:
# Do NOT auto-install vLLM/transformers here. They may upgrade numpy/scipy.
# If you really need vLLM:
#   1) In a separate cell: !pip install -qU vllm transformers accelerate bitsandbytes
#   2) Then re-pin:        !pip install -qU "numpy==1.26.4" "scipy==1.11.4"
#   3) Restart session     (Runtime -> Restart session), then run this script again.

# 1) Imports after installs
import asyncio
import zipfile
import re
import warnings
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import jsonlines

from pydantic import BaseModel
from tqdm import tqdm

# Notebook helpers
import nest_asyncio
nest_asyncio.apply()

# vLLM optional imports: we do NOT install here; just try to import if user installed separately
try:
    from vllm import AsyncLLM, SamplingParams
    from transformers import AutoTokenizer
    HAS_VLLM = True
except Exception:
    HAS_VLLM = False

# 2) Logging
def setup_logging(level=logging.INFO):
    logger = logging.getLogger("curebench")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger

logger = setup_logging()

# 3) Plot style
warnings.filterwarnings("ignore")
sns.set_style("whitegrid", {"axes.facecolor": "#f2f2f2", "figure.facecolor": "#f2f2f2"})
plt.rcParams["figure.figsize"] = (16, 9)
plt.rcParams["axes.titlesize"] = 20
plt.rcParams["axes.labelsize"] = 14

# 4) Config
class Config(BaseModel):
    # Will be updated dynamically to Phase 2 if the file exists
    TEST_FILE: str = "/kaggle/input/cure-bench/curebench_testset_phase1.jsonl"
    OUTPUT_DIR: Path = Path("/kaggle/working/")
    MODEL_PATH: str = "/kaggle/input/deepseek-coder-v2-lite-instruct/DeepSeek-Coder-V2-Lite-Instruct-AWQ"

    # vLLM params (used only if HAS_VLLM and model path is valid)
    TENSOR_PARALLEL_SIZE: int = 1
    GPU_MEMORY_UTILIZATION: float = 0.90

    # Generation params
    TEMPERATURE: float = 0.01
    MAX_TOKENS: int = 512
    TOP_P: float = 0.95

    # Debug
    DEBUG_MODE: bool = True
    DEBUG_SAMPLES: int = 5

def print_versions():
    import numpy as _np, scipy as _sp, seaborn as _sns, pandas as _pd
    logger.info(f"Versions -> numpy={_np.__version__}, scipy={_sp.__version__}, seaborn={_sns.__version__}, pandas={_pd.__version__}")
    if not HAS_VLLM:
        logger.warning("vLLM not available. The pipeline will fall back to dummy predictions.")
    else:
        logger.info("vLLM is available (ensure you did not upgrade numpy/scipy inadvertently).")

# 5) Kaggle CLI helpers to fetch data
def kaggle_download_competition(competition: str = "cure-bench", dest: Path = Path("/kaggle/working")):
    """Download competition files to dest using Kaggle CLI and unzip them."""
    dest.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.check_call([
            "kaggle", "competitions", "download", "-c", competition, "-p", str(dest), "-q"
        ])
    except FileNotFoundError:
        logger.error("Kaggle CLI not found. In Kaggle notebooks it should exist. If running elsewhere, install and authenticate.")
        return
    except subprocess.CalledProcessError as e:
        logger.error(f"Kaggle download failed: {e}")
        return

    # Unzip all zip files downloaded
    for zfile in dest.glob("*.zip"):
        try:
            with zipfile.ZipFile(zfile, "r") as zf:
                zf.extractall(dest)
            logger.info(f"Unzipped: {zfile}")
        except zipfile.BadZipFile:
            logger.error(f"Bad zip file: {zfile}")

def auto_select_test_file(cfg: Config) -> str:
    """Prefer Phase 2 if available, else Phase 1, else the configured default."""
    candidates = [
        "/kaggle/working/curebench_testset_phase2.jsonl",
        "/kaggle/input/cure-bench/curebench_testset_phase2.jsonl",
        "/kaggle/working/curebench_testset_phase1.jsonl",
        "/kaggle/input/cure-bench/curebench_testset_phase1.jsonl",
        cfg.TEST_FILE,
    ]
    for p in candidates:
        if Path(p).exists():
            return p
    return cfg.TEST_FILE

# 6) IO utilities
def load_data(file_path: str) -> pd.DataFrame:
    try:
        with jsonlines.open(file_path) as reader:
            data = list(reader)
        logger.info(f"Loaded {len(data)} records from {file_path}")
        return pd.DataFrame(data)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return pd.DataFrame()

# 7) EDA
def run_eda(df: pd.DataFrame):
    logger.info("Starting EDA...")
    if df.empty:
        logger.warning("DataFrame is empty. Skipping EDA.")
        return

    expected_cols = {"id", "question", "options"}
    missing = expected_cols - set(df.columns)
    if missing:
        logger.warning(f"Missing expected columns for EDA: {missing}. EDA will be partial.")

    # Plot 1: question_type distribution
    if "question_type" in df.columns:
        plt.figure(figsize=(12, 7))
        ax = sns.countplot(
            y=df["question_type"],
            order=df["question_type"].value_counts().index,
            palette="viridis",
            hue=df["question_type"],
            dodge=False,
            legend=False,
        )
        total = len(df)
        for p in ax.patches:
            percentage = f"{100 * p.get_width() / total:.1f}%"
            x = p.get_width() + 10
            y = p.get_y() + p.get_height() / 2
            ax.annotate(f"{int(p.get_width())} ({percentage})", (x, y), ha="left", va="center", fontsize=12)
        plt.title("Phân Phối Các Loại Câu Hỏi", fontsize=20, fontweight="bold")
        plt.xlabel("Số Lượng", fontsize=14)
        plt.ylabel("Loại Câu Hỏi", fontsize=14)
        plt.xlim(0, ax.get_xlim()[1] * 1.1)
        sns.despine(left=True, bottom=True)
        plt.show()
    else:
        logger.info("Column 'question_type' not found. Skipping that plot.")

    # Plot 2: question length
    if "question" in df.columns:
        df = df.copy()
        df["question_length"] = df["question"].apply(lambda x: len(str(x)))
        plt.figure(figsize=(16, 7))
        sns.histplot(df["question_length"], bins=50, kde=True, color="darkcyan", line_kws={"linewidth": 3})
        mean_len = df["question_length"].mean()
        plt.axvline(mean_len, color="red", linestyle="--", linewidth=2, label=f"Độ dài trung bình: {mean_len:.0f}")
        plt.title("Phân Phối Độ Dài Câu Hỏi", fontsize=20, fontweight="bold")
        plt.xlabel("Độ Dài (số ký tự)", fontsize=14)
        plt.ylabel("Tần suất", fontsize=14)
        plt.legend()
        sns.despine()
        plt.show()

    # Plot 3: number of options
    if "options" in df.columns:
        df = df.copy()
        df["num_options"] = df["options"].apply(lambda x: len(x) if isinstance(x, dict) else 0)
        plt.figure(figsize=(12, 7))
        ax = sns.countplot(x=df["num_options"], palette="plasma", hue=df["num_options"], legend=False)
        total = len(df)
        for p in ax.patches:
            height = p.get_height()
            if height > 0:
                percentage = f"{100 * height / total:.1f}%"
                ax.annotate(
                    f"{int(height)}\n({percentage})",
                    (p.get_x() + p.get_width() / 2.0, height),
                    ha="center",
                    va="bottom",
                    fontsize=12,
                    xytext=(0, 5),
                    textcoords="offset points",
                )
        plt.title("Số Lượng Lựa Chọn (Options) Cho Mỗi Câu Hỏi", fontsize=20, fontweight="bold")
        plt.xlabel("Số Lựa Chọn", fontsize=14)
        plt.ylabel("Số Lượng Câu Hỏi", fontsize=14)
        plt.ylim(0, ax.get_ylim()[1] * 1.1)
        sns.despine()
        plt.show()

    # Plot 4: Word cloud
    if "question" in df.columns:
        logger.info("Generating Word Cloud...")
        custom_stopwords = set(STOPWORDS)
        custom_stopwords.update(
            [
                "patient", "mg", "day", "week", "treatment", "disease", "drug", "therapy",
                "study", "group", "risk", "effect", "symptoms", "associated",
                "which", "what", "following", "recommended", "should",
            ]
        )
        text = " ".join(df["question"].astype(str).tolist())
        if text.strip():
            wordcloud = WordCloud(
                width=1600, height=800, background_color="white", colormap="viridis", stopwords=custom_stopwords
            ).generate(text)
            plt.figure(figsize=(20, 10))
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
            plt.title("Các Từ Khóa Phổ Biến Trong Câu Hỏi (Đã lọc từ dừng y tế)", fontsize=24, fontweight="bold")
            plt.show()
        else:
            logger.info("Empty text for wordcloud. Skipping.")

# 8) LLM prompt + processor
class MedicalPromptEngine:
    def create_prompt(self, item: Dict) -> str:
        question = item.get("question", "")
        options = item.get("options", {})

        prompt = (
            "You are an expert medical AI. Analyze the following question and provide a step-by-step reasoning. "
            "Conclude with your final answer in the format 'ANSWER: [Letter]'.\n\n"
        )
        prompt += f"Question: {question}\n\n"
        if options and isinstance(options, dict):
            prompt += "Options:\n"
            for key, value in options.items():
                prompt += f"{key}. {value}\n"
        prompt += "\nReasoning and Answer:"
        return prompt

class CureBenchProcessor:
    def __init__(self, config: Config):
        self.config = config
        self.llm: Optional["AsyncLLM"] = None
        self.tokenizer = None
        self.prompt_engine = MedicalPromptEngine()

        if HAS_VLLM and Path(config.MODEL_PATH).exists():
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH, trust_remote_code=True)
                self.llm = AsyncLLM(
                    model=config.MODEL_PATH,
                    tensor_parallel_size=config.TENSOR_PARALLEL_SIZE,
                    gpu_memory_utilization=config.GPU_MEMORY_UTILIZATION,
                    quantization="awq",
                    trust_remote_code=True,
                    disable_log_stats=True,
                )
                logger.info("Initialized vLLM with provided model path.")
            except Exception as e:
                logger.error(f"Failed to initialize vLLM. Falling back to dummy predictions. Error: {e}")
                self.llm = None
        else:
            if not HAS_VLLM:
                logger.warning("vLLM not available. Falling back to dummy predictions.")
            else:
                logger.error(f"Model path not found: {config.MODEL_PATH}. Falling back to dummy predictions.")

    def _parse_answer(self, response_text: str) -> str:
        match = re.search(r"ANSWER:\s*([A-D])", response_text.upper())
        if match:
            return match.group(1)
        matches = re.findall(r"\b([A-D])\b", response_text.upper())
        if matches:
            return matches[-1]
        return "A"

    async def process_batch(self, batch_data: List[Dict]) -> List[Dict]:
        if not self.llm:
            return [{"id": item.get("id"), "prediction": "A"} for item in batch_data]

        prompts = [self.prompt_engine.create_prompt(item) for item in batch_data]
        sampling_params = SamplingParams(
            temperature=self.config.TEMPERATURE,
            max_tokens=self.config.MAX_TOKENS,
            top_p=self.config.TOP_P,
        )
        request_outputs = await self.llm.generate(prompts, sampling_params, use_tqdm=False)
        results = []
        for i, output in enumerate(request_outputs):
            try:
                text = output.outputs[0].text if output.outputs else ""
            except Exception:
                text = ""
            results.append({"id": batch_data[i].get("id"), "prediction": self._parse_answer(text)})
        return results

    def create_submission(self, all_predictions: List[Dict], test_file: str):
        if not all_predictions:
            logger.warning("No predictions to save. Skipping submission creation.")
            return

        submission_df = pd.DataFrame(all_predictions)
        test_df = load_data(test_file)
        if test_df.empty:
            logger.error("Test file is empty or missing. Cannot create submission.")
            return

        if self.config.DEBUG_MODE:
            test_df = test_df.head(self.config.DEBUG_SAMPLES)

        final_df = pd.merge(test_df[["id"]], submission_df, on="id", how="left")
        final_df["prediction"] = final_df["prediction"].fillna("A")

        self.config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        csv_path = self.config.OUTPUT_DIR / "submission.csv"
        zip_path = self.config.OUTPUT_DIR / "submission.zip"

        final_df[["id", "prediction"]].to_csv(csv_path, index=False)
        logger.info(f"Saved CSV to: {csv_path}")

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(csv_path, arcname="submission.csv")
        logger.info(f"✅ Created ZIP submission: {zip_path}")
        try:
            print(final_df.head())
        except Exception:
            pass

# 9) Main
async def main():
    config = Config()
    logger.info(f"Config:\n{config.model_dump_json(indent=2)}")
    print_versions()

    # Try to ensure data is available via Kaggle CLI (optional).
    # If you've already added the dataset via "Add Data", you can comment these out.
    kaggle_download_competition("cure-bench", Path("/kaggle/working"))

    # Prefer Phase 2 if present
    config.TEST_FILE = auto_select_test_file(config)
    logger.info(f"Using TEST_FILE: {config.TEST_FILE}")

    if not Path(config.TEST_FILE).exists():
        logger.error(f"Test file not found: {config.TEST_FILE}")
        logger.error("Please Add Data (competition files) or ensure Kaggle CLI download succeeded.")
        return

    # Load data
    df = load_data(config.TEST_FILE)
    if df.empty:
        logger.error("Empty dataframe after loading test file. Exiting.")
        return

    # EDA
    run_eda(df)

    # Modeling
    logger.info("Proceeding to Modeling (Submission generation)...")
    processor = CureBenchProcessor(config)

    if config.DEBUG_MODE:
        logger.warning(f"--- DEBUG MODE ON: processing only {config.DEBUG_SAMPLES} samples ---")
        data_to_process = df.head(config.DEBUG_SAMPLES).to_dict("records")
    else:
        data_to_process = df.to_dict("records")

    all_predictions: List[Dict] = []
    batch_size = 8
    for i in tqdm(range(0, len(data_to_process), batch_size), desc="Xử lý câu hỏi"):
        batch_data = data_to_process[i : i + batch_size]
        batch_results = await processor.process_batch(batch_data)
        all_predictions.extend(batch_results)

    processor.create_submission(all_predictions, test_file=config.TEST_FILE)

def _run():
    try:
        from IPython import get_ipython
        shell = get_ipython()
        if shell and "IPKernelApp" in shell.config:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return loop.create_task(main())
            else:
                loop.run_until_complete(main())
        else:
            asyncio.run(main())
    except Exception as e:
        logger.error(f"Failed to run main(): {e}")

if __name__ == "__main__":
    _run()

