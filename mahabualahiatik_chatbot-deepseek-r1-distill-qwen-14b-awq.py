





import os
import torch
import pandas as pd
import gc
import re
from vllm import LLM, SamplingParams
from torch.cuda.amp import autocast

# Configuration Constants
MODEL_PATH = "/kaggle/input/deepseek-r1/transformers/deepseek-r1-distill-qwen-14b-awq/1"
DATA_PATH = "/kaggle/input/wsdm-cup-multilingual-chatbot-arena/test.parquet"
BATCH_SIZE = 128  # Optimized for 2 GPUs
MAX_TOKENS = 5  # Strict output control
MAX_INPUT_TOKENS = 8192  # Hard limit for input tokenization

class PreferencePredictor:
    def __init__(self):
        self._configure_self()
        self.llm = self._initialize_model()
        self.tokenizer = self.llm.get_tokenizer()
        self._augment_tokenizer()

    def _configure_self(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # Adjusted for 2 GPUs
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        torch.set_float32_matmul_precision("high")

    def _initialize_model(self):
        return LLM(
            MODEL_PATH,
            tensor_parallel_size=2,  # Adjusted for 2 GPUs
            max_num_seqs=60,
            max_model_len=8192,
            gpu_memory_utilization=0.95,
            dtype="float16",  # Explicitly use float16 for Tesla T4
            trust_remote_code=True,
            seed=2024,
        )

    def _augment_tokenizer(self):
        special_tokens = [
            "<system>",
            "</system>",
            "<prompt>",
            "</prompt>",
            "<response_a>",
            "</response_a>",
            "<response_b>",
            "</response_b>",
        ]
        self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Vectorized text cleaning"""
        return df.map(lambda x: x.replace("null", "") if isinstance(x, str) else x)

    def _format_prompt(self, row) -> str:
        """Formats the prompt with truncation to avoid exceeding token limits"""
        system_msg = """You are an impartial evaluator tasked with judging the quality of responses provided by two AI assistants to the same user question. Your goal is to determine which assistant's response is superior based on the following criteria:

1. **Helpfulness**: How well does the response address the user's needs and answer their question?
2. **Relevance**: Does the response directly relate to the user's question without going off-topic?
3. **Accuracy**: Is the information provided correct and reliable?
4. **Depth**: Does the response provide a thorough and detailed explanation or solution?
5. **Clarity**: Is the response easy to understand, well-structured, and free of ambiguity?
6. **Creativity**: Does the response demonstrate originality or offer unique insights (if applicable)?
Your evaluation should be unbiased, and the order in which the responses are presented must not affect your decision. Avoid letting response length, specific assistant names, or stylistic preferences influence your judgment.

Strict Rules:
- Respond ONLY with "model_a" or "model_b".
- NO explanations, punctuation, or extra words.

Final verdict:"""

        prompt_text = f"""<system>{system_msg}</system>
<prompt>{row['prompt']}</prompt>
<response_a>{row['response_a']}</response_a>
<response_b>{row['response_b']}</response_b>
Final verdict:"""

        # Ensure token limit is not exceeded
        tokenized = self.tokenizer(
            prompt_text, truncation=True, max_length=MAX_INPUT_TOKENS, return_tensors="pt"
        )
        return self.tokenizer.decode(tokenized["input_ids"][0], skip_special_tokens=True)

    @torch.inference_mode()
    def predict(self, test_df: pd.DataFrame) -> pd.DataFrame:
        torch.cuda.synchronize()  # Ensure GPU readiness
        test_df = self._preprocess_data(test_df)
        prompts = test_df.apply(self._format_prompt, axis=1).tolist()

        sampling_params = SamplingParams(
            temperature=0.6,  # Keep deterministic for consistency
            max_tokens=MAX_TOKENS,
            stop=["</s>", "\n"],
            top_p=1.0,
            frequency_penalty=0.4,
            best_of=3,
        )

        outputs = []
        for i in range(0, len(prompts), BATCH_SIZE):
            batch = prompts[i : i + BATCH_SIZE]

            # ðŸš€ Enable autocast for mixed precision inference
            with torch.amp.autocast("cuda"):
                results = self.llm.generate(batch, sampling_params)

            outputs.extend([self._parse_output(r.outputs[0].text) for r in results])

            # Efficient memory management
            if i % (BATCH_SIZE * 4) == 0:
                torch.cuda.empty_cache()
                gc.collect()

        return pd.DataFrame({"id": test_df["id"], "winner": outputs})

    def _parse_output(self, text: str) -> str:
        """Parses output strictly and ensures valid model selection"""
        text = text.lower().strip()  # Normalize case and remove spaces

        # Direct strict match
        if text in {"model_a", "model_b"}:
            return text

        # Strict regex match (ensures valid extraction)
        match = re.search(r"\b(model_a|model_b)\b", text)
        if match:
            return match.group(1)

        # Final fallback (if no clear winner is found, default to model_a)
        return "model_a"  # Ensures consistency in ambiguous cases


if __name__ == "__main__":
    predictor = PreferencePredictor()
    test_data = pd.read_parquet(DATA_PATH)
    predictions = predictor.predict(test_data)
    predictions.to_csv("submission.csv", index=False)


 predictions































