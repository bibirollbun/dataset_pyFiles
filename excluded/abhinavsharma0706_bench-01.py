# Import necessary libraries
!pip install -q openai
!pip install -q transformers
!pip install -q accelerate
!pip install -q bitsandbytes
!pip install -q pandas
!pip install -q numpy
!pip install -q tqdm
!pip install -q jsonlines
!pip install -q pydantic
!pip install -q tenacity
!pip install -q sentencepiece 
import os
import json
import jsonlines
import pandas as pd
import numpy as np
import torch
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import re
import time
from datetime import datetime
import zipfile
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydantic import BaseModel, Field

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('medical_qa.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# Define model types
class ModelType(Enum):
    RULE_BASED = "rule_based"
    AZURE_GPT4 = "azure_gpt4"
    AZURE_GPT35 = "azure_gpt35"
    OPENAI_GPT4 = "openai_gpt4"
    HF_PIPELINE = "hf_pipeline"
    LOCAL_DEEPSEEK = "local_deepseek"

@dataclass
class ModelConfig:
    name: str
    type: ModelType
    enabled: bool = True
    weight: float = 1.0
    max_tokens: int = 1000
    temperature: float = 0.1
    api_key: Optional[str] = None
    endpoint: Optional[str] = None
    deployment: Optional[str] = None
    model_id: Optional[str] = None
    load_in_8bit: bool = False

class Config(BaseModel):
    use_ensemble: bool = True
    models: List[ModelConfig] = Field(default_factory=list)
    use_chain_of_thought: bool = True
    use_few_shot: bool = True
    use_self_consistency: bool = False
    consistency_samples: int = 3
    batch_size: int = 1
    max_workers: int = 2
    rate_limit_delay: float = 0.1
    timeout: int = 60
    input_path: str = "/kaggle/input/cure-bench"
    output_path: str = "/kaggle/working"
    test_file: str = "curebench_testset_phase1.jsonl"
    debug_mode: bool = False
    sample_size: Optional[int] = None
    save_intermediate: bool = True

    class Config:
        arbitrary_types_allowed = True


def get_default_config() -> Config:
    config = Config()

    # Add models
    config.models.append(ModelConfig(
        name="deepseek_model",
        type=ModelType.LOCAL_DEEPSEEK,
        weight=2.5,
        model_id="./deepseek_model",
        max_tokens=1500,
        temperature=0.1,
        load_in_8bit=True
    ))

    config.models.append(ModelConfig(
        name="hf_model",
        type=ModelType.HF_PIPELINE,
        weight=1.5,
        model_id="google/flan-t5-base"
    ))

    config.models.append(ModelConfig(
        name="rule_based_model",
        type=ModelType.RULE_BASED,
        weight=0.5
    ))

    return config



class BaseModelHandler:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.logger = logging.getLogger(f"{self.__class__.__name__}.{config.name}")

    def generate(self, question: str, options: List[str] = None, question_type: str = "open_ended") -> Dict[str, Any]:
        raise NotImplementedError

    def format_prompt(self, question: str, options: List[str] = None, question_type: str = "open_ended") -> str:
        prompt = f"Medical Question: {question}\n"
        if options and question_type in ["multi_choice", "open_ended_multi_choice"]:
            prompt += "\nOptions:\n" + "\n".join([f"{chr(65+i)}. {option}" for i, option in enumerate(options)])
        if self.config.type != ModelType.RULE_BASED:
            prompt += "\nProvide your reasoning step by step, then give your final answer."
        return prompt


class RuleBasedModelHandler(BaseModelHandler):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.medical_keywords = {
            'safety': ['safe', 'adverse', 'contraindication', 'risk', 'warning'],
            'efficacy': ['effective', 'efficacy', 'benefit', 'outcome', 'response'],
            'pediatric': ['child', 'pediatric', 'juvenile', 'young'],
            'genetic': ['CYP', 'metabolizer', 'genetic', 'polymorphism'],
            'dosage': ['dose', 'dosage', 'mg', 'administration']
        }

    def generate(self, question: str, options: List[str] = None, question_type: str = "open_ended") -> Dict[str, Any]:
        reasoning = self._analyze_question(question, options)
        answer = self._determine_answer(question, options, reasoning)
        return {'reasoning': reasoning, 'answer': answer, 'confidence': 0.3, 'model': self.config.name}

    def _analyze_question(self, question: str, options: List[str]) -> str:
        question_lower = question.lower()
        reasoning_parts = []
        for category, keywords in self.medical_keywords.items():
            if any(kw in question_lower for kw in keywords):
                reasoning_parts.append(f"This question involves {category} considerations.")
        if options:
            for i, opt in enumerate(options):
                if 'none' in opt.lower() or 'avoid' in opt.lower():
                    reasoning_parts.append(f"Option {chr(65+i)} suggests avoiding treatment.")
        return " ".join(reasoning_parts) if reasoning_parts else "Based on general medical principles."

    def _determine_answer(self, question: str, options: List[str], reasoning: str) -> str:
        if not options:
            return "Consult with a healthcare provider for personalized recommendations."
        question_lower = question.lower()
        if 'poor metabolizer' in question_lower:
            for i, opt in enumerate(options):
                if 'low' in opt.lower() or 'reduced' in opt.lower():
                    return chr(65+i)
        if any('none' in opt.lower() for opt in options):
            for i, opt in enumerate(options):
                if 'none' in opt.lower():
                    return chr(65+i)
        if 'child' in question_lower:
            for i, opt in enumerate(options):
                if 'aspirin' not in opt.lower():
                    return chr(65+i)
        return "A"


class HFPipelineModelHandler(BaseModelHandler):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.pipeline = pipeline(
            "text2text-generation",
            model=self.config.model_id,
            device=0 if torch.cuda.is_available() else -1,
            max_length=self.config.max_tokens
        )

    def generate(self, question: str, options: List[str] = None, question_type: str = "open_ended") -> Dict[str, Any]:
        prompt = self.format_prompt(question, options, question_type)
        try:
            response = self.pipeline(
                prompt,
                max_length=self.config.max_tokens,
                temperature=self.config.temperature,
                do_sample=True,
                top_p=0.95
            )[0]['generated_text']
            reasoning, answer = self._parse_response(response, options)
            return {'reasoning': reasoning, 'answer': answer, 'confidence': 0.6, 'model': self.config.name}
        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            return {'reasoning': f"Error in generation: {str(e)}", 'answer': "A" if options else "Unable to determine", 'confidence': 0.1, 'model': self.config.name}

    def _parse_response(self, response: str, options: List[str]) -> Tuple[str, str]:
        if options:
            for i in range(len(options)):
                letter = chr(65+i)
                if letter in response.upper():
                    return response, letter
        return response, response[:50] if not options else "A"



class LocalDeepSeekModelHandler(BaseModelHandler):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_id, trust_remote_code=True, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.config.load_in_8bit:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True, bnb_8bit_compute_dtype=torch.float16)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_id,
                trust_remote_code=True,
                device_map="auto",
                quantization_config=bnb_config,
                torch_dtype=torch.float16
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_id,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype=torch.float16
            )
        self.model.eval()

    def generate(self, question: str, options: List[str] = None, question_type: str = "open_ended") -> Dict[str, Any]:
        system_prompt = "You are an expert medical AI assistant specializing in drug decision-making and precision therapeutics."
        prompt = f"{system_prompt}\n\n{self.format_prompt(question, options, question_type)}\n\nPlease think step by step and provide your medical reasoning, then give your final answer."
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda" if torch.cuda.is_available() else "cpu")
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    do_sample=True,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):].strip()
            reasoning, answer = self._parse_response(response, options)
            return {'reasoning': reasoning, 'answer': answer, 'confidence': 0.85, 'model': self.config.name}
        except Exception as e:
            self.logger.error(f"Generation error: {e}")
            return {'reasoning': f"Error during generation: {str(e)}", 'answer': "A" if options else "Unable to determine", 'confidence': 0.1, 'model': self.config.name}

    def _parse_response(self, response: str, options: List[str]) -> Tuple[str, str]:
        if "REASONING:" in response and "ANSWER:" in response:
            parts = response.split("ANSWER:")
            reasoning = parts[0].replace("REASONING:", "").strip()
            answer = parts[1].strip()
        elif any(marker in response.lower() for marker in ["final answer", "therefore", "my answer"]):
            for marker in ["final answer:", "therefore, the answer is", "my answer is", "the answer is"]:
                if marker.lower() in response.lower():
                    idx = response.lower().find(marker.lower())
                    reasoning = response[:idx].strip()
                    answer = response[idx + len(marker):].strip()
                    break
            else:
                sentences = response.split('.')
                reasoning = '.'.join(sentences[:-1]).strip()
                answer = sentences[-1].strip()
        else:
            reasoning = response
            answer = ""
        if options:
            text_to_search = answer if answer else reasoning[-100:]
            patterns = [r'\b([A-D])\b(?:\)|\.|\s|$)', r'\[([A-D])\]', r'\(([A-D])\)', r'^([A-D])$']
            for pattern in patterns:
                match = re.search(pattern, text_to_search.upper())
                if match:
                    answer = match.group(1)
                    break
            else:
                for i, opt in enumerate(options):
                    if opt.lower() in response.lower():
                        answer = chr(65 + i)
                        break
                else:
                    answer = "A"
        return reasoning, answer



class ModelFactory:
    @staticmethod
    def create_model(config: ModelConfig) -> Optional[BaseModelHandler]:
        try:
            if config.type == ModelType.RULE_BASED:
                return RuleBasedModelHandler(config)
            elif config.type == ModelType.HF_PIPELINE:
                return HFPipelineModelHandler(config)
            elif config.type == ModelType.LOCAL_DEEPSEEK:
                return LocalDeepSeekModelHandler(config)
            else:
                logger.warning(f"Model type {config.type} not implemented")
                return None
        except Exception as e:
            logger.error(f"Failed to create model {config.name}: {e}")
            return None


class EnsembleSystem:
    def __init__(self, models: List[BaseModelHandler], config: Config):
        self.models = models
        self.config = config
        self.logger = logging.getLogger("EnsembleSystem")

    def generate_answer(self, question: str, options: List[str] = None, question_type: str = "open_ended") -> Dict[str, Any]:
        results = []
        for model in self.models:
            try:
                result = model.generate(question, options, question_type)
                results.append(result)
                self.logger.info(f"Result from {model.config.name}")
            except Exception as e:
                self.logger.error(f"Model {model.config.name} failed: {e}")
        if not results:
            return {'reasoning': "All models failed. Using default answer.", 'answer': "A" if options else "Unable to determine", 'confidence': 0.0, 'ensemble_details': {'error': 'All models failed'}}
        final_result = self._weighted_voting(results)
        final_result['ensemble_details'] = {'num_models': len(results), 'model_results': results}
        return final_result

    def _weighted_voting(self, results: List[Dict]) -> Dict[str, Any]:
        if len(results) == 1:
            return results[0]
        answer_votes = defaultdict(float)
        answer_reasoning = defaultdict(list)
        for result in results:
            model_name = result.get('model', 'unknown')
            weight = next((m.config.weight for m in self.models if m.config.name == model_name), 1.0)
            confidence = result.get('confidence', 0.5)
            vote_weight = weight * confidence
            answer = result['answer']
            answer_votes[answer] += vote_weight
            answer_reasoning[answer].append(result['reasoning'])
        winning_answer = max(answer_votes.items(), key=lambda x: x[1])[0]
        reasonings = answer_reasoning[winning_answer]
        combined_reasoning = "Based on multiple models:\n" + "\n".join(f"- {r[:200]}" for r in reasonings[:3]) if len(reasonings) > 1 else reasonings[0]
        total_weight = sum(answer_votes.values())
        winning_weight = answer_votes[winning_answer]
        consensus = winning_weight / total_weight if total_weight > 0 else 0
        return {'reasoning': combined_reasoning, 'answer': winning_answer, 'confidence': consensus}


class MedicalQAProcessor:
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger("MedicalQAProcessor")
        self.models = []
        self.ensemble = None
        self._initialize_models()

    def _initialize_models(self):
        self.logger.info("Initializing models...")
        for model_config in self.config.models:
            if not model_config.enabled:
                continue
            model = ModelFactory.create_model(model_config)
            if model:
                self.models.append(model)
                self.logger.info(f"Loaded: {model_config.name}")
            else:
                self.logger.warning(f"Failed to load: {model_config.name}")
        if not self.models:
            self.logger.warning("No models loaded, adding rule-based fallback")
            fallback = ModelFactory.create_model(ModelConfig(name="emergency_fallback", type=ModelType.RULE_BASED))
            self.models.append(fallback)
        if self.config.use_ensemble and len(self.models) > 1:
            self.ensemble = EnsembleSystem(self.models, self.config)
            self.logger.info(f"Created ensemble with {len(self.models)} models")
        else:
            self.logger.info(f"Using single model: {self.models[0].config.name}")

    def process_dataset(self, input_file: str) -> pd.DataFrame:
        self.logger.info(f"Processing dataset: {input_file}")
        questions = self._load_questions(input_file)
        if self.config.debug_mode and self.config.sample_size:
            questions = questions[:self.config.sample_size]
            self.logger.info(f"Debug mode: Processing {len(questions)} questions")
        results = []
        for question_data in tqdm(questions, desc="Processing questions"):
            try:
                result = self._process_question(question_data)
                results.append(result)
                time.sleep(self.config.rate_limit_delay)
            except Exception as e:
                self.logger.error(f"Failed on question {question_data.get('id')}: {e}")
                results.append({'id': question_data.get('id'), 'prediction': 'A', 'reasoning_trace': f'Processing error: {str(e)}', 'choice': 'A' if question_data.get('options') else ''})
        df = pd.DataFrame(results)
        self.logger.info(f"Processed {len(df)} questions")
        return df

    def _load_questions(self, filepath: str) -> List[Dict]:
        questions = []
        with jsonlines.open(filepath) as reader:
            for obj in reader:
                questions.append(obj)
        return questions

    def _process_question(self, question_data: Dict) -> Dict:
        q_id = question_data.get('id')
        question = question_data.get('question')
        q_type = question_data.get('question_type', 'open_ended')
        options = question_data.get('options', {})
        options_list = [options.get(chr(65+i), '') for i in range(len(options))] if isinstance(options, dict) else options
        if self.ensemble:
            result = self.ensemble.generate_answer(question, options_list, q_type)
        else:
            result = self.models[0].generate(question, options_list, q_type)
        return {'id': q_id, 'prediction': result['answer'], 'reasoning_trace': result['reasoning'], 'choice': result['answer'] if options_list else ''}


def create_submission(df: pd.DataFrame, config: Config) -> str:
    logger = logging.getLogger("SubmissionCreator")
    required_columns = ['id', 'prediction', 'reasoning_trace', 'choice']
    for col in required_columns:
        if col not in df.columns:
            df[col] = ''
    csv_path = os.path.join(config.output_path, "submission.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved submission CSV: {csv_path}")
    metadata = {
        "meta_data": {
            "model_name": "medical_qa_ensemble",
            "track": "internal_reasoning",
            "model_type": "Ensemble" if config.use_ensemble else "Single",
            "base_model_type": "Mixed",
            "base_model_name": f"{len(config.models)} models",
            "dataset": "medical_qa_dataset",
            "additional_info": {"models_used": [m.name for m in config.models if m.enabled], "ensemble": config.use_ensemble}
        }
    }
    metadata_path = os.path.join(config.output_path, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_path = os.path.join(config.output_path, f"medical_qa_submission_{timestamp}.zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(csv_path, "submission.csv")
        zipf.write(metadata_path, "metadata.json")
    logger.info(f"Created submission: {zip_path}")
    return zip_path


def main():
    print("\n" + "="*60)
    print("Medical QA Submission System")
    print("="*60)
    config = get_default_config()
    deepseek_available = any(m.type == ModelType.LOCAL_DEEPSEEK for m in config.models)
    if deepseek_available:
        print("\nâœ… DeepSeek model detected!")
    print("\nConfiguration:")
    print(f"- Models: {len([m for m in config.models if m.enabled])} enabled")
    for model in config.models:
        if model.enabled:
            icon = "ğŸš€" if model.type == ModelType.LOCAL_DEEPSEEK else "â€¢"
            print(f"  {icon} {model.name} ({model.type.value})")
    print(f"- Ensemble: {config.use_ensemble}")
    print(f"- Debug mode: {config.debug_mode}")
    input_file = os.path.join(config.input_path, config.test_file)
    if not os.path.exists(input_file):
        print(f"\nâ�Œ ERROR: Input file not found: {input_file}")
        print("Please ensure the dataset is in the input directory")
        return
    print("\nğŸ“¥ Initializing processor...")
    processor = MedicalQAProcessor(config)
    print(f"\nğŸ”¬ Processing dataset...")
    start_time = time.time()
    try:
        df = processor.process_dataset(input_file)
        print(f"\nğŸ“¦ Creating submission...")
        submission_path = create_submission(df, config)
        elapsed = time.time() - start_time
        print(f"\nâœ… COMPLETE!")
        print(f"- Processed {len(df)} questions")
        print(f"- Time: {elapsed:.1f} seconds")
        print(f"- Output: {submission_path}")
    except Exception as e:
        print(f"\nâ�Œ ERROR: {e}")
        logger.exception("Fatal error")

if __name__ == "__main__":
    if os.path.exists('./data'):
        print("ğŸ�¯ Detected local environment")
    else:
        print("ğŸ’» Running in default environment")
    print("\nUsage:")
    print("- Run with default config: main()")
    main()

