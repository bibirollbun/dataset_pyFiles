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


#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CURE-Bench Competition - Advanced Submission System v3.0
Fixed version with proper string literals and pip installs

Author: CURE-Bench Advanced System
Version: 3.0
"""

# ============================================================================
# CELL 1: Package Installation
# ============================================================================

# Install required packages first
!pip install -q openai>=1.0.0
!pip install -q transformers>=4.36.0
!pip install -q accelerate>=0.25.0
!pip install -q bitsandbytes>=0.41.0
!pip install -q pandas>=2.0.0
!pip install -q numpy>=1.24.0
!pip install -q tqdm>=4.65.0
!pip install -q jsonlines>=3.1.0
!pip install -q pydantic>=2.0.0
!pip install -q tenacity>=8.2.0
!pip install -q sentencepiece  # Required for some tokenizers

# ============================================================================
# CELL 2: Core Imports
# ============================================================================

import os
import gc
import warnings
warnings.filterwarnings('ignore')

# Core imports
import json
import jsonlines
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import re
import time
from datetime import datetime
import zipfile
import logging
from pathlib import Path
from collections import defaultdict
import random

# ML imports
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Try to import quantization support
try:
    from transformers import BitsAndBytesConfig
    HAS_QUANTIZATION = True
except ImportError:
    HAS_QUANTIZATION = False
    print("Quantization support not available - will use fp16")

# Advanced imports
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydantic import BaseModel, Field

# Try to import OpenAI (optional)
try:
    from openai import AzureOpenAI, OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("OpenAI not available - will use local models only")

# ============================================================================
# CELL 3: Logging Setup
# ============================================================================

# Create output directory
output_dir = "/kaggle/working" if os.path.exists("/kaggle/working") else "."
log_dir = os.path.join(output_dir, "cure_bench_logs")
os.makedirs(log_dir, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'cure_bench.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# CELL 4: Configuration System
# ============================================================================

class ModelType(Enum):
    """Available model types"""
    RULE_BASED = "rule_based"
    AZURE_GPT4 = "azure_gpt4"
    AZURE_GPT35 = "azure_gpt35"
    OPENAI_GPT4 = "openai_gpt4"
    HF_PIPELINE = "hf_pipeline"
    LOCAL_MISTRAL = "local_mistral"
    LOCAL_LLAMA = "local_llama"
    LOCAL_DEEPSEEK = "local_deepseek"

@dataclass
class ModelConfig:
    """Configuration for individual models"""
    name: str
    type: ModelType
    enabled: bool = True
    weight: float = 1.0
    max_tokens: int = 1000
    temperature: float = 0.1
    # API settings
    api_key: Optional[str] = None
    endpoint: Optional[str] = None
    deployment: Optional[str] = None
    # Local model settings
    model_id: Optional[str] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    load_in_8bit: bool = False
    
class Config(BaseModel):
    """Main configuration"""
    # Model settings
    use_ensemble: bool = True
    models: List[ModelConfig] = Field(default_factory=list)
    
    # Reasoning enhancements
    use_chain_of_thought: bool = True
    use_few_shot: bool = True
    use_self_consistency: bool = False
    consistency_samples: int = 3
    
    # Processing
    batch_size: int = 1
    max_workers: int = 2
    rate_limit_delay: float = 0.1
    timeout: int = 60
    
    # Paths
    input_path: str = "/kaggle/input/cure-bench"
    output_path: str = "/kaggle/working"
    test_file: str = "curebench_testset_phase1.jsonl"
    
    # Debug
    debug_mode: bool = False
    sample_size: Optional[int] = None
    save_intermediate: bool = True
    
    class Config:
        arbitrary_types_allowed = True

def get_default_config() -> Config:
    """Get default configuration with fallback models"""
    config = Config()
    
    # Add models in order of preference
    # 1. DeepSeek R1 Distill (if available in Kaggle)
    deepseek_path = "/kaggle/input/deepseek-r1-distill-qwen-7b"
    if os.path.exists(deepseek_path):
        config.models.append(ModelConfig(
            name="deepseek_r1_7b",
            type=ModelType.LOCAL_DEEPSEEK,
            weight=2.5,
            model_id=deepseek_path,
            device="cuda" if torch.cuda.is_available() else "cpu",
            max_tokens=1500,
            temperature=0.1,
            load_in_8bit=True  # Use 8-bit quantization to save memory
        ))
    
    # 2. Azure OpenAI (if available)
    if HAS_OPENAI and os.environ.get("AZURE_OPENAI_API_KEY"):
        config.models.append(ModelConfig(
            name="azure_gpt4",
            type=ModelType.AZURE_GPT4,
            weight=2.0,
            api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
            endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
            deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4")
        ))
    
    # 3. Hugging Face Pipeline (always available)
    config.models.append(ModelConfig(
        name="hf_medical_qa",
        type=ModelType.HF_PIPELINE,
        weight=1.5,
        model_id="google/flan-t5-base",  # Lightweight, works on CPU
        device="cuda" if torch.cuda.is_available() else "cpu"
    ))
    
    # 4. Rule-based fallback (always works)
    config.models.append(ModelConfig(
        name="rule_based",
        type=ModelType.RULE_BASED,
        weight=0.5
    ))
    
    return config

# ============================================================================
# CELL 5: Base Model Class
# ============================================================================

class BaseModel:
    """Base class for all models"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.logger = logging.getLogger(f"{self.__class__.__name__}.{config.name}")
        
    def generate(self, question: str, options: List[str] = None, 
                question_type: str = "open_ended") -> Dict[str, Any]:
        """Generate answer with reasoning"""
        raise NotImplementedError
        
    def format_prompt(self, question: str, options: List[str] = None,
                     question_type: str = "open_ended") -> str:
        """Format prompt for the model"""
        prompt = f"Medical Question: {question}\n"
        
        if options and question_type in ["multi_choice", "open_ended_multi_choice"]:
            prompt += "\nOptions:\n"
            for i, option in enumerate(options):
                prompt += f"{chr(65+i)}. {option}\n"
        
        if self.config.type != ModelType.RULE_BASED:
            prompt += "\nProvide your reasoning step by step, then give your final answer."
            prompt += "\nFormat: REASONING: [your reasoning] ANSWER: [your answer]"
        
        return prompt

# ============================================================================
# CELL 6: Rule-Based Model (Always Works)
# ============================================================================

class RuleBasedModel(BaseModel):
    """Simple rule-based model as fallback"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.medical_keywords = {
            'safety': ['safe', 'adverse', 'contraindication', 'risk', 'warning'],
            'efficacy': ['effective', 'efficacy', 'benefit', 'outcome', 'response'],
            'pediatric': ['child', 'pediatric', 'juvenile', 'young'],
            'genetic': ['CYP', 'metabolizer', 'genetic', 'polymorphism'],
            'dosage': ['dose', 'dosage', 'mg', 'administration']
        }
        
    def generate(self, question: str, options: List[str] = None,
                question_type: str = "open_ended") -> Dict[str, Any]:
        """Generate rule-based answer"""
        
        reasoning = self._analyze_question(question, options)
        answer = self._determine_answer(question, options, reasoning)
        
        return {
            'reasoning': reasoning,
            'answer': answer,
            'confidence': 0.3,  # Low confidence for rule-based
            'model': self.config.name
        }
    
    def _analyze_question(self, question: str, options: List[str]) -> str:
        """Analyze question using rules"""
        question_lower = question.lower()
        reasoning_parts = []
        
        # Check for safety concerns
        if any(kw in question_lower for kw in self.medical_keywords['safety']):
            reasoning_parts.append("This question involves safety considerations.")
            
        # Check for pediatric patient
        if any(kw in question_lower for kw in self.medical_keywords['pediatric']):
            reasoning_parts.append("The patient is pediatric, requiring special dosing considerations.")
            
        # Check for genetic factors
        if any(kw in question_lower for kw in self.medical_keywords['genetic']):
            reasoning_parts.append("Genetic factors affect drug metabolism.")
            
        # Analyze options if available
        if options:
            safe_options = []
            for i, opt in enumerate(options):
                opt_lower = opt.lower()
                if 'none' in opt_lower or 'avoid' in opt_lower:
                    reasoning_parts.append(f"Option {chr(65+i)} suggests avoiding treatment.")
                elif any(kw in opt_lower for kw in ['low', 'reduced', 'pediatric']):
                    safe_options.append(chr(65+i))
                    reasoning_parts.append(f"Option {chr(65+i)} appears safer with adjusted dosing.")
                    
        return " ".join(reasoning_parts) if reasoning_parts else "Based on general medical principles."
    
    def _determine_answer(self, question: str, options: List[str], reasoning: str) -> str:
        """Determine answer based on rules"""
        if not options:
            return "Consult with a healthcare provider for personalized recommendations."
            
        question_lower = question.lower()
        
        # Safety-first approach
        if 'poor metabolizer' in question_lower:
            # Look for lowest dose or alternative
            for i, opt in enumerate(options):
                if 'low' in opt.lower() or 'reduced' in opt.lower():
                    return chr(65+i)
                    
        # Default to most conservative option
        if any('none' in opt.lower() for opt in options):
            for i, opt in enumerate(options):
                if 'none' in opt.lower():
                    return chr(65+i)
                    
        # Otherwise, choose first non-aspirin option for children
        if 'child' in question_lower:
            for i, opt in enumerate(options):
                if 'aspirin' not in opt.lower():
                    return chr(65+i)
                    
        # Default to first option
        return "A"

# ============================================================================
# CELL 7: Hugging Face Pipeline Model
# ============================================================================

class HFPipelineModel(BaseModel):
    """Hugging Face pipeline model"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._initialize_pipeline()
        
    def _initialize_pipeline(self):
        """Initialize HF pipeline"""
        try:
            self.logger.info(f"Loading HF model: {self.config.model_id}")
            
            # For medical QA, we'll use a general model with medical prompting
            self.pipeline = pipeline(
                "text2text-generation",
                model=self.config.model_id,
                device=0 if self.config.device == "cuda" and torch.cuda.is_available() else -1,
                max_length=self.config.max_tokens
            )
            
            self.logger.info("HF pipeline initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize HF pipeline: {e}")
            raise
            
    def generate(self, question: str, options: List[str] = None,
                question_type: str = "open_ended") -> Dict[str, Any]:
        """Generate answer using HF pipeline"""
        
        # Create medical-focused prompt
        prompt = self._create_medical_prompt(question, options, question_type)
        
        try:
            # Generate response
            response = self.pipeline(
                prompt,
                max_length=self.config.max_tokens,
                temperature=self.config.temperature,
                do_sample=True,
                top_p=0.95
            )[0]['generated_text']
            
            # Parse response
            reasoning, answer = self._parse_response(response, options)
            
            return {
                'reasoning': reasoning,
                'answer': answer,
                'confidence': 0.6,
                'model': self.config.name
            }
            
        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            # Fallback to simple answer
            return {
                'reasoning': f"Error in generation: {str(e)}",
                'answer': "A" if options else "Unable to determine",
                'confidence': 0.1,
                'model': self.config.name
            }
    
    def _create_medical_prompt(self, question: str, options: List[str], 
                              question_type: str) -> str:
        """Create medical-focused prompt"""
        prompt = "You are a medical expert. Answer the following question:\n\n"
        prompt += f"Question: {question}\n"
        
        if options:
            prompt += "\nOptions:\n"
            for i, opt in enumerate(options):
                prompt += f"{chr(65+i)}. {opt}\n"
            prompt += "\nSelect the best option and explain why.\n"
        
        prompt += "\nAnswer:"
        return prompt
    
    def _parse_response(self, response: str, options: List[str]) -> Tuple[str, str]:
        """Parse model response"""
        # Simple parsing logic
        if options:
            # Look for option letters
            for i in range(len(options)):
                letter = chr(65+i)
                if letter in response.upper():
                    return response, letter
                    
        # Default parsing
        return response, response[:50] if not options else "A"

# ============================================================================
# CELL 8: Azure OpenAI Model (Optional)
# ============================================================================

class AzureOpenAIModel(BaseModel):
    """Azure OpenAI model"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        if not HAS_OPENAI:
            raise ImportError("OpenAI package not available")
        self._initialize_client()
        
    def _initialize_client(self):
        """Initialize Azure OpenAI client"""
        if not self.config.api_key or not self.config.endpoint:
            raise ValueError("Missing Azure OpenAI credentials")
            
        self.client = AzureOpenAI(
            api_key=self.config.api_key,
            api_version="2024-02-01",
            azure_endpoint=self.config.endpoint
        )
        self.logger.info("Azure OpenAI client initialized")
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate(self, question: str, options: List[str] = None,
                question_type: str = "open_ended") -> Dict[str, Any]:
        """Generate using Azure OpenAI"""
        
        messages = [
            {
                "role": "system",
                "content": "You are an expert medical AI assistant specializing in drug decision-making. Provide clear reasoning followed by your answer."
            },
            {
                "role": "user",
                "content": self.format_prompt(question, options, question_type)
            }
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.deployment,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            full_response = response.choices[0].message.content
            reasoning, answer = self._parse_response(full_response, options)
            
            return {
                'reasoning': reasoning,
                'answer': answer,
                'confidence': 0.9,
                'model': self.config.name
            }
            
        except Exception as e:
            self.logger.error(f"Azure OpenAI error: {e}")
            raise
            
    def _parse_response(self, response: str, options: List[str]) -> Tuple[str, str]:
        """Parse Azure OpenAI response"""
        if "REASONING:" in response and "ANSWER:" in response:
            parts = response.split("ANSWER:")
            reasoning = parts[0].replace("REASONING:", "").strip()
            answer = parts[1].strip()
            
            # Extract letter for multiple choice
            if options and answer:
                match = re.search(r'^[(\[]?([A-Z])[)\].]?', answer)
                if match:
                    answer = match.group(1)
                    
            return reasoning, answer
        else:
            # Fallback parsing
            return response, "A" if options else response[:100]

# ============================================================================
# CELL 9: Local DeepSeek Model
# ============================================================================

class LocalDeepSeekModel(BaseModel):
    """Local DeepSeek model from Kaggle input"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize DeepSeek model"""
        try:
            self.logger.info(f"Loading DeepSeek model from: {self.config.model_id}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_id,
                trust_remote_code=True,
                use_fast=True
            )
            
            # Set padding token if needed
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with memory optimization
            if self.config.load_in_8bit and HAS_QUANTIZATION:
                # Use 8-bit quantization
                bnb_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_compute_dtype=torch.float16
                )
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_id,
                    trust_remote_code=True,
                    device_map="auto",
                    quantization_config=bnb_config,
                    torch_dtype=torch.float16
                )
            else:
                if self.config.load_in_8bit and not HAS_QUANTIZATION:
                    self.logger.warning("8-bit quantization requested but not available, using fp16")
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_id,
                    trust_remote_code=True,
                    device_map="auto",
                    torch_dtype=torch.float16
                )
            
            self.model.eval()
            self.logger.info("DeepSeek model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load DeepSeek model: {e}")
            raise
            
    def generate(self, question: str, options: List[str] = None,
                question_type: str = "open_ended") -> Dict[str, Any]:
        """Generate answer using DeepSeek"""
        
        # Create medical-focused prompt
        system_prompt = """You are an expert medical AI assistant specializing in drug decision-making and precision therapeutics. 
You have deep knowledge of pharmacology, drug interactions, contraindications, and treatment guidelines.
When answering, always provide clear medical reasoning before giving your final answer."""

        prompt = f"""{system_prompt}

{self.format_prompt(question, options, question_type)}

Please think step by step and provide your medical reasoning, then give your final answer."""

        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.config.device)
            
            # Generate with appropriate settings for medical reasoning
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
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the input prompt from response
            response = response[len(prompt):].strip()
            
            # Parse response
            reasoning, answer = self._parse_response(response, options)
            
            # Clean GPU memory
            del inputs, outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return {
                'reasoning': reasoning,
                'answer': answer,
                'confidence': 0.85,  # High confidence for DeepSeek
                'model': self.config.name
            }
            
        except Exception as e:
            self.logger.error(f"DeepSeek generation error: {e}")
            # Return a fallback response
            return {
                'reasoning': f"Error during generation: {str(e)}",
                'answer': "A" if options else "Unable to determine",
                'confidence': 0.1,
                'model': self.config.name
            }
            
    def _parse_response(self, response: str, options: List[str]) -> Tuple[str, str]:
        """Parse DeepSeek response"""
        # Try multiple parsing strategies
        
        # Strategy 1: Look for explicit REASONING/ANSWER format
        if "REASONING:" in response and "ANSWER:" in response:
            parts = response.split("ANSWER:")
            reasoning = parts[0].replace("REASONING:", "").strip()
            answer = parts[1].strip()
        # Strategy 2: Look for "Final answer" or similar
        elif any(marker in response.lower() for marker in ["final answer", "therefore", "my answer"]):
            for marker in ["final answer:", "therefore, the answer is", "my answer is", "the answer is"]:
                if marker.lower() in response.lower():
                    idx = response.lower().find(marker.lower())
                    reasoning = response[:idx].strip()
                    answer = response[idx + len(marker):].strip()
                    break
            else:
                # Split at last sentence
                sentences = response.split('.')
                if len(sentences) > 1:
                    reasoning = '.'.join(sentences[:-1]).strip()
                    answer = sentences[-1].strip()
                else:
                    reasoning = response
                    answer = ""
        else:
            # Default: use full response as reasoning
            reasoning = response
            answer = ""
        
        # Extract letter for multiple choice
        if options:
            # Look for letter patterns in the answer or end of reasoning
            text_to_search = answer if answer else reasoning[-100:]
            
            # Pattern 1: Explicit letter (A), [A], A., etc.
            patterns = [
                r'\b([A-D])\b(?:\)|\.|\s|$)',  # Letter followed by ), ., space, or end
                r'\[([A-D])\]',  # [Letter]
                r'\(([A-D])\)',  # (Letter)
                r'^([A-D])$'     # Just the letter
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text_to_search.upper())
                if match:
                    answer = match.group(1)
                    break
            else:
                # If no pattern found, look for option text
                for i, opt in enumerate(options):
                    if opt.lower() in response.lower():
                        answer = chr(65 + i)
                        break
                else:
                    # Default to A if nothing found
                    answer = "A"
                    
        return reasoning, answer

# ============================================================================
# CELL 10: Model Factory
# ============================================================================

class ModelFactory:
    """Factory for creating models"""
    
    @staticmethod
    def create_model(config: ModelConfig) -> Optional[BaseModel]:
        """Create model based on configuration"""
        try:
            if config.type == ModelType.RULE_BASED:
                return RuleBasedModel(config)
            elif config.type == ModelType.HF_PIPELINE:
                return HFPipelineModel(config)
            elif config.type == ModelType.LOCAL_DEEPSEEK:
                return LocalDeepSeekModel(config)
            elif config.type in [ModelType.AZURE_GPT4, ModelType.AZURE_GPT35]:
                if HAS_OPENAI:
                    return AzureOpenAIModel(config)
                else:
                    logger.warning(f"OpenAI not available for {config.name}")
                    return None
            else:
                logger.warning(f"Model type {config.type} not implemented")
                return None
                
        except Exception as e:
            logger.error(f"Failed to create model {config.name}: {e}")
            return None

# ============================================================================
# CELL 11: Ensemble System
# ============================================================================

class EnsembleSystem:
    """Simple ensemble system"""
    
    def __init__(self, models: List[BaseModel], config: Config):
        self.models = models
        self.config = config
        self.logger = logging.getLogger("EnsembleSystem")
        
    def generate_answer(self, question: str, options: List[str] = None,
                       question_type: str = "open_ended") -> Dict[str, Any]:
        """Generate answer using ensemble"""
        
        results = []
        
        # Get predictions from all models
        for model in self.models:
            try:
                result = model.generate(question, options, question_type)
                results.append(result)
                self.logger.info(f"Got result from {model.config.name}")
            except Exception as e:
                self.logger.error(f"Model {model.config.name} failed: {e}")
                
        if not results:
            # Emergency fallback
            return {
                'reasoning': "All models failed. Using default answer.",
                'answer': "A" if options else "Unable to determine",
                'confidence': 0.0,
                'ensemble_details': {'error': 'All models failed'}
            }
            
        # Simple weighted voting
        final_result = self._weighted_voting(results)
        final_result['ensemble_details'] = {
            'num_models': len(results),
            'model_results': results
        }
        
        return final_result
        
    def _weighted_voting(self, results: List[Dict]) -> Dict[str, Any]:
        """Simple weighted voting"""
        if len(results) == 1:
            return results[0]
            
        # For multiple choice, vote on answers
        answer_votes = defaultdict(float)
        answer_reasoning = defaultdict(list)
        
        for result in results:
            model_name = result.get('model', 'unknown')
            weight = next((m.config.weight for m in self.models 
                         if m.config.name == model_name), 1.0)
            confidence = result.get('confidence', 0.5)
            
            vote_weight = weight * confidence
            answer = result['answer']
            
            answer_votes[answer] += vote_weight
            answer_reasoning[answer].append(result['reasoning'])
            
        # Get winning answer
        winning_answer = max(answer_votes.items(), key=lambda x: x[1])[0]
        
        # Combine reasoning
        reasonings = answer_reasoning[winning_answer]
        if len(reasonings) > 1:
            combined_reasoning = "Based on multiple models:\n" + "\n".join(
                f"- {r[:200]}" for r in reasonings[:3]
            )
        else:
            combined_reasoning = reasonings[0]
            
        # Calculate final confidence
        total_weight = sum(answer_votes.values())
        winning_weight = answer_votes[winning_answer]
        consensus = winning_weight / total_weight if total_weight > 0 else 0
        
        return {
            'reasoning': combined_reasoning,
            'answer': winning_answer,
            'confidence': consensus
        }

# ============================================================================
# CELL 12: Main Processing Pipeline
# ============================================================================

class CUREBenchProcessor:
    """Main processing pipeline"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger("CUREBenchProcessor")
        self.models = []
        self.ensemble = None
        
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize all configured models"""
        self.logger.info("Initializing models...")
        
        for model_config in self.config.models:
            if not model_config.enabled:
                continue
                
            model = ModelFactory.create_model(model_config)
            if model:
                self.models.append(model)
                self.logger.info(f"Successfully loaded: {model_config.name}")
            else:
                self.logger.warning(f"Failed to load: {model_config.name}")
                
        if not self.models:
            # Ensure we always have at least the rule-based model
            self.logger.warning("No models loaded, adding rule-based fallback")
            fallback = ModelFactory.create_model(
                ModelConfig(name="emergency_fallback", type=ModelType.RULE_BASED)
            )
            self.models.append(fallback)
            
        # Create ensemble if requested
        if self.config.use_ensemble and len(self.models) > 1:
            self.ensemble = EnsembleSystem(self.models, self.config)
            self.logger.info(f"Created ensemble with {len(self.models)} models")
        else:
            self.logger.info(f"Using single model: {self.models[0].config.name}")
            
    def process_dataset(self, input_file: str) -> pd.DataFrame:
        """Process the test dataset"""
        self.logger.info(f"Processing dataset: {input_file}")
        
        # Load questions
        questions = self._load_questions(input_file)
        
        if self.config.debug_mode and self.config.sample_size:
            questions = questions[:self.config.sample_size]
            self.logger.info(f"Debug mode: Processing {len(questions)} questions")
            
        # Process questions
        results = []
        for question_data in tqdm(questions, desc="Processing questions"):
            try:
                result = self._process_question(question_data)
                results.append(result)
                
                # Rate limiting
                time.sleep(self.config.rate_limit_delay)
                
            except Exception as e:
                self.logger.error(f"Failed on question {question_data.get('id')}: {e}")
                # Add error result
                results.append({
                    'id': question_data.get('id'),
                    'prediction': 'A',  # Default
                    'reasoning_trace': f'Processing error: {str(e)}',
                    'choice': 'A' if question_data.get('options') else ''
                })
                
        # Convert to DataFrame
        df = pd.DataFrame(results)
        self.logger.info(f"Processed {len(df)} questions")
        
        return df
        
    def _load_questions(self, filepath: str) -> List[Dict]:
        """Load questions from JSONL"""
        questions = []
        with jsonlines.open(filepath) as reader:
            for obj in reader:
                questions.append(obj)
        return questions
        
    def _process_question(self, question_data: Dict) -> Dict:
        """Process single question"""
        q_id = question_data.get('id')
        question = question_data.get('question')
        q_type = question_data.get('question_type', 'open_ended')
        options = question_data.get('options', {})
        
        # Convert options dict to list
        if isinstance(options, dict):
            options_list = [options.get(chr(65+i), '') for i in range(len(options))]
        else:
            options_list = options
            
        # Generate answer
        if self.ensemble:
            result = self.ensemble.generate_answer(question, options_list, q_type)
        else:
            result = self.models[0].generate(question, options_list, q_type)
            
        # Format for submission
        return {
            'id': q_id,
            'prediction': result['answer'],
            'reasoning_trace': result['reasoning'],
            'choice': result['answer'] if options_list else ''
        }

# ============================================================================
# CELL 13: Submission Creation
# ============================================================================

def create_submission(df: pd.DataFrame, config: Config) -> str:
    """Create submission package"""
    logger = logging.getLogger("SubmissionCreator")
    
    # Ensure required columns
    required_columns = ['id', 'prediction', 'reasoning_trace', 'choice']
    for col in required_columns:
        if col not in df.columns:
            df[col] = ''
            
    # Save CSV
    csv_path = os.path.join(config.output_path, "submission.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved submission CSV: {csv_path}")
    
    # Create metadata
    metadata = {
        "meta_data": {
            "model_name": "cure_bench_ensemble_v3",
            "track": "internal_reasoning",
            "model_type": "Ensemble" if config.use_ensemble else "Single",
            "base_model_type": "Mixed",
            "base_model_name": f"{len(config.models)} models",
            "dataset": "cure_bench_phase_1",
            "additional_info": {
                "models_used": [m.name for m in config.models if m.enabled],
                "ensemble": config.use_ensemble
            }
        }
    }
    
    metadata_path = os.path.join(config.output_path, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
        
    # Create zip
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_path = os.path.join(config.output_path, f"cure_bench_submission_{timestamp}.zip")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(csv_path, "submission.csv")
        zipf.write(metadata_path, "metadata.json")
        
    logger.info(f"Created submission: {zip_path}")
    return zip_path

# ============================================================================
# CELL 14: Main Execution
# ============================================================================

def main():
    """Main execution function"""
    print("\n" + "="*60)
    print("CURE-Bench Submission System v3.0")
    print("="*60)
    
    # Get configuration
    config = get_default_config()
    
    # Check for DeepSeek model
    deepseek_available = any(m.type == ModelType.LOCAL_DEEPSEEK for m in config.models)
    if deepseek_available:
        print("\nâœ… DeepSeek R1 Distill model detected!")
    
    # Print configuration
    print("\nConfiguration:")
    print(f"- Models: {len([m for m in config.models if m.enabled])} enabled")
    for model in config.models:
        if model.enabled:
            icon = "ğŸš€" if model.type == ModelType.LOCAL_DEEPSEEK else "â€¢"
            print(f"  {icon} {model.name} ({model.type.value})")
    print(f"- Ensemble: {config.use_ensemble}")
    print(f"- Debug mode: {config.debug_mode}")
    
    # Check for input file
    input_file = os.path.join(config.input_path, config.test_file)
    if not os.path.exists(input_file):
        print(f"\nâ�Œ ERROR: Input file not found: {input_file}")
        print("Please ensure the dataset is in the input directory")
        return
        
    # Create processor
    print("\nğŸ“¥ Initializing processor...")
    processor = CUREBenchProcessor(config)
    
    # Process dataset
    print(f"\nğŸ”¬ Processing dataset...")
    start_time = time.time()
    
    try:
        df = processor.process_dataset(input_file)
        
        # Create submission
        print(f"\nğŸ“¦ Creating submission...")
        submission_path = create_submission(df, config)
        
        # Summary
        elapsed = time.time() - start_time
        print(f"\nâœ… COMPLETE!")
        print(f"- Processed {len(df)} questions")
        print(f"- Time: {elapsed:.1f} seconds")
        print(f"- Output: {submission_path}")
        
    except Exception as e:
        print(f"\nâ�Œ ERROR: {e}")
        logger.exception("Fatal error")
        
# ============================================================================
# CELL 15: Quick Start Functions
# ============================================================================

def quick_test(n_samples: int = 5):
    """Quick test with n samples"""
    global config
    config = get_default_config()
    config.debug_mode = True
    config.sample_size = n_samples
    config.use_ensemble = False  # Faster
    print(f"ğŸš€ Running quick test with {n_samples} samples...")
    
    # Update global config before running main
    processor = CUREBenchProcessor(config)
    
    # Process with limited samples
    input_file = os.path.join(config.input_path, config.test_file)
    if os.path.exists(input_file):
        df = processor.process_dataset(input_file)
        submission_path = create_submission(df, config)
        print(f"âœ… Test complete! Output: {submission_path}")
    else:
        print(f"â�Œ Input file not found: {input_file}")

def full_run():
    """Full competition run"""
    global config
    config = get_default_config()
    config.debug_mode = False
    config.sample_size = None
    print("ğŸ�¯ Running full submission...")
    main()

# ============================================================================
# EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Auto-detect environment
    if os.path.exists('/kaggle/working'):
        print("ğŸ�¯ Detected Kaggle environment")
    else:
        print("ğŸ’» Running locally")
        
    print("\nUsage:")
    print("- quick_test(5)  # Test with 5 samples")
    print("- full_run()     # Full submission")
    print("- main()         # Run with current config")
    
    # Default: run full submission
    main()

