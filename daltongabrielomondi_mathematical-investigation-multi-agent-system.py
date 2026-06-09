!pip install -q transformers accelerate bitsandbytes


# Import essential libraries only
import os
import json
import re
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from typing import Dict, Any, List

# Suppress warnings
import warnings
warnings.simplefilter('ignore')


# GPU and CUDA setup
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # Adjust based on available GPUs
os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda/bin/ptxas"


class Config:
    """Lightweight configuration for Kaggle environment"""
    MODELS = {
        "explorer": "/kaggle/input/deepseek-math-7b-instruct/transformers/main/1",
        "prover": "/kaggle/input/deepseek-prover/transformers/deepseek-prover-v1.5-rl/1",
        "verifier": "/kaggle/input/distillbert/transformers/default/1",
        "code_generator": "/kaggle/input/deepseek-coder-v2/transformers/deepseek-coder-v2-instruct/1"
    }
    
    MAX_TOKENS = 2048
    TEMPERATURES = {
        "explorer": 0.7,
        "prover": 0.3,
        "verifier": 0.1,
        "code_generator": 0.2
    }
    
    TOPICS = {
        "riemann": "Riemann Hypothesis about zeta function zeros",
        "twin_primes": "Twin Prime Conjecture about infinite prime pairs",
        "goldbach": "Goldbach Conjecture about even numbers as prime sums",
        "prime_recurrence": "Finding recurrence relations for primes"
    }

config = Config()


# Model Manager with memory optimization
class ModelManager:
    """Manages model loading/unloading with Kaggle constraints"""
    def __init__(self):
        self.loaded_models = {}
        self.active_model = None
        
    def load_model(self, role: str):
        """Load model with 4-bit quantization to save memory"""
        if role == self.active_model:
            return self.loaded_models[role]
            
        # Unload previous model if memory constrained
        if len(self.loaded_models) >= 1 and torch.cuda.memory_reserved() > 0.8 * torch.cuda.get_device_properties(0).total_memory:
            self.unload_model(self.active_model)
            
        print(f"Loading {role} model: {config.MODELS[role]}")
        
        tokenizer = AutoTokenizer.from_pretrained(config.MODELS[role])
        model = AutoModelForCausalLM.from_pretrained(
            config.MODELS[role],
            device_map="auto",
            load_in_4bit=True,
            torch_dtype=torch.float16
        )
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=config.MAX_TOKENS
        )
        
        self.loaded_models[role] = pipe
        self.active_model = role
        return pipe
    
    def unload_model(self, role: str):
        """Unload model to free GPU memory"""
        if role in self.loaded_models:
            del self.loaded_models[role]
            torch.cuda.empty_cache()
            print(f"Unloaded {role} model")
            
    def generate(self, role: str, prompt: str) -> str:
        """Generate response with the appropriate model"""
        pipe = self.load_model(role)
        response = pipe(
            prompt,
            temperature=config.TEMPERATURES[role],
            top_p=0.95,
            do_sample=True
        )
        return response[0]['generated_text']

model_manager = ModelManager()


# Specialized agents with role-specific prompts
class MathAgent:
    """Base agent class with role-specific capabilities"""
    def __init__(self, role: str):
        self.role = role
        self.history = []
    
    def generate_prompt(self, topic: str, user_input: str = "") -> str:
        """Generate role-specific prompt"""
        base_prompts = {
            "explorer": (
                "You're a creative math explorer. Investigate novel approaches to: "
                f"{config.TOPICS.get(topic.lower(), topic)}. Consider established techniques "
                "and innovative perspectives. Break into manageable components."
            ),
            "prover": (
                "You're a formal proof assistant. Translate this into precise notation: "
                f"{user_input}. Provide clear definitions and logical steps. "
                "Identify assumptions and gaps."
            ),
            "verifier": (
                "You're a math critic. Analyze for logical errors: "
                f"{user_input}. Rate reliability 1-10 and identify issues."
            ),
            "code_generator": (
                "You're a math code expert. Implement this concept in Python: "
                f"{user_input}. Use efficient algorithms with documentation."
            )
        }
        return base_prompts[self.role]
    
    def execute(self, topic: str, user_input: str = "") -> str:
        """Execute the agent's task"""
        prompt = self.generate_prompt(topic, user_input)
        response = model_manager.generate(self.role, prompt)
        self.history.append({"input": user_input, "output": response})
        return response


class ResearchAssistant:
    """Interactive assistant for mathematical research"""
    def __init__(self):
        self.agents = {
            "explorer": MathAgent("explorer"),
            "prover": MathAgent("prover"),
            "verifier": MathAgent("verifier"),
            "coder": MathAgent("code_generator")
        }
        self.conversation = []
        
    def handle_command(self, command: str, user_input: str) -> str:
        """Process user commands with efficient resource usage"""
        cmd = command.lower()
        
        if cmd == "/explore":
            return self.agents["explorer"].execute(user_input)
        
        elif cmd == "/prove":
            # Use verifier to check before proving
            verification = self.agents["verifier"].execute("", user_input)
            if "reliability: [7-10]" in verification:
                return self.agents["prover"].execute("", user_input)
            return f"Verification issues:\n{verification}"
        
        elif cmd == "/code":
            return self.agents["coder"].execute("", user_input)
        
        elif cmd == "/help":
            return (
                "Available commands:\n"
                "/explore [topic] - Investigate mathematical topic\n"
                "/prove [statement] - Formalize mathematical argument\n"
                "/code [concept] - Generate implementation code\n"
                "/topics - List available research topics\n"
                "/exit - End session"
            )
        
        elif cmd == "/topics":
            return "\n".join([f"- {topic}: {desc}" for topic, desc in config.TOPICS.items()])
        
        return "Invalid command. Type /help for options."

    def chat_loop(self):
        """Interactive chat interface for research assistance"""
        print("Math Research Assistant initialized! Type /help for commands.")
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if not user_input:
                    continue
                    
                if user_input.lower() == "/exit":
                    print("Ending research session.")
                    break
                
                # Handle commands
                if user_input.startswith("/"):
                    parts = user_input.split(maxsplit=1)
                    command = parts[0]
                    text = parts[1] if len(parts) > 1 else ""
                    response = self.handle_command(command, text)
                
                # Default to exploration
                else:
                    response = self.agents["explorer"].execute(user_input)
                
                # Format and display response
                print("\nAssistant:")
                for paragraph in self._chunk_response(response):
                    print(paragraph)
                    time.sleep(0.5)  # Simulate thinking
                    
                self.conversation.append({"user": user_input, "assistant": response})
                
            except KeyboardInterrupt:
                print("\nSession interrupted.")
                break
            except Exception as e:
                print(f"Error: {str(e)}")
    
    def _chunk_response(self, text: str, max_len: int = 80) -> List[str]:
        """Split long responses into readable chunks"""
        words = text.split()
        chunks = []
        current_chunk = []
        
        for word in words:
            if len(' '.join(current_chunk + [word])) > max_len:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
            else:
                current_chunk.append(word)
                
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks


#if __name__ == "__main__":
#    assistant = ResearchAssistant()
#    assistant.chat_loop()
    
    # Save conversation history
#    with open("research_session.json", "w") as f:
#        json.dump(assistant.conversation, f, indent=2)
#    print("Conversation history saved.")


# class Config:
#     """Configuration for the Mathematical Investigation Multi-Agent System"""
#     
#     # API server settings
#     host = "127.0.0.1"
#     explorer_port = 8000
#     prover_port = 8001
#     verifier_port = 8002
#     translator_port = 8003
#     code_generator_port = 8004
#     
#     # GPU allocation (for vLLM)
#     explorer_gpus = "0"
#     prover_gpus = "1"
#     verifier_gpus = "2"
#     translator_gpus = "3"
#     code_generator_gpus = "0"  # Can share with explorer if needed
#     
#     # Model Registry - Define available models for each role
#     
#     # Explorer Models - for creative mathematical reasoning
#     explorer_models = {
#         "deepseek-math-7b": {
#             "id": "deepseek-ai/deepseek-math-7b-instruct",
#             "path": "/kaggle/input/deepseek-math-7b-instruct/transformers/main/1",
#             "description": "Specialized for mathematical exploration and problem-solving",
#             "vllm_args": {"tensor_parallel_size": 1, "gpu_memory_utilization": 0.9}
#         },
#         "deepseek-distill-qwen-7b": {
#             "id": "deepseek-ai/deepseek-coder-instruct-qwen2-1.5b",
#             "path": "/kaggle/input/deepseek-r1/transformers/deepseek-r1-distill-qwen-7b-awq-casperhansen/1",
#             "description": "Efficient distilled model for creative approaches",
#             "vllm_args": {"tensor_parallel_size": 1, "gpu_memory_utilization": 0.85}
#         }
#     }
#     
#     # Prover Models - for formal mathematical reasoning
#     prover_models = {
#         "deepseek-prover-7b": {
#             "id": "deepseek-ai/deepseek-prover-7b-instruct",
#             "path": "/kaggle/input/deepseek-prover/transformers/deepseek-prover-v1.5-rl/1",
#             "description": "Specialized for formal mathematical proofs and Lean integration",
#             "vllm_args": {"tensor_parallel_size": 1, "gpu_memory_utilization": 0.9}
#         }
#     }
#     
#     # Verifier Models - for checking mathematical reasoning
#     verifier_models = {
#         "distilbert-base": {
#             "id": "distilbert-base-uncased",
#             "path": "/kaggle/input/distillbert/transformers/default/1",
#             "description": "Lightweight model for efficient verification",
#             "vllm_args": {"tensor_parallel_size": 1, "gpu_memory_utilization": 0.7}
#         },
#         "deepseek-prover-7b": {  # Can reuse prover model for verification
#             "id": "deepseek-ai/deepseek-prover-7b-instruct",
#             "path": "/kaggle/input/deepseek-prover/transformers/deepseek-prover-v1.5-rl/1",
#             "description": "Specialized for formal verification",
#             "vllm_args": {"tensor_parallel_size": 1, "gpu_memory_utilization": 0.9}
#         }
#     }
#     
#     # Translator Models - for translating between formal and natural language
#     translator_models = {
#         "deepseek-distill-qwen-7b": {  # Reuse from explorer
#             "id": "deepseek-ai/deepseek-r1-distill-qwen-7b-awq-casperhansen/1",
#             "path": "/kaggle/input/deepseek-r1/transformers/deepseek-r1-distill-qwen-7b-awq-casperhansen/1",
#             "description": "Efficient model for translating math to natural language",
#             "vllm_args": {"tensor_parallel_size": 1, "gpu_memory_utilization": 0.85}
#         }
#     }
#     
#     # Code Generator Models - for implementing mathematical algorithms
#     code_generator_models = {
#         "deepseek-coder-33b": {
#             "id": "deepseek-ai/deepseek-coder-33b-instruct",
#             "path": "/kaggle/input/deepseek-coder-gguf/transformers/deepseek/1",
#             "description": "Specialized for mathematical code generation",
#             "vllm_args": {"tensor_parallel_size": 1, "gpu_memory_utilization": 0.9}
#         }
#     }
#     
#     # Default model selections
#     explorer_model = "deepseek-math-7b"
#     prover_model = "deepseek-prover-7b"
#     verifier_model = "deepseek-prover-7b"  # Use prover for verification by default
#     translator_model = "deepseek-distill-qwen-7b"
#     code_generator_model = "deepseek-coder-33b"
#     
#     # Model ID and paths (derived from selections)
#     @property
#     def explorer_model_id(self):
#         return self.explorer_models[self.explorer_model]["id"]
#     
#     @property
#     def prover_model_id(self):
#         return self.prover_models[self.prover_model]["id"]
#     
#     @property
#     def verifier_model_id(self):
#         return self.verifier_models[self.verifier_model]["id"]
#     
#     @property
#     def translator_model_id(self):
#         return self.translator_models[self.translator_model]["id"]
#     
#     @property
#     def code_generator_model_id(self):
#         return self.code_generator_models[self.code_generator_model]["id"]
#     
#     @property
#     def explorer_model_path(self):
#         return self.explorer_models[self.explorer_model]["path"]
#     
#     @property
#     def prover_model_path(self):
#         return self.prover_models[self.prover_model]["path"]
#     
#     @property
#     def verifier_model_path(self):
#         return self.verifier_models[self.verifier_model]["path"]
#     
#     @property
#     def translator_model_path(self):
#         return self.translator_models[self.translator_model]["path"]
#     
#     @property
#     def code_generator_model_path(self):
#         return self.code_generator_models[self.code_generator_model]["path"]
#     
#     # API names
#     explorer_name = "explorer"
#     prover_name = "prover"
#     verifier_name = "verifier"
#     translator_name = "translator"
#     code_generator_name = "code-generator"
#     
#     # Runtime settings
#     use_vllm = True       # Use vLLM for inference if available
#     use_8bit = True       # Use 8-bit quantization for HF models
#     use_4bit = False      # Use 4-bit quantization for HF models
#     batch_size = 1        # Default batch size
#     
#     # Formal verification tools
#     use_lean = True       # Use Lean theorem prover for formal verification
#     lean_path = "/kaggle/input/lean/lean"  # Path to Lean executable
#     lean_timeout = 30     # Lean execution timeout in seconds
#     
#     # Generation parameters
#     max_tokens = 4096
#     temperature_creative = 0.7   # Higher for creative exploration
#     temperature_formal = 0.2     # Lower for formal reasoning
#     temperature_verification = 0.1  # Very low for verification
#     temperature_translation = 0.3   # Moderate for translation
#     temperature_code_gen = 0.2      # Low for code generation
#     top_p = 0.95
#     top_k = 50
#     
#     # Agent parameters
#     max_reasoning_steps = 10
#     max_verification_attempts = 3
#     hallucination_threshold = 0.7
#     
#     # Timeout settings
#     request_timeout = 90
#     server_startup_timeout = 300
#     
#     # Mathematical topics
#     topics = {
#         "riemann": {
#             "description": "The Riemann Hypothesis states that all non-trivial zeros of the zeta function have real part 1/2.",
#             "key_concepts": ["zeta function", "non-trivial zeros", "critical line", "prime number theorem"],
#             "potential_approaches": ["analytic methods", "computational verification", "spectral theory"]
#         },
#         "twin_primes": {
#             "description": "The Twin Prime Conjecture states that there are infinitely many pairs of primes that differ by 2.",
#             "key_concepts": ["prime gaps", "sieve methods", "Brun's constant", "prime counting function"],
#             "potential_approaches": ["sieve theory", "analytic number theory", "probabilistic methods"]
#         },
#         "goldbach": {
#             "description": "The Goldbach Conjecture states that every even integer greater than 2 can be expressed as the sum of two primes.",
#             "key_concepts": ["prime sums", "even numbers", "circle method", "additive number theory"],
#             "potential_approaches": ["analytic methods", "computational verification", "probabilistic bounds"]
#         },
#         "prime_recurrence": {
#             "description": "Finding a recurrence relation that generates prime numbers.",
#             "key_concepts": ["recurrence relations", "prime generating functions", "prime testing", "number theory"],
#             "potential_approaches": ["polynomial methods", "sieve algorithms", "modular arithmetic"]
#         }
#     }
#     
#     # Agent system prompts
#     agent_prompts = {
#         "explorer": """You are a creative mathematical explorer specializing in number theory and advanced mathematics.
# Your goal is to investigate novel approaches to unsolved mathematical conjectures.
# Think outside the box while maintaining mathematical rigor, exploring both established techniques and innovative perspectives.
# Break down complex problems into manageable components and consider multiple angles of attack.""",
#         
#         "prover": """You are DeepSeek Prover, an expert in formal mathematical verification and proof assistance.
# Your specialization is translating mathematical arguments into formal systems like Lean theorem prover.
# For each theorem or lemma:
# 1. Provide formal definitions of all concepts using proper mathematical notation
# 2. Break down proofs into clear logical steps with appropriate tactics
# 3. Identify any hidden assumptions or gaps in reasoning
# 4. When appropriate, use Lean syntax for formal verification
# 5. Ensure all steps follow rigorously from axioms and previous results""",
#         
#         "verifier": """You are a rigorous mathematical critic with expertise in detecting logical inconsistencies and hallucinations.
# Your task is to carefully analyze mathematical arguments for logical errors, unfounded claims, or mathematical hallucinations.
# Identify specific statements that lack justification or contain errors, being precise about where reasoning breaks down.
# Rate claims on a scale of 1-10 for reliability, where 1-3 is clearly incorrect, 4-6 is partially correct with issues,
# 7-8 is mostly correct with minor issues, and 9-10 is mathematically sound.""",
#         
#         "translator": """You are an expert at translating between formal mathematical notation and natural language.
# Your task is to make complex mathematical ideas accessible while preserving their precision and rigor.
# Explain formal proofs in clear, intuitive language that maintains mathematical accuracy.""",
#         
#         "code_generator": """You are a mathematical code generation expert who specializes in implementing mathematical algorithms.
# Your task is to translate mathematical concepts, proofs, and algorithms into efficient, well-documented code.
# Follow these principles:
# 1. Choose appropriate data structures for mathematical objects
# 2. Implement algorithms with the correct time and space complexity
# 3. Add clear documentation explaining the mathematical background
# 4. Include test cases that verify correctness
# 5. Optimize for numerical stability when implementing formulas"""
#     }
#     
#     @classmethod
#     def for_kaggle(cls):
#         """Configure settings optimized for Kaggle environment"""
#         config = cls()
#         
#         # Check available GPUs
#         gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
#         print(f"Detected {gpu_count} GPUs")
#         
#         if gpu_count == 0:
#             print(" No GPUs detected")
#             config.use_vllm = False
#             config.use_8bit = True
#             config.use_4bit = True  # Use more aggressive quantization if no GPUs
#         
#         elif gpu_count == 1:
#             print(" Single GPU detected ")
#             # Assign all models to the same GPU
#             config.explorer_gpus = "0"
#             config.prover_gpus = "0"
#             config.verifier_gpus = "0"
#             config.translator_gpus = "0"
#             config.code_generator_gpus = "0"
#             
#             # Use smaller/more efficient models
#             config.explorer_model = "deepseek-distill-qwen-7b"
#             config.verifier_model = "distilbert-base"
#             
#             # If using vLLM, we'll load models sequentially
#             if config.use_vllm:
#                 print("With single GPU and vLLM, models will be loaded sequentially")
#             else:
#                 # Use quantization for HF models to save memory
#                 config.use_8bit = True
#         
#         elif gpu_count >= 2:
#             print(f" {gpu_count} GPUs detected - distributing models across GPUs")
#             
#             if gpu_count == 2:
#                 # Optimize for 2 GPUs
#                 config.explorer_gpus = "0"
#                 config.prover_gpus = "1"
#                 config.verifier_gpus = "1"
#                 config.translator_gpus = "0"
#                 config.code_generator_gpus = "0"
#             elif gpu_count >= 4:
#                 # Optimize for 4+ GPUs
#                 config.explorer_gpus = "0"
#                 config.prover_gpus = "1"
#                 config.verifier_gpus = "2"
#                 config.translator_gpus = "3"
#                 config.code_generator_gpus = "0"
#             
#             # With multiple GPUs, prefer vLLM
#             config.use_vllm = True
#         
#         # Update vLLM tensor parallelism based on GPU count
#         for model_category in [config.explorer_models, config.prover_models, 
#                               config.verifier_models, config.translator_models,
#                               config.code_generator_models]:
#             for model_info in model_category.values():
#                 if gpu_count == 1:
#                     model_info["vllm_args"]["tensor_parallel_size"] = 1
#                 elif gpu_count >= 2:
#                     model_info["vllm_args"]["tensor_parallel_size"] = min(2, gpu_count)
#         
#         print(" Kaggle configuration complete!")
#         return config
# 
# # Initialize configuration
# config = Config.for_kaggle() if IS_KAGGLE else Config()


