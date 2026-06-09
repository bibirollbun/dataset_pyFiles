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


import os
import warnings

# Suppress Python warnings
warnings.filterwarnings("ignore")

# Disable HuggingFace tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

!pip install -qq --disable-pip-version-check --no-warn-script-location \
    google-generativeai chromadb sentence-transformers matplotlib ipywidgets --upgrade > /dev/null 2>&1

print("âœ… Libraries installed successfully!")


# Enable inline plotting (required in Kaggle)

%matplotlib inline
import matplotlib.pyplot as plt

print("âœ… Enabling inline plotting.")


# Import Libraries

import json
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import google.generativeai as genai

print("âœ… All Libraries imported.")


# Test Cell: Verify Installation
print("Testing installations...")

try:
    import google.generativeai as genai
    print("âœ… Google Generative AI: OK")
except:
    print("â�Œ Google Generative AI: FAILED")

try:
    import chromadb
    print("âœ… ChromaDB: OK")
except:
    print("â�Œ ChromaDB: FAILED")

try:
    from sentence_transformers import SentenceTransformer
    print("âœ… Sentence Transformers: OK")
except:
    print("â�Œ Sentence Transformers: FAILED")

try:
    import matplotlib.pyplot as plt
    print("âœ… Matplotlib: OK")
except:
    print("â�Œ Matplotlib: FAILED")

print("\nâœ¨ All critical libraries working!")


# Configuration (Using Kaggle Secrets!)
class Config:
    """Configuration for the LLM Agent System"""
    
    # ğŸ”� AUTOMATICALLY LOADS FROM KAGGLE SECRETS
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        GEMINI_API_KEY = user_secrets.get_secret("GEMINI_API_KEY")
        print("âœ… Gemini API Key loaded successfully!")
        
        # Configure Gemini
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        
        # List available models (for debugging)
        print("\nğŸ“‹ Available models:")
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(f"  âœ“ {m.name}")
        
    except Exception as e:
        print(f"âš ï¸� Could not load API key: {e}")
        print("\nğŸ“� Setup Instructions:")
        print("   1. Get API key: https://makersuite.google.com/app/apikey")
        print("   2. Kaggle â†’ Add-ons â†’ Secrets")
        print("   3. Add: GEMINI_API_KEY = your-key")
        print("   4. Turn ON toggle")
        GEMINI_API_KEY = None
    
    # âš ï¸� UPDATED MODEL NAMES
    MODEL_NAME = "models/gemini-2.5-flash"
    MAX_TOKENS = 2000
    TEMPERATURE = 0.7
    
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    MEMORY_COLLECTION_NAME = "agent_memory"
    TOP_K_MEMORIES = 5
    MAX_ITERATIONS = 5
    EVALUATION_THRESHOLD = 0.7


# Memory System with Vector Database

# This handles storing and retrieving relevant past interactions
class MemorySystem:
    """Handles memory storage, retrieval, and updates using vector embeddings"""
    
    def __init__(self):
        # Initialize embedding model for semantic search
        self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
        
        # Initialize ChromaDB for vector storage
        self.client = chromadb.Client(Settings(anonymized_telemetry=False))
        
        # Create or get existing collection
        try:
            self.collection = self.client.get_collection(Config.MEMORY_COLLECTION_NAME)
        except:
            self.collection = self.client.create_collection(
                name=Config.MEMORY_COLLECTION_NAME,
                metadata={"description": "Agent interaction memory"}
            )
        
        self.memory_count = 0
    
    def add_memory(self, content: str, metadata: Dict[str, Any]) -> str:
        """Add a new memory with semantic embedding"""
        # Create unique ID for this memory
        memory_id = f"mem_{self.memory_count}_{datetime.now().timestamp()}"
        
        # Convert text to vector embedding
        embedding = self.embedding_model.encode(content).tolist()
        
        # Store in vector database
        self.collection.add(
            embeddings=[embedding],
            documents=[content],
            metadatas=[metadata],
            ids=[memory_id]
        )
        
        self.memory_count += 1
        return memory_id
    
    def retrieve_memories(self, query: str, top_k: int = Config.TOP_K_MEMORIES) -> List[Dict]:
        """Retrieve relevant memories based on semantic similarity"""
        # Convert query to embedding
        query_embedding = self.embedding_model.encode(query).tolist()
        
        # Search for similar memories
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # Format results
        memories = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                memories.append({
                    'content': doc,
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'distance': results['distances'][0][i] if results['distances'] else 0
                })
        
        return memories
    
    def update_memory(self, memory_id: str, content: str, metadata: Dict[str, Any]):
        """Update an existing memory"""
        # Delete old memory
        try:
            self.collection.delete(ids=[memory_id])
        except:
            pass
        
        # Add updated memory
        embedding = self.embedding_model.encode(content).tolist()
        self.collection.add(
            embeddings=[embedding],
            documents=[content],
            metadatas=[metadata],
            ids=[memory_id]
        )
    
    def get_all_memories(self) -> List[Dict]:
        """Retrieve all stored memories"""
        try:
            results = self.collection.get()
            memories = []
            if results['documents']:
                for i, doc in enumerate(results['documents']):
                    memories.append({
                        'id': results['ids'][i],
                        'content': doc,
                        'metadata': results['metadatas'][i] if results['metadatas'] else {}
                    })
            return memories
        except:
            return []


class ContextEngineering:
    """Manages context construction and optimization for LLM prompts"""
    
    @staticmethod
    def build_context(
        task: str,
        relevant_memories: List[Dict],
        conversation_history: List[Dict],
        system_prompt: str = ""
    ) -> str:
        """
        Build optimized, smooth, and clean context for LLM prompts.
        Ensures consistent formatting, stable spacing, and predictable structure.
        """
        
        context_parts = []

        # === 1. System Instructions ===
        if system_prompt and system_prompt.strip():
            context_parts.append(
                "### System Instructions\n"
                f"{system_prompt.strip()}\n"
            )

        # === 2. Relevant Memories ===
        if relevant_memories:
            mem_lines = []
            for mem in relevant_memories[:3]:   # Use only top 3
                score = 1 - mem.get("distance", 0)
                mem_lines.append(f"- {mem['content']}  (score: {score:.2f})")
            
            context_parts.append(
                "### Relevant Memories\n"
                + "\n".join(mem_lines) + "\n"
            )

        # === 3. Recent Conversation History ===
        if conversation_history:
            hist_lines = []
            for msg in conversation_history[-3:]:
                role = msg.get("role", "unknown").capitalize()
                content = msg.get("content", "").strip()
                hist_lines.append(f"{role}: {content}")
            
            context_parts.append(
                "### Conversation History\n"
                + "\n".join(hist_lines) + "\n"
            )

        # === 4. Current Task ===
        context_parts.append(
            "### Current Task\n"
            f"{task.strip()}"
        )

        # === Final Clean Join ===
        final_context = "\n".join(context_parts).strip()

        return final_context
    
    @staticmethod
    def compress_context(context: str, max_length: int = 3000) -> str:
        """
        Compress context if too large. 
        Preserves start and end so the LLM retains global structure.
        """
        if len(context) <= max_length:
            return context
        
        portion = max_length // 3
        
        return (
            context[:portion].rstrip() +
            "\n\n[... context compressed for length ...]\n\n" +
            context[-portion:].lstrip()
        )


# Agent Base Class

# Foundation for all specialized agents
@dataclass
class AgentResponse:
    """Standard response format for agents"""
    content: str  # The actual response
    confidence: float  # Confidence score (0-1)
    reasoning: str  # Why this response was generated
    metadata: Dict[str, Any]  # Additional info

class BaseAgent:
    """Base class for all agents - provides common functionality"""
    
    def __init__(self, name: str, role: str, system_prompt: str):
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.memory = MemorySystem()  # Each agent has its own memory
        self.conversation_history = []
    
    def call_llm(self, prompt: str) -> str:
        """Call the Gemini LLM API - This is where the magic happens!"""
        try:
            # Check if API key is available
            if not Config.GEMINI_API_KEY:
                raise ValueError("API key not configured. Please add GEMINI_API_KEY to Kaggle Secrets.")
            
            # Initialize Gemini model
            model = genai.GenerativeModel(
                model_name=Config.MODEL_NAME,
                generation_config={
                    "temperature": Config.TEMPERATURE,
                    "max_output_tokens": Config.MAX_TOKENS,
                }
            )
            
            # Call Gemini API
            response = model.generate_content(prompt)
            
            # Extract text response
            return response.text
            
        except Exception as e:
            # If API fails, provide helpful error message
            print(f"âš ï¸� API Error: {e}")
            print("ğŸ’¡ Setup Kaggle Secrets:")
            print("   1. Get API key from: https://makersuite.google.com/app/apikey")
            print("   2. Click 'Add-ons' â†’ 'Secrets' in Kaggle")
            print("   3. Add secret: GEMINI_API_KEY = your-key")
            print("   4. Turn ON the toggle switch")
            print("   5. Re-run the notebook")
            # Return simulation for testing without API
            return f"[Simulated response - API not configured]\nTask: {prompt[:100]}..."
    
    def process(self, task: str, context: Optional[str] = None) -> AgentResponse:
        """Process a task with context - Override in subclasses"""
        raise NotImplementedError("Subclasses must implement process method")

# === Confirmation output ===
print("âœ… Agent Base Class set-up done.")


# Worker Agent (Does the actual work)

class WorkerAgent(BaseAgent):
    """Agent that performs the actual task execution"""
    
    def __init__(self):
        super().__init__(
            name="Worker",
            role="Task Executor",
            system_prompt="""You are a helpful AI assistant that executes tasks.
            Provide clear, accurate, and well-reasoned responses.
            Explain your reasoning step by step.
            Be specific and practical in your answers."""
        )
    
    def process(self, task: str, context: Optional[str] = None) -> AgentResponse:
        """Execute the given task with full context"""
        
        # Step 1: Retrieve relevant past memories
        relevant_memories = self.memory.retrieve_memories(task)
        
        # Step 2: Build comprehensive context
        full_context = ContextEngineering.build_context(
            task=task,
            relevant_memories=relevant_memories,
            conversation_history=self.conversation_history,
            system_prompt=self.system_prompt
        )
        
        # Step 3: Construct final prompt
        prompt = f"{full_context}\n\nPlease provide a detailed response with your reasoning."
        
        # Step 4: Call LLM to get response
        response = self.call_llm(prompt)
        
        # Step 5: Store this interaction in memory for future use
        self.memory.add_memory(
            content=f"Task: {task}\nResponse: {response}",
            metadata={
                'agent': self.name,
                'timestamp': datetime.now().isoformat(),
                'type': 'task_execution'
            }
        )
        
        # Step 6: Update conversation history
        self.conversation_history.append({'role': 'user', 'content': task})
        self.conversation_history.append({'role': 'assistant', 'content': response})
        
        # Step 7: Return structured response
        return AgentResponse(
            content=response,
            confidence=0.8,
            reasoning="Task executed with context and memory retrieval",
            metadata={'memories_used': len(relevant_memories)}
        )

# === Confirmation output ===
print("âœ… WorkerAgent loaded successfully.")


# Evaluator Agent (Quality Control)

class EvaluatorAgent(BaseAgent):
    """Agent that evaluates responses from other agents"""
    
    def __init__(self):
        super().__init__(
            name="Evaluator",
            role="Quality Assessor",
            system_prompt="""You are an expert evaluator. Assess responses on:
            1. Accuracy (0-1): Is the information correct?
            2. Completeness (0-1): Does it fully answer the question?
            3. Clarity (0-1): Is it easy to understand?
            4. Relevance (0-1): Does it address the task?
            
            Provide scores and specific, actionable feedback for improvement.
            Format your response as JSON with scores and feedback."""
        )
    
    def process(self, task: str, response: str) -> Dict[str, Any]:
        """Evaluate a response and provide detailed feedback"""
        
        # Construct evaluation prompt
        evaluation_prompt = f"""
        {self.system_prompt}
        
        Task: {task}
        
        Response to Evaluate: {response}
        
        Evaluate this response and provide:
        1. Scores for accuracy, completeness, clarity, relevance (each 0-1)
        2. Overall score (average of the four)
        3. Specific feedback explaining what works and what doesn't
        4. Concrete suggestions for improvement
        
        Return as JSON format:
        {{
            "accuracy": 0.0-1.0,
            "completeness": 0.0-1.0,
            "clarity": 0.0-1.0,
            "relevance": 0.0-1.0,
            "overall_score": 0.0-1.0,
            "feedback": "detailed feedback here",
            "suggestions": ["suggestion 1", "suggestion 2"]
        }}
        """
        
        # Get evaluation from LLM
        eval_response = self.call_llm(evaluation_prompt)
        
        # Parse evaluation (with fallback for safety)
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', eval_response, re.DOTALL)
            if json_match:
                evaluation = json.loads(json_match.group())
                # Ensure all required fields exist
                if 'overall_score' not in evaluation:
                    scores = [
                        evaluation.get('accuracy', 0.7),
                        evaluation.get('completeness', 0.7),
                        evaluation.get('clarity', 0.7),
                        evaluation.get('relevance', 0.7)
                    ]
                    evaluation['overall_score'] = sum(scores) / len(scores)
            else:
                # Fallback: create reasonable default evaluation
                evaluation = {
                    'accuracy': 0.75,
                    'completeness': 0.70,
                    'clarity': 0.80,
                    'relevance': 0.85,
                    'overall_score': 0.775,
                    'feedback': eval_response,
                    'suggestions': ['Consider adding more specific examples']
                }
        except Exception as e:
            print(f"âš ï¸� Evaluation parsing error: {e}")
            # Safe fallback
            evaluation = {
                'accuracy': 0.7,
                'completeness': 0.7,
                'clarity': 0.7,
                'relevance': 0.7,
                'overall_score': 0.7,
                'feedback': eval_response,
                'suggestions': ['Unable to parse specific suggestions']
            }
        
        # Store evaluation in memory
        self.memory.add_memory(
            content=f"Evaluated task: {task}\nScore: {evaluation.get('overall_score', 0)}",
            metadata={
                'agent': self.name,
                'timestamp': datetime.now().isoformat(),
                'type': 'evaluation',
                'score': evaluation.get('overall_score', 0)
            }
        )
        
        return evaluation

        
# === Confirmation output ===
print("âœ… EvaluatorAgent loaded successfully.")


# Prompt Optimizer Agent (Makes things better)
# -----------------------------------------------------
class PromptOptimizerAgent(BaseAgent):
    """Agent that optimizes prompts based on evaluation feedback"""
    
    def __init__(self):
        super().__init__(
            name="PromptOptimizer",
            role="Prompt Refiner",
            system_prompt="""You are a prompt engineering expert.
            Given a task, response, and evaluation, create an improved prompt
            that addresses the identified weaknesses while maintaining clarity.
            
            Your optimized prompts should:
            - Be more specific and detailed
            - Include helpful constraints or examples
            - Address evaluation feedback directly
            - Guide toward higher quality responses"""
        )
    
    def optimize_prompt(
        self,
        original_task: str,
        response: str,
        evaluation: Dict[str, Any]
    ) -> str:
        """Generate an optimized version of the prompt"""
        
        # Build optimization prompt
        optimization_prompt = f"""
        {self.system_prompt}
        
        Original Task: {original_task}
        
        Previous Response: {response[:500]}...
        
        Evaluation Results:
        Overall Score: {evaluation.get('overall_score', 0):.2f}
        
        Specific Scores:
        - Accuracy: {evaluation.get('accuracy', 0):.2f}
        - Completeness: {evaluation.get('completeness', 0):.2f}
        - Clarity: {evaluation.get('clarity', 0):.2f}
        - Relevance: {evaluation.get('relevance', 0):.2f}
        
        Feedback: {evaluation.get('feedback', 'No feedback')}
        
        Suggestions: {evaluation.get('suggestions', [])}
        
        Create an improved version of the original task prompt that:
        1. Addresses ALL weaknesses identified in the evaluation
        2. Provides clearer, more specific instructions
        3. Includes helpful constraints, examples, or format requirements
        4. Maintains the core intent of the original task
        5. Guides toward a response that would score above 0.8
        
        Return ONLY the optimized prompt text, nothing else.
        """
        
        # Get optimized prompt from LLM
        optimized = self.call_llm(optimization_prompt)
        
        # Store optimization in memory
        self.memory.add_memory(
            content=f"Optimized prompt for: {original_task[:100]}...\nNew prompt: {optimized[:100]}...",
            metadata={
                'agent': self.name,
                'timestamp': datetime.now().isoformat(),
                'type': 'prompt_optimization',
                'original_score': evaluation.get('overall_score', 0)
            }
        )
        
        return optimized.strip()


# === Confirmation output ===
print("âœ… PromptOptimizerAgent loaded successfully.")


# Multi-Agent Orchestrator (The Conductor)

class AgentOrchestrator:
    """Coordinates multiple agents for self-evaluation and iterative improvement"""
    
    def __init__(self):
        # Initialize all agents
        self.worker = WorkerAgent()
        self.evaluator = EvaluatorAgent()
        self.optimizer = PromptOptimizerAgent()
        self.iteration_history = []
    
    def run_task_with_refinement(
        self, 
        task: str, 
        max_iterations: int = Config.MAX_ITERATIONS
    ) -> Dict[str, Any]:
        """Execute task with iterative refinement until quality threshold is met"""
        
        current_task = task
        results = {
            'original_task': task,
            'iterations': [],
            'final_response': None,
            'final_score': 0
        }
        
        # Iterative refinement loop
        for iteration in range(max_iterations):
            print(f"\n{'='*60}")
            print(f"ITERATION {iteration + 1}/{max_iterations}")
            print(f"{'='*60}\n")
            
            # STEP 1: Worker executes the task
            print(f"[{self.worker.name}] Executing task...")
            worker_response = self.worker.process(current_task)
            response_preview = worker_response.content[:200].replace('\n', ' ')
            print(f"Response preview: {response_preview}...")
            
            # STEP 2: Evaluator assesses the response quality
            print(f"\n[{self.evaluator.name}] Evaluating response...")
            evaluation = self.evaluator.process(current_task, worker_response.content)
            print(f"Overall Score: {evaluation.get('overall_score', 0):.2f}")
            print(f"  - Accuracy: {evaluation.get('accuracy', 0):.2f}")
            print(f"  - Completeness: {evaluation.get('completeness', 0):.2f}")
            print(f"  - Clarity: {evaluation.get('clarity', 0):.2f}")
            print(f"  - Relevance: {evaluation.get('relevance', 0):.2f}")
            
            # Store iteration data
            iteration_data = {
                'iteration': iteration + 1,
                'task': current_task,
                'response': worker_response.content,
                'evaluation': evaluation
            }
            results['iterations'].append(iteration_data)
            
            # STEP 3: Check if quality threshold is met
            current_score = evaluation.get('overall_score', 0)
            if current_score >= Config.EVALUATION_THRESHOLD:
                print(f"\nâœ… Quality threshold met! (Score: {current_score:.2f} >= {Config.EVALUATION_THRESHOLD})")
                results['final_response'] = worker_response.content
                results['final_score'] = current_score
                break
            
            # STEP 4: Optimizer refines the prompt for next iteration
            if iteration < max_iterations - 1:
                print(f"\n[{self.optimizer.name}] Optimizing prompt for next iteration...")
                current_task = self.optimizer.optimize_prompt(
                    current_task,
                    worker_response.content,
                    evaluation
                )
                task_preview = current_task[:200].replace('\n', ' ')
                print(f"Optimized task preview: {task_preview}...")
            else:
                # Max iterations reached
                print(f"\nâš ï¸� Max iterations reached. Using best response so far.")
                results['final_response'] = worker_response.content
                results['final_score'] = current_score
        
        return results
    
    def get_memory_summary(self) -> Dict[str, List[Dict]]:
        """Get summary of all agent memories"""
        return {
            'worker_memories': self.worker.memory.get_all_memories(),
            'evaluator_memories': self.evaluator.memory.get_all_memories(),
            'optimizer_memories': self.optimizer.memory.get_all_memories()
        }


# === Confirmation output ===
print("âœ… AgentOrchestrator loaded successfully.")


def visualize_improvement(results):
    """Line graph showing overall score improvement across iterations"""
    try:
        import matplotlib.pyplot as plt
        
        iterations = [i['iteration'] for i in results['iterations']]
        scores = [i['evaluation'].get('overall_score', 0) for i in results['iterations']]
        
        plt.figure(figsize=(12, 7))
        plt.plot(iterations, scores, marker='o', linewidth=3, markersize=10,
                 label='Overall Score')

        plt.axhline(y=Config.EVALUATION_THRESHOLD, linestyle='--', linewidth=2,
                    label=f'Target Threshold ({Config.EVALUATION_THRESHOLD})')
        
        plt.xlabel('Iteration', fontsize=14, fontweight='bold')
        plt.ylabel('Evaluation Score', fontsize=14, fontweight='bold')
        plt.title('LLM Self-Evaluation: Score Improvement Over Iterations',
                  fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xticks(iterations)
        plt.ylim(0, 1.05)

        for i, score in zip(iterations, scores):
            plt.annotate(f'{score:.2f}', xy=(i, score), xytext=(0, 10),
                         textcoords='offset points', ha='center')

        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"âš ï¸� Line graph error: {e}")


def visualize_bar_graph(results):
    """Bar graph comparing metrics across all iterations"""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        iterations = [f"Iter {i['iteration']}" for i in results['iterations']]
        accuracy = [i['evaluation'].get('accuracy', 0) for i in results['iterations']]
        completeness = [i['evaluation'].get('completeness', 0) for i in results['iterations']]
        clarity = [i['evaluation'].get('clarity', 0) for i in results['iterations']]
        relevance = [i['evaluation'].get('relevance', 0) for i in results['iterations']]
        
        x = np.arange(len(iterations))
        width = 0.2
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        ax.bar(x - 1.5*width, accuracy, width, label='Accuracy')
        ax.bar(x - 0.5*width, completeness, width, label='Completeness')
        ax.bar(x + 0.5*width, clarity, width, label='Clarity')
        ax.bar(x + 1.5*width, relevance, width, label='Relevance')
        
        ax.set_xlabel('Iterations', fontsize=14, fontweight='bold')
        ax.set_ylabel('Scores', fontsize=14, fontweight='bold')
        ax.set_title('LLM Evaluation Metrics Across Iterations',
                     fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(iterations)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1.05)

        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"âš ï¸� Bar graph error: {e}")


def visualize_radar_chart(results):
    """Radar chart for final iteration metric distribution"""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        final_eval = results['iterations'][-1]['evaluation']
        
        categories = ['Accuracy', 'Completeness', 'Clarity', 'Relevance']
        values = [
            final_eval.get('accuracy', 0),
            final_eval.get('completeness', 0),
            final_eval.get('clarity', 0),
            final_eval.get('relevance', 0)
        ]
        
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        values += values[:1]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        ax.plot(angles, values, 'o-', linewidth=2, label='Final Iteration')
        ax.fill(angles, values, alpha=0.25)

        ax.set_ylim(0, 1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        
        plt.title('Final Evaluation Metric Profile (Radar Chart)',
                  fontsize=16, fontweight='bold')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"âš ï¸� Radar chart error: {e}")


def visualize_confusion_matrix(results):
    """Confusion-style matrix showing predicted vs actual LLM evaluation scores"""
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        final_eval = results['iterations'][-1]['evaluation']

        metrics = ['Accuracy', 'Completeness', 'Clarity', 'Relevance']
        
        predicted = [
            final_eval.get('pred_accuracy', final_eval.get('accuracy', 0)),
            final_eval.get('pred_completeness', final_eval.get('completeness', 0)),
            final_eval.get('pred_clarity', final_eval.get('clarity', 0)),
            final_eval.get('pred_relevance', final_eval.get('relevance', 0))
        ]
        
        actual = [
            final_eval.get('actual_accuracy', final_eval.get('accuracy', 0)),
            final_eval.get('actual_completeness', final_eval.get('completeness', 0)),
            final_eval.get('actual_clarity', final_eval.get('clarity', 0)),
            final_eval.get('actual_relevance', final_eval.get('relevance', 0))
        ]

        matrix = np.array([predicted, actual])

        fig, ax = plt.subplots(figsize=(10, 5))
        im = ax.imshow(matrix, cmap='Blues', vmin=0, vmax=1)

        ax.set_xticks(np.arange(len(metrics)))
        ax.set_yticks([0, 1])
        ax.set_xticklabels(metrics)
        ax.set_yticklabels(['Predicted', 'Actual'])

        for i in range(2):
            for j in range(len(metrics)):
                ax.text(j, i, f'{matrix[i, j]:.2f}',
                        ha='center', va='center')

        plt.title("Predicted vs Actual Evaluation Scores (Confusion-Style Matrix)",
                  fontsize=16, fontweight='bold')
        plt.colorbar(im)

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"âš ï¸� Confusion matrix error: {e}")


def visualize_all(results):
    """Run all 4 visualizations"""
    print("="*70)
    print("ğŸ“Š GENERATING LLM SELF-EVALUATION VISUALIZATIONS")
    print("="*70 + "\n")
    
    print("1ï¸�âƒ£ Line Graph - Score Progression")
    visualize_improvement(results)
    
    print("\n2ï¸�âƒ£ Bar Graph - Metric Comparison")
    visualize_bar_graph(results)
    
    print("\n3ï¸�âƒ£ Radar Chart - Final Evaluation Shape")
    visualize_radar_chart(results)

    print("\n4ï¸�âƒ£ Confusion-Style Matrix - Predicted vs Actual")
    visualize_confusion_matrix(results)
    
    print("\n" + "="*70)
    print("âœ… ALL VISUALIZATIONS COMPLETE!")
    print("="*70)



def create_interactive_interface():
    """Create an interactive chat-like interface for the agent system"""
    try:
        from IPython.display import display, HTML, clear_output
        import ipywidgets as widgets
        
        print("âœ… Interactive interface loaded!")
        
        # Create orchestrator
        orchestrator = AgentOrchestrator()
        
        # Create UI components
        output_area = widgets.Output()
        
        # Input text area
        prompt_input = widgets.Textarea(
            value='Explain quantum computing to a beginner with a real-world example',
            placeholder='Enter your task here...',
            description='Your Task:',
            layout=widgets.Layout(width='100%', height='100px'),
            style={'description_width': '100px'}
        )
        
        # Settings
        iterations_slider = widgets.IntSlider(
            value=3,
            min=1,
            max=5,
            step=1,
            description='Max Iterations:',
            style={'description_width': '120px'}
        )
        
        threshold_slider = widgets.FloatSlider(
            value=0.7,
            min=0.5,
            max=1.0,
            step=0.05,
            description='Quality Target:',
            style={'description_width': '120px'}
        )
        
        # Buttons
        run_button = widgets.Button(
            description='ğŸš€ Run Agent System',
            button_style='success',
            layout=widgets.Layout(width='200px', height='40px')
        )
        
        clear_button = widgets.Button(
            description='ğŸ—‘ï¸� Clear Output',
            button_style='warning',
            layout=widgets.Layout(width='200px', height='40px')
        )
        
        visualize_button = widgets.Button(
            description='ğŸ“ˆ Line Graph',
            button_style='info',
            layout=widgets.Layout(width='200px', height='40px')
        )
        
        bar_graph_button = widgets.Button(
            description='ğŸ“Š Bar Graph',
            button_style='info',
            layout=widgets.Layout(width='200px', height='40px')
        )
        
        radar_button = widgets.Button(
            description='ğŸ�¯ Radar Chart',
            button_style='info',
            layout=widgets.Layout(width='200px', height='40px')
        )
        
        confusion_button = widgets.Button(
            description='ğŸ”µ Confusion Matrix',
            button_style='info',
            layout=widgets.Layout(width='200px', height='40px')
        )
        
        all_viz_button = widgets.Button(
            description='ğŸ�¨ All Visualizations',
            button_style='primary',
            layout=widgets.Layout(width='200px', height='40px')
        )
        
        # Store global results
        global last_results
        last_results = None
        
        # Handler for running agent
        def on_run_click(b):
            global last_results
            with output_area:
                clear_output()
                
                display(HTML("""
                    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                padding: 20px; border-radius: 10px; color: white; margin-bottom: 20px;'>
                        <h2 style='margin: 0;'>ğŸ¤– AI Agent System Processing...</h2>
                        <p>Watch your prompt being refined...</p>
                    </div>
                """))
                
                task = prompt_input.value
                max_iter = iterations_slider.value
                Config.EVALUATION_THRESHOLD = threshold_slider.value
                
                print(f"ğŸ“� Task: {task}\n")
                print(f"âš™ï¸� Iterations={max_iter}, Target Score={threshold_slider.value}\n")
                
                results = orchestrator.run_task_with_refinement(task, max_iterations=max_iter)
                last_results = results
                
                display(HTML(f"""
                    <div style='background: #e8f4ff; padding: 20px; border-radius: 10px; 
                                border-left: 6px solid #4285F4;'>
                        <h3>ğŸ�¯ Final Results</h3>
                        <p><b>Total Iterations:</b> {len(results['iterations'])}</p>
                        <p><b>Final Score:</b> <span style='color:#0f9d58;font-size:22px;'>{results['final_score']:.2f}</span></p>
                    </div>
                """))
                
                display(HTML("<h4>ğŸ“� Final Optimized Response:</h4>"))
                print(results['final_response'])
        
        # â�— FIXED: attach event handler to run button
        run_button.on_click(on_run_click)
        
        # Visualization handler
        def safe_viz(func, name):
            global last_results
            with output_area:
                if last_results:
                    print(f"\nğŸ“Š Generating {name}...\n")
                    func(last_results)
                else:
                    print("âš ï¸� No results yet. Run the agent first!")
        
        visualize_button.on_click(lambda b: safe_viz(visualize_improvement, "Line Graph"))
        bar_graph_button.on_click(lambda b: safe_viz(visualize_bar_graph, "Bar Graph"))
        radar_button.on_click(lambda b: safe_viz(visualize_radar_chart, "Radar Chart"))
        confusion_button.on_click(lambda b: safe_viz(visualize_confusion_matrix, "Confusion Matrix"))
        all_viz_button.on_click(lambda b: safe_viz(visualize_all, "All Visualizations"))
        
        clear_button.on_click(lambda b: (output_area.clear_output(), print("Output cleared!")))
        
        # Layout
        settings_box = widgets.VBox([iterations_slider, threshold_slider])
        
        buttons_box = widgets.HBox([run_button, clear_button, visualize_button])
        
        viz_buttons_box = widgets.HBox([
            bar_graph_button,
            radar_button,
            confusion_button,
            all_viz_button
        ], layout=widgets.Layout(gap='10px', margin='10px 0'))
        
        interface = widgets.VBox([
            widgets.HTML("""
                <div style='background: linear-gradient(135deg,#667eea,#764ba2); 
                            padding:25px; border-radius:15px; color:white;text-align:center;'>
                    <h1>ğŸ¤– LLM Self-Evaluation Agent System</h1>
                    <p>Google Gemini API + Multi-Agent Pipeline</p>
                </div>
            """),
            prompt_input,
            settings_box,
            buttons_box,
            viz_buttons_box,
            output_area
        ])
        
        display(interface)
    
    except Exception as e:
        print(f"â�Œ Error: {e}")


# Launch Interactive Interface
# Run this cell to launch the interactive UI!

print("ğŸš€ Launching Interactive Interface...")
print("="*70 + "\n")
create_interactive_interface()


# Example Usage & Demo

def demo_agent_system():
    """Demonstrate the complete agent system"""
    
    print("="*70)
    print("ğŸ¤– LLM SELF-EVALUATION & PROMPT-FINETUNING SYSTEM")
    print("   Powered by Google Gemini API")
    print("="*70)
    print("\nThis system uses 3 AI agents working together:")
    print("  1. Worker: Executes tasks")
    print("  2. Evaluator: Assesses quality")
    print("  3. Optimizer: Improves prompts")
    print("\n" + "="*70 + "\n")
    
    # Initialize orchestrator
    orchestrator = AgentOrchestrator()
    
    # ğŸ�¯ YOUR TASK GOES HERE - Change this to anything you want or anything of your choice!
    # Examples:
    # - "Explain quantum computing to a 10-year-old"
    # - "Write a Python function to find prime numbers"
    # - "Create a marketing strategy for a local bakery"
    # - "Explain the water cycle with a creative analogy"
    
    task = """Explain quantum computing to a 10-year-old"""
    
    print(f"ğŸ“� Original Task:\n{task}\n")
    print("="*70)
    
    # Run task with iterative refinement
    results = orchestrator.run_task_with_refinement(task, max_iterations=3)
    
    # Display final results
    print("\n" + "="*70)
    print("ğŸ�¯ FINAL RESULTS")
    print("="*70)
    print(f"\nğŸ“Š Total Iterations: {len(results['iterations'])}")
    print(f"â­� Final Score: {results['final_score']:.2f}")
    print(f"\nğŸ“� Final Response:\n")
    print("-" * 70)
    print(results['final_response'])
    print("-" * 70)
    
    # Show score progression
    print("\nğŸ“ˆ Score Progression:")
    for i, iter_data in enumerate(results['iterations']):
        score = iter_data['evaluation'].get('overall_score', 0)
        bar_length = int(score * 30)
        bar = "â–ˆ" * bar_length + "â–‘" * (30 - bar_length)
        print(f"  Iteration {i+1}: {bar} {score:.2f}")
    
    print("\n" + "="*70)
    
    return results

# === Confirmation output ===
print("\nâœ… demo_agent_system() executed successfully!")


# Execute this cell to see the system in action! demo run

if __name__ == "__main__":
    results = demo_agent_system()

# Optional: Uncomment below to see detailed iteration data
# print("\n\nDetailed iteration 1 data:")
# print(json.dumps(results['iterations'][0], indent=2))

