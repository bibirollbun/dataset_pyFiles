# ==================== ğŸš€ IMPORTS & CONFIGURATION ğŸš€ ====================
import os
# Suppress all TensorFlow and CUDA warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['NVIDIA_TF32_OVERRIDE'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '0'
os.environ['XLA_FLAGS'] = '--xla_gpu_force_compilation_parallelism=1'

# Configure logging before importing any other packages
import logging
logging.basicConfig(level=logging.CRITICAL)
logging.getLogger().setLevel(logging.CRITICAL)
logging.getLogger('transformers').setLevel(logging.CRITICAL)
logging.getLogger('torch').setLevel(logging.CRITICAL)
logging.getLogger('numba').setLevel(logging.CRITICAL)

# Disable all warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Main imports
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict, deque
from sympy import symbols, Eq, solve
from tqdm.auto import tqdm
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import cv2
cv2.setLogLevel(0)  # Disable OpenCV logging
import hashlib
import time
from functools import lru_cache
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from numba import jit
import sympy as sp
import unittest
from transformers import ViTModel, ViTConfig, AutoModel, AutoTokenizer
from enum import Enum, auto
import zlib
from difflib import SequenceMatcher
from scipy.ndimage import label, generate_binary_structure
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import networkx as nx

# Configure PyTorch
torch.set_num_threads(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# Initialize CUDA carefully
if torch.cuda.is_available():
    try:
        torch.cuda.init()
        torch.cuda.empty_cache()
        # Additional CUDA configurations
        torch._C._jit_set_nvfuser_enabled(False)
        torch._C._jit_set_texpr_fuser_enabled(False)
        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_set_profiling_mode(False)
    except Exception:
        pass


# ==================== âš™ï¸� HYPERPARAMETER CONFIGURATION âš™ï¸� ====================
class Config:
    """Optimized configuration for competition performance"""
    
    # System paths
    DATA_PATH = '/kaggle/input/arc-prize-2025'
    MODEL_CACHE = '/kaggle/working/model_cache'
    SUBMISSION_FILE = '/kaggle/working/submission.json'
    
    # Hardware settings
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    VERBOSE = True
    DEBUG_MODE = False
    
    # Image processing
    IMAGE = {
        'target_size': 64,
        'color_clusters': 12,
        'kmeans_init': 20,
        'contour_min_area': 2,
        'default_channels': 3,
        'patch_size': 8
    }
    
    # Model parameters
    MODEL = {
        'vit_hidden_size': 256,
        'vit_num_layers': 6,
        'vit_num_heads': 8,
        'cnn_channels': [32, 64, 128],
        'llm_model': 'microsoft/phi-2',
        'llm_max_length': 512,
        'embedding_size': 128
    }
    
    # Learning parameters
    LEARNING = {
        'meta_learning_rate': 1e-4,
        'max_train_iter': 10,
        'hypothesis_samples': 5,
        'few_shot_k': 3
    }
    
    # Performance tuning
    PERFORMANCE = {
        'max_workers': 8,
        'timeout_per_task': 30,
        'shape_cache_size': 10000,
        'pattern_cache_size': 5000,
        'max_rule_depth': 5
    }
    
    # Thresholds
    THRESHOLDS = {
        'similarity': 0.92,
        'color_mapping': 0.95,
        'shape_match': 0.88,
        'min_confidence': 0.85,
        'llm_confidence': 0.7
    }


# ==================== ğŸ§  CORE MODELS ARCHITECTURE ğŸ§  ====================
class DynamicViT(nn.Module):
    """Vision Transformer with Dynamic Patch Processing"""
    def __init__(self):
        super().__init__()
        config = ViTConfig(
            image_size=Config.IMAGE['target_size'],
            patch_size=Config.IMAGE['patch_size'],
            num_classes=0,
            hidden_size=Config.MODEL['vit_hidden_size'],
            num_hidden_layers=Config.MODEL['vit_num_layers'],
            num_attention_heads=Config.MODEL['vit_num_heads'],
            num_channels=Config.IMAGE['default_channels']
        )
        self.vit = ViTModel(config).to(Config.DEVICE)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x):
        features = self.vit(x).last_hidden_state
        features = features.permute(0, 2, 1).view(features.size(0), 
                    Config.MODEL['vit_hidden_size'],
                    Config.IMAGE['target_size']//Config.IMAGE['patch_size'],
                    Config.IMAGE['target_size']//Config.IMAGE['patch_size'])
        return self.adaptive_pool(features).squeeze()

class MultiScaleCNN(nn.Module):
    """Hierarchical Feature Extraction CNN"""
    def __init__(self):
        super().__init__()
        layers = []
        in_channels = 1
        for out_channels in Config.MODEL['cnn_channels']:
            layers.extend([
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(2)
            ])
            in_channels = out_channels
        self.network = nn.Sequential(*layers)
        self.final_pool = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x):
        return self.final_pool(self.network(x)).squeeze()

class HybridReasoner(nn.Module):
    """Multi-Modal Reasoning Module"""
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, Config.MODEL['embedding_size'])
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# ==================== ğŸ�† ENUMERATIONS & TYPES ğŸ�† ====================
class SolutionStrategy(Enum):
    COLOR_MAPPING = auto()
    SPATIAL_TRANSFORM = auto()
    PATTERN_MATCHING = auto()
    SYMBOLIC_REASONING = auto()
    DEEP_LEARNING = auto()
    HYBRID_APPROACH = auto()
    LLM_REASONING = auto()
    GRAPH_BASED = auto()
    META_REASONING = auto()

    def __str__(self):
        return self.name.replace('_', ' ').title()

class TaskComplexity(Enum):
    SIMPLE = auto()
    MODERATE = auto()
    COMPLEX = auto()
    EXTREME = auto()


# ==================== ğŸ�¨ ADVANCED COLOR PROCESSING ğŸ�¨ ====================
class ColorEngine:
    """Next-Gen Color Analysis with Semantic Understanding"""
    
    def __init__(self):
        self.palette = self._init_palette()
        self.cached_mappings = {}
        
    def _init_palette(self):
        """Enhanced color palette with semantic meanings"""
        return {
            0: ('Black', (0, 0, 0)),
            1: ('Blue', (0, 116, 217)),
            2: ('Red', (255, 65, 54)),
            3: ('Green', (46, 204, 64)),
            4: ('Yellow', (255, 220, 0)),
            5: ('Gray', (170, 170, 170)),
            6: ('Pink', (240, 18, 190)),
            7: ('Orange', (255, 133, 27)),
            8: ('Light Blue', (127, 219, 255)),
            9: ('Dark Red', (135, 12, 37)),
            10: ('White', (255, 255, 255)),
            11: ('Purple', (128, 0, 128))
        }
    
    def semantic_color_map(self, input_grid, output_grid):
        """Advanced color mapping with semantic relationships"""
        input_colors = np.unique(input_grid)
        output_colors = np.unique(output_grid)
        
        # First try exact pixel mapping
        mapping = {}
        remaining_input = set(input_colors)
        remaining_output = set(output_colors)
        
        for ic in input_colors:
            for oc in output_colors:
                if np.array_equal(np.where(input_grid == ic, 1, 0),
                                 np.where(output_grid == oc, 1, 0)):
                    mapping[int(ic)] = int(oc)
                    remaining_input.discard(ic)
                    remaining_output.discard(oc)
        
        # Semantic mapping for remaining colors
        for ic in remaining_input:
            ic_semantic = self.palette[ic][0]
            best_match = None
            best_score = 0
            
            for oc in remaining_output:
                oc_semantic = self.palette[oc][0]
                similarity = SequenceMatcher(None, ic_semantic, oc_semantic).ratio()
                if similarity > best_score:
                    best_score = similarity
                    best_match = oc
            
            if best_match and best_score > Config.THRESHOLDS['color_mapping']:
                mapping[int(ic)] = int(best_match)
                remaining_output.discard(best_match)
        
        return mapping



# ==================== ğŸ”� SHAPE & STRUCTURE ANALYSIS ğŸ”� ====================
class ShapeAnalyzerPro:
    """Advanced Shape and Topology Analysis"""
    
    def __init__(self):
        self.cache = {}
        self.structure = generate_binary_structure(2, 2)
        
    def extract_topology(self, grid):
        """Extract topological features including connectivity and holes"""
        binary = (grid > 0).astype(np.uint8)
        labeled, num_features = label(binary, structure=self.structure)
        
        features = {
            'component_count': num_features,
            'component_sizes': [],
            'component_centroids': [],
            'holes': 0
        }
        
        for i in range(1, num_features + 1):
            component = (labeled == i)
            features['component_sizes'].append(np.sum(component))
            features['component_centroids'].append(self._calculate_centroid(component))
        
        # Calculate holes using Euler characteristic
        if num_features > 0:
            inverted = 1 - binary
            labeled_inv, num_inv = label(inverted, structure=self.structure)
            features['holes'] = num_inv - num_features
        
        return features
    
    def _calculate_centroid(self, component):
        """Calculate centroid of a connected component"""
        rows, cols = np.where(component)
        return (np.mean(rows), np.mean(cols))


# ==================== ğŸ§© PATTERN RECOGNITION ENGINE ğŸ§© ====================
class PatternMaster:
    """Hierarchical Pattern Recognition with Graph-Based Matching"""
    
    def __init__(self):
        self.pattern_db = []
        self.graph_db = []
        self.similarity_cache = {}
        
    def build_pattern_graph(self, grid):
        """Convert grid to a graph representation"""
        binary = (grid > 0).astype(np.uint8)
        labeled, num_features = label(binary)
        
        G = nx.Graph()
        centroids = []
        
        for i in range(1, num_features + 1):
            component = (labeled == i)
            rows, cols = np.where(component)
            centroid = (np.mean(rows), np.mean(cols))
            area = len(rows)
            G.add_node(i, pos=centroid, area=area)
            centroids.append(centroid)
        
        # Connect nodes based on spatial relationships
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                dist = np.linalg.norm(np.array(centroids[i]) - np.array(centroids[j]))
                G.add_edge(i + 1, j + 1, weight=1/dist)
        
        return G
    
    def graph_similarity(self, G1, G2):
        """Calculate similarity between two graphs"""
        def extract_features(G):
            return {
                'degrees': sorted([d for n, d in G.degree()]),
                'clustering': nx.clustering(G),
                'centrality': nx.degree_centrality(G)
            }
        
        f1 = extract_features(G1)
        f2 = extract_features(G2)
        
        # Compare degree sequences
        degree_sim = cosine_similarity(
            [list(f1['degrees'])], 
            [list(f2['degrees'])]
        )[0][0]
        
        # Compare clustering coefficients
        clust_sim = cosine_similarity(
            [list(f1['clustering'].values())],
            [list(f2['clustering'].values())]
        )[0][0]
        
        return (degree_sim + clust_sim) / 2


# ==================== ğŸ¤– LLM REASONING MODULE ğŸ¤– ====================
class LLMReasoner:
    """Language-Guided Reasoning with Small LLMs"""
    
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL['llm_model'])
        self.model = AutoModel.from_pretrained(Config.MODEL['llm_model']).to(Config.DEVICE)
        self.prompt_template = """Analyze this ARC task:
Input Grid: {input_grid}
Possible Output Patterns: {patterns}
Identify the transformation rules and suggest the output grid."""
    
    def generate_hypothesis(self, input_grid, patterns):
        """Generate reasoning hypotheses using LLM"""
        prompt = self.prompt_template.format(
            input_grid=str(input_grid),
            patterns=str(patterns[:3])  # Show top 3 patterns
        )
        
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            max_length=Config.MODEL['llm_max_length'],
            truncation=True
        ).to(Config.DEVICE)
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=100)
        
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self._parse_llm_output(decoded)
    
    def _parse_llm_output(self, text):
        """Extract structured rules from LLM output"""
        # Implementation of text parsing to extract transformation rules
        rules = []
        if "rotate" in text.lower():
            if "90" in text:
                rules.append(('rotate', 90))
            elif "180" in text:
                rules.append(('rotate', 180))
        if "flip" in text.lower():
            rules.append(('flip', 'horizontal'))
        return rules


# ==================== ğŸ› ï¸� FALLBACK GENERATOR CLASS ğŸ› ï¸� ====================
class FallbackGenerator:
    """Generates fallback solutions when primary methods fail"""
    
    def __init__(self):
        self.color_palette = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # Basic color palette
        self.shape_templates = [
            np.zeros((5, 5)), 
            np.ones((5, 5)),
            np.eye(5),
            np.array([[1 if (i+j)%2==0 else 0 for j in range(5)] for i in range(5)])
        ]
    
    def generate(self, input_grid):
        """Generate fallback solution based on input grid characteristics"""
        try:
            # Analyze input grid
            input_colors = np.unique(input_grid)
            input_shape = (input_grid > 0).astype(int)
            
            # Strategy 1: Try to preserve color distribution
            if len(input_colors) > 1:
                output = self._preserve_color_pattern(input_grid)
                return {
                    'output_1': output,
                    'output_2': self._random_symmetrical_pattern(input_grid),
                    'confidence': 0.5,
                    'strategy': 'ColorPreservation'
                }
            
            # Strategy 2: Try to preserve shape
            labeled, num_features = label(input_shape)
            if num_features > 0:
                output = self._preserve_shape_pattern(input_grid)
                return {
                    'output_1': output,
                    'output_2': self._random_symmetrical_pattern(input_grid),
                    'confidence': 0.4,
                    'strategy': 'ShapePreservation'
                }
            
            # Default fallback
            return self._default_fallback()
            
        except Exception:
            return self._default_fallback()
    
    def _preserve_color_pattern(self, input_grid):
        """Generate output preserving color patterns"""
        output = np.zeros_like(input_grid)
        color_map = {c: np.random.choice(self.color_palette) for c in np.unique(input_grid)}
        for c in color_map:
            output[input_grid == c] = color_map[c]
        return output
    
    def _preserve_shape_pattern(self, input_grid):
        """Generate output preserving shape patterns"""
        binary = (input_grid > 0).astype(int)
        labeled, num_features = label(binary)
        output = np.zeros_like(input_grid)
        for i in range(1, num_features+1):
            output[labeled == i] = np.random.choice(self.color_palette)
        return output
    
    def _random_symmetrical_pattern(self, input_grid):
        """Generate random symmetrical pattern"""
        template = np.random.choice(self.shape_templates)
        color = np.random.choice(self.color_palette)
        return (template * color).astype(int)
    
    def _default_fallback(self):
        """Default fallback solution"""
        return {
            'output_1': np.zeros((5, 5), dtype=int),
            'output_2': np.zeros((5, 5), dtype=int),
            'confidence': 0.1,
            'strategy': 'DefaultFallback'
        }



# ==================== ğŸ§  META-REASONING CONTROLLER ğŸ§  ====================
class MetaReasoner:
    """Orchestrates Multiple Reasoning Strategies"""
    
    def __init__(self):
        self.color_engine = ColorEngine()
        self.shape_analyzer = ShapeAnalyzerPro()
        self.pattern_master = PatternMaster()
        self.llm_reasoner = LLMReasoner()
        
        # Initialize models
        self.vision_model = DynamicViT()
        self.cnn_model = MultiScaleCNN()
        self.reasoner = HybridReasoner(
            Config.MODEL['vit_hidden_size'] + 
            Config.MODEL['cnn_channels'][-1] +
            128  # Additional features
        )
        
        # Load weights if available
        self._load_models()
        
    def _load_models(self):
        """Load pre-trained model weights"""
        try:
            state = torch.load(f"{Config.MODEL_CACHE}/arc_model.pth")
            self.vision_model.load_state_dict(state['vision'])
            self.cnn_model.load_state_dict(state['cnn'])
            self.reasoner.load_state_dict(state['reasoner'])
        except:
            if Config.VERBOSE:
                print("No pre-trained models found, using random initialization")

    def analyze_task(self, train_examples):
        """Meta-analysis of task to determine best approach"""
        complexity = self._assess_complexity(train_examples)
        
        if complexity == TaskComplexity.SIMPLE:
            return SolutionStrategy.COLOR_MAPPING
        elif complexity == TaskComplexity.MODERATE:
            return SolutionStrategy.PATTERN_MATCHING
        elif complexity == TaskComplexity.COMPLEX:
            return SolutionStrategy.HYBRID_APPROACH
        else:
            return SolutionStrategy.LLM_REASONING
    
    def _assess_complexity(self, examples):
        """Determine task complexity level"""
        color_changes = 0
        structural_changes = 0
        
        for ex in examples:
            input_grid = np.array(ex['input'])
            output_grid = np.array(ex['output'])
            
            # Check color changes
            if not np.array_equal(np.unique(input_grid), np.unique(output_grid)):
                color_changes += 1
            
            # Check structural changes
            input_shape = self.shape_analyzer.extract_topology(input_grid)
            output_shape = self.shape_analyzer.extract_topology(output_grid)
            
            if input_shape['component_count'] != output_shape['component_count'] or \
               input_shape['holes'] != output_shape['holes']:
                structural_changes += 1
        
        if color_changes == len(examples) and structural_changes == 0:
            return TaskComplexity.SIMPLE
        elif structural_changes < len(examples) / 2:
            return TaskComplexity.MODERATE
        elif structural_changes < len(examples):
            return TaskComplexity.COMPLEX
        else:
            return TaskComplexity.EXTREME


# ==================== ğŸ�† ARC GRANDMASTER SOLVER ğŸ�† ====================
class ARCGrandmasterPro:
    """Complete Competition-Ready Solution"""
    
    def __init__(self):
        self.meta_reasoner = MetaReasoner()
        self.solution_cache = {}
        self.rule_cache = {}
        self.fallback_generator = FallbackGenerator()
        
    def solve_task(self, task):
        """End-to-end task solution with optimal strategy selection"""
        task_id = task['task_id']
        train_examples = task['train']
        test_cases = task['test'] if isinstance(task['test'], list) else [task['test']]
        
        # Learn from training examples
        strategy = self.meta_reasoner.analyze_task(train_examples)
        self._learn_rules(train_examples, strategy)
        
        # Generate solutions for test cases
        solutions = []
        for test_case in test_cases:
            input_grid = np.array(test_case['input'])
            solution = self._generate_solution(input_grid, strategy)
            solutions.append(solution)
        
        return {'solutions': solutions, 'strategy': str(strategy)}
    
    def _learn_rules(self, examples, strategy):
        """Learn transformation rules based on selected strategy"""
        if strategy in [SolutionStrategy.COLOR_MAPPING, SolutionStrategy.HYBRID_APPROACH]:
            for ex in examples:
                input_grid = np.array(ex['input'])
                output_grid = np.array(ex['output'])
                color_map = self.meta_reasoner.color_engine.semantic_color_map(input_grid, output_grid)
                self.rule_cache[hashlib.sha256(input_grid.tobytes()).hexdigest()] = {
                    'type': 'color_mapping',
                    'mapping': color_map
                }
        
        if strategy in [SolutionStrategy.PATTERN_MATCHING, SolutionStrategy.HYBRID_APPROACH]:
            for ex in examples:
                input_grid = np.array(ex['input'])
                output_grid = np.array(ex['output'])
                self.meta_reasoner.pattern_master.pattern_db.append({
                    'input': input_grid,
                    'output': output_grid,
                    'graph': self.meta_reasoner.pattern_master.build_pattern_graph(input_grid)
                })
    
    def _generate_solution(self, input_grid, strategy):
        """Generate solution using selected strategy"""
        grid_hash = hashlib.sha256(input_grid.tobytes()).hexdigest()
        
        # Check cache first
        if grid_hash in self.solution_cache:
            return self.solution_cache[grid_hash]
        
        # Apply selected strategy
        if strategy == SolutionStrategy.COLOR_MAPPING:
            solution = self._apply_color_mapping(input_grid)
        elif strategy == SolutionStrategy.PATTERN_MATCHING:
            solution = self._apply_pattern_matching(input_grid)
        elif strategy == SolutionStrategy.LLM_REASONING:
            solution = self._apply_llm_reasoning(input_grid)
        else:  # HYBRID_APPROACH
            solution = self._apply_hybrid_approach(input_grid)
        
        # Store in cache
        self.solution_cache[grid_hash] = solution
        return solution
    
    def _apply_color_mapping(self, input_grid):
        """Apply color mapping strategy"""
        grid_hash = hashlib.sha256(input_grid.tobytes()).hexdigest()
        if grid_hash in self.rule_cache:
            mapping = self.rule_cache[grid_hash]['mapping']
            output = np.zeros_like(input_grid)
            for src, dst in mapping.items():
                output[input_grid == src] = dst
            return {
                'output_1': output,
                'output_2': self.fallback_generator._random_symmetrical_pattern(input_grid),
                'confidence': 0.9,
                'strategy': 'ColorMapping'
            }
        return self.fallback_generator.generate(input_grid)
    
    def _apply_pattern_matching(self, input_grid):
        """Apply pattern matching strategy"""
        input_graph = self.meta_reasoner.pattern_master.build_pattern_graph(input_grid)
        best_match = None
        highest_score = 0
        
        for pattern in self.meta_reasoner.pattern_master.pattern_db:
            similarity = self.meta_reasoner.pattern_master.graph_similarity(input_graph, pattern['graph'])
            if similarity > highest_score:
                highest_score = similarity
                best_match = pattern
        
        if best_match and highest_score > Config.THRESHOLDS['similarity']:
            return {
                'output_1': best_match['output'],
                'output_2': self.fallback_generator._random_symmetrical_pattern(input_grid),
                'confidence': highest_score,
                'strategy': 'PatternMatching'
            }
        return self.fallback_generator.generate(input_grid)
    
    def _apply_llm_reasoning(self, input_grid):
        """Apply LLM reasoning strategy"""
        patterns = [p['output'] for p in self.meta_reasoner.pattern_master.pattern_db[:3]]
        rules = self.meta_reasoner.llm_reasoner.generate_hypothesis(input_grid, patterns)
        
        if rules:
            output = input_grid.copy()
            for rule in rules:
                if rule[0] == 'rotate':
                    output = np.rot90(output, k=rule[1]//90)
                elif rule[0] == 'flip':
                    output = np.fliplr(output)
            
            return {
                'output_1': output,
                'output_2': self.fallback_generator._random_symmetrical_pattern(input_grid),
                'confidence': Config.THRESHOLDS['llm_confidence'],
                'strategy': 'LLMReasoning'
            }
        return self.fallback_generator.generate(input_grid)
    
    def _apply_hybrid_approach(self, input_grid):
        """Advanced hybrid reasoning with fallback mechanisms"""
        # Step 1: Try color mapping
        color_solution = self._apply_color_mapping(input_grid)
        if color_solution['confidence'] > Config.THRESHOLDS['min_confidence']:
            return color_solution
        
        # Step 2: Try pattern matching
        pattern_solution = self._apply_pattern_matching(input_grid)
        if pattern_solution['confidence'] > Config.THRESHOLDS['min_confidence']:
            return pattern_solution
        
        # Step 3: Try LLM reasoning
        llm_solution = self._apply_llm_reasoning(input_grid)
        if llm_solution['confidence'] > Config.THRESHOLDS['llm_confidence']:
            return llm_solution
        
        # Final fallback
        return self.fallback_generator.generate(input_grid)


# ==================== ğŸš€ MAIN EXECUTION ğŸš€ ====================
if __name__ == "__main__":
    print("""
    ğŸš€ ARC Grandmaster Pro - Competition Edition ğŸš€
    --------------------------------------------
    System Configuration:
      Device: {device}
      Workers: {workers}
      Vision: Dynamic ViT
      Reasoning: Hybrid (Symbolic + LLM + Graph)
    """.format(
        device=Config.DEVICE,
        workers=Config.PERFORMANCE['max_workers']
    ))
    
    # Initialize solver
    solver = ARCGrandmasterPro()
    
    # Load competition data
    with open(f'{Config.DATA_PATH}/arc-agi_test_challenges.json') as f:
        tasks = json.load(f)
    
    # Process all tasks
    submission = {}
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=Config.PERFORMANCE['max_workers']) as executor:
        futures = {
            executor.submit(solver.solve_task, {'task_id': tid, **task}): tid
            for tid, task in tasks.items()
        }
        
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Processing Tasks"):
            task_id = futures[future]
            try:
                result = future.result()
                submission[task_id] = []
                
                for sol in result['solutions']:
                    submission[task_id].append({
                        'attempt_1': sol['output_1'].tolist() if isinstance(sol['output_1'], np.ndarray) else sol['output_1'],
                        'attempt_2': sol['output_2'].tolist() if isinstance(sol['output_2'], np.ndarray) else sol['output_2']
                    })
                    
            except Exception as e:
                if Config.VERBOSE:
                    print(f"Task {task_id} failed: {str(e)}")
                submission[task_id] = [{
                    'attempt_1': [[0]*5]*5,
                    'attempt_2': [[0]*5]*5
                }]
    
    # Save submission
    with open(Config.SUBMISSION_FILE, 'w') as f:
        json.dump(submission, f, indent=2)
    
    print(f"\nâœ… Processing completed in {time.time()-start_time:.2f}s")
    print(f"ğŸ“� Results saved to {Config.SUBMISSION_FILE}")

