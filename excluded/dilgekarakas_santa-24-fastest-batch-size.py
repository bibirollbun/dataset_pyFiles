import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm
import random, pickle, math, warnings
import itertools
import torch
import transformers
import time
from typing import List, Dict, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
from statistics import mean, stdev

p = '/kaggle/input/santa-2024/sample_submission.csv'
df = pd.read_csv(p)


@dataclass
class BenchmarkResult:
    batch_size: int
    avg_time_per_sample: float
    throughput: float
    memory_peak: float
    std_dev: float

class PerplexityBenchmark:
    def __init__(self, model_name: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.setup_model()
        
    def setup_model(self):
        """Initialize model and tokenizer"""
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_name,
            model_max_length=512,
            padding_side='right',
            truncation_side='right'
        )
        
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
            device_map='auto' if self.device.type == 'cuda' else None
        ).eval()

    def reset_gpu_memory(self):
        """Reset GPU memory stats"""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
    def get_gpu_memory_peak(self) -> float:
        """Get peak GPU memory usage in GB"""
        if self.device.type == 'cuda':
            return torch.cuda.max_memory_allocated() / 1024**3
        return 0
        
    @torch.inference_mode()
    def calculate_perplexity_batch(self, texts: List[str]) -> List[float]:
        """Calculate perplexity for a batch of texts"""
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=512
        ).to(self.device)
        
        outputs = self.model(**inputs)
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = inputs['input_ids'][..., 1:].contiguous()
        
        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction='none'
        ).view(shift_labels.size())
        
        mask = (shift_labels != self.tokenizer.pad_token_id).float()
        sequence_loss = (loss * mask).sum(dim=1) / mask.sum(dim=1)
        return torch.exp(sequence_loss).cpu().numpy().tolist()

    def benchmark_batch_size(
        self,
        texts: List[str],
        batch_size: int,
        num_runs: int = 3
    ) -> BenchmarkResult:
        """Benchmark specific batch size"""
        self.reset_gpu_memory()
        times = []
        
        for _ in range(num_runs):
            start_time = time.time()
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                _ = self.calculate_perplexity_batch(batch)
                
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = mean(times)
        return BenchmarkResult(
            batch_size=batch_size,
            avg_time_per_sample=avg_time / len(texts),
            throughput=len(texts) / avg_time,
            memory_peak=self.get_gpu_memory_peak(),
            std_dev=stdev(times) if len(times) > 1 else 0
        )

    def find_optimal_batch_size(
        self,
        sample_texts: List[str],
        batch_sizes: List[int] = None,
        num_runs: int = 3
    ) -> List[BenchmarkResult]:
        """Find optimal batch size by testing multiple options"""
        if batch_sizes is None:
            batch_sizes = [1, 2, 4, 8, 16, 32, 64]
            
        results = []
        for batch_size in tqdm(batch_sizes, desc="Testing batch sizes"):
            try:
                result = self.benchmark_batch_size(sample_texts, batch_size, num_runs)
                results.append(result)
                print(f"\nBatch size {batch_size}:")
                print(f"Average time per sample: {result.avg_time_per_sample:.4f} seconds")
                print(f"Throughput: {result.throughput:.2f} samples/second")
                print(f"Peak GPU memory: {result.memory_peak:.2f} GB")
                print(f"Standard deviation: {result.std_dev:.4f} seconds")
            except RuntimeError as e:
                print(f"\nBatch size {batch_size} failed: {str(e)}")
                break
                
        return results

    def plot_results(self, results: List[BenchmarkResult]):
        """Plot benchmark results"""
        batch_sizes = [r.batch_size for r in results]
        throughputs = [r.throughput for r in results]
        memories = [r.memory_peak for r in results]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        ax1.plot(batch_sizes, throughputs, 'bo-')
        ax1.set_xlabel('Batch Size')
        ax1.set_ylabel('Throughput (samples/second)')
        ax1.set_title('Throughput vs Batch Size')
        ax1.grid(True)
        
        ax2.plot(batch_sizes, memories, 'ro-')
        ax2.set_xlabel('Batch Size')
        ax2.set_ylabel('Peak GPU Memory (GB)')
        ax2.set_title('Memory Usage vs Batch Size')
        ax2.grid(True)
        
        plt.tight_layout()
        return fig

def run_benchmark(
    input_file: str,
    model_name: str = "google/gemma-2b",
    num_samples: int = 100,
    batch_sizes: List[int] = None
):
    """Run complete benchmark"""
    df = pd.read_csv(input_file)
    sample_texts = df['text'].head(num_samples).tolist()
    
    benchmark = PerplexityBenchmark(model_name)
    
    results = benchmark.find_optimal_batch_size(
        sample_texts,
        batch_sizes=batch_sizes,
        num_runs=3
    )
    
    fig = benchmark.plot_results(results)
    optimal_result = min(results, key=lambda x: x.avg_time_per_sample)
    
    print("\nOptimal configuration:")
    print(f"Batch size: {optimal_result.batch_size}")
    print(f"Average time per sample: {optimal_result.avg_time_per_sample:.4f} seconds")
    print(f"Throughput: {optimal_result.throughput:.2f} samples/second")
    print(f"Peak GPU memory: {optimal_result.memory_peak:.2f} GB")
    
    return results, fig


batch_sizes = [1, 2, 4, 8, 16, 32]
results, fig = run_benchmark(
    input_file="/kaggle/input/santa-2024/sample_submission.csv",
    model_name="/kaggle/input/gemma-2/transformers/gemma-2-9b/2",
    num_samples=100,
    batch_sizes=batch_sizes
)

