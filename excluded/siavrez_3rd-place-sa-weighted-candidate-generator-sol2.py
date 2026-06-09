import datetime
import gc
import math
import os
import random
import time
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import transformers

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
PAD_TOKEN_LABEL_ID = torch.nn.CrossEntropyLoss().ignore_index
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ParticipantVisibleError(Exception):
    pass


@dataclass(frozen=True)
class OptimizerConfig:
    """Configuration for the optimizer using clear parameter names."""

    model_path: str
    batch_size: int = 80
    generated_neighborhood_candidates: int = 240
    score_threshold: float = 0.0
    max_stagnant_steps: int = 100
    stagnation_count_for_path_increase: int = 5
    max_score_deviation: float = 0.35
    num_parallel_paths: int = 2
    max_parallel_paths: int = 4
    initial_temperature: float = 2.0
    final_temperature: float = -5.0
    total_steps: int = 40000
    log_frequency: int = 200
    random_seed: int = 1992
    cooling_strategy: str = "cyclic"
    num_restarts: int = 10
    boltzmann_constant: float = 10.0


class Operation:
    """Base class for text manipulation operations."""

    def __call__(
        self, arr: np.ndarray, rng: np.random.Generator, block_size: int
    ) -> np.ndarray:
        raise NotImplementedError


class SwapOperation(Operation):
    """Swap two random positions in the array."""

    def __call__(
        self, arr: np.ndarray, rng: np.random.Generator, block_size: int
    ) -> np.ndarray:
        result = arr.copy()
        positions = rng.choice(len(result), 2, replace=False)
        result[positions[0]], result[positions[1]] = (
            result[positions[1]],
            result[positions[0]],
        )
        return result


class BlockMoveOperation(Operation):
    """Move a block of elements to a new position."""

    def __call__(
        self, arr: np.ndarray, rng: np.random.Generator, block_size: int
    ) -> np.ndarray:
        length = len(arr)
        result = arr.copy()
        block_size = min(block_size, length // 2)
        start = rng.integers(0, length - block_size + 1)
        possible_destinations = list(range(0, start - block_size + 1)) + list(
            range(start + block_size, length - block_size + 1)
        )
        if not possible_destinations:
            return result
        dest = rng.choice(possible_destinations)
        block = result[start : start + block_size].copy()
        if dest < start:
            result[dest + block_size : start + block_size] = result[dest:start]
        else:
            result[start:dest] = result[start + block_size : dest + block_size]
        result[dest : dest + block_size] = block
        return result


class BlockSwapOperation(Operation):
    """Swap two blocks of equal size."""

    def __call__(
        self, arr: np.ndarray, rng: np.random.Generator, block_size: int
    ) -> np.ndarray:
        length = len(arr)
        result = arr.copy()
        pos1 = rng.integers(0, length - block_size + 1)
        valid_positions = [
            pos
            for pos in range(length - block_size + 1)
            if pos < pos1 - block_size or pos > pos1 + block_size
        ]
        if not valid_positions:
            return result
        pos2 = rng.choice(valid_positions)
        temp = result[pos1 : pos1 + block_size].copy()
        result[pos1 : pos1 + block_size] = result[pos2 : pos2 + block_size]
        result[pos2 : pos2 + block_size] = temp
        return result


class ReverseBlockOperation(Operation):
    """Reverse a block of elements."""

    def __call__(
        self, arr: np.ndarray, rng: np.random.Generator, block_size: int
    ) -> np.ndarray:
        length = len(arr)
        result = arr.copy()
        start = rng.integers(0, length - block_size + 1)
        result[start : start + block_size] = result[start : start + block_size][::-1]
        return result


class ShuffleBlockOperation(Operation):
    """Shuffle elements within a block."""

    def __call__(
        self, arr: np.ndarray, rng: np.random.Generator, block_size: int
    ) -> np.ndarray:
        length = len(arr)
        result = arr.copy()
        start = rng.integers(0, length - block_size + 1)
        block = result[start : start + block_size].copy()
        rng.shuffle(block)
        result[start : start + block_size] = block
        return result


class CyclicShiftOperation(Operation):
    """Perform a cyclic shift on a section of the array."""

    def __call__(
        self, arr: np.ndarray, rng: np.random.Generator, block_size: int
    ) -> np.ndarray:
        length = len(arr)
        result = arr.copy()
        start = rng.integers(0, length - block_size + 1)
        shift = rng.integers(1, block_size)
        section = result[start : start + block_size]
        result[start : start + block_size] = np.roll(section, shift)
        return result


class ShiftOneOperation(Operation):
    """Shift one element to a new position."""

    def __call__(
        self, arr: np.ndarray, rng: np.random.Generator, block_size: int
    ) -> np.ndarray:
        result = arr.copy()
        length = len(result)
        pos1, pos2 = rng.choice(length, 2, replace=False)

        temp = result[pos1]

        if pos1 < pos2:
            result[pos1:pos2] = result[pos1 + 1 : pos2 + 1]
        else:
            result[pos2 + 1 : pos1 + 1] = result[pos2:pos1]

        result[pos2] = temp
        return result


class TripleMoveOperation(Operation):
    """Move three elements in coordination."""

    def __call__(
        self, arr: np.ndarray, rng: np.random.Generator, block_size: int
    ) -> np.ndarray:
        length = len(arr)
        if length < 6:
            return arr.copy()

        result = arr.copy()
        positions = rng.choice(length, 6, replace=False)

        source = positions[:3]
        dest = positions[3:]
        source_elements = result[source].copy()
        dest_elements = result[dest].copy()

        for i in range(3):
            result[source[i]] = dest_elements[i]
            result[dest[i]] = source_elements[i]

        return result


class QuadMoveOperation(Operation):
    """Move four elements in coordination."""

    def __call__(
        self, arr: np.ndarray, rng: np.random.Generator, block_size: int
    ) -> np.ndarray:
        length = len(arr)
        if length < 8:
            return arr.copy()

        result = arr.copy()
        positions = rng.choice(length, 8, replace=False)

        source = positions[:4]
        dest = positions[4:]
        source_elements = result[source].copy()
        dest_elements = result[dest].copy()

        for i in range(4):
            result[source[i]] = dest_elements[i]
            result[dest[i]] = source_elements[i]

        return result


class InterleaveOperation(Operation):
    """Interleave two blocks while preserving length."""

    def __call__(
        self, arr: np.ndarray, rng: np.random.Generator, block_size: int
    ) -> np.ndarray:
        length = len(arr)
        max_start = length - (2 * block_size)
        if max_start < 0:
            return arr.copy()

        result = arr.copy()
        start1 = rng.integers(0, max_start + 1)
        start2 = start1 + block_size

        block1 = result[start1 : start1 + block_size].copy()
        block2 = result[start2 : start2 + block_size].copy()

        for i in range(block_size):
            result[start1 + 2 * i] = block1[i]
            result[start1 + 2 * i + 1] = block2[i]

        return result


class SplitMergeOperation(Operation):
    """Split and merge segments while preserving length."""

    def __call__(
        self, arr: np.ndarray, rng: np.random.Generator, block_size: int
    ) -> np.ndarray:
        length = len(arr)
        n_segments = 2
        result = arr.copy()

        segments = []
        remaining = length
        start = 0

        for i in range(n_segments - 1):
            size = remaining // (n_segments - i)
            segments.append(result[start : start + size])
            start += size
            remaining -= size

        segments.append(result[start:])

        rng.shuffle(segments)
        return np.concatenate(segments)


class WindowSlideOperation(Operation):
    """Apply sliding window permutations."""

    def __call__(
        self, arr: np.ndarray, rng: np.random.Generator, block_size: int
    ) -> np.ndarray:
        length = len(arr)
        result = arr.copy()
        start = rng.integers(0, length - block_size + 1)

        window = result[start : start + block_size].copy()
        rng.shuffle(window)

        result[start : start + block_size] = window
        return result


class PivotRotateOperation(Operation):
    """Rotate around pivot while preserving length."""

    def __call__(
        self, arr: np.ndarray, rng: np.random.Generator, block_size: int
    ) -> np.ndarray:
        length = len(arr)
        result = arr.copy()
        pivot = block_size
        rotation = rng.integers(-2, 3)

        left = result[:pivot]
        right = result[pivot:]

        result[:pivot] = np.roll(left, rotation)
        result[pivot:] = np.roll(right, -rotation)
        return result


class CrossBridgeOperation(Operation):
    """Create bridges between sections while preserving length."""

    def __call__(
        self, arr: np.ndarray, rng: np.random.Generator, block_size: int
    ) -> np.ndarray:
        length = len(arr)
        if length < 4:
            return arr.copy()

        result = arr.copy()
        pos1 = rng.integers(0, length - block_size)

        valid_positions = [
            pos
            for pos in range(length - block_size)
            if pos < pos1 - block_size or pos > pos1 + block_size
        ]

        if not valid_positions:
            return result

        pos2 = rng.choice(valid_positions)

        temp = result[pos1 : pos1 + block_size].copy()
        result[pos1 : pos1 + block_size] = result[pos2 : pos2 + block_size]
        result[pos2 : pos2 + block_size] = temp

        return result


class RotateBlockOperation(Operation):
    """Rotate a block of elements."""

    def __call__(
        self, arr: np.ndarray, rng: np.random.Generator, block_size: int
    ) -> np.ndarray:
        length = len(arr)
        result = arr.copy()
        start = rng.integers(0, length - block_size + 1)
        rotation = rng.integers(1, block_size)

        block = result[start : start + block_size]
        result[start : start + block_size] = np.roll(block, rotation)
        return result


class MultiSwapOperation(Operation):
    """Apply multiple swaps simultaneously."""

    def __call__(
        self, arr: np.ndarray, rng: np.random.Generator, block_size: int
    ) -> np.ndarray:
        result = arr.copy()
        length = len(result)
        n_swaps = rng.integers(2, 4)

        positions = rng.choice(length, n_swaps * 2, replace=False)

        for i in range(0, len(positions), 2):
            result[positions[i]], result[positions[i + 1]] = (
                result[positions[i + 1]],
                result[positions[i]],
            )

        return result


OPERATIONS = {
    "swap": SwapOperation(),
    "block_move": BlockMoveOperation(),
    "block_swap": BlockSwapOperation(),
    "reverse_block": ReverseBlockOperation(),
    "shuffle_block": ShuffleBlockOperation(),
    "cyclic_shift": CyclicShiftOperation(),
    "shift_one": ShiftOneOperation(),
    "triple_move": TripleMoveOperation(),
    "quad_move": QuadMoveOperation(),
    "interleave": InterleaveOperation(),
    "split_merge": SplitMergeOperation(),
    "window_slide": WindowSlideOperation(),
    "pivot_rotate": PivotRotateOperation(),
    "cross_bridge": CrossBridgeOperation(),
    "rotate_block": RotateBlockOperation(),
    "multi_swap": MultiSwapOperation(),
}


class PerplexityCalculator:
    """Calculates perplexity scores for text sequences."""

    def __init__(
        self,
        model_path: str,
        load_in_8bit: bool = False,
        device_map: str = "auto",
    ):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_path, padding_side="right"
        )

        if load_in_8bit:
            if DEVICE.type != "cuda":
                raise ValueError("8-bit quantization requires CUDA device")

            quantization_config = transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="fp4",
                bnb_4bit_use_double_quant=False,
                bnb_4bit_compute_dtype=torch.float16,
            )

            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                device_map=device_map,
            )
        else:
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if DEVICE.type == "cuda" else torch.float32,
                device_map=device_map,
            )

        self.loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        self.model.eval()

    def get_perplexity(
        self, input_texts: str | list[str], batch_size: int = 1
    ) -> float | list[float]:
        """Calculate perplexity for input texts."""
        single_input = isinstance(input_texts, str)
        input_texts = [input_texts] if single_input else input_texts

        loss_list = []
        batches = len(input_texts) // batch_size + (len(input_texts) % batch_size != 0)

        for j in range(batches):
            a = j * batch_size
            b = (j + 1) * batch_size
            input_batch = input_texts[a:b]

            with torch.no_grad():
                text_with_special = [
                    f"{self.tokenizer.bos_token}{text}{self.tokenizer.eos_token}"
                    for text in input_batch
                ]

                model_inputs = self.tokenizer(
                    text_with_special,
                    return_tensors="pt",
                    add_special_tokens=False,
                    padding=True,
                )

                if "token_type_ids" in model_inputs:
                    model_inputs.pop("token_type_ids")

                model_inputs = {k: v.to(DEVICE) for k, v in model_inputs.items()}

                output = self.model(**model_inputs, use_cache=False)
                logits = output["logits"]

                label = model_inputs["input_ids"]
                label[label == self.tokenizer.pad_token_id] = PAD_TOKEN_LABEL_ID

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = label[..., 1:].contiguous()

                loss = self.loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
                )

                loss = loss.view(len(logits), -1)
                valid_length = (shift_labels != PAD_TOKEN_LABEL_ID).sum(dim=-1)
                loss = torch.sum(loss, -1) / valid_length

                loss_list.extend(loss.cpu().tolist())

        ppl = [math.exp(i) for i in loss_list]
        return ppl[0] if single_input else ppl

    def clear_gpu_memory(self) -> None:
        """Clear GPU memory and reset cache."""
        if not torch.cuda.is_available():
            return

        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "tokenizer"):
            del self.tokenizer

        gc.collect()

        with DEVICE:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.reset_peak_memory_stats()


class CandidateGenerator:
    """Generates candidate text permutations with improved diversity tracking."""

    def __init__(self, max_cache_size: int = 100000000):
        self.max_cache_size = max_cache_size
        self.global_cache: set[str] = set()
        self.operation_history: list[str] = []
        self.diversity_score: float = 0.0
        self.mutation_rate: float = 0.1
        self.adaptive_block_size: int = 2
        self.operation_weights = np.ones(len(OPERATIONS))
        self.rng = np.random.default_rng()
        self.failed_attempts = 0

    def calculate_diversity_score(self, candidates: list[str]) -> float:
        """Calculate diversity score based on edit distances between candidates."""
        if len(candidates) < 2:
            return 0.0

        n = len(candidates)
        possible_pairs = (n * (n - 1)) // 2
        n_pairs = min(100, possible_pairs)

        if n_pairs < possible_pairs:
            idx1 = self.rng.choice(len(candidates), size=n_pairs, replace=True)
            idx2 = self.rng.choice(len(candidates), size=n_pairs, replace=True)
            for i in range(n_pairs):
                while idx2[i] == idx1[i]:
                    idx2[i] = self.rng.integers(0, len(candidates))
            pairs = list(
                zip([candidates[i] for i in idx1], [candidates[i] for i in idx2])
            )
        else:
            pairs = [
                (candidates[i], candidates[j])
                for i in range(len(candidates))
                for j in range(i + 1, len(candidates))
            ]

        distances = []
        for c1, c2 in pairs:
            words1 = c1.split()
            words2 = c2.split()
            common = sum(w1 == w2 for w1, w2 in zip(words1, words2))
            distance = 1 - (common / len(words1))
            distances.append(distance)

        return float(np.mean(distances)) if distances else 0.0

    def adjust_parameters(self, success_rate: float, diversity_score: float) -> None:
        """Adjust generator parameters based on performance metrics."""
        if success_rate < 0.3:
            self.adaptive_block_size = min(self.adaptive_block_size + 1, 6)
        elif success_rate > 0.7:
            self.adaptive_block_size = max(2, self.adaptive_block_size - 1)

        if diversity_score < 0.3:
            self.mutation_rate = min(self.mutation_rate * 1.5, 0.5)
        elif diversity_score > 0.7:
            self.mutation_rate = max(self.mutation_rate * 0.8, 0.05)

        if len(self.operation_history) > 50:
            recent_ops = self.operation_history[-50:]
            op_success = Counter(recent_ops)
            total = len(recent_ops)

            for op, count in op_success.items():
                idx = list(OPERATIONS.keys()).index(op)
                success_rate = count / total
                if success_rate > 0.3:
                    self.operation_weights[idx] *= 1.2
                else:
                    self.operation_weights[idx] *= 0.8

            self.operation_weights /= np.sum(self.operation_weights)

        if len(self.operation_history) > 1000:
            self.operation_history = self.operation_history[-500:]

    def targeted_modify_text_batch(
        self, text: str, batch_size: int
    ) -> tuple[list[str], list[str]]:
        """Generate batch of candidates with improved diversity tracking."""
        words = text.split()
        solution_array = np.array(words)

        successful_mods = 0
        neighbor_texts = []
        success_methods = []

        normalized_weights = self.operation_weights / np.sum(self.operation_weights)
        methods = self.rng.choice(
            list(OPERATIONS.keys()), batch_size, p=normalized_weights
        )

        for method in methods:
            operation = OPERATIONS[method]

            result = operation(solution_array, self.rng, self.adaptive_block_size)

            if self.rng.random() < self.mutation_rate:
                mutation_method = self.rng.choice(list(OPERATIONS.keys()))
                if mutation_method != method:
                    result = OPERATIONS[mutation_method](
                        result, self.rng, self.adaptive_block_size
                    )

            if Counter(result) != Counter(words):
                continue

            new_text = " ".join(result)
            if new_text not in self.global_cache:
                neighbor_texts.append(new_text)
                self.global_cache.add(new_text)
                successful_mods += 1
                success_methods.append(method)

        diversity_score = self.calculate_diversity_score(neighbor_texts)
        self.diversity_score = diversity_score

        success_rate = successful_mods / batch_size
        self.adjust_parameters(success_rate, diversity_score)

        self.operation_history.extend(success_methods)

        if len(self.global_cache) > self.max_cache_size:
            self.global_cache = set(
                list(self.global_cache)[-self.max_cache_size // 2 :]
            )

        return neighbor_texts, success_methods

    def update_weights(self, operation_scores: dict[str, float]) -> None:
        """Update operation weights with aggressive adaptation."""
        min_score = min(operation_scores.values())
        max_score = max(operation_scores.values())

        for i, operation in enumerate(OPERATIONS.keys()):
            score = operation_scores.get(operation, max_score)
            if score == min_score:
                self.operation_weights[i] *= 1.5
            else:
                penalty = 0.8 + (
                    0.2 * (score - min_score) / (max_score - min_score + 1e-6)
                )
                self.operation_weights[i] *= penalty

        total_weight = np.sum(self.operation_weights)
        if total_weight <= 0 or not np.isfinite(total_weight):
            self.operation_weights = np.ones(len(OPERATIONS))
        else:
            self.operation_weights /= total_weight


class SAOptimizer:
    """Optimizes text permutations using simulated annealing with improved tracking."""

    def __init__(self, scorer: PerplexityCalculator, config: OptimizerConfig):
        self.scorer = scorer
        self.config = config
        self.generator = CandidateGenerator()
        self.rng = np.random.default_rng(config.random_seed)

        self.temperature = config.initial_temperature
        self.best_solutions: set[tuple[str, float]] = set()
        self.steps_since_improvement = 0
        self.max_solutions = 10
        self.stagnation_count = 0
        self.improvement_history: list[dict] = []
        self.diversity_metrics: list[dict] = []
        self.acceptance_history: list[dict] = []

        if config.cooling_strategy == "cyclic":
            self.temp_schedule = self._create_cyclic_schedule()

    def _create_cyclic_schedule(self) -> np.ndarray:
        """Create temperature schedule for cyclic cooling."""
        steps_per_cycle = self.config.total_steps // self.config.num_restarts
        schedule = np.zeros(self.config.total_steps)

        for cycle in range(self.config.num_restarts):
            start_idx = cycle * steps_per_cycle
            end_idx = start_idx + steps_per_cycle
            schedule[start_idx:end_idx] = np.logspace(
                self.config.initial_temperature,
                self.config.final_temperature,
                steps_per_cycle,
            )
        return schedule

    def _acceptance_probability(
        self, current_energy: float, new_energy: float, temperature: float
    ) -> float:
        """Calculate acceptance probability using Metropolis criterion."""
        if new_energy <= current_energy:
            return 1.0
        elif (
            (new_energy - current_energy ) / current_energy
            < self.config.max_score_deviation
        ):
            return np.exp(
                self.config.boltzmann_constant
                * (current_energy - new_energy)
                / temperature
            )
        else:
            return 0.0

    def _track_improvement(
        self,
        step: int,
        current_score: float,
        best_score: float,
        new_candidates: list[str],
    ) -> None:
        """Track detailed improvement metrics."""
        self.improvement_history.append(
            {
                "step": step,
                "current_score": current_score,
                "best_score": best_score,
                "n_candidates": len(new_candidates),
                "temperature": self.temperature,
                "cache_size": len(self.generator.global_cache),
            }
        )

        if new_candidates:
            diversity = self.generator.calculate_diversity_score(new_candidates)
            self.diversity_metrics.append({"step": step, "diversity": diversity})

    def process_candidates(
        self, candidates: list[str], scores: list[float], n_paths: int
    ) -> list[tuple[str, float]]:
        """Process initial candidates and return top n_paths candidates."""
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i])
        top_n_indices = sorted_indices[:n_paths]
        return [(candidates[i], scores[i]) for i in top_n_indices]

    def process_repeats(
        self,
        candidate_text: str,
        generated_neighborhood_candidates: int,
        n_repeats: int,
        batch_size: int = 1,
    ) -> tuple[list[str], list[float]]:
        """Generate and process repeat candidates for a given text."""
        all_repeat_candidates = []
        all_repeat_scores = []

        for _ in range(n_repeats):
            repeat_candidates, _ = self.generator.targeted_modify_text_batch(
                candidate_text, generated_neighborhood_candidates
            )

            if not repeat_candidates:
                continue

            repeat_scores = self.scorer.get_perplexity(
                repeat_candidates, batch_size=batch_size
            )
            all_repeat_candidates.extend(repeat_candidates)
            all_repeat_scores.extend(repeat_scores)

        return all_repeat_candidates, all_repeat_scores

    def find_best_candidate(
        self, all_candidates: list[str], all_scores: list[float], current_score: float
    ) -> tuple[str | None, float | None]:
        """Find the best acceptable candidate among all candidates."""
        sorted_indices = sorted(range(len(all_scores)), key=lambda i: all_scores[i])

        for idx in sorted_indices:
            new_score = all_scores[idx]
            acceptance = self._acceptance_probability(
                current_score, new_score, self.temperature
            )

            if acceptance > self.rng.random():
                return all_candidates[idx], new_score

        return None, None

    def optimize(self, initial_text: str) -> tuple[str, float, dict]:
        """Main optimization loop with improved tracking and debugging."""
        start_time = time.time()
        current_text = initial_text
        current_score = self.scorer.get_perplexity(current_text, batch_size=1)
        best_text = current_text
        best_score = current_score

        self.best_solutions.add((best_text, best_score))
        print(f"\nInitial score: {best_score:.3f}")

        for step in range(self.config.total_steps):
            if self.stagnation_count >= self.config.stagnation_count_for_path_increase:
                self.steps_since_improvement = 0
                self.config.num_parallel_paths += 1
                self.config.num_parallel_paths = min(
                    self.config.num_parallel_paths, self.config.max_parallel_paths
                )
            if self.config.cooling_strategy == "cyclic":
                self.temperature = self.temp_schedule[step]

            if self.steps_since_improvement >= self.config.max_stagnant_steps:
                self.stagnation_count += 1
                if len(self.best_solutions) > 1:
                    current_text, current_score = random.choice(
                        list(self.best_solutions)
                    )
                    self.steps_since_improvement = 0
                    self.temperature = self.config.initial_temperature
                    print(f"\nSwitched to solution with score: {current_score:.3f}")
                else:
                    current_text = sorted(initial_text.split(), key=lambda x: self.rng.random())

            candidates, operations = self.generator.targeted_modify_text_batch(
                current_text, self.config.generated_neighborhood_candidates
            )

            if not candidates:
                continue

            candidate_scores = self.scorer.get_perplexity(
                candidates, batch_size=self.config.batch_size
            )

            self._track_improvement(step, current_score, best_score, candidates)

            top_candidates = self.process_candidates(
                candidates, candidate_scores, self.config.num_parallel_paths
            )

            all_candidates = []
            all_scores = []
            for candidate_text, _ in top_candidates:
                repeat_candidates, repeat_scores = self.process_repeats(
                    candidate_text,
                    self.config.generated_neighborhood_candidates,
                    self.config.num_parallel_paths,
                    self.config.batch_size,
                )
                all_candidates.extend(repeat_candidates)
                all_scores.extend(repeat_scores)

            new_text, new_score = self.find_best_candidate(
                all_candidates, all_scores, current_score
            )

            if new_text is not None:
                self.best_solutions.add((new_text, new_score))
                if len(self.best_solutions) > self.max_solutions:
                    self.best_solutions = set(
                        sorted(self.best_solutions, key=lambda x: x[1])[
                            : self.max_solutions
                        ]
                    )

                if new_score < best_score:
                    best_text = new_text
                    best_score = new_score
                    self.steps_since_improvement = 0
                    print(f"\nNew best score: {best_score:8.3f} at step {step}")
                    print(f"Text: {best_text}")
                else:
                    self.steps_since_improvement += 1

                current_text = new_text
                current_score = new_score

            if step % self.config.log_frequency == 0:
                print(
                    f"\nStep {step:6d} "
                    f"Current score: {current_score:8.3f} "
                    f"Best score: {best_score:8.3f} "
                    f"Temperature: {self.temperature:8.3f} "
                    f"Steps since improvement: {self.steps_since_improvement}"
                )

            if best_score < self.config.score_threshold:
                print(f"\nTarget score achieved: {best_score:8.3f}")
                break

            operation_scores = {op: float("inf") for op in OPERATIONS.keys()}
            for op, score in zip(operations, candidate_scores):
                operation_scores[op] = min(operation_scores[op], score)
            self.generator.update_weights(operation_scores)

        elapsed = str(datetime.timedelta(seconds=int(round(time.time() - start_time))))
        print(f"\nOptimization completed in {elapsed}")
        print(f"Final best score: {best_score:8.3f}")

        return (
            best_text,
            best_score,
            {
                "improvements": pd.DataFrame(self.improvement_history),
                "diversity": pd.DataFrame(self.diversity_metrics),
                "acceptance": pd.DataFrame(self.acceptance_history),
            },
        )


def main() -> None:
    """Main function to run the optimization."""
    config = OptimizerConfig(
        model_path="/kaggle/input/gemma-2/transformers/gemma-2-9b/2",
        batch_size=80,
        generated_neighborhood_candidates=640,
        score_threshold=0,
        max_stagnant_steps=1000,
        stagnation_count_for_path_increase=5,
        max_score_deviation=0.5,
        num_parallel_paths=1,
        max_parallel_paths=4,
        initial_temperature=2.0,
        final_temperature=-5.0,
        total_steps=100,
        log_frequency=20,
        random_seed=1982,
        cooling_strategy="cyclic",
        num_restarts=10,
        boltzmann_constant=5.0,
    )

    initial_text = "from and and the and as is in it of of not that the the to we with you card candy chocolate cookie doll drive eat eggnog family game give have holiday hope jump laugh milk naughty nice night night peace puzzle relax season sing sleep toy unwrap visit walk wish wrapping paper yuletide advent angel bake beard believe bow carol candle cheer cheer chimney chimney decorations dream elf fireplace fireplace fruitcake gifts gingerbread grinch greeting holly hohoho jingle joy kaggle magi merry mistletoe nutcracker ornament ornament peppermint polar poinsettia reindeer scrooge sleigh snowglobe star stocking wreath wonder workshop workshop"
    words = initial_text.split()
    word_counts = Counter(words)
    random.seed(config.random_seed)
    shuffled_words = sorted(words, reverse=False)
    assert (
        Counter(shuffled_words) == word_counts
    ), "Initial text should be a permutation of the original text"
    shuffled_text = " ".join(shuffled_words)

    scorer = PerplexityCalculator(config.model_path)
    optimizer = SAOptimizer(scorer, config)

    best_text, best_score, debug_info = optimizer.optimize(shuffled_text)

    print("\nFinal Results:")
    print(f"Best Score: {best_score:.3f}")
    print(f"Best Text: {best_text}")

    debug_info["improvements"].to_csv("optimization_history.csv", index=False)
    debug_info["diversity"].to_csv("diversity_metrics.csv", index=False)
    debug_info["acceptance"].to_csv("acceptance_history.csv", index=False)



