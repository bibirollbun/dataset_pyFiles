# ============================================================
# Standard Library
# ============================================================

import os
import re
import json
import time
import logging
import warnings
import pathlib
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Union,
    Tuple,
    Callable,
    Set,
)


# ============================================================
# Data Handling
# ============================================================

import numpy as np
import pandas as pd


# ============================================================
# HTTP / Networking
# ============================================================

import requests  # used for reference checks and URL health verification


# ============================================================
# Notebook / Jupyter Display
# ============================================================

from IPython.display import display, Markdown, HTML


# ============================================================
# Kaggle Secrets (Gemini API Key)
# ============================================================

from kaggle_secrets import UserSecretsClient


# ============================================================
# Google GenAI (Gemini SDK)
# ============================================================

import google.genai as genai
from google.genai import types
from google.genai import errors   # APIError for retry handling


# ============================================================
# Google ADK Core Components
# ============================================================

"""
Imports for core ADK objects. These classes enable:
- LLM-backed agent execution (LlmAgent)
- Tool integration (FunctionTool)
- In-memory sessions and memory for deterministic execution
- Runners that orchestrate tool execution

All imports must remain stable to preserve ADK compatibility.
"""

from google.adk.models.google_llm import Gemini
from google.adk.agents import Agent, LlmAgent
from google.adk.runners import InMemoryRunner, Runner
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService
from google.adk.tools.function_tool import FunctionTool


# ============================================================
# Optional Built-in ADK Tools
# ============================================================

"""
Optional toolset from ADK that exposes Google Search capabilities.
Used only if the V-AID pipeline includes a web-evidence stage.
"""

from google.adk.tools import google_search


# ============================================================
# Optional MCP / Advanced Tools
# ============================================================

from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from google.adk.tools.tool_context import ToolContext
from mcp import StdioServerParameters


# ============================================================
# Local Models (Hugging Face)
# ============================================================

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import logging as hf_logging

# Silence HF Hub and Transformers warnings
hf_logging.set_verbosity_error()
transformers.logging.set_verbosity_error()

# Cache for local model instances
LOCAL_MODEL_CACHE: Dict[str, Tuple[AutoModelForCausalLM, AutoTokenizer]] = {}


# ============================================================
# Runtime Noise Suppression and Protobuf Shim
# ============================================================

"""
Reduces non-critical runtime noise produced by TensorFlow, JAX and
third-party libraries inside the Kaggle environment. Also includes a
compatibility shim required to silence repeated protobuf errors caused
by deprecated MessageFactory APIs.
"""

# 1) Suppress TensorFlow / XLA / JAX noise
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
logging.getLogger("absl").setLevel(logging.ERROR)

# 2) Suppress unsupported Pydantic field attribute warnings
try:
    from pydantic._internal._generate_schema import UnsupportedFieldAttributeWarning

    warnings.filterwarnings(
        "ignore",
        category=UnsupportedFieldAttributeWarning,
        module="pydantic._internal._generate_schema",
    )
except Exception:
    pass

# 3) Protobuf compatibility shim for MessageFactory.GetPrototype
try:
    from google.protobuf import message_factory as _message_factory

    if not hasattr(_message_factory.MessageFactory, "GetPrototype"):

        def _get_prototype(self, descriptor):
            """Compatibility shim mapping deprecated GetPrototype() to GetMessageClass()."""
            return self.GetMessageClass(descriptor)

        _message_factory.MessageFactory.GetPrototype = _get_prototype

except Exception:
    pass

# 4) Silence generic absl and pandas RuntimeWarnings
warnings.filterwarnings("ignore", category=UserWarning, module="absl")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="pandas.io.formats.format")

# 5) Silence asyncio/aiohttp connector warnings inside notebook
logging.getLogger("asyncio").setLevel(logging.CRITICAL)
logging.getLogger("aiohttp.client").setLevel(logging.CRITICAL)


print("Global imports and environment configuration loaded.")
print("ADK components loaded.")



# ============================================================
# Gemini API Key Configuration (Kaggle Secrets)
# ============================================================

"""
This block retrieves and validates the Gemini API key stored in
Kaggle Secrets. The environment variable GOOGLE_API_KEY is exported
so that all underlying Google client libraries (GenAI SDK and ADK)
operate correctly.

Integration:
- Required before initializing the Gemini client, ADK agents or tools.
- Must run once at notebook startup.

Raises:
- ValueError if the key is missing or invalid.
"""

try:
    secrets = UserSecretsClient()
    GOOGLE_API_KEY = secrets.get_secret("GOOGLE_API_KEY_AGENTE2")

    if not GOOGLE_API_KEY or not isinstance(GOOGLE_API_KEY, str):
        raise ValueError("Gemini API key is missing or invalid.")

    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

    print("Gemini API key configured.")

except Exception as exc:
    raise RuntimeError(
        f"Failed to configure Gemini API key. "
        f"Ensure the Kaggle secret 'GOOGLE_API_KEY_AGENTE2' exists. "
        f"Details: {exc}"
    )


# ============================================================
# HTTP Retry Policy for Gemini Requests
# ============================================================

"""
Defines a retry policy for Gemini calls.

This configuration handles transient errors (rate limits, server
timeouts, temporary unavailability) by applying exponential backoff.

Integration:
- Passed to all genai.Client calls inside V-AID helpers.
- Ensures stability during competitive Kaggle workloads.
"""

retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)

if retry_config.attempts < 1:
    raise ValueError("Retry attempts must be >= 1.")

print("Retry configuration initialized.")


# ------------------------------------------------------------
# Text sanitization for logging
# ------------------------------------------------------------

"""
_sanitize_text_for_log

Purpose in the global pipeline:
- Normalizes text fields (queries, context, reasons) before they are
  stored in log rows or DataFrames.
- Prevents excessively long or malformed strings from polluting the
  Kaggle analysis outputs.
- Keeps logging robust even when upstream models return unexpected types.

Integration:
- Used by build_vaid_log_row to clean the textual fields that come from
  upstream models and V-AID summaries.
"""

def _sanitize_text_for_log(value: Any, max_length: int = 2000) -> str:
    """
    Sanitize arbitrary values for logging and DataFrame storage.

    Parameters
    ----------
    value : Any
        Arbitrary value to be converted into a safe text representation.
    max_length : int
        Maximum length of the returned string. Longer strings are truncated.

    Returns
    -------
    str
        Cleaned and truncated string representation suitable for logging.
    """
    if value is None:
        return ""

    # Convert to string and normalize whitespace
    text = str(value)
    text = text.replace("\n", " ").replace("\r", " ").strip()

    if len(text) > max_length:
        # Truncate but keep a clear indication that the content was shortened
        return text[: max_length - 3] + "..."
    return text


# ------------------------------------------------------------
# Log row construction for V-AID results
# ------------------------------------------------------------

"""
build_vaid_log_row

Purpose in the global pipeline:
- Converts a single V-AID evaluation (input + result) into a flat dictionary
  that can be appended to a pandas DataFrame.
- Enables systematic offline analysis of V-AID performance across many
  scenarios in a Kaggle competition setting.

Integration:
- Called by the test suite and experimentation cells to build structured
  logs.
- Designed to be robust even when V-AID returns an error status or partial
  metadata.
"""

def build_vaid_log_row(
    test_id: str,
    scenario: str,
    vaid_input: Dict[str, Any],
    vaid_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a flat log row from a V-AID input and evaluation result.

    Parameters
    ----------
    test_id : str
        Identifier of the test case (e.g. "T01_claims_basic").
    scenario : str
        Human-readable scenario label describing the test.
    vaid_input : Dict[str, Any]
        Original V-AID input dictionary containing query, context and metadata.
    vaid_result : Dict[str, Any]
        V-AID evaluation result as returned by the orchestrator.

    Returns
    -------
    Dict[str, Any]
        Flattened dictionary with fields ready to be stored in a DataFrame.

    Notes
    -----
    - This function does not modify its inputs.
    - It is safe to call even when V-AID returned an error status or partial
      metadata, defaulting to empty values when necessary.
    """
    # --- Input side ---
    query = _sanitize_text_for_log(vaid_input.get("query", ""))
    context_list = vaid_input.get("context", [])
    if isinstance(context_list, list):
        context_repr = _sanitize_text_for_log(" || ".join(str(c) for c in context_list))
    else:
        context_repr = _sanitize_text_for_log(str(context_list))

    metadata_input = vaid_input.get("metadata", {}) or {}
    model_source = str(vaid_input.get("model_source", "")).strip()

    # --- Result side ---
    status = str(vaid_result.get("status", "error")).strip()
    verdict = str(vaid_result.get("verdict", "")).strip()
    reason = _sanitize_text_for_log(vaid_result.get("reason", ""))

    summary = vaid_result.get("summary", {}) or {}
    num_claims = int(summary.get("num_claims", 0) or 0)
    num_claim_errors = int(summary.get("num_claim_errors", 0) or 0)
    num_claim_warnings = int(summary.get("num_claim_warnings", 0) or 0)
    num_ref_errors = int(summary.get("num_ref_errors", 0) or 0)
    num_ref_warnings = int(summary.get("num_ref_warnings", 0) or 0)

    meta_res = vaid_result.get("metadata", {}) or {}
    started_at = meta_res.get("started_at", "")
    finished_at = meta_res.get("finished_at", "")
    model_internal = meta_res.get("model_internal", "")
    model_web = meta_res.get("model_web", "")
    model_ref = meta_res.get("model_ref", "")
    upstream_model_source = meta_res.get("upstream_model_source", model_source)

    error_message = meta_res.get("error_message", "")

    return {
        # identifiers
        "test_id": test_id,
        "scenario": scenario,
        # upstream input
        "query": query,
        "context": context_repr,
        "input_metadata": json.dumps(metadata_input, ensure_ascii=False),
        "upstream_model_source": upstream_model_source,
        # global V-AID result
        "status": status,
        "verdict": verdict,
        "reason": reason,
        # counts
        "num_claims": num_claims,
        "num_claim_errors": num_claim_errors,
        "num_claim_warnings": num_claim_warnings,
        "num_ref_errors": num_ref_errors,
        "num_ref_warnings": num_ref_warnings,
        # models used inside V-AID
        "model_internal": model_internal,
        "model_web": model_web,
        "model_ref": model_ref,
        # timing and errors
        "started_at": started_at,
        "finished_at": finished_at,
        "error_message": error_message,
    }


# ------------------------------------------------------------
# Local model device resolution
# ------------------------------------------------------------

"""
_resolve_device

Purpose in the global pipeline:
- Chooses the execution device for local Hugging Face models.
- Prefers GPU when available in the Kaggle environment, but gracefully
  falls back to CPU.

Integration:
- Used by all local model helpers to ensure consistent device selection.
- Keeps device logic centralized and easy to adjust if competition
  constraints change.
"""

def _resolve_device(prefer_gpu: bool = True) -> str:
    """
    Resolve the device to run local models on.

    Parameters
    ----------
    prefer_gpu : bool
        If True, attempt to use CUDA if available.

    Returns
    -------
    str
        Device string ("cuda" or "cpu").
    """
    if prefer_gpu and torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ------------------------------------------------------------
# Local Hugging Face model loading with cache
# ------------------------------------------------------------

"""
_load_local_hf_model

Purpose in the global pipeline:
- Loads and caches Hugging Face causal language models and tokenizers
  so that repeated calls in the Kaggle notebook do not reload weights.

Integration:
- Used by generate_with_local_hf_model and preload_local_models.
- Relies on the global LOCAL_MODEL_CACHE defined in the imports section.
"""

def _load_local_hf_model(
    model_id: str,
    device: Optional[str] = None,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a local Hugging Face causal language model and tokenizer with caching.

    Parameters
    ----------
    model_id : str
        Hugging Face model identifier (must be public if no HF token is set).
    device : Optional[str]
        Optional device override. If None, a device will be resolved automatically.

    Returns
    -------
    Tuple[AutoModelForCausalLM, AutoTokenizer]
        Loaded model and tokenizer.

    Raises
    ------
    RuntimeError
        If model loading fails (for example gated repository or connectivity issues).
    """
    if model_id in LOCAL_MODEL_CACHE:
        return LOCAL_MODEL_CACHE[model_id]

    if device is None:
        device = _resolve_device(prefer_gpu=True)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        )
        model.to(device)
        model.eval()
    except Exception as exc:
        raise RuntimeError(f"Failed to load local model '{model_id}': {exc}") from exc

    LOCAL_MODEL_CACHE[model_id] = (model, tokenizer)
    return model, tokenizer


# ------------------------------------------------------------
# Local model preloading for experiments
# ------------------------------------------------------------

"""
preload_local_models

Purpose in the global pipeline:
- Warms up a set of local Hugging Face models at notebook startup.
- Reduces latency when running multiple V-AID experiments that reuse
  the same upstream generators.

Integration:
- Optional utility called in a setup cell if local models are used
  in the competition experiments.
"""

def preload_local_models(
    model_ids: List[str],
    prefer_gpu: bool = True,
) -> None:
    """
    Preload a list of local Hugging Face models into memory.

    Parameters
    ----------
    model_ids : List[str]
        List of Hugging Face model identifiers to preload.
    prefer_gpu : bool
        If True, load models on GPU when available.

    Notes
    -----
    This function is intended to be called once at notebook startup.
    It will skip models already present in LOCAL_MODEL_CACHE.
    """
    device = _resolve_device(prefer_gpu=prefer_gpu)
    for model_id in model_ids:
        if model_id in LOCAL_MODEL_CACHE:
            continue
        try:
            print(f"Loading local model '{model_id}' on device '{device}'...")
            _load_local_hf_model(model_id=model_id, device=device)
            print(f"  -> Loaded: {model_id}")
        except Exception as exc:
            print(f"  -> Error loading '{model_id}': {exc}")


# ------------------------------------------------------------
# Local model wrapper to produce V-AID inputs
# ------------------------------------------------------------

"""
generate_with_local_hf_model

Purpose in the global pipeline:
- Wraps a Hugging Face causal language model as an upstream generator
  that produces standardized V-AID inputs.
- Allows comparing Gemini-based pipelines against local or open-source
  models under the same V-AID evaluation logic.

Integration:
- Used in experimental cells to generate an answer with a local model,
  then pass the resulting V-AID input to the orchestrator.
- Keeps the format of the returned dictionary aligned with V-AID expectations.
"""

def generate_with_local_hf_model(
    model_id: str,
    query: str,
    context: Optional[Union[str, List[str]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    model_source: Optional[str] = None,
    max_new_tokens: int = 256,
    temperature: float = 0.9,
    top_k: int = 50,
    top_p: float = 0.95,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate a response using a local Hugging Face causal model and wrap it as V-AID input.

    Parameters
    ----------
    model_id : str
        Hugging Face model identifier (for example 'gpt2'). Must be public if no HF token is set.
    query : str
        User question or instruction.
    context : Optional[Union[str, List[str]]]
        Optional context to prepend to the prompt as plain text.
    metadata : Optional[Dict[str, Any]]
        Optional metadata dictionary to attach to the V-AID input.
    model_source : Optional[str]
        Identifier string for the upstream model. If None, model_id is used.
    max_new_tokens : int
        Maximum number of tokens to generate.
    temperature : float
        Sampling temperature for generation.
    top_k : int
        Top-k sampling parameter.
    top_p : float
        Nucleus sampling (top-p) parameter.
    device : Optional[str]
        Device override, for example "cuda" or "cpu". If None, it will be resolved.

    Returns
    -------
    Dict[str, Any]
        Standardized V-AID input dictionary including query, response, context, metadata and model_source.

    Raises
    ------
    ValueError
        If the query is empty or not a string.
    RuntimeError
        If generation fails for any reason.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string.")

    if context is None:
        context_list: List[str] = []
    elif isinstance(context, str):
        context_list = [context]
    elif isinstance(context, list) and all(isinstance(c, str) for c in context):
        context_list = context
    else:
        raise ValueError("Context must be None, a string, or a list of strings.")

    if device is None:
        device = _resolve_device(prefer_gpu=True)

    model, tokenizer = _load_local_hf_model(model_id=model_id, device=device)

    # Build an instruction-like prompt to encourage more structured answers
    prompt_parts: List[str] = []
    prompt_parts.append("You are a general language model without external tools.")
    if context_list:
        prompt_parts.append("")
        prompt_parts.append("Context:")
        prompt_parts.append("\n".join(context_list))
    prompt_parts.append("")
    prompt_parts.append("Question:")
    prompt_parts.append(query)
    prompt_parts.append("")
    prompt_parts.append("Answer:")
    prompt = "\n".join(prompt_parts)

    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True,
        )
    except Exception as exc:
        raise RuntimeError(f"Local model generation failed for '{model_id}': {exc}") from exc

    # Try to keep only the answer portion after the prompt
    if generated.startswith(prompt):
        response_text = generated[len(prompt):].strip()
    else:
        response_text = generated.strip()

    if not response_text:
        response_text = "[Empty or invalid generation from local model.]"

    if model_source is None:
        model_source = f"local:{model_id}"

    enriched_metadata: Dict[str, Any] = {
        "timestamp": datetime.utcnow().isoformat(),
        "model_source": model_source,
        "local_model_id": model_id,
        "device": device,
    }
    if metadata:
        enriched_metadata.update(metadata)

    return {
        "query": query,
        "response": response_text,
        "context": context_list,
        "metadata": enriched_metadata,
        "model_source": model_source,
    }


# ------------------------------------------------------------
# Gemini-based upstream generator
# ------------------------------------------------------------

"""
generate_with_gemini

Purpose in the global pipeline:
- Uses a Gemini model as the upstream generator and returns a V-AID
  compatible input dictionary.
- Allows evaluating V-AID directly on answers produced by the same
  family of models that will be used in production.

Integration:
- Called in experiment and test cells to obtain a single standardized
  input object for the V-AID orchestrator.
"""

def generate_with_gemini(
    client: genai.Client,
    query: str,
    context: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    model: str = "gemini-2.0-flash",
    model_source: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate a text response using a Gemini model and wrap it into
    a V-AID compatible input structure.

    Parameters
    ----------
    client : genai.Client
        Configured Google Gemini client instance.
    query : str
        Main question or prompt to send to the model.
    context : Optional[List[str]]
        Optional list of context strings (instructions, background, retrieved passages).
    metadata : Optional[Dict[str, Any]]
        Optional metadata dictionary to attach to the V-AID input (for example run_id).
    model : str
        Model name to use (for example 'gemini-2.0-flash').
    model_source : Optional[str]
        Identifier for the upstream model configuration. If None, a default
        based on the model name is used.

    Returns
    -------
    Dict[str, Any]
        Standardized V-AID input dictionary with keys:
        - "query"
        - "response"
        - "context"
        - "metadata"
        - "model_source"

    Raises
    ------
    ValueError
        If the query is empty or invalid.
    RuntimeError
        If the generation call fails or returns no valid text.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string.")

    # Normalize context to a list of strings
    if context is None:
        context_list: List[str] = []
    elif isinstance(context, list) and all(isinstance(c, str) for c in context):
        context_list = context
    else:
        raise ValueError("Context must be a list of strings or None.")

    # 1) Build full prompt
    full_prompt_parts: List[str] = []
    if context_list:
        full_prompt_parts.append("\n".join(context_list))
        full_prompt_parts.append("")  # blank line separator
    full_prompt_parts.append(f"Query: {query}")
    full_prompt = "\n".join(full_prompt_parts)

    # 2) Call Gemini API
    try:
        contents = [types.Content(parts=[types.Part(text=full_prompt)])]

        response = client.models.generate_content(
            model=model,
            contents=contents,
        )

        response_text = getattr(response, "text", None)
        if not isinstance(response_text, str) or not response_text.strip():
            raise ValueError("Gemini returned an empty or invalid response.")

    except Exception as exc:
        raise RuntimeError(f"Gemini generation failed: {exc}") from exc

    # 3) Build enriched metadata
    if model_source is None:
        model_source = f"gemini:{model}"

    enriched_metadata: Dict[str, Any] = {
        "timestamp": datetime.utcnow().isoformat(),
        "model_source": model_source,
    }
    if metadata:
        enriched_metadata.update(metadata)

    # 4) Return standardized input
    return {
        "query": query,
        "response": response_text.strip(),
        "context": context_list,
        "metadata": enriched_metadata,
        "model_source": model_source,
    }


# ------------------------------------------------------------
# Local model variant A (small baseline)
# ------------------------------------------------------------

"""
generate_with_local_model_a

Purpose in the global pipeline:
- Uses a small public Hugging Face model as an upstream generator.
- Produces relatively short and simple answers, often lacking strong
  reasoning or grounding, which helps to stress-test V-AID on weaker
  model outputs.

Integration:
- Thin wrapper around generate_with_local_hf_model, keeping a fixed
  configuration for reproducible experiments in Kaggle.
"""

def generate_with_local_model_a(
    query: str,
    context: Optional[Union[str, List[str]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    max_new_tokens: int = 256,
    temperature: float = 0.9,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate a response using a small public local model (Model A) and wrap it as V-AID input.

    This variant aims to produce roughly coherent, short answers that still
    lack deep reasoning, grounding or explicit uncertainty handling, which
    makes them good candidates to stress-test V-AID.

    Parameters
    ----------
    query : str
        User question or instruction.
    context : Optional[Union[str, List[str]]]
        Optional context to prepend to the prompt.
    metadata : Optional[Dict[str, Any]]
        Optional metadata dictionary to attach to the V-AID input.
    max_new_tokens : int
        Maximum number of tokens to generate.
    temperature : float
        Sampling temperature for generation.
    device : Optional[str]
        Device override (for example "cuda", "cpu"). If None, it will be resolved.

    Returns
    -------
    Dict[str, Any]
        Standardized V-AID input dictionary.
    """
    model_id = "gpt2"  # Public baseline model
    return generate_with_local_hf_model(
        model_id=model_id,
        query=query,
        context=context,
        metadata=metadata,
        model_source="local-model-a-gpt2",
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=50,
        top_p=0.95,
        device=device,
    )


# ------------------------------------------------------------
# Local model variant B (noisier baseline)
# ------------------------------------------------------------

"""
generate_with_local_model_b

Purpose in the global pipeline:
- Uses a slightly larger or more expressive local model with a more
  aggressive sampling configuration.
- Intentionally produces more creative and noisy answers, often
  off-topic or overconfident, which are useful to probe V-AID's
  detection of hallucinations and weakly grounded claims.

Integration:
- Also wraps generate_with_local_hf_model with a fixed configuration
  to keep experiments reproducible.
"""

def generate_with_local_model_b(
    query: str,
    context: Optional[Union[str, List[str]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    max_new_tokens: int = 256,
    temperature: float = 1.2,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate a response using a medium public local model (Model B) and wrap it as V-AID input.

    This variant is configured to be more creative and noisy (higher temperature
    and sampling diversity), which should produce answers that are readable but
    often off-topic, incomplete or overconfident, providing challenging test
    cases for V-AID.

    Parameters
    ----------
    query : str
        User question or instruction.
    context : Optional[Union[str, List[str]]]
        Optional context to prepend to the prompt.
    metadata : Optional[Dict[str, Any]]
        Optional metadata dictionary to attach to the V-AID input.
    max_new_tokens : int
        Maximum number of tokens to generate.
    temperature : float
        Sampling temperature for generation (higher values increase randomness).
    device : Optional[str]
        Device override (for example "cuda", "cpu"). If None, it will be resolved.

    Returns
    -------
    Dict[str, Any]
        Standardized V-AID input dictionary.
    """
    model_id = "gpt2-medium"  # If OOM in Kaggle, change to "gpt2"
    return generate_with_local_hf_model(
        model_id=model_id,
        query=query,
        context=context,
        metadata=metadata,
        model_source="local-model-b-gpt2-medium",
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=40,
        top_p=0.9,
        device=device,
    )


# ------------------------------------------------------------
# Manual upstream response wrapper
# ------------------------------------------------------------

"""
wrap_manual_response

Purpose in the global pipeline:
- Wraps a manually written answer into the same structure used for
  LLM outputs, so that V-AID can evaluate human-crafted or synthetic
  examples without calling any model.

Integration:
- Used in debugging, unit tests and illustrative examples.
- Enables testing specific failure patterns that are hard to obtain
  deterministically from live models.
"""

def wrap_manual_response(
    query: str,
    response: str,
    context: Optional[Union[str, List[str]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    model_source: str = "manual-input",
) -> Dict[str, Any]:
    """
    Wrap a manually provided response into a V-AID compatible input structure.

    This is useful for debugging, unit tests, or simulating upstream LLM outputs
    without calling a real model.

    Parameters
    ----------
    query : str
        User question or instruction originally posed to the upstream model.
    response : str
        Manually provided response text to be evaluated by V-AID.
    context : Optional[Union[str, List[str]]]
        Optional context associated with this interaction.
    metadata : Optional[Dict[str, Any]]
        Optional metadata dictionary to attach to the input.
    model_source : str
        Identifier used to mark this as a manual or synthetic source.

    Returns
    -------
    Dict[str, Any]
        Standardized V-AID input dictionary.

    Raises
    ------
    ValueError
        If query or response are empty.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string.")
    if not isinstance(response, str) or not response.strip():
        raise ValueError("Response must be a non-empty string.")

    if context is None:
        context_list: List[str] = []
    elif isinstance(context, str):
        context_list = [context]
    elif isinstance(context, list) and all(isinstance(c, str) for c in context):
        context_list = context
    else:
        raise ValueError("Context must be None, a string, or a list of strings.")

    enriched_metadata: Dict[str, Any] = {
        "timestamp": datetime.utcnow().isoformat(),
        "model_source": model_source,
    }
    if metadata:
        enriched_metadata.update(metadata)

    return {
        "query": query,
        "response": response.strip(),
        "context": context_list,
        "metadata": enriched_metadata,
        "model_source": model_source,
    }



# ------------------------------------------------------------
# Schema assertion helper
# ------------------------------------------------------------

"""
_assert_vaid_input_schema

Purpose in the global pipeline:
- Centralizes validation of the standardized V-AID input format.
- Ensures that all upstream generators conform to the same contract
  before their outputs are passed to the orchestrator.

Integration:
- Used by the smoke test function to validate each generator result.
"""

def _assert_vaid_input_schema(source_label: str, vaid_input: Dict[str, Any]) -> None:
    """
    Validate that a dictionary follows the expected V-AID input schema.

    Parameters
    ----------
    source_label : str
        Human-readable label identifying the upstream generator under test.
    vaid_input : Dict[str, Any]
        Dictionary returned by a synthetic generator (manual, Gemini or local).

    Raises
    ------
    ValueError
        If any required field is missing or has an invalid type.
    """
    required_keys = ["query", "response", "context", "metadata", "model_source"]
    for key in required_keys:
        if key not in vaid_input:
            raise ValueError(f"[{source_label}] Missing required key: '{key}'.")

    if not isinstance(vaid_input["query"], str) or not vaid_input["query"].strip():
        raise ValueError(f"[{source_label}] 'query' must be a non-empty string.")

    if not isinstance(vaid_input["response"], str) or not vaid_input["response"].strip():
        raise ValueError(f"[{source_label}] 'response' must be a non-empty string.")

    if not isinstance(vaid_input["context"], list):
        raise ValueError(f"[{source_label}] 'context' must be a list of strings.")
    if not all(isinstance(c, str) for c in vaid_input["context"]):
        raise ValueError(f"[{source_label}] 'context' must contain only strings.")

    if not isinstance(vaid_input["metadata"], dict):
        raise ValueError(f"[{source_label}] 'metadata' must be a dictionary.")

    if not isinstance(vaid_input["model_source"], str) or not vaid_input["model_source"].strip():
        raise ValueError(f"[{source_label}] 'model_source' must be a non-empty string.")


# ------------------------------------------------------------
# Smoke test runner for synthetic generators
# ------------------------------------------------------------

"""
run_synthetic_generators_smoke_tests

Purpose in the global pipeline:
- Executes a minimal set of end-to-end checks for all synthetic upstream
  generators (manual, local models and optionally Gemini).
- Provides a quick diagnostic table that confirms the generators are
  usable before V-AID evaluations are launched.

Integration:
- Can be called once at the beginning of an experiment notebook.
- Returns a pandas DataFrame summarizing pass/fail status and any error
  messages, which is convenient for Kaggle analysis and logging.
"""


def run_synthetic_generators_smoke_tests(
    client: Optional[genai.Client] = None,
    run_gemini: bool = True,
) -> pd.DataFrame:
    """
    Run simple smoke tests for all synthetic upstream generators.

    Parameters
    ----------
    client : Optional[genai.Client]
        Configured Gemini client. Required if `run_gemini` is True.
    run_gemini : bool
        If True, the Gemini-based generator is tested. If False, it is skipped.

    Returns
    -------
    pd.DataFrame
        Summary table with one row per generator, including:
        - generator_name
        - status ("ok" or "error")
        - error_message
        - query
        - response
        - context
        - metadata
        - model_source
        - response_preview (first characters of the generated response)
    """
    results: List[Dict[str, Any]] = []

    # Common test payload
    test_query = "Explain how rain forms in the atmosphere using simple language."
    test_context = ["This question is about basic weather phenomena and everyday atmospheric processes."]


    # Helper to build a result row from a valid V-AID input
    def _build_result_row(
        generator_name: str,
        vaid_input: Dict[str, Any],
        status: str = "ok",
        error_message: str = "",
    ) -> Dict[str, Any]:
        context_joined = " || ".join(vaid_input.get("context", []))
        # Use json.dumps for metadata so that it is easy to inspect in the DataFrame
        metadata_str = json.dumps(vaid_input.get("metadata", {}), ensure_ascii=False)

        return {
            "generator_name": generator_name,
            "status": status,
            "error_message": error_message,
            "query": vaid_input.get("query", ""),
            "response": vaid_input.get("response", ""),
            "context": context_joined,
            "metadata": metadata_str,
            "model_source": vaid_input.get("model_source", ""),
            "response_preview": vaid_input.get("response", "")[:120],
        }

    # Helper to build a result row for failed generators
    def _build_error_row(generator_name: str, exc: Exception) -> Dict[str, Any]:
        return {
            "generator_name": generator_name,
            "status": "error",
            "error_message": str(exc),
            "query": "",
            "response": "",
            "context": "",
            "metadata": "",
            "model_source": "",
            "response_preview": "",
        }

    # 1) Manual wrapper
    try:
        manual_input = wrap_manual_response(
            query=test_query,
            response="Manual TEST",
            context=test_context,
            metadata={"test_id": "smoke_manual"},
        )
        _assert_vaid_input_schema("manual", manual_input)
        results.append(_build_result_row("manual", manual_input))
    except Exception as exc:
        results.append(_build_error_row("manual", exc))

    # 2) Local model A
    try:
        local_a_input = generate_with_local_model_a(
            query=test_query,
            context=test_context,
            metadata={"test_id": "smoke_local_a"},
        )
        _assert_vaid_input_schema("local_model_a", local_a_input)
        results.append(_build_result_row("local_model_a", local_a_input))
    except Exception as exc:
        results.append(_build_error_row("local_model_a", exc))

    # 3) Local model B
    try:
        local_b_input = generate_with_local_model_b(
            query=test_query,
            context=test_context,
            metadata={"test_id": "smoke_local_b"},
        )
        _assert_vaid_input_schema("local_model_b", local_b_input)
        results.append(_build_result_row("local_model_b", local_b_input))
    except Exception as exc:
        results.append(_build_error_row("local_model_b", exc))

    # 4) Gemini-based generator (optional)
    if run_gemini:
        try:
            if client is None:
                raise RuntimeError("Gemini client is None but run_gemini=True.")
            gemini_input = generate_with_gemini(
                client=client,
                query=test_query,
                context=test_context,
                metadata={"test_id": "smoke_gemini"},
                model="gemini-2.0-flash",
            )
            _assert_vaid_input_schema("gemini", gemini_input)
            results.append(_build_result_row("gemini", gemini_input))
        except Exception as exc:
            results.append(_build_error_row("gemini", exc))

    return pd.DataFrame(results)



client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
df_smoke = run_synthetic_generators_smoke_tests(
    client=client,      # Gemini client
    run_gemini=True     # include Gemini generator
)

df_smoke




# ------------------------------------------------------------
# Lazy Gemini client initialization
# ------------------------------------------------------------

# Explanation:
# A single global Gemini client is reused across calls to reduce overhead.
# The API key is configured earlier via environment variables. This helper
# is intentionally minimal to keep the tool layer focused on claim logic.

_GENAI_CLIENT: Optional[genai.Client] = None


def get_genai_client() -> genai.Client:
    """
    Lazily initialize and cache a global Gemini client instance.

    Returns
    -------
    genai.Client
        Configured Gemini client that uses the GOOGLE_API_KEY environment variable.

    Notes
    -----
    - This helper avoids repeatedly constructing client objects.
    - It assumes that GOOGLE_API_KEY has been set beforehand.
    """
    global _GENAI_CLIENT
    if _GENAI_CLIENT is None:
        _GENAI_CLIENT = genai.Client()
    return _GENAI_CLIENT


# ------------------------------------------------------------
# Data structure for extracted claims
# ------------------------------------------------------------

# Explanation:
# VaidClaim is the canonical structure used to represent a single atomic
# factual claim extracted from an upstream response. It encapsulates both
# the text and minimal metadata required for downstream validation steps.

@dataclass
class VaidClaim:
    """
    Structured representation of a single atomic factual claim.

    Attributes
    ----------
    claim_id : str
        Stable identifier for the claim within a single response.
    text : str
        Text of the factual claim as a standalone statement.
    claim_type : str
        Coarse-grained category of the claim (for example "factual").
    criticality : str
        Importance level of the claim ("low", "medium", "high", "critical").
    source_span : Optional[str]
        Exact or approximate substring from the original response where
        this claim was extracted, useful for traceability.
    """
    claim_id: str
    text: str
    claim_type: str = "factual"
    criticality: str = "medium"
    source_span: Optional[str] = None


# ------------------------------------------------------------
# Prompt builders (extraction and structural filtering)
# ------------------------------------------------------------

# Explanation:
# The claim extraction process is deliberately split into two prompts:
# 1) Extraction prompt: enumerates all candidate atomic factual claims.
# 2) Structural validation prompt: removes items that are not true claims
#    without performing any factual verification.

def _build_claim_extraction_prompt(response_text: str) -> str:
    """
    Build the prompt used to extract atomic factual claims from a response.

    Parameters
    ----------
    response_text : str
        Full text of the upstream response to be analyzed.

    Returns
    -------
    str
        Prompt string instructing the model to output a JSON object with
        a "claims" list.
    """
    return f"""
You are a CLAIM EXTRACTION ENGINE.

Your task is to extract ATOMIC FACTUAL CLAIMS from the following RESPONSE.
A claim is a single factual statement that can be verified or disproven.

### DEFINITION OF CLAIM
- Declarative factual statement about the system, world, or process.
- Must be explicitly present in the RESPONSE.
- Must be atomic (one fact per claim).
- Must be verifiable.
- Must not be an opinion, question, command, or meta-comment.

### VALID CLAIM EXAMPLES
- "RAG combines a retriever and a generator."
- "AI agents require human oversight in high-risk scenarios."
- "This system uses three components: retrieval, ranking, and synthesis."

### NOT CLAIMS
- "What is RAG?"
- "Click the button to continue."
- "This approach is interesting."
- "AI agents could theoretically replace managers." (not stated explicitly)
- "RAG always improves accuracy." (adds qualifiers not in text)

### OUTPUT FORMAT (STRICT JSON)
{{
  "claims": [
    {{
      "claim_id": "c01",
      "text": "the factual claim here",
      "criticality": "medium",
      "claim_type": "factual",
      "source_span": "exact substring from the RESPONSE"
    }}
  ]
}}

### RESPONSE TO ANALYZE
\"\"\"{response_text.strip()}\"\"\""""


def _build_claim_validation_prompt(
    response_text: str,
    extracted_claims: List[Dict[str, Any]],
) -> str:
    """
    Build the prompt used to filter out non-claim items from the extraction.

    Parameters
    ----------
    response_text : str
        Full upstream response text (ORIGINAL_TEXT).
    extracted_claims : List[Dict[str, Any]]
        List of candidate claims produced by the extraction step.

    Returns
    -------
    str
        Prompt string instructing the model to output a JSON object with
        a "validated_claims" list containing only structurally valid claims.
    """
    serialized_claims = json.dumps(extracted_claims, indent=2)

    return f"""
You are a CLAIM STRUCTURAL VALIDATOR.

You receive:
- ORIGINAL_TEXT
- CANDIDATE_CLAIMS (JSON list)

Your job is to REMOVE items that:
- Are not factual claims
- Cannot be verified as statements
- Modify or extend meaning beyond ORIGINAL_TEXT
- Are questions, instructions, or opinions

Do NOT create new claims.
Do NOT rewrite claims creatively.
If uncertain, remove the claim.

### VALID OUTPUT FORMAT
{{
  "validated_claims": [...]
}}

### ORIGINAL_TEXT
\"\"\"{response_text.strip()}\"\"\"


### CANDIDATE_CLAIMS
\"\"\"{serialized_claims}\"\"\""""


# ------------------------------------------------------------
# Robust JSON extraction helper
# ------------------------------------------------------------

# Explanation:
# LLMs may prepend or append text around JSON. This helper tolerates
# minor formatting noise by searching for the first JSON object and
# then extracting the list under the expected key.

def _safe_json_extract(raw: str, key: str) -> List[Dict[str, Any]]:
    """
    Safely extract a list of dictionaries from an LLM JSON-like response.

    Parameters
    ----------
    raw : str
        Raw text returned by the model.
    key : str
        Top-level key expected to contain a list (for example "claims").

    Returns
    -------
    List[Dict[str, Any]]
        Parsed list associated with the given key.

    Raises
    ------
    ValueError
        If the content is empty, invalid JSON or does not contain the key.
    """
    if not raw or not isinstance(raw, str):
        raise ValueError("Extractor returned empty output.")

    raw = raw.strip()

    # First try direct JSON
    try:
        parsed = json.loads(raw)
        if key in parsed and isinstance(parsed[key], list):
            return parsed[key]
    except json.JSONDecodeError:
        # Fall back to fragment search below
        pass

    # Fallback: find first '{ ... }' fragment
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"Expected JSON but got:\n{raw}")

    fragment = raw[start : end + 1]
    parsed = json.loads(fragment)

    if key not in parsed or not isinstance(parsed[key], list):
        raise ValueError(f"JSON parsed but key '{key}' is missing or not a list.")

    return parsed[key]


# ------------------------------------------------------------
# Dynamic limit for number of claims
# ------------------------------------------------------------

# Explanation:
# To keep the verification pipeline efficient, the number of claims is
# capped based on the length and approximate sentence count of the input
# response. Longer texts tolerate more claims but are still bounded.

def _determine_dynamic_max_claims(response_text: str) -> int:
    """
    Determine an upper bound for the number of claims based on text size.

    Parameters
    ----------
    response_text : str
        Original upstream response text.

    Returns
    -------
    int
        Maximum number of claims to keep after prioritization.
    """
    length = len(response_text)
    sentences = (
        response_text.count(".")
        + response_text.count("!")
        + response_text.count("?")
    )

    if length < 800:
        return min(40, sentences * 2)
    elif length < 2500:
        return max(15, sentences // 2)
    else:
        return max(10, sentences // 3)


# ------------------------------------------------------------
# Prioritization when there are too many claims
# ------------------------------------------------------------

# Explanation:
# When the model returns more claims than the dynamic limit, this helper
# selects the most important ones according to criticality and length,
# ensuring that verification effort focuses on the most relevant content.

def _prioritize_claims(
    claims: List[Dict[str, Any]],
    max_claims: int,
) -> List[Dict[str, Any]]:
    """
    Prioritize and truncate a list of claims according to criticality.

    Parameters
    ----------
    claims : List[Dict[str, Any]]
        List of claim dictionaries as produced by the model.
    max_claims : int
        Maximum number of claims to keep.

    Returns
    -------
    List[Dict[str, Any]]
        Subset of claims sorted and truncated according to priority.
    """
    priority = {"critical": 4, "high": 3, "medium": 2, "low": 1}

    sorted_claims = sorted(
        claims,
        key=lambda c: (
            -priority.get(c.get("criticality", "medium"), 2),
            len(c.get("text", "")),
            c.get("claim_id", ""),
        ),
    )

    return sorted_claims[:max_claims]


# ------------------------------------------------------------
# Core extraction + structural validation routine
# ------------------------------------------------------------

# Explanation:
# This function coordinates the full claim extraction step:
# - Builds prompts
# - Calls Gemini
# - Parses JSON
# - Applies dynamic limit and prioritization
# - Normalizes into VaidClaim records

def extract_and_validate_claims_core(
    client: genai.Client,
    vaid_input: Dict[str, Any],
    model: str = "gemini-2.0-flash",
) -> Dict[str, Any]:
    """
    Run the full claim extraction and structural validation process.

    Parameters
    ----------
    client : genai.Client
        Configured Gemini client instance.
    vaid_input : Dict[str, Any]
        Standard V-AID input containing at least a 'response' field.
    model : str
        Gemini model name to use for both extraction and structural validation.

    Returns
    -------
    Dict[str, Any]
        Dictionary with:
        - status : "success"
        - num_claims : int
        - claims : List[Dict[str, Any]]
        - metadata : Dict[str, Any] with processing information.

    Raises
    ------
    ValueError
        If the input response is missing or empty.
    RuntimeError
        If JSON parsing or model calls fail unexpectedly.
    """
    response_text = vaid_input.get("response", "")
    if not isinstance(response_text, str) or not response_text.strip():
        raise ValueError("vaid_input['response'] must be non-empty text.")

    # ---------------- EXTRACT ----------------
    prompt_ex = _build_claim_extraction_prompt(response_text)
    raw_ex = client.models.generate_content(
        model=model,
        contents=[types.Content(parts=[types.Part(text=prompt_ex)])],
    ).text

    extracted = _safe_json_extract(raw_ex, "claims")

    # ---------------- VALIDATE ----------------
    prompt_val = _build_claim_validation_prompt(response_text, extracted)
    raw_val = client.models.generate_content(
        model=model,
        contents=[types.Content(parts=[types.Part(text=prompt_val)])],
    ).text

    validated = _safe_json_extract(raw_val, "validated_claims")

    # ---------------- LIMIT + PRIORITIZE ----------------
    max_claims = _determine_dynamic_max_claims(response_text)
    if len(validated) > max_claims:
        validated = _prioritize_claims(validated, max_claims)

    # ---------------- Normalize ----------------
    final_claims: List[Dict[str, Any]] = []
    for idx, c in enumerate(validated, start=1):
        claim = VaidClaim(
            claim_id=c.get("claim_id", f"c{idx:02d}"),
            text=c["text"],
            claim_type=c.get("claim_type", "factual"),
            criticality=c.get("criticality", "medium"),
            source_span=c.get("source_span", c["text"]),
        )
        final_claims.append(asdict(claim))

    return {
        "status": "success",
        "num_claims": len(final_claims),
        "claims": final_claims,
        "metadata": {
            "processed_at": datetime.utcnow().isoformat(),
            "model_source": vaid_input.get("model_source"),
        },
    }


# ------------------------------------------------------------
# ADK-compatible tool wrapper
# ------------------------------------------------------------

# Explanation:
# This wrapper exposes the core functionality with a simple, tool-like
# interface. It handles errors defensively and returns a structured error
# payload instead of raising exceptions, which is safer for orchestration.

def claim_extractor_vaid_tool(
    vaid_input: Dict[str, Any],
    model: str = "gemini-2.0-flash",
) -> Dict[str, Any]:
    """
    ADK-compatible wrapper around the claim extraction core.

    Parameters
    ----------
    vaid_input : Dict[str, Any]
        Standard V-AID input containing upstream response text.
    model : str
        Gemini model name to use for extraction and structural validation.

    Returns
    -------
    Dict[str, Any]
        On success:
        - status : "success"
        - num_claims : int
        - claims : List[Dict[str, Any]]
        - metadata : Dict[str, Any]
        On error:
        - status : "error"
        - error_message : str
        - claims : []
        - num_claims : 0
    """
    try:
        client = get_genai_client()
        return extract_and_validate_claims_core(
            client=client,
            vaid_input=vaid_input,
            model=model,
        )
    except Exception as exc:
        return {
            "status": "error",
            "error_message": str(exc),
            "claims": [],
            "num_claims": 0,
        }



# ============================================================
# 0) Retry utilities (shared across internal and web checks)
# ============================================================

"""
The retry helpers centralize the logic that interprets HttpRetryOptions
and converts them into concrete parameters for exponential backoff.

They are used by:
- _llm_internal_check
- _llm_web_check
- fetch_web_evidence

This design keeps retry behavior consistent across the tool.
"""


def _extract_retry_params(
    retry_config: Optional[types.HttpRetryOptions],
) -> Tuple[int, float, float, float, Set[int]]:
    """
    Extract retry parameters from HttpRetryOptions, with safe defaults.

    Parameters
    ----------
    retry_config : Optional[types.HttpRetryOptions]
        High-level retry configuration provided to the tool.

    Returns
    -------
    attempts : int
        Maximum number of attempts (including the first call).
    exp_base : float
        Exponential base for backoff growth.
    initial_delay : float
        Initial delay in seconds before the first retry.
    max_delay : float
        Maximum delay in seconds between retries.
    status_codes : set[int]
        HTTP status codes that should trigger a retry.
    """
    if retry_config is None:
        return 1, 2.0, 1.0, 30.0, {429, 500, 503, 504}

    attempts = getattr(retry_config, "attempts", 1) or 1
    exp_base = getattr(retry_config, "exp_base", 2.0) or 2.0
    initial_delay = getattr(retry_config, "initial_delay", 1.0) or 1.0
    max_delay = getattr(retry_config, "max_delay", 30.0) or 30.0

    codes = getattr(retry_config, "http_status_codes", None)
    if not codes:
        status_codes = {429, 500, 503, 504}
    else:
        status_codes = {int(c) for c in codes}

    return int(attempts), float(exp_base), float(initial_delay), float(max_delay), status_codes


def _compute_backoff_delay(
    attempt_index: int,
    initial_delay: float,
    exp_base: float,
    max_delay: float,
) -> float:
    """
    Compute exponential backoff delay based on attempt index.

    Parameters
    ----------
    attempt_index : int
        Zero-based index of the current retry attempt.
    initial_delay : float
        Initial delay in seconds.
    exp_base : float
        Exponential base for backoff growth.
    max_delay : float
        Upper bound on delay in seconds.

    Returns
    -------
    float
        Delay in seconds to wait before the next attempt.
    """
    delay = initial_delay * (exp_base ** attempt_index)
    return float(min(delay, max_delay))


# ============================================================
# 1) Web search agent (ADK + google_search)
# ============================================================

"""
The web search component provides lightweight external evidence using
the google_search tool exposed through ADK. It is intentionally narrow:
it only retrieves short factual snippets for a single claim.

The create_vaid_web_search_agent function configures a minimal agent
with strict JSON output requirements. The fetch_web_evidence coroutine
wraps the agent into a convenient helper that returns a list of
{title, url, snippet} dictionaries.
"""


def create_vaid_web_search_agent(
    model: str = "gemini-2.0-flash",
) -> Agent:
    """
    Create a minimal ADK agent that uses ONLY google_search to fetch factual snippets.

    Parameters
    ----------
    model : str
        Gemini model name used by the agent for tool-augmented reasoning.

    Returns
    -------
    Agent
        Configured ADK agent restricted to the google_search tool.
    """
    return Agent(
        model=model,
        name="vaid_web_search",
        instruction=(
            "You verify factual claims using strictly Google Search via the google_search tool.\n"
            "\n"
            "RULES:\n"
            "- Only return results obtained directly from google_search.\n"
            "- Do NOT invent URLs, authors, institutions, or citations.\n"
            "- Do NOT summarize beyond short factual snippets.\n"
            "- Return ONLY JSON with this format:\n"
            "{\n"
            '  \"evidence\": [\n'
            '    { \"title\": \"...\", \"url\": \"...\", \"snippet\": \"...\" }\n'
            "  ]\n"
            "}\n"
        ),
        tools=[google_search],
    )


async def fetch_web_evidence(
    claim: str,
    max_results: int = 3,
    model: str = "gemini-2.0-flash",
    session_id: str = "vaid-web-001",
    retry_config: Optional[types.HttpRetryOptions] = None,
) -> List[Dict[str, Any]]:
    """
    Fetch evidence from the web for a single claim using ADK + google_search.

    Returns a list of dicts: [{\"title\", \"url\", \"snippet\"}, ...]
    """
    if not claim or not isinstance(claim, str):
        raise ValueError("claim must be a non-empty string.")

    attempts, exp_base, initial_delay, max_delay, status_codes = _extract_retry_params(retry_config)

    async def _single_attempt() -> List[Dict[str, Any]]:
        agent = create_vaid_web_search_agent(model=model)

        session_service = InMemorySessionService()
        session = await session_service.create_session(
            app_name="vaid_web",
            user_id="vaid",
            session_id=session_id,
        )

        runner = Runner(
            agent=agent,
            app_name="vaid_web",
            session_service=session_service,
        )

        try:
            prompt = (
                "Search the web for evidence to verify this claim.\n\n"
                f'CLAIM: \"{claim}\"\n\n'
                "Return ONLY JSON with field \"evidence\" (list of {title,url,snippet}).\n"
                f"Maximum {max_results} items."
            )

            content = types.Content(
                role="user",
                parts=[types.Part(text=prompt)],
            )

            events = runner.run_async(
                user_id="vaid",
                session_id=session.id,
                new_message=content,
            )

            final_text: Optional[str] = None

            async for event in events:
                if event.is_final_response():
                    if event.content and event.content.parts:
                        final_text = event.content.parts[0].text
                    break

            if not final_text:
                return []

            def _safe_json_from_text(text: str) -> Dict[str, Any]:
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("Empty text; cannot parse JSON.")
                raw = text.strip()
                # Try direct JSON parse
                try:
                    return json.loads(raw)
                except Exception:
                    pass
                # Try fenced code block
                if raw.startswith("```"):
                    raw = raw.strip("`")
                    parts = raw.split("\n", 1)
                    if len(parts) == 2:
                        raw = parts[1].strip()
                    try:
                        return json.loads(raw)
                    except Exception:
                        pass
                # Try substring between first '{' and last '}'
                start = raw.find("{")
                end = raw.rfind("}")
                if start == -1 or end == -1 or end <= start:
                    raise ValueError(f"Could not locate JSON object in text: {text[:200]}...")
                fragment = raw[start : end + 1]
                return json.loads(fragment)

            try:
                parsed = _safe_json_from_text(final_text)
            except Exception:
                return []

            evidence_raw = parsed.get("evidence", [])
            if not isinstance(evidence_raw, list):
                return []

            evidence: List[Dict[str, Any]] = []
            for item in evidence_raw:
                if not isinstance(item, dict):
                    continue
                url = str(item.get("url", "")).strip()
                if not url:
                    continue
                evidence.append(
                    {
                        "title": str(item.get("title", "")).strip(),
                        "url": url,
                        "snippet": str(item.get("snippet", "")).strip(),
                    }
                )

            return evidence[:max_results]
        finally:
            # Ensure runner closes underlying aiohttp resources when possible.
            close_coro = getattr(runner, "aclose", None)
            if callable(close_coro):
                try:
                    await close_coro()
                except Exception:
                    # Do not propagate close errors; this is best-effort cleanup.
                    pass

    # Retry loop for web evidence
    for attempt_index in range(attempts):
        try:
            return await _single_attempt()
        except errors.APIError as exc:
            code = getattr(exc, "code", None)
            if code not in status_codes or attempt_index == attempts - 1:
                raise
            delay = _compute_backoff_delay(
                attempt_index=attempt_index,
                initial_delay=initial_delay,
                exp_base=exp_base,
                max_delay=max_delay,
            )
            await asyncio.sleep(delay)

    return []



# ============================================================
# 2) Internal + web claim validation dataclasses
# ============================================================

"""
These dataclasses define the structured output of the validation process
for each claim. They encapsulate internal and web checks as separate
components, plus a final severity label used by the orchestrator.
"""


@dataclass
class InternalClaimCheck:
    """
    Result of an internal consistency check for a single claim.

    Attributes
    ----------
    verdict : str
        One of {"supported", "contradicted", "not_mentioned"}.
    confidence : float
        Confidence score in [0.0, 1.0] as reported by the model.
    explanation : str
        Short natural language justification of the verdict.
    """
    verdict: str
    confidence: float
    explanation: str


@dataclass
class WebClaimCheck:
    """
    Result of a web-based validation for a single claim.

    Attributes
    ----------
    verdict : str
        One of {"supported", "refuted", "uncertain"}.
    confidence : float
        Confidence score in [0.0, 1.0] as reported by the model.
    reasoning : str
        Short explanation of how the evidence supports the verdict.
    evidence : List[Dict[str, Any]]
        Subset of evidence snippets deemed most relevant.
    """
    verdict: str
    confidence: float
    reasoning: str
    evidence: List[Dict[str, Any]]


@dataclass
class ClaimValidationResult:
    """
    Aggregated validation result for a single claim.

    Attributes
    ----------
    claim_id : str
        Identifier of the claim (inherited from extraction stage).
    text : str
        Text of the claim.
    criticality : str
        Criticality level from the extraction phase.
    internal_check : InternalClaimCheck
        Result of internal consistency checking.
    web_check : Optional[WebClaimCheck]
        Result of web-based validation, if web evidence was available.
    severity : str
        Final severity label: "ok", "warning" or "error".
    """
    claim_id: str
    text: str
    criticality: str
    internal_check: InternalClaimCheck
    web_check: Optional[WebClaimCheck]
    severity: str


# ============================================================
# 3) JSON parsing and prompt builders
# ============================================================

def _safe_json_loads(text: str) -> Dict[str, Any]:
    """
    Safely parse JSON from model output, handling code fences and extra text.

    Parameters
    ----------
    text : str
        Raw text returned by the model.

    Returns
    -------
    Dict[str, Any]
        Parsed JSON object.

    Raises
    ------
    ValueError
        If no valid JSON object can be located.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Cannot parse JSON from empty text.")

    raw = text.strip()
    try:
        return json.loads(raw)
    except Exception:
        pass

    # Handle fenced code blocks
    if raw.startswith("```"):
        raw = raw.strip("`")
        parts = raw.split("\n", 1)
        if len(parts) == 2:
            raw = parts[1].strip()
        try:
            return json.loads(raw)
        except Exception:
            pass

    # Fallback: locate first JSON object
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Could not locate JSON object in text: {text[:200]}...")

    fragment = raw[start : end + 1]
    return json.loads(fragment)


def _build_internal_prompt(query: str, answer: str, claim_text: str) -> str:
    """
    Build the prompt for internal consistency checking of a single claim.
    """
    return f"""
You are an INTERNAL CONSISTENCY CHECKER inside a verification system.

You receive:
1) ORIGINAL QUERY
2) MODEL RESPONSE (the full answer produced by the model)
3) CLAIM (a single factual statement that was extracted from the response)

Your task is to decide whether the CLAIM is:

- "supported": the content of the CLAIM is clearly stated or strongly implied in the MODEL RESPONSE.
- "contradicted": the MODEL RESPONSE clearly states the opposite of the CLAIM.
- "not_mentioned": the MODEL RESPONSE does NOT clearly support or contradict the CLAIM.

IMPORTANT RULES:
- Use ONLY the MODEL RESPONSE. Ignore any world knowledge.
- If the CLAIM is only loosely related or depends on external knowledge, use "not_mentioned".
- Be conservative. Do not assume facts that are not clearly present in the response.

You MUST output a single JSON object with this structure:

{{
  "verdict": "supported" | "contradicted" | "not_mentioned",
  "confidence": 0.0,
  "explanation": "short explanation"
}}

ORIGINAL_QUERY:
\"\"\"{query.strip()}\"\"\"


MODEL_RESPONSE:
\"\"\"{answer.strip()}\"\"\"


CLAIM:
\"\"\"{claim_text.strip()}\"\"\""""


def _build_web_prompt(claim_text: str, evidence: List[Dict[str, Any]]) -> str:
    """
    Build the prompt for web-based fact checking of a single claim.
    """
    lines: List[str] = []
    for idx, ev in enumerate(evidence):
        title = str(ev.get("title", "")).strip()
        url = str(ev.get("url", "")).strip()
        snippet = str(ev.get("snippet", "")).strip()
        lines.append(f"[{idx}] title={title!r} url={url} snippet={snippet!r}")
    evidence_block = "\n".join(lines) if lines else "(no evidence)"
    return f"""
You are a careful FACT-CHECKING ASSISTANT.

You receive:
1) CLAIM: a single factual statement.
2) WEB_EVIDENCE: a small list of snippets retrieved from web search.

Your task is to decide whether the CLAIM is:

- "supported": clearly backed by one or more snippets.
- "refuted": clearly contradicted by one or more snippets.
- "uncertain": snippets are irrelevant, too vague, or mixed.

IMPORTANT RULES:
- Use ONLY the WEB_EVIDENCE; do not use any other knowledge.
- If no snippet clearly supports or contradicts the claim, choose "uncertain".
- When in doubt, choose "uncertain" rather than guessing.
- Do NOT invent URLs, titles, or snippets.

You MUST output a single JSON object:

{{
  "verdict": "supported" | "refuted" | "uncertain",
  "confidence": 0.0,
  "reasoning": "short explanation",
  "evidence_indices": [0, 2]
}}

CLAIM:
\"\"\"{claim_text.strip()}\"\"\"


WEB_EVIDENCE:
{evidence_block}"""


# ============================================================
# 4) Low-level LLM checks (internal and web)
# ============================================================

def _llm_internal_check(
    client: genai.Client,
    model: str,
    query: str,
    answer: str,
    claim_text: str,
    retry_config: Optional[types.HttpRetryOptions] = None,
) -> InternalClaimCheck:
    """
    Run internal consistency check with optional retry logic.

    Parameters
    ----------
    client : genai.Client
        Configured Gemini client instance.
    model : str
        Gemini model name used for the check.
    query : str
        Original user query.
    answer : str
        Full upstream model response.
    claim_text : str
        Text of the claim being checked.
    retry_config : Optional[types.HttpRetryOptions]
        Retry policy for HTTP / API errors.

    Returns
    -------
    InternalClaimCheck
        Structured result with verdict, confidence and explanation.
    """
    attempts, exp_base, initial_delay, max_delay, status_codes = _extract_retry_params(retry_config)
    prompt = _build_internal_prompt(query=query, answer=answer, claim_text=claim_text)
    contents = [types.Content(parts=[types.Part(text=prompt)])]

    last_exc: Optional[Exception] = None

    for attempt_index in range(attempts):
        try:
            response = client.models.generate_content(model=model, contents=contents)
            raw = getattr(response, "text", None)
            if not isinstance(raw, str) or not raw.strip():
                raise RuntimeError("Empty response from Gemini in internal claim check.")
            parsed = _safe_json_loads(raw)

            verdict = str(parsed.get("verdict", "not_mentioned")).strip().lower()
            if verdict not in {"supported", "contradicted", "not_mentioned"}:
                verdict = "not_mentioned"

            try:
                confidence = float(parsed.get("confidence", 0.0))
            except Exception:
                confidence = 0.0
            confidence = max(0.0, min(1.0, confidence))

            explanation = str(parsed.get("explanation", "")).strip()

            return InternalClaimCheck(
                verdict=verdict,
                confidence=confidence,
                explanation=explanation,
            )
        except errors.APIError as exc:
            last_exc = exc
            code = getattr(exc, "code", None)
            if code not in status_codes or attempt_index == attempts - 1:
                raise
            delay = _compute_backoff_delay(
                attempt_index=attempt_index,
                initial_delay=initial_delay,
                exp_base=exp_base,
                max_delay=max_delay,
            )
            time.sleep(delay)

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Internal claim check failed without a valid response.")


def _llm_web_check(
    client: genai.Client,
    model: str,
    claim_text: str,
    evidence: List[Dict[str, Any]],
    retry_config: Optional[types.HttpRetryOptions] = None,
) -> WebClaimCheck:
    """
    Run web-based check (LLM over web evidence) with optional retry logic.

    Parameters
    ----------
    client : genai.Client
        Configured Gemini client instance.
    model : str
        Gemini model name used for the check.
    claim_text : str
        Text of the claim being checked.
    evidence : List[Dict[str, Any]]
        Evidence snippets obtained from fetch_web_evidence.
    retry_config : Optional[types.HttpRetryOptions]
        Retry policy for HTTP / API errors.

    Returns
    -------
    WebClaimCheck
        Structured result summarizing web-based validation.
    """
    attempts, exp_base, initial_delay, max_delay, status_codes = _extract_retry_params(retry_config)
    prompt = _build_web_prompt(claim_text=claim_text, evidence=evidence)
    contents = [types.Content(parts=[types.Part(text=prompt)])]

    last_exc: Optional[Exception] = None

    for attempt_index in range(attempts):
        try:
            response = client.models.generate_content(model=model, contents=contents)
            raw = getattr(response, "text", None)
            if not isinstance(raw, str) or not raw.strip():
                raise RuntimeError("Empty response from Gemini in web claim check.")
            parsed = _safe_json_loads(raw)

            verdict = str(parsed.get("verdict", "uncertain")).strip().lower()
            if verdict not in {"supported", "refuted", "uncertain"}:
                verdict = "uncertain"

            try:
                confidence = float(parsed.get("confidence", 0.0))
            except Exception:
                confidence = 0.0
            confidence = max(0.0, min(1.0, confidence))

            reasoning = str(parsed.get("reasoning", "")).strip()

            indices = parsed.get("evidence_indices", [])
            selected: List[Dict[str, Any]] = []
            if isinstance(indices, list):
                for idx in indices:
                    try:
                        i = int(idx)
                    except Exception:
                        continue
                    if 0 <= i < len(evidence):
                        selected.append(evidence[i])

            return WebClaimCheck(
                verdict=verdict,
                confidence=confidence,
                reasoning=reasoning,
                evidence=selected,
            )
        except errors.APIError as exc:
            last_exc = exc
            code = getattr(exc, "code", None)
            if code not in status_codes or attempt_index == attempts - 1:
                raise
            delay = _compute_backoff_delay(
                attempt_index=attempt_index,
                initial_delay=initial_delay,
                exp_base=exp_base,
                max_delay=max_delay,
            )
            time.sleep(delay)

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Web claim check failed without a valid response.")


# ============================================================
# 5) Severity computation helper
# ============================================================

def _compute_severity(
    criticality: str,
    internal_check: InternalClaimCheck,
    web_check: Optional[WebClaimCheck],
) -> str:
    """
    Compute a final severity label for a claim based on validation signals.

    Parameters
    ----------
    criticality : str
        Criticality label from the extraction phase.
    internal_check : InternalClaimCheck
        Internal consistency result.
    web_check : Optional[WebClaimCheck]
        Web-based validation result, if available.

    Returns
    -------
    str
        One of {"ok", "warning", "error"}.
    """
    crit = (criticality or "").lower()

    if web_check is not None and web_check.verdict == "refuted":
        return "error"

    if internal_check.verdict in {"contradicted", "not_mentioned"}:
        return "warning"

    if web_check is not None and web_check.verdict == "uncertain" and crit in {"high", "critical"}:
        return "warning"

    return "ok"


# ============================================================
# 6) High-level validation tool (ADK-compatible)
# ============================================================

async def validate_claims_vaid_tool(
    client: genai.Client,
    vaid_input: Dict[str, Any],
    claims_result: Dict[str, Any],
    *,
    model_internal: str = "gemini-2.0-flash",
    model_web: str = "gemini-2.0-flash",
    max_web_snippets: int = 3,
    retry_config: Optional[types.HttpRetryOptions] = None,
) -> Dict[str, Any]:
    """
    Validate claims using:
    - Internal consistency (query + response + claim)
    - Web-based validation (claim + google_search evidence)

    This function is designed as an ADK-compatible tool entrypoint.

    Parameters
    ----------
    client : genai.Client
        Configured Gemini client instance.
    vaid_input : Dict[str, Any]
        Standard V-AID input, must contain non-empty 'query' and 'response'.
    claims_result : Dict[str, Any]
        Output of the claim extraction tool, with 'status' and 'claims'.
    model_internal : str
        Model name used for internal consistency checks.
    model_web : str
        Model name used for web-based validation and web evidence agent.
    max_web_snippets : int
        Maximum number of search snippets fetched per claim.
    retry_config : Optional[types.HttpRetryOptions]
        Retry policy shared across internal, web checks and web evidence.

    Returns
    -------
    Dict[str, Any]
        On success:
        - status : "success"
        - validated_claims : list of structured records
        - summary : aggregate counts (ok / warnings / errors)
        - metadata : processing metadata and model identifiers

        On error:
        - status : "error"
        - validated_claims : []
        - summary : zeros
        - metadata : includes error_message
    """
    try:
        query = str(vaid_input.get("query", "")).strip()
        answer = str(vaid_input.get("response", "")).strip()
        if not query or not answer:
            raise ValueError("vaid_input must contain non-empty 'query' and 'response'.")

        if claims_result.get("status") != "success":
            raise ValueError("claims_result status must be 'success'.")

        claims = claims_result.get("claims", [])
        if not isinstance(claims, list):
            raise ValueError("claims_result['claims'] must be a list.")

        validated: List[ClaimValidationResult] = []

        for c in claims:
            claim_id = str(c.get("claim_id", "")).strip() or "unknown"
            claim_text = str(c.get("text", "")).strip()
            criticality = str(c.get("criticality", "medium")).strip()

            if not claim_text:
                continue

            # 1) Internal consistency check
            internal_check = _llm_internal_check(
                client=client,
                model=model_internal,
                query=query,
                answer=answer,
                claim_text=claim_text,
                retry_config=retry_config,
            )

            # 2) Web evidence + web validation
            web_evidence = await fetch_web_evidence(
                claim=claim_text,
                max_results=max_web_snippets,
                model=model_web,
                retry_config=retry_config,
            )

            web_check: Optional[WebClaimCheck] = None
            if web_evidence:
                web_check = _llm_web_check(
                    client=client,
                    model=model_web,
                    claim_text=claim_text,
                    evidence=web_evidence,
                    retry_config=retry_config,
                )

            severity = _compute_severity(
                criticality=criticality,
                internal_check=internal_check,
                web_check=web_check,
            )

            validated.append(
                ClaimValidationResult(
                    claim_id=claim_id,
                    text=claim_text,
                    criticality=criticality,
                    internal_check=internal_check,
                    web_check=web_check,
                    severity=severity,
                )
            )

        num_ok = sum(1 for v in validated if v.severity == "ok")
        num_warn = sum(1 for v in validated if v.severity == "warning")
        num_err = sum(1 for v in validated if v.severity == "error")

        return {
            "status": "success",
            "validated_claims": [
                {
                    "claim_id": v.claim_id,
                    "text": v.text,
                    "criticality": v.criticality,
                    "internal_check": asdict(v.internal_check),
                    "web_check": asdict(v.web_check) if v.web_check is not None else None,
                    "severity": v.severity,
                }
                for v in validated
            ],
            "summary": {
                "total_claims": len(validated),
                "num_ok": num_ok,
                "num_warnings": num_warn,
                "num_errors": num_err,
            },
            "metadata": {
                "validated_at": datetime.utcnow().isoformat(),
                "model_internal": model_internal,
                "model_web": model_web,
                "upstream_model_source": vaid_input.get("model_source", ""),
            },
        }

    except Exception as exc:
        return {
            "status": "error",
            "validated_claims": [],
            "summary": {
                "total_claims": 0,
                "num_ok": 0,
                "num_warnings": 0,
                "num_errors": 0,
            },
            "metadata": {
                "validated_at": datetime.utcnow().isoformat(),
                "model_internal": model_internal,
                "model_web": model_web,
                "error_message": f"validate_claims_vaid_tool failed: {exc}",
            },
        }


# ============================================================
# Claim tools test: user-defined prompt → Gemini → V-AID tools
# ============================================================

"""
This section provides a compact, user-driven test harness for the claim tools.

End-to-end flow:
1) The user defines a test question and optional context.
2) Gemini is called once to generate an upstream response using `generate_with_gemini`.
3) The resulting V-AID input is passed into:
   - `claim_extractor_vaid_tool` (claim extraction)
   - `validate_claims_vaid_tool` (internal + web validation)
4) A pandas DataFrame is returned with one row per claim, including:
   - upstream_query
   - upstream_response
   - claim_text and criticality
   - internal verdict + confidence
   - web verdict + confidence (if available)
   - final severity

The goal is to quickly verify that:
- the upstream generation is correctly wrapped,
- the extractor operates on real model outputs,
- the validation tool produces structured, interpretable results.
"""


def build_claim_test_input_with_gemini_from_user(
    test_query: str,
    test_context: Optional[List[str]] = None,
    *,
    client: Optional[genai.Client] = None,
    model_upstream: str = "gemini-2.0-flash",
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a V-AID input by calling Gemini once with a user-defined query and context.

    Parameters
    ----------
    test_query : str
        User-defined question to send to Gemini as upstream generator.
    test_context : Optional[List[str]]
        Optional list of context strings to prepend to the prompt.
    client : Optional[genai.Client]
        Existing Gemini client. If None, a new client is obtained via get_genai_client().
    model_upstream : str
        Gemini model name used for upstream generation.
    extra_metadata : Optional[Dict[str, Any]]
        Optional metadata that will be merged into the V-AID input metadata.

    Returns
    -------
    Dict[str, Any]
        V-AID input dictionary produced by `generate_with_gemini`, with fields:
        - query
        - response
        - context
        - metadata
        - model_source

    Raises
    ------
    ValueError
        If the test_query is empty or invalid.
    RuntimeError
        If upstream generation fails.
    """
    if not isinstance(test_query, str) or not test_query.strip():
        raise ValueError("test_query must be a non-empty string.")

    # Normalize context to a list of strings
    if test_context is None:
        context_list: List[str] = []
    elif isinstance(test_context, list) and all(isinstance(c, str) for c in test_context):
        context_list = test_context
    else:
        raise ValueError("test_context must be a list of strings or None.")

    # Ensure client
    if client is None:
        client = get_genai_client()

    # Base metadata for the test
    metadata: Dict[str, Any] = {
        "test_id": "claims_tools_test",
        "test_origin": "user_defined_prompt",
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    # Call the upstream Gemini helper to build a V-AID input
    vaid_input = generate_with_gemini(
        client=client,
        query=test_query,
        context=context_list,
        metadata=metadata,
        model=model_upstream,
        model_source=None,  # let the helper derive a default model_source
    )

    return vaid_input


# ============================================================
# Claim tools test: user-defined prompt → Gemini → V-AID tools
# (clean numeric confidences for pandas)
# ============================================================

def _to_float_or_nan(value: Any) -> float:
    """
    Safely convert a value to float, returning np.nan on failure.

    This helper is used to ensure that confidence columns in the
    resulting DataFrame are always numeric, which avoids pandas
    RuntimeWarnings during formatting.
    """
    try:
        if value is None:
            return np.nan
        return float(value)
    except Exception:
        return np.nan


async def run_claim_tools_test_from_prompt(
    test_query: str,
    test_context: Optional[List[str]] = None,
    *,
    client: Optional[genai.Client] = None,
    model_upstream: str = "gemini-2.0-flash",
    model_extractor: str = "gemini-2.0-flash",
    model_internal: str = "gemini-2.0-flash",
    model_web: str = "gemini-2.0-flash",
    max_web_snippets: int = 2,
    retry_config: Optional[types.HttpRetryOptions] = None,
) -> pd.DataFrame:
    """
    Run an end-to-end test of the claim tools using a user-defined query and context.

    Workflow:
    - Generate an upstream response with Gemini (upstream model).
    - Run the claim extraction tool on that response.
    - Run the claim validation tool (internal + web) on the extracted claims.
    - Return a DataFrame summarizing per-claim validation results.

    Parameters
    ----------
    test_query : str
        User-defined question to send to Gemini as upstream generator.
    test_context : Optional[List[str]]
        Optional list of context strings to provide additional background.
    client : Optional[genai.Client]
        Existing Gemini client. If None, a new client is obtained via get_genai_client().
    model_upstream : str
        Gemini model used for upstream answer generation.
    model_extractor : str
        Gemini model used by the claim extraction tool.
    model_internal : str
        Gemini model used for internal consistency checks.
    model_web : str
        Gemini model used for web-based validation (and the web-search agent).
    max_web_snippets : int
        Maximum number of web snippets fetched per claim.
    retry_config : Optional[types.HttpRetryOptions]
        Retry configuration shared across validation calls.

    Returns
    -------
    pd.DataFrame
        Table with one row per validated claim, including:
        - upstream_query
        - upstream_response
        - claim_id
        - claim_text
        - criticality
        - internal_verdict
        - internal_confidence
        - web_verdict
        - web_confidence
        - severity

        If any tool fails, the DataFrame contains a single row with
        severity="error" and a tool_error message.
    """
    # 1) Ensure client
    if client is None:
        client = get_genai_client()

    # 2) Build a V-AID input based on user-defined query and context
    vaid_input = build_claim_test_input_with_gemini_from_user(
        test_query=test_query,
        test_context=test_context,
        client=client,
        model_upstream=model_upstream,
        extra_metadata={"stage": "claims_tools_test"},
    )

    upstream_query = vaid_input.get("query", "")
    upstream_response = vaid_input.get("response", "")

    # 3) Run claim extraction tool
    claims_result = claim_extractor_vaid_tool(
        vaid_input=vaid_input,
        model=model_extractor,
    )

    if claims_result.get("status") != "success":
        meta = claims_result.get("metadata", {}) or {}
        error_msg = str(meta.get("error_message", "")).strip()
        if not error_msg:
            error_msg = "claim_extractor_vaid_tool failed without an explicit error_message."
        return pd.DataFrame(
            [
                {
                    "upstream_query": upstream_query,
                    "upstream_response": upstream_response,
                    "claim_id": None,
                    "claim_text": None,
                    "criticality": None,
                    "internal_verdict": None,
                    "internal_confidence": np.nan,
                    "web_verdict": None,
                    "web_confidence": np.nan,
                    "severity": "error",
                    "tool_error": error_msg,
                }
            ]
        )

    # 4) Run claim validation tool (internal + web)
    validation_result = await validate_claims_vaid_tool(
        client=client,
        vaid_input=vaid_input,
        claims_result=claims_result,
        model_internal=model_internal,
        model_web=model_web,
        max_web_snippets=max_web_snippets,
        retry_config=retry_config,
    )

    if validation_result.get("status") != "success":
        meta = validation_result.get("metadata", {}) or {}
        error_msg = str(meta.get("error_message", "")).strip()
        if not error_msg:
            error_msg = "validate_claims_vaid_tool failed without an explicit error_message."
        return pd.DataFrame(
            [
                {
                    "upstream_query": upstream_query,
                    "upstream_response": upstream_response,
                    "claim_id": None,
                    "claim_text": None,
                    "criticality": None,
                    "internal_verdict": None,
                    "internal_confidence": np.nan,
                    "web_verdict": None,
                    "web_confidence": np.nan,
                    "severity": "error",
                    "tool_error": error_msg,
                }
            ]
        )

    # 5) Build final DataFrame with one row per validated claim
    validated_claims = validation_result.get("validated_claims", []) or []

    rows: List[Dict[str, Any]] = []
    for item in validated_claims:
        internal = item.get("internal_check", {}) or {}
        web = item.get("web_check", {}) or {}

        internal_conf = _to_float_or_nan(internal.get("confidence", None))
        web_conf = _to_float_or_nan(web.get("confidence", None) if web else None)

        rows.append(
            {
                "upstream_query": upstream_query,
                "upstream_response": upstream_response,
                "claim_id": item.get("claim_id", ""),
                "claim_text": item.get("text", ""),
                "criticality": item.get("criticality", ""),
                "internal_verdict": internal.get("verdict", ""),
                "internal_confidence": internal_conf,
                "web_verdict": web.get("verdict", "") if web else "",
                "web_confidence": web_conf,
                "severity": item.get("severity", ""),
            }
        )

    return pd.DataFrame(rows)



test_query = "Explain how rain forms in the atmosphere using simple language."
test_context = ["This question is about basic weather and everyday atmospheric processes."]

df_claims = await run_claim_tools_test_from_prompt(
    test_query=test_query,
    test_context=test_context,
    client=None,                 # usa get_genai_client() internamente
    model_upstream="gemini-2.0-flash",
    model_extractor="gemini-2.0-flash",
    model_internal="gemini-2.0-flash",
    model_web="gemini-2.0-flash",
    max_web_snippets=2,
    retry_config=retry_config,   # o None si prefieres
)

df_claims



@dataclass
class UrlReachability:
    """
    HTTP reachability result for a single URL.

    Attributes
    ----------
    url : str
        URL that was probed.
    reachable : bool
        True if an HTTP status in the 2xx–3xx range was obtained.
    http_status : Optional[int]
        Last HTTP status code observed, if any.
    error : Optional[str]
        Error message describing the failure if reachable is False.
    """
    url: str
    reachable: bool
    http_status: Optional[int]
    error: Optional[str]


@dataclass
class ReferenceWebCheck:
    """
    Web-based validation outcome for a bibliographic reference.

    Attributes
    ----------
    verdict : str
        One of {"valid", "inconsistent", "unknown"}.
    confidence : float
        Normalised confidence in [0.0, 1.0].
    reasoning : str
        Short explanation of how the verdict was reached.
    evidence : List[Dict[str, Any]]
        Subset of web snippets that were considered most relevant.
    """
    verdict: str
    confidence: float
    reasoning: str
    evidence: List[Dict[str, Any]]


@dataclass
class ReferenceValidationResult:
    """
    Full validation result for a single extracted reference.

    Attributes
    ----------
    ref_id : str
        Stable identifier within the current evaluation run (e.g. "ref-1").
    raw_text : str
        Original text fragment where the reference was detected.
    kind : str
        Reference kind ("url", "arxiv", "doi").
    value : str
        Normalised value (URL, arXiv ID or DOI string).
    url_check : Optional[UrlReachability]
        HTTP reachability result for URL references, otherwise None.
    web_check : Optional[ReferenceWebCheck]
        LLM-based validation result for arXiv/DOI references, otherwise None.
    severity : str
        Severity label for V-AID: "ok", "warning" or "error".
    """
    ref_id: str
    raw_text: str
    kind: str
    value: str
    url_check: Optional[UrlReachability]
    web_check: Optional[ReferenceWebCheck]
    severity: str


def _extract_references_from_text(text: str) -> List[Dict[str, Any]]:
    """
    Extract candidate references (URLs, arXiv IDs, DOIs) from free text.

    The logic is intentionally conservative and pattern-based. It focuses on
    clear structural markers that can be validated deterministically or with
    light web support, without attempting full citation parsing.

    Parameters
    ----------
    text : str
        Arbitrary model response text.

    Returns
    -------
    List[Dict[str, Any]]
        List of dictionaries with keys:
        - "kind": "url" | "arxiv" | "doi"
        - "value": normalised reference value
        - "raw_text": fragment as found in the text
    """
    if not isinstance(text, str) or not text.strip():
        return []

    refs: List[Dict[str, Any]] = []

    # --- URLs ---
    url_pattern = re.compile(r"(https?://[^\s\)\]\}\"'>]+)")
    for match in url_pattern.finditer(text):
        url = match.group(1).rstrip(".,);]")
        span = match.group(1)
        refs.append(
            {
                "kind": "url",
                "value": url,
                "raw_text": span,
            }
        )

    # --- arXiv IDs (simple pattern) ---
    arxiv_pattern = re.compile(r"(arxiv:\s*\d{4}\.\d{4,5})", re.IGNORECASE)
    for match in arxiv_pattern.finditer(text):
        raw = match.group(1)
        value = raw.split(":", 1)[1].strip()
        refs.append(
            {
                "kind": "arxiv",
                "value": value,
                "raw_text": raw,
            }
        )

    # --- DOIs ---
    doi_pattern = re.compile(r"(10\.\d{4,9}/\S+)", re.IGNORECASE)
    for match in doi_pattern.finditer(text):
        raw = match.group(1).rstrip(".,);]")
        refs.append(
            {
                "kind": "doi",
                "value": raw,
                "raw_text": raw,
            }
        )

    # Deduplicate by (kind, value)
    seen: Set[Tuple[str, str]] = set()
    unique_refs: List[Dict[str, Any]] = []
    for r in refs:
        key = (r["kind"], r["value"])
        if key in seen:
            continue
        seen.add(key)
        unique_refs.append(r)

    return unique_refs


def _check_url_reachability(
    url: str,
    timeout: float = 5.0,
    max_redirects: int = 5,
) -> UrlReachability:
    """
    Check whether a URL appears reachable using a lightweight HTTP probe.

    The function first attempts an HTTP HEAD request with redirects allowed.
    If that fails, it falls back to a GET request. The goal is to flag URLs
    that are clearly broken (4xx/5xx) while remaining tolerant to minor
    network issues.

    Parameters
    ----------
    url : str
        URL to probe.
    timeout : float
        Timeout per request in seconds.
    max_redirects : int
        Maximum number of redirects to follow. Requests handles the
        redirect chain internally; this parameter is kept for future
        tuning and documentation.

    Returns
    -------
    UrlReachability
        Structured reachability result with status and error description.
    """
    try:
        # Try HEAD first
        resp = requests.head(url, allow_redirects=True, timeout=timeout)
        status = resp.status_code
        reachable = 200 <= status < 400
        return UrlReachability(
            url=url,
            reachable=reachable,
            http_status=status,
            error=None if reachable else f"Non-success status code (HEAD): {status}",
        )
    except Exception as exc:
        # Fallback to GET to be more tolerant
        try:
            resp = requests.get(url, allow_redirects=True, timeout=timeout)
            status = resp.status_code
            reachable = 200 <= status < 400
            return UrlReachability(
                url=url,
                reachable=reachable,
                http_status=status,
                error=None if reachable else f"Non-success status code (GET): {status}",
            )
        except Exception as exc2:
            return UrlReachability(
                url=url,
                reachable=False,
                http_status=None,
                error=f"Request failed: {exc2}",
            )


def _safe_json_loads_ref(text: str) -> Dict[str, Any]:
    """
    Safely parse JSON from model output for reference validation.

    This helper is resilient to:
    - raw JSON strings,
    - fenced code blocks,
    - extra explanatory text before or after the JSON object.

    Parameters
    ----------
    text : str
        Raw LLM output.

    Returns
    -------
    Dict[str, Any]
        Parsed JSON object.

    Raises
    ------
    ValueError
        If no valid JSON object can be located and parsed.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Cannot parse JSON from empty text.")

    raw = text.strip()
    try:
        return json.loads(raw)
    except Exception:
        pass

    if raw.startswith("```"):
        raw = raw.strip("`")
        parts = raw.split("\n", 1)
        if len(parts) == 2:
            raw = parts[1].strip()
        try:
            return json.loads(raw)
        except Exception:
            pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Could not locate JSON object in text: {text[:200]}...")

    fragment = raw[start : end + 1]
    return json.loads(fragment)


def _build_reference_web_prompt(
    ref_text: str,
    evidence: List[Dict[str, Any]],
) -> str:
    """
    Build a focused prompt for checking bibliographic references using web evidence.

    The prompt is designed to:
    - keep the model in an evaluative mode instead of generative mode,
    - restrict reasoning strictly to the provided web snippets,
    - force a normalised JSON response for downstream parsing.

    Parameters
    ----------
    ref_text : str
        Reference string as written in the model response.
    evidence : List[Dict[str, Any]]
        List of web snippets with keys {title, url, snippet}.

    Returns
    -------
    str
        Prompt text to send to Gemini.
    """
    lines: List[str] = []
    for idx, ev in enumerate(evidence):
        title = str(ev.get("title", "")).strip()
        url = str(ev.get("url", "")).strip()
        snippet = str(ev.get("snippet", "")).strip()
        lines.append(f"[{idx}] title={title!r} url={url} snippet={snippet!r}")

    evidence_block = "\n".join(lines) if lines else "(no evidence)"

    return f"""
You are a FACT-CHECKING ASSISTANT specialized in bibliographic references.

You receive:
1) REF_TEXT: a citation or reference string as written by the model.
2) WEB_EVIDENCE: small list of snippets from web search (titles, urls, snippets).

Your task is to decide whether REF_TEXT looks:

- "valid": it matches or closely matches a real reference implied by the snippets.
- "inconsistent": it clearly conflicts with names, title, year, or identifier in the snippets.
- "unknown": the snippets are too vague, unrelated, or no evidence is available.

IMPORTANT RULES:
- Use ONLY the WEB_EVIDENCE. Do not rely on your own knowledge.
- If there is no clear match, use "unknown".
- If parts of the reference are correct but key fields (title, main author, identifier)
  do not match, treat it as "inconsistent".
- Do NOT invent URLs, authors, or titles.

You MUST output a single JSON object:

{{
  "verdict": "valid" | "inconsistent" | "unknown",
  "confidence": 0.0,
  "reasoning": "short explanation",
  "evidence_indices": [0, 2]
}}

REF_TEXT:
\"\"\"{ref_text.strip()}\"\"\"

WEB_EVIDENCE:
{evidence_block}
"""


def _llm_reference_check(
    client: genai.Client,
    model: str,
    ref_text: str,
    evidence: List[Dict[str, Any]],
    retry_config: Optional[types.HttpRetryOptions] = None,
) -> ReferenceWebCheck:
    """
    Use Gemini to compare REF_TEXT with WEB_EVIDENCE and classify the reference.

    The call is guarded by the same retry configuration used in other V-AID
    tools to keep behaviour consistent across the pipeline.

    Parameters
    ----------
    client : genai.Client
        Configured Gemini client instance.
    model : str
        Gemini model used for the reference check.
    ref_text : str
        Reference string to validate.
    evidence : List[Dict[str, Any]]
        List of web snippets used as external context.
    retry_config : Optional[types.HttpRetryOptions]
        Optional retry policy controlling attempts, backoff and HTTP codes.

    Returns
    -------
    ReferenceWebCheck
        Normalised validation result.

    Raises
    ------
    RuntimeError
        If no valid response can be obtained after retries.
    """
    attempts, exp_base, initial_delay, max_delay, status_codes = _extract_retry_params(retry_config)
    prompt = _build_reference_web_prompt(ref_text=ref_text, evidence=evidence)
    contents = [types.Content(parts=[types.Part(text=prompt)])]

    last_exc: Optional[Exception] = None

    for attempt_index in range(attempts):
        try:
            response = client.models.generate_content(model=model, contents=contents)
            raw = getattr(response, "text", None)
            if not isinstance(raw, str) or not raw.strip():
                raise RuntimeError("Empty response from Gemini in reference check.")
            parsed = _safe_json_loads_ref(raw)

            verdict = str(parsed.get("verdict", "unknown")).strip().lower()
            if verdict not in {"valid", "inconsistent", "unknown"}:
                verdict = "unknown"

            try:
                confidence = float(parsed.get("confidence", 0.0))
            except Exception:
                confidence = 0.0
            confidence = max(0.0, min(1.0, confidence))

            reasoning = str(parsed.get("reasoning", "")).strip()

            indices = parsed.get("evidence_indices", [])
            selected: List[Dict[str, Any]] = []
            if isinstance(indices, list):
                for idx in indices:
                    try:
                        i = int(idx)
                    except Exception:
                        continue
                    if 0 <= i < len(evidence):
                        selected.append(evidence[i])

            return ReferenceWebCheck(
                verdict=verdict,
                confidence=confidence,
                reasoning=reasoning,
                evidence=selected,
            )

        except errors.APIError as exc:
            last_exc = exc
            code = getattr(exc, "code", None)
            if code not in status_codes or attempt_index == attempts - 1:
                raise
            delay = _compute_backoff_delay(
                attempt_index=attempt_index,
                initial_delay=initial_delay,
                exp_base=exp_base,
                max_delay=max_delay,
            )
            time.sleep(delay)

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Reference web check failed without a valid response.")


def _compute_reference_severity(
    kind: str,
    url_check: Optional[UrlReachability],
    web_check: Optional[ReferenceWebCheck],
) -> str:
    """
    Compute a severity label for a reference based on URL and web checks.

    Rules
    -----
    - URL references with hard HTTP failures (4xx/5xx or unreachable) are "error".
    - References with bibliographic verdict "inconsistent" are "error".
    - References with bibliographic verdict "unknown" are "warning".
    - Everything else is treated as "ok".

    Parameters
    ----------
    kind : str
        Reference kind ("url", "arxiv", "doi").
    url_check : Optional[UrlReachability]
        HTTP reachability result, if applicable.
    web_check : Optional[ReferenceWebCheck]
        Bibliographic web-based validation result, if applicable.

    Returns
    -------
    str
        Severity label: "ok", "warning" or "error".
    """
    # URL hard failures are errors
    if kind == "url" and url_check is not None:
        if not url_check.reachable or (url_check.http_status is not None and url_check.http_status >= 400):
            return "error"

    # Bibliographic inconsistency is an error
    if web_check is not None and web_check.verdict == "inconsistent":
        return "error"

    # Unknown references are warnings
    if web_check is not None and web_check.verdict == "unknown":
        return "warning"

    # Everything else is OK
    return "ok"


# ============================================================
# TOOL: VALIDATE REFERENCES (URLs + bibliographic)
# ============================================================

async def validate_references_vaid_tool(
    client: genai.Client,
    vaid_input: Dict[str, Any],
    *,
    model_ref: str = "gemini-2.0-flash",
    max_refs: int = 10,
    max_evidence: int = 3,
    retry_config: Optional[types.HttpRetryOptions] = None,
) -> Dict[str, Any]:
    """
    Validate references found in `vaid_input['response']`.

    This tool is designed to detect structurally plausible references and
    distinguish between:
    - reachable vs broken URLs,
    - bibliographic references that are likely real vs inconsistent vs unknown.

    It does NOT attempt to enforce a specific citation style. Instead, it
    focuses on structural patterns that can be checked deterministically
    (URLs) or with light web evidence (arXiv/DOI).

    Parameters
    ----------
    client : genai.Client
        Configured Gemini client (GOOGLE_API_KEY already set).
    vaid_input : Dict[str, Any]
        Standard V-AID input dict with keys `query`, `response`, etc.
    model_ref : str
        Gemini model used to interpret web evidence for bibliographic refs.
    max_refs : int
        Maximum number of references to validate from the response.
    max_evidence : int
        Maximum number of web snippets per reference when using web evidence.
    retry_config : Optional[types.HttpRetryOptions]
        Retry configuration for Gemini and ADK-based calls.

    Returns
    -------
    Dict[str, Any]
        Structured result with keys:
        - "status": "success" | "error"
        - "references": list of per-reference dictionaries
        - "summary": aggregate counts
        - "metadata": traceability information (timestamps, model_ref, upstream source)
    """
    try:
        response_text = str(vaid_input.get("response", "")).strip()
        if not response_text:
            raise ValueError("vaid_input['response'] must be a non-empty string.")

        # 1) Extract raw references
        refs_raw = _extract_references_from_text(response_text)
        if not refs_raw:
            return {
                "status": "success",
                "references": [],
                "summary": {
                    "total_references": 0,
                    "num_ok": 0,
                    "num_warnings": 0,
                    "num_errors": 0,
                },
                "metadata": {
                    "validated_at": datetime.utcnow().isoformat(),
                    "model_ref": model_ref,
                    "upstream_model_source": vaid_input.get("model_source", ""),
                    "note": "No references detected in response.",
                },
            }

        refs_raw = refs_raw[:max_refs]
        validated: List[ReferenceValidationResult] = []

        for idx, r in enumerate(refs_raw, start=1):
            kind = r.get("kind", "")
            value = r.get("value", "")
            raw_text = r.get("raw_text", value)

            url_check: Optional[UrlReachability] = None
            web_check: Optional[ReferenceWebCheck] = None

            # 2) URL reachability for URL references
            if kind == "url":
                url_check = _check_url_reachability(value)

            # 3) Web-based validation for bibliographic references (arxiv, doi)
            if kind in {"arxiv", "doi"}:
                evidence = await fetch_web_evidence(
                    claim=value,
                    max_results=max_evidence,
                    model=model_ref,
                    retry_config=retry_config,
                )
                if evidence:
                    web_check = _llm_reference_check(
                        client=client,
                        model=model_ref,
                        ref_text=raw_text,
                        evidence=evidence,
                        retry_config=retry_config,
                    )

            severity = _compute_reference_severity(
                kind=kind,
                url_check=url_check,
                web_check=web_check,
            )

            validated.append(
                ReferenceValidationResult(
                    ref_id=f"ref-{idx}",
                    raw_text=raw_text,
                    kind=kind,
                    value=value,
                    url_check=url_check,
                    web_check=web_check,
                    severity=severity,
                )
            )

        num_ok = sum(1 for v in validated if v.severity == "ok")
        num_warn = sum(1 for v in validated if v.severity == "warning")
        num_err = sum(1 for v in validated if v.severity == "error")

        return {
            "status": "success",
            "references": [
                {
                    "ref_id": v.ref_id,
                    "raw_text": v.raw_text,
                    "kind": v.kind,
                    "value": v.value,
                    "url_check": asdict(v.url_check) if v.url_check is not None else None,
                    "web_check": asdict(v.web_check) if v.web_check is not None else None,
                    "severity": v.severity,
                }
                for v in validated
            ],
            "summary": {
                "total_references": len(validated),
                "num_ok": num_ok,
                "num_warnings": num_warn,
                "num_errors": num_err,
            },
            "metadata": {
                "validated_at": datetime.utcnow().isoformat(),
                "model_ref": model_ref,
                "upstream_model_source": vaid_input.get("model_source", ""),
            },
        }

    except Exception as exc:
        return {
            "status": "error",
            "references": [],
            "summary": {
                "total_references": 0,
                "num_ok": 0,
                "num_warnings": 0,
                "num_errors": 0,
            },
            "metadata": {
                "validated_at": datetime.utcnow().isoformat(),
                "model_ref": model_ref,
                "upstream_model_source": vaid_input.get("model_source", ""),
                "error_message": f"validate_references_vaid_tool failed: {exc}",
            },
        }



# ============================================================
# Helpers for reference tool testing
# ============================================================

def _to_float_or_nan(value: Any) -> float:
    """
    Convert an arbitrary value to float, returning NaN on failure.

    This is used only for reporting confidence values in a numeric
    column without raising errors if the field is missing or invalid.
    """
    try:
        if value is None:
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def _build_reference_test_input_with_gemini_from_user(
    test_query: str,
    test_context: Optional[List[str]] = None,
    *,
    client: Optional[genai.Client] = None,
    model_upstream: str = "gemini-2.0-flash",
) -> Dict[str, Any]:
    """
    Generate a V-AID input containing references using Gemini as upstream model.
    """
    if client is None:
        client = get_genai_client()

    vaid_input = generate_with_gemini(
        client=client,
        query=test_query,
        context=test_context,
        metadata={"stage": "references_tools_test"},
        model=model_upstream,
        model_source=None,
    )
    return vaid_input


async def run_reference_tools_test_from_prompt(
    test_query: str,
    test_context: Optional[List[str]] = None,
    *,
    client: Optional[genai.Client] = None,
    model_upstream: str = "gemini-2.0-flash",
    model_ref: str = "gemini-2.0-flash",
    max_refs: int = 10,
    max_evidence: int = 3,
    retry_config: Optional[types.HttpRetryOptions] = None,
) -> pd.DataFrame:
    """
    End-to-end test for the reference validation tool using a user-defined prompt.
    """
    if client is None:
        client = get_genai_client()

    # 1) Generate upstream answer with references
    vaid_input = _build_reference_test_input_with_gemini_from_user(
        test_query=test_query,
        test_context=test_context,
        client=client,
        model_upstream=model_upstream,
    )

    upstream_query = vaid_input.get("query", "")
    upstream_response = vaid_input.get("response", "")

    # 2) Run reference validation tool
    ref_result = await validate_references_vaid_tool(
        client=client,
        vaid_input=vaid_input,
        model_ref=model_ref,
        max_refs=max_refs,
        max_evidence=max_evidence,
        retry_config=retry_config,
    )

    if ref_result.get("status") != "success":
        meta = ref_result.get("metadata", {}) or {}
        error_msg = str(meta.get("error_message", "")).strip() or \
            "validate_references_vaid_tool failed without an explicit error_message."
        return pd.DataFrame(
            [
                {
                    "upstream_query": upstream_query,
                    "upstream_response": upstream_response,
                    "ref_id": None,
                    "raw_text": None,
                    "kind": None,
                    "value": None,
                    "url_reachable": None,
                    "url_http_status": None,
                    "web_verdict": None,
                    "web_confidence": float("nan"),
                    "severity": "error",
                    "tool_error": error_msg,
                }
            ]
        )

    references = ref_result.get("references", []) or []

    rows: List[Dict[str, Any]] = []
    for r in references:
        url_check = r.get("url_check") or {}
        web_check = r.get("web_check") or {}

        rows.append(
            {
                "upstream_query": upstream_query,
                "upstream_response": upstream_response,
                "ref_id": r.get("ref_id", ""),
                "raw_text": r.get("raw_text", ""),
                "kind": r.get("kind", ""),
                "value": r.get("value", ""),
                "url_reachable": url_check.get("reachable", None),
                "url_http_status": url_check.get("http_status", None),
                "web_verdict": web_check.get("verdict", "") if web_check else "",
                "web_confidence": _to_float_or_nan(
                    web_check.get("confidence") if web_check else None
                ),
                "severity": r.get("severity", ""),
            }
        )

    return pd.DataFrame(rows)



test_query = (
    "Give a short paragraph about recent advances in machine learning. "
    "Include at least one real DOI, one arXiv ID and one URL in your answer."
)

test_context = [
    "Example formats:",
    "DOI: 10.1038/nature14539",
    "arXiv: 1706.03762",
    "URL: https://doi.org/10.1038/nature14539",
]

df_refs = await run_reference_tools_test_from_prompt(
    test_query=test_query,
    test_context=test_context,
    model_upstream="gemini-2.0-flash",
    model_ref="gemini-2.0-flash",
    max_refs=8,
    max_evidence=2,
    retry_config=retry_config,
)

df_refs



# ============================================================
# 1) Risk policy aggregation (ALLOW / WARN / ESCALATE)
# ============================================================


@dataclass
class VaidRiskSummary:
    """
    Aggregated risk view for the full V-AID evaluation cycle.

    This object captures the global decision derived from:
      - structural + web claim validation
      - structural + web reference validation

    The summary exposes:
      verdict : global decision ("ALLOW", "WARN", "ESCALATE")
      reason  : human-readable explanation
      num_*   : severity counts for both claims and references
    """
    verdict: str
    reason: str
    num_claims: int
    num_claim_errors: int
    num_claim_warnings: int
    num_ref_errors: int
    num_ref_warnings: int


def _compute_risk_summary(
    claims_validation: Optional[Dict[str, Any]],
    refs_validation: Optional[Dict[str, Any]],
) -> VaidRiskSummary:
    """
    Compute a global V-AID risk verdict based solely on
    the outputs of the claim tool and reference tool.

    No model calls are made here. This is a deterministic rule-based
    aggregator that receives the validated structures and produces
    a final risk level.

    Policy:
      - ESCALATE  : any "error" in claims or references
      - WARN      : no errors but at least one warning
      - ALLOW     : clean results (no warnings, no errors)
    """

    # Extract structural status signals
    claims_status = (claims_validation or {}).get("status", "error")
    refs_status = (refs_validation or {}).get("status", "error")

    # Claims
    claims_list: List[Dict[str, Any]] = []
    if claims_status == "success":
        # claim extractor returns { "claims": [...] }
        # claim validator returns { "validated_claims": [...] }
        claims_list = (
            claims_validation.get("validated_claims")
            or claims_validation.get("claims")
            or []
        )

    # References
    refs_list: List[Dict[str, Any]] = []
    if refs_status == "success":
        refs_list = refs_validation.get("references", []) or []

    # Count severities for claims
    claim_errors = sum(1 for c in claims_list if c.get("severity") == "error")
    claim_warnings = sum(1 for c in claims_list if c.get("severity") == "warning")

    # Count severities for references
    ref_errors = sum(1 for r in refs_list if r.get("severity") == "error")
    ref_warnings = sum(1 for r in refs_list if r.get("severity") == "warning")

    total_claims = len(claims_list)

    # ============================================================
    # Risk policy decisions
    # ============================================================

    if claim_errors > 0 or ref_errors > 0:
        verdict = "ESCALATE"
        reason = (
            "One or more critical issues were detected in claims or references. "
            "A human review is required before using this answer."
        )

    elif claim_warnings > 0 or ref_warnings > 0:
        verdict = "WARN"
        reason = (
            "Some elements show uncertainty or weak support. "
            "Use with caution and consider a quick human review."
        )

    else:
        verdict = "ALLOW"
        reason = (
            "All checks completed successfully with no warnings or errors. "
            "The answer can be used safely in low-risk contexts."
        )

    return VaidRiskSummary(
        verdict=verdict,
        reason=reason,
        num_claims=total_claims,
        num_claim_errors=claim_errors,
        num_claim_warnings=claim_warnings,
        num_ref_errors=ref_errors,
        num_ref_warnings=ref_warnings,
    )



# ============================================================
# Simple smoke test for the risk aggregation layer
# Does not call any model or external service
# ============================================================

def smoke_test_risk_aggregation():
    test_cases = {}

    # 1) Clean case → ALLOW
    claims_ok = {
        "status": "success",
        "validated_claims": [
            {"severity": "ok"},
            {"severity": "ok"},
        ],
    }
    refs_ok = {
        "status": "success",
        "references": [
            {"severity": "ok"},
        ],
    }
    test_cases["ALLOW_case"] = _compute_risk_summary(claims_ok, refs_ok)

    # 2) Warning case → WARN
    claims_warn = {
        "status": "success",
        "validated_claims": [
            {"severity": "ok"},
            {"severity": "warning"},
        ],
    }
    refs_warn = {
        "status": "success",
        "references": [
            {"severity": "ok"},
        ],
    }
    test_cases["WARN_case"] = _compute_risk_summary(claims_warn, refs_warn)

    # 3) Error in claims → ESCALATE
    claims_err = {
        "status": "success",
        "validated_claims": [
            {"severity": "error"},
            {"severity": "ok"},
        ],
    }
    refs_clean = {
        "status": "success",
        "references": [],
    }
    test_cases["ESCALATE_claim_error"] = _compute_risk_summary(claims_err, refs_clean)

    # 4) Error in references → ESCALATE
    claims_clean = {
        "status": "success",
        "validated_claims": [
            {"severity": "ok"},
            {"severity": "ok"},
        ],
    }
    refs_err = {
        "status": "success",
        "references": [
            {"severity": "error"},
        ],
    }
    test_cases["ESCALATE_ref_error"] = _compute_risk_summary(claims_clean, refs_err)

    # Print results
    for name, result in test_cases.items():
        print(f"\n=== {name} ===")
        print(result)

    return test_cases


# Run the smoke test (safe to call)
smoke_test_results = smoke_test_risk_aggregation()






async def run_vaid_evaluation(
    client: genai.Client,
    vaid_input: Dict[str, Any],
    *,
    model_internal: str = "gemini-2.0-flash",
    model_web: str = "gemini-2.0-flash",
    model_ref: str = "gemini-2.0-flash",
    max_web_snippets: int = 3,
    max_refs: int = 10,
    retry_config: Optional[types.HttpRetryOptions] = None,
) -> Dict[str, Any]:
    """
    Run the full V-AID evaluation pipeline on a single V-AID input.

    Steps
    -----
    1) Extract claims from the upstream answer.
    2) Validate claims for internal consistency and web support.
    3) Validate references (URLs, arXiv IDs, DOIs).
    4) Aggregate a global risk verdict (ALLOW / WARN / ESCALATE).

    Parameters
    ----------
    client : genai.Client
        Configured Gemini client instance.
    vaid_input : Dict[str, Any]
        Upstream model output in standardized V-AID format:
        {
          "query": str,
          "response": str,
          "context": List[str],
          "metadata": Dict[str, Any],
          "model_source": str,
        }
    model_internal : str
        Gemini model used for internal consistency checks of claims.
    model_web : str
        Gemini model used for web-based claim validation.
    model_ref : str
        Gemini model used for reference validation.
    max_web_snippets : int
        Maximum number of web snippets per claim.
    max_refs : int
        Maximum number of references to validate.
    retry_config : Optional[types.HttpRetryOptions]
        Optional retry configuration shared across web-based calls.

    Returns
    -------
    Dict[str, Any]
        Structured V-AID evaluation report:
        {
          "status": "success" | "error",
          "verdict": "ALLOW" | "WARN" | "ESCALATE",
          "reason": str,
          "inputs": {...},                 # original vaid_input (lightly sanitized)
          "claims_extraction": {...},      # output of claim_extractor_vaid_tool
          "claims_validation": {...},      # output of validate_claims_vaid_tool
          "references_validation": {...},  # output of validate_references_vaid_tool
          "summary": {...},                # aggregated counters
          "metadata": {...},               # timestamps, models, error info if any
        }
    """
    started_at = datetime.utcnow().isoformat()

    try:
        # ------------------------------------------------
        # Basic structural validation of the incoming V-AID input
        # ------------------------------------------------
        query = str(vaid_input.get("query", "")).strip()
        answer = str(vaid_input.get("response", "")).strip()
        if not query or not answer:
            raise ValueError(
                "vaid_input must contain non-empty 'query' and 'response' fields."
            )

        # ------------------------------------------------
        # 1) Claim extraction (synchronous tool)
        # ------------------------------------------------
        claims_extraction = claim_extractor_vaid_tool(
            vaid_input=vaid_input,
            model=model_internal,
        )

        # ------------------------------------------------
        # 2) Claim validation (internal consistency + web evidence)
        # ------------------------------------------------
        claims_validation = await validate_claims_vaid_tool(
            client=client,
            vaid_input=vaid_input,
            claims_result=claims_extraction,
            model_internal=model_internal,
            model_web=model_web,
            max_web_snippets=max_web_snippets,
            retry_config=retry_config,
        )

        # ------------------------------------------------
        # 3) Reference validation (URLs, DOIs, arXiv IDs)
        # ------------------------------------------------
        references_validation = await validate_references_vaid_tool(
            client=client,
            vaid_input=vaid_input,
            model_ref=model_ref,
            max_refs=max_refs,
            max_evidence=3,
            retry_config=retry_config,
        )

        # ------------------------------------------------
        # 4) Global risk aggregation
        # ------------------------------------------------
        risk_summary = _compute_risk_summary(
            claims_validation=claims_validation,
            refs_validation=references_validation,
        )

        # Summaries from tools (if available)
        claims_summary = (
            claims_validation.get("summary", {})
            if claims_validation.get("status") == "success"
            else {}
        )
        refs_summary = (
            references_validation.get("summary", {})
            if references_validation.get("status") == "success"
            else {}
        )

        finished_at = datetime.utcnow().isoformat()

        return {
            "status": "success",
            "verdict": risk_summary.verdict,
            "reason": risk_summary.reason,
            "inputs": {
                "query": query,
                "response": answer,
                "context": vaid_input.get("context", []),
                "model_source": vaid_input.get("model_source", ""),
                "metadata": vaid_input.get("metadata", {}),
            },
            "claims_extraction": claims_extraction,
            "claims_validation": claims_validation,
            "references_validation": references_validation,
            "summary": {
                "num_claims": risk_summary.num_claims,
                "num_claim_errors": risk_summary.num_claim_errors,
                "num_claim_warnings": risk_summary.num_claim_warnings,
                "num_ref_errors": risk_summary.num_ref_errors,
                "num_ref_warnings": risk_summary.num_ref_warnings,
                "claims_tool_summary": claims_summary,
                "refs_tool_summary": refs_summary,
            },
            "metadata": {
                "started_at": started_at,
                "finished_at": finished_at,
                "model_internal": model_internal,
                "model_web": model_web,
                "model_ref": model_ref,
                "upstream_model_source": vaid_input.get("model_source", ""),
            },
        }

    except Exception as exc:
        finished_at = datetime.utcnow().isoformat()
        return {
            "status": "error",
            "verdict": "ESCALATE",
            "reason": f"V-AID pipeline failed: {exc}",
            "inputs": {
                "query": vaid_input.get("query", ""),
                "response": vaid_input.get("response", ""),
                "context": vaid_input.get("context", []),
                "model_source": vaid_input.get("model_source", ""),
                "metadata": vaid_input.get("metadata", {}),
            },
            "claims_extraction": {},
            "claims_validation": {},
            "references_validation": {},
            "summary": {
                "num_claims": 0,
                "num_claim_errors": 0,
                "num_claim_warnings": 0,
                "num_ref_errors": 0,
                "num_ref_warnings": 0,
            },
            "metadata": {
                "started_at": started_at,
                "finished_at": finished_at,
                "error_message": f"run_vaid_evaluation failed: {exc}",
                "model_internal": model_internal,
                "model_web": model_web,
                "model_ref": model_ref,
                "upstream_model_source": vaid_input.get("model_source", ""),
            },
        }



# ============================================================
# End-to-end smoke test using a manual LLM-style answer
# ============================================================

def _build_manual_vaid_input_for_smoke_test() -> Dict[str, Any]:
    """
    Build a compact V-AID input using a manually crafted LLM-style answer.

    The answer is designed to:
      - contain several factual claims about a verification assistant,
      - mention V-AID-like behavior (claim checking and references),
      - include multiple reference patterns: URL, DOI and arXiv ID.

    This avoids spending tokens on upstream generation while still
    providing enough structure to exercise all downstream tools.
    """
    test_query = (
        "Explain what a verification assistant for model-generated answers does, "
        "and mention a few external references in your explanation."
    )

    manual_response = (
        "A verification assistant for model-generated answers is a system that analyzes an answer, "
        "splits it into atomic factual claims and checks each claim against internal context and web evidence. "
        "When it detects that a critical claim looks refuted or unsupported, the assistant recommends human review "
        "instead of returning a confident answer.\n\n"
        "For background on modern language models, see Vaswani et al., 'Attention Is All You Need' "
        "(arXiv:1706.03762). A broader discussion of business adoption is often attributed to reports such as "
        "'The GenAI Divide: State of AI in Business 2025' (DOI 10.5555/genai.divide.2025.001, "
        "URL https://example.com/genai-divide-2025), although many summaries are informal. "
        "For a generic example of a DOI-based reference, you can also refer to https://doi.org/10.1038/nature14539."
    )

    manual_context = [
        "The assistant is intended to run after an upstream model has already produced an answer.",
        "It focuses on factual claims and references, not on style or tone.",
    ]

    vaid_input = wrap_manual_response(
        query=test_query,
        response=manual_response,
        context=manual_context,
        metadata={"stage": "manual_smoke_test"},
        model_source="manual-llm-smoke-test",
    )
    return vaid_input


def _to_float_or_nan(value: Any) -> float:
    """
    Safely convert arbitrary values to float for reporting.

    Returns NaN when the input cannot be interpreted as a numeric value.
    This is used only for confidence columns in DataFrames.
    """
    try:
        if value is None:
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


async def run_vaid_smoke_test_manual(
    *,
    client: Optional[genai.Client] = None,
    model_internal: str = "gemini-2.0-flash",
    model_web: str = "gemini-2.0-flash",
    model_ref: str = "gemini-2.0-flash",
    max_web_snippets: int = 1,
    max_refs: int = 5,
    retry_config: Optional[types.HttpRetryOptions] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Run a full V-AID evaluation using a manual upstream answer
    and return compact DataFrames for inspection.

    The function:
      1) Builds a manual V-AID input.
      2) Runs the full orchestrator (claims, references, risk aggregation).
      3) Converts the results into three DataFrames:
         - overview: one row with verdict and aggregated counters
         - claims: one row per validated claim
         - references: one row per validated reference
    """
    if client is None:
        client = get_genai_client()

    # 1) Build manual V-AID input
    vaid_input = _build_manual_vaid_input_for_smoke_test()

    # 2) Run full V-AID pipeline
    eval_result = await run_vaid_evaluation(
        client=client,
        vaid_input=vaid_input,
        model_internal=model_internal,
        model_web=model_web,
        model_ref=model_ref,
        max_web_snippets=max_web_snippets,
        max_refs=max_refs,
        retry_config=retry_config,
    )

    # 3) Build overview DataFrame
    summary = eval_result.get("summary", {}) or {}
    overview_row = {
        "verdict": eval_result.get("verdict", ""),
        "reason": eval_result.get("reason", ""),
        "num_claims": summary.get("num_claims", 0),
        "num_claim_errors": summary.get("num_claim_errors", 0),
        "num_claim_warnings": summary.get("num_claim_warnings", 0),
        "num_ref_errors": summary.get("num_ref_errors", 0),
        "num_ref_warnings": summary.get("num_ref_warnings", 0),
    }
    df_overview = pd.DataFrame([overview_row])

    # 4) Build claims DataFrame
    claims_validation = eval_result.get("claims_validation", {}) or {}
    claims_list = claims_validation.get("validated_claims", []) or []

    claim_rows: List[Dict[str, Any]] = []
    for c in claims_list:
        internal = c.get("internal_check") or {}
        web = c.get("web_check") or {}
        claim_rows.append(
            {
                "claim_id": c.get("claim_id", ""),
                "text": c.get("text", ""),
                "criticality": c.get("criticality", ""),
                "severity": c.get("severity", ""),
                "internal_verdict": internal.get("verdict", ""),
                "internal_confidence": _to_float_or_nan(internal.get("confidence")),
                "web_verdict": web.get("verdict", "") if web else "",
                "web_confidence": _to_float_or_nan(
                    web.get("confidence") if web else None
                ),
            }
        )

    df_claims = pd.DataFrame(claim_rows) if claim_rows else pd.DataFrame(
        columns=[
            "claim_id",
            "text",
            "criticality",
            "severity",
            "internal_verdict",
            "internal_confidence",
            "web_verdict",
            "web_confidence",
        ]
    )

    # 5) Build references DataFrame
    refs_validation = eval_result.get("references_validation", {}) or {}
    refs_list = refs_validation.get("references", []) or []

    ref_rows: List[Dict[str, Any]] = []
    for r in refs_list:
        url_check = r.get("url_check") or {}
        web_check = r.get("web_check") or {}

        ref_rows.append(
            {
                "ref_id": r.get("ref_id", ""),
                "raw_text": r.get("raw_text", ""),
                "kind": r.get("kind", ""),
                "value": r.get("value", ""),
                "url_reachable": url_check.get("reachable", None),
                "url_http_status": url_check.get("http_status", None),
                "web_verdict": web_check.get("verdict", "") if web_check else "",
                "web_confidence": _to_float_or_nan(
                    web_check.get("confidence") if web_check else None
                ),
                "severity": r.get("severity", ""),
            }
        )

    df_refs = pd.DataFrame(ref_rows) if ref_rows else pd.DataFrame(
        columns=[
            "ref_id",
            "raw_text",
            "kind",
            "value",
            "url_reachable",
            "url_http_status",
            "web_verdict",
            "web_confidence",
            "severity",
        ]
    )

    return {
        "overview": df_overview,
        "claims": df_claims,
        "references": df_refs,
    }



results_manual = await run_vaid_smoke_test_manual(
    model_internal="gemini-2.0-flash",
    model_web="gemini-2.0-flash",
    model_ref="gemini-2.0-flash",
    max_web_snippets=1,
    max_refs=5,
    retry_config=retry_config,
)

results_manual["overview"], results_manual["claims"], results_manual["references"]



# ============================================================
# Shared helper: build DataFrames from a full V-AID evaluation
# ============================================================

def build_vaid_result_frames(
    eval_result: Dict[str, Any]
) -> Dict[str, pd.DataFrame]:
    """
    Convert a full V-AID evaluation result into three pandas DataFrames.

    This helper is shared by all smoke tests (manual, Gemini, local models)
    so that the output structure is consistent and easy to compare.

    The returned dict contains:
      - "overview"   : one row with global verdict and aggregated counters
      - "claims"     : one row per validated claim
      - "references" : one row per validated reference
    """
    summary = eval_result.get("summary", {}) or {}

    # 1) Overview
    overview_row = {
        "verdict": eval_result.get("verdict", ""),
        "reason": eval_result.get("reason", ""),
        "num_claims": summary.get("num_claims", 0),
        "num_claim_errors": summary.get("num_claim_errors", 0),
        "num_claim_warnings": summary.get("num_claim_warnings", 0),
        "num_ref_errors": summary.get("num_ref_errors", 0),
        "num_ref_warnings": summary.get("num_ref_warnings", 0),
    }
    df_overview = pd.DataFrame([overview_row])

    # 2) Claims
    claims_validation = eval_result.get("claims_validation", {}) or {}
    claims_list = claims_validation.get("validated_claims", []) or []

    claim_rows: List[Dict[str, Any]] = []
    for c in claims_list:
        internal = c.get("internal_check") or {}
        web = c.get("web_check") or {}
        claim_rows.append(
            {
                "claim_id": c.get("claim_id", ""),
                "text": c.get("text", ""),
                "criticality": c.get("criticality", ""),
                "severity": c.get("severity", ""),
                "internal_verdict": internal.get("verdict", ""),
                "internal_confidence": _to_float_or_nan(internal.get("confidence")),
                "web_verdict": web.get("verdict", "") if web else "",
                "web_confidence": _to_float_or_nan(
                    web.get("confidence") if web else None
                ),
            }
        )

    df_claims = (
        pd.DataFrame(claim_rows)
        if claim_rows
        else pd.DataFrame(
            columns=[
                "claim_id",
                "text",
                "criticality",
                "severity",
                "internal_verdict",
                "internal_confidence",
                "web_verdict",
                "web_confidence",
            ]
        )
    )

    # 3) References
    refs_validation = eval_result.get("references_validation", {}) or {}
    refs_list = refs_validation.get("references", []) or []

    ref_rows: List[Dict[str, Any]] = []
    for r in refs_list:
        url_check = r.get("url_check") or {}
        web_check = r.get("web_check") or {}

        ref_rows.append(
            {
                "ref_id": r.get("ref_id", ""),
                "raw_text": r.get("raw_text", ""),
                "kind": r.get("kind", ""),
                "value": r.get("value", ""),
                "url_reachable": url_check.get("reachable", None),
                "url_http_status": url_check.get("http_status", None),
                "web_verdict": web_check.get("verdict", "") if web_check else "",
                "web_confidence": _to_float_or_nan(
                    web_check.get("confidence") if web_check else None
                ),
                "severity": r.get("severity", ""),
            }
        )

    df_refs = (
        pd.DataFrame(ref_rows)
        if ref_rows
        else pd.DataFrame(
            columns=[
                "ref_id",
                "raw_text",
                "kind",
                "value",
                "url_reachable",
                "url_http_status",
                "web_verdict",
                "web_confidence",
                "severity",
            ]
        )
    )

    return {
        "overview": df_overview,
        "claims": df_claims,
        "references": df_refs,
    }


# ============================================================
# Common prompt for model-based smoke tests
# ============================================================

SMOKE_TEST_QUERY = (
    "In one short paragraph, explain what a verification assistant "
    "for AI-generated answers does, and mention at least one external reference."
)

SMOKE_TEST_CONTEXT: List[str] = [
    "The assistant runs after an upstream model has produced an answer.",
    "It focuses on factual claims and references, not on style or tone.",
]


# ============================================================
# 1) Smoke test using Gemini as upstream model
# ============================================================

async def run_vaid_smoke_test_gemini(
    *,
    client: Optional[genai.Client] = None,
    model_upstream: str = "gemini-2.0-flash",
    model_internal: str = "gemini-2.0-flash",
    model_web: str = "gemini-2.0-flash",
    model_ref: str = "gemini-2.0-flash",
    max_web_snippets: int = 2,
    max_refs: int = 5,
    retry_config: Optional[types.HttpRetryOptions] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Run a full V-AID evaluation using Gemini as the upstream model.

    Steps:
      1) Generate an answer with Gemini using a short, general-knowledge prompt.
      2) Wrap the result into a standardized V-AID input.
      3) Execute the full V-AID pipeline via run_vaid_evaluation().
      4) Convert the outcome into three DataFrames for inspection.
    """
    if client is None:
        client = get_genai_client()

    # 1) Upstream generation with Gemini
    vaid_input = generate_with_gemini(
        client=client,
        query=SMOKE_TEST_QUERY,
        context=SMOKE_TEST_CONTEXT,
        metadata={"stage": "smoke_test_gemini"},
        model=model_upstream,
        model_source=f"gemini:{model_upstream}",
    )

    # 2) Full V-AID evaluation
    eval_result = await run_vaid_evaluation(
        client=client,
        vaid_input=vaid_input,
        model_internal=model_internal,
        model_web=model_web,
        model_ref=model_ref,
        max_web_snippets=max_web_snippets,
        max_refs=max_refs,
        retry_config=retry_config,
    )

    # 3) Frame conversion
    return build_vaid_result_frames(eval_result)


# ============================================================
# 2) Smoke test using Local Model A as upstream model
# ============================================================

async def run_vaid_smoke_test_local_a(
    *,
    client: Optional[genai.Client] = None,
    model_internal: str = "gemini-2.0-flash",
    model_web: str = "gemini-2.0-flash",
    model_ref: str = "gemini-2.0-flash",
    max_web_snippets: int = 2,
    max_refs: int = 5,
    retry_config: Optional[types.HttpRetryOptions] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Run a full V-AID evaluation using Local Model A as upstream model.

    Local Model A is configured as a small public model (e.g. GPT-2) that
    produces short, noisy answers. The goal is to stress-test V-AID under
    limited reasoning and no explicit uncertainty handling.
    """
    if client is None:
        client = get_genai_client()

    # 1) Upstream generation with Local Model A
    vaid_input = generate_with_local_model_a(
        query=SMOKE_TEST_QUERY,
        context=SMOKE_TEST_CONTEXT,
        metadata={"stage": "smoke_test_local_a"},
    )

    # 2) Full V-AID evaluation
    eval_result = await run_vaid_evaluation(
        client=client,
        vaid_input=vaid_input,
        model_internal=model_internal,
        model_web=model_web,
        model_ref=model_ref,
        max_web_snippets=max_web_snippets,
        max_refs=max_refs,
        retry_config=retry_config,
    )

    # 3) Frame conversion
    return build_vaid_result_frames(eval_result)


# ============================================================
# 3) Smoke test using Local Model B as upstream model
# ============================================================

async def run_vaid_smoke_test_local_b(
    *,
    client: Optional[genai.Client] = None,
    model_internal: str = "gemini-2.0-flash",
    model_web: str = "gemini-2.0-flash",
    model_ref: str = "gemini-2.0-flash",
    max_web_snippets: int = 2,
    max_refs: int = 5,
    retry_config: Optional[types.HttpRetryOptions] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Run a full V-AID evaluation using Local Model B as upstream model.

    Local Model B is configured as a larger or more stochastic public model
    (e.g. gpt2-medium) with higher temperature. It tends to produce more
    creative and potentially off-topic answers, which are useful to probe
    the robustness of the verification pipeline.
    """
    if client is None:
        client = get_genai_client()

    # 1) Upstream generation with Local Model B
    vaid_input = generate_with_local_model_b(
        query=SMOKE_TEST_QUERY,
        context=SMOKE_TEST_CONTEXT,
        metadata={"stage": "smoke_test_local_b"},
    )

    # 2) Full V-AID evaluation
    eval_result = await run_vaid_evaluation(
        client=client,
        vaid_input=vaid_input,
        model_internal=model_internal,
        model_web=model_web,
        model_ref=model_ref,
        max_web_snippets=max_web_snippets,
        max_refs=max_refs,
        retry_config=retry_config,
    )

    # 3) Frame conversion
    return build_vaid_result_frames(eval_result)



# Gemini upstream
results_gemini = await run_vaid_smoke_test_gemini(
    model_upstream="gemini-2.0-flash",
    model_internal="gemini-2.0-flash",
    model_web="gemini-2.0-flash",
    model_ref="gemini-2.0-flash",
    retry_config=retry_config,
)

# Local model A upstream
results_local_a = await run_vaid_smoke_test_local_a(
    model_internal="gemini-2.0-flash",
    model_web="gemini-2.0-flash",
    model_ref="gemini-2.0-flash",
    retry_config=retry_config,
)

# Local model B upstream
results_local_b = await run_vaid_smoke_test_local_b(
    model_internal="gemini-2.0-flash",
    model_web="gemini-2.0-flash",
    model_ref="gemini-2.0-flash",
    retry_config=retry_config,
)

results_gemini["overview"], results_local_a["overview"], results_local_b["overview"]


