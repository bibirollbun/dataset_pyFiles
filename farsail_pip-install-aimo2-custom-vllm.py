!pip install --target=/kaggle/working vllm==0.7.1


%%writefile /kaggle/working/vllm/entrypoints/llm.py

import re
import os
import subprocess
import tempfile
from collections import Counter as _Counter
import time

import itertools
import warnings
from contextlib import contextmanager
from typing import (Any, Callable, ClassVar, Dict, List, Optional, Sequence,
                    Tuple, Type, Union, cast, overload)

import cloudpickle
import torch
import torch.nn as nn
from tqdm import tqdm
from typing_extensions import TypeVar, deprecated

from vllm import envs
from vllm.beam_search import (BeamSearchInstance, BeamSearchOutput,
                              BeamSearchSequence, get_beam_search_score)
from vllm.config import CompilationConfig
from vllm.engine.arg_utils import (EngineArgs, HfOverrides, PoolerConfig,
                                   TaskOption)
from vllm.engine.llm_engine import LLMEngine
from vllm.entrypoints.chat_utils import (ChatCompletionMessageParam,
                                         ChatTemplateContentFormatOption,
                                         apply_hf_chat_template,
                                         apply_mistral_chat_template,
                                         parse_chat_messages,
                                         resolve_chat_template_content_format)
from vllm.inputs import PromptType, SingletonPrompt, TextPrompt, TokensPrompt
from vllm.inputs.parse import is_token_prompt, parse_and_batch_prompt
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor.guided_decoding.guided_fields import (
    GuidedDecodingRequest, LLMGuidedOptions)
from vllm.outputs import (ClassificationRequestOutput, EmbeddingRequestOutput,
                          PoolingRequestOutput, RequestOutput,
                          ScoringRequestOutput)
from vllm.pooling_params import PoolingParams
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import (BeamSearchParams, GuidedDecodingParams,
                                  RequestOutputKind, SamplingParams)
from vllm.transformers_utils.tokenizer import (AnyTokenizer, MistralTokenizer,
                                               get_cached_tokenizer)
from vllm.transformers_utils.tokenizer_group import TokenizerGroup
from vllm.usage.usage_lib import UsageContext
from vllm.utils import Counter, deprecate_args, deprecate_kwargs, is_list_of

logger = init_logger(__name__)

_R = TypeVar("_R", default=Any)

class PythonREPL:
    def __init__(self, timeout=5):
        self.timeout = timeout

    def __call__(self, query: str):
        # 最終行にprint(...)がない場合は自動的に付加
        lines = query.strip().split("\n")
        if "print(" not in lines[-1]:
            # もし行末にコメントがあるなら削る
            if "#" in lines[-1]:
                lines[-1] = lines[-1].split("#")[0]
            lines[-1] = f"print({lines[-1]})"
        query = "\n".join(lines)
       
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = os.path.join(temp_dir, "tmp.py")
            with open(temp_file_path, "w", encoding="utf-8") as f:
                f.write(query)

            try:
                result = subprocess.run(
                    ["python3", temp_file_path],
                    capture_output=True,
                    check=False,
                    text=True,
                    timeout=self.timeout,
                )
            except subprocess.TimeoutExpired:
                return False, ["Time limit exceeded."]


            if result.returncode == 0:
                # 成功時は標準出力を返す
                return True, result.stdout.strip()

            # 失敗時は標準エラーを整形して返す
            error_msg = result.stderr.strip()
            lines = error_msg.split("\n")

            new_msgs = []
            want_next = False
            for i, m in enumerate(lines):
                if "Traceback" in m:
                    new_msgs.append(m)
                elif i == len(lines) - 1:
                    # 最終行 (エラー原因行)
                    new_msgs.append(m)
                elif temp_file_path in m:
                    # ファイルパスを削除
                    if '"/' in m:
                        st = m.index('"/') + 1
                        ed = m.index(temp_file_path) + len(temp_file_path)
                        # エラー文言からパス部分を除去
                        clr = m[st:ed]
                        m = m.replace(clr, "")
                    new_msgs.append(m)
                    want_next = True
                elif want_next:
                    new_msgs.append(m)
                    want_next = False

            error_msg = "\n".join(new_msgs)
            return False, error_msg.strip()




class LLM:
    """An LLM for generating texts from given prompts and sampling parameters.

    This class includes a tokenizer, a language model (possibly distributed
    across multiple GPUs), and GPU memory space allocated for intermediate
    states (aka KV cache). Given a batch of prompts and sampling parameters,
    this class generates texts from the model, using an intelligent batching
    mechanism and efficient memory management.

    Args:
        model: The name or path of a HuggingFace Transformers model.
        tokenizer: The name or path of a HuggingFace Transformers tokenizer.
        tokenizer_mode: The tokenizer mode. "auto" will use the fast tokenizer
            if available, and "slow" will always use the slow tokenizer.
        skip_tokenizer_init: If true, skip initialization of tokenizer and
            detokenizer. Expect valid prompt_token_ids and None for prompt
            from the input.
        trust_remote_code: Trust remote code (e.g., from HuggingFace) when
            downloading the model and tokenizer.
        allowed_local_media_path: Allowing API requests to read local images
            or videos from directories specified by the server file system.
            This is a security risk. Should only be enabled in trusted
            environments.
        tensor_parallel_size: The number of GPUs to use for distributed
            execution with tensor parallelism.
        dtype: The data type for the model weights and activations. Currently,
            we support `float32`, `float16`, and `bfloat16`. If `auto`, we use
            the `torch_dtype` attribute specified in the model config file.
            However, if the `torch_dtype` in the config is `float32`, we will
            use `float16` instead.
        quantization: The method used to quantize the model weights. Currently,
            we support "awq", "gptq", and "fp8" (experimental).
            If None, we first check the `quantization_config` attribute in the
            model config file. If that is None, we assume the model weights are
            not quantized and use `dtype` to determine the data type of
            the weights.
        revision: The specific model version to use. It can be a branch name,
            a tag name, or a commit id.
        tokenizer_revision: The specific tokenizer version to use. It can be a
            branch name, a tag name, or a commit id.
        seed: The seed to initialize the random number generator for sampling.
        gpu_memory_utilization: The ratio (between 0 and 1) of GPU memory to
            reserve for the model weights, activations, and KV cache. Higher
            values will increase the KV cache size and thus improve the model's
            throughput. However, if the value is too high, it may cause out-of-
            memory (OOM) errors.
        swap_space: The size (GiB) of CPU memory per GPU to use as swap space.
            This can be used for temporarily storing the states of the requests
            when their `best_of` sampling parameters are larger than 1. If all
            requests will have `best_of=1`, you can safely set this to 0.
            Otherwise, too small values may cause out-of-memory (OOM) errors.
        cpu_offload_gb: The size (GiB) of CPU memory to use for offloading
            the model weights. This virtually increases the GPU memory space
            you can use to hold the model weights, at the cost of CPU-GPU data
            transfer for every forward pass.
        enforce_eager: Whether to enforce eager execution. If True, we will
            disable CUDA graph and always execute the model in eager mode.
            If False, we will use CUDA graph and eager execution in hybrid.
        max_seq_len_to_capture: Maximum sequence len covered by CUDA graphs.
            When a sequence has context length larger than this, we fall back
            to eager mode. Additionally for encoder-decoder models, if the
            sequence length of the encoder input is larger than this, we fall
            back to the eager mode.
        disable_custom_all_reduce: See :class:`~vllm.config.ParallelConfig`
        disable_async_output_proc: Disable async output processing.
            This may result in lower performance.
        hf_overrides: If a dictionary, contains arguments to be forwarded to the
            HuggingFace config. If a callable, it is called to update the
            HuggingFace config.
        compilation_config: Either an integer or a dictionary. If it is an
            integer, it is used as the level of compilation optimization. If it
            is a dictionary, it can specify the full compilation configuration.
        **kwargs: Arguments for :class:`~vllm.EngineArgs`. (See
            :ref:`engine-args`)

    Note:
        This class is intended to be used for offline inference. For online
        serving, use the :class:`~vllm.AsyncLLMEngine` class instead.
    """

    DEPRECATE_LEGACY: ClassVar[bool] = True
    """A flag to toggle whether to deprecate the legacy generate/encode API."""

    DEPRECATE_INIT_POSARGS: ClassVar[bool] = True
    """
    A flag to toggle whether to deprecate positional arguments in
    :meth:`LLM.__init__`.
    """

    @classmethod
    @contextmanager
    def deprecate_legacy_api(cls):
        cls.DEPRECATE_LEGACY = True

        yield

        cls.DEPRECATE_LEGACY = False

    @deprecate_args(
        start_index=2,  # Ignore self and model
        is_deprecated=lambda: LLM.DEPRECATE_INIT_POSARGS,
        additional_message=(
            "All positional arguments other than `model` will be "
            "replaced with keyword arguments in an upcoming version."),
    )
    def __init__(
        self,
        model: str,
        tokenizer: Optional[str] = None,
        tokenizer_mode: str = "auto",
        skip_tokenizer_init: bool = False,
        trust_remote_code: bool = False,
        allowed_local_media_path: str = "",
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        quantization: Optional[str] = None,
        revision: Optional[str] = None,
        tokenizer_revision: Optional[str] = None,
        seed: int = 0,
        gpu_memory_utilization: float = 0.9,
        swap_space: float = 4,
        cpu_offload_gb: float = 0,
        enforce_eager: Optional[bool] = None,
        max_seq_len_to_capture: int = 8192,
        disable_custom_all_reduce: bool = False,
        disable_async_output_proc: bool = False,
        hf_overrides: Optional[HfOverrides] = None,
        mm_processor_kwargs: Optional[Dict[str, Any]] = None,
        # After positional args are removed, move this right below `model`
        task: TaskOption = "auto",
        override_pooler_config: Optional[PoolerConfig] = None,
        compilation_config: Optional[Union[int, Dict[str, Any]]] = None,
        start_time: Optional[float] = None,
        **kwargs,
    ) -> None:
        '''
        LLM constructor.

        Note: if enforce_eager is unset (enforce_eager is None)
        it defaults to False.
        '''

        if "disable_log_stats" not in kwargs:
            kwargs["disable_log_stats"] = True

        if "worker_cls" in kwargs:
            worker_cls = kwargs["worker_cls"]
            # if the worker_cls is not qualified string name,
            # we serialize it using cloudpickle to avoid pickling issues
            if isinstance(worker_cls, type):
                kwargs["worker_cls"] = cloudpickle.dumps(worker_cls)

        if compilation_config is not None:
            if isinstance(compilation_config, (int, dict)):
                compilation_config_instance = CompilationConfig.from_cli(
                    str(compilation_config))
            else:
                compilation_config_instance = compilation_config
        else:
            compilation_config_instance = None

        engine_args = EngineArgs(
            model=model,
            task=task,
            tokenizer=tokenizer,
            tokenizer_mode=tokenizer_mode,
            skip_tokenizer_init=skip_tokenizer_init,
            trust_remote_code=trust_remote_code,
            allowed_local_media_path=allowed_local_media_path,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            quantization=quantization,
            revision=revision,
            tokenizer_revision=tokenizer_revision,
            seed=seed,
            gpu_memory_utilization=gpu_memory_utilization,
            swap_space=swap_space,
            cpu_offload_gb=cpu_offload_gb,
            enforce_eager=enforce_eager,
            max_seq_len_to_capture=max_seq_len_to_capture,
            disable_custom_all_reduce=disable_custom_all_reduce,
            disable_async_output_proc=disable_async_output_proc,
            hf_overrides=hf_overrides,
            mm_processor_kwargs=mm_processor_kwargs,
            override_pooler_config=override_pooler_config,
            compilation_config=compilation_config_instance,
            **kwargs,
        )
        # Logic to switch between engines is done at runtime instead of import
        # to avoid import order issues
        self.engine_class = self.get_engine_class()
        self.llm_engine = self.engine_class.from_engine_args(
            engine_args, usage_context=UsageContext.LLM_CLASS)

        self.request_counter = Counter()
        
        self.start_time = start_time

    @staticmethod
    def get_engine_class() -> Type[LLMEngine]:
        if envs.VLLM_USE_V1:
            # Lazy import: the v1 package isn't distributed
            from vllm.v1.engine.llm_engine import LLMEngine as V1LLMEngine
            return V1LLMEngine  # type: ignore
        return LLMEngine

    def get_tokenizer(self) -> AnyTokenizer:
        return self.llm_engine.get_tokenizer_group(TokenizerGroup).tokenizer

    def set_tokenizer(self, tokenizer: AnyTokenizer) -> None:
        tokenizer_group = self.llm_engine.get_tokenizer_group(TokenizerGroup)

        # While CachedTokenizer is dynamic, have no choice but
        # compare class name. Misjudgment will arise from
        # user-defined tokenizer started with 'Cached'
        if tokenizer.__class__.__name__.startswith("Cached"):
            tokenizer_group.tokenizer = tokenizer
        else:
            tokenizer_group.tokenizer = get_cached_tokenizer(tokenizer)

    def get_default_sampling_params(self) -> SamplingParams:
        diff_sampling_param = (
            self.llm_engine.model_config.get_diff_sampling_param())
        if diff_sampling_param:
            return SamplingParams.from_optional(**diff_sampling_param)
        return SamplingParams()

    @overload
    def generate(
        self,
        prompts: Union[PromptType, Sequence[PromptType]],
        /,
        sampling_params: Optional[Union[SamplingParams,
                                        Sequence[SamplingParams]]] = None,
        *,
        use_tqdm: bool = True,
        lora_request: Optional[Union[List[LoRARequest], LoRARequest]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        guided_options_request: Optional[Union[LLMGuidedOptions,
                                               GuidedDecodingRequest]] = None,
    ) -> List[RequestOutput]:
        ...

    @overload  # LEGACY: single (prompt + optional token ids)
    @deprecated("'prompt_token_ids' will become part of 'prompts'")
    def generate(
        self,
        prompts: str,
        sampling_params: Optional[Union[SamplingParams,
                                        List[SamplingParams]]] = None,
        prompt_token_ids: Optional[List[int]] = None,
        use_tqdm: bool = True,
        lora_request: Optional[Union[List[LoRARequest], LoRARequest]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        guided_options_request: Optional[Union[LLMGuidedOptions,
                                               GuidedDecodingRequest]] = None,
    ) -> List[RequestOutput]:
        ...

    @overload  # LEGACY: multi (prompt + optional token ids)
    @deprecated("'prompt_token_ids' will become part of 'prompts'")
    def generate(
        self,
        prompts: List[str],
        sampling_params: Optional[Union[SamplingParams,
                                        List[SamplingParams]]] = None,
        prompt_token_ids: Optional[List[List[int]]] = None,
        use_tqdm: bool = True,
        lora_request: Optional[Union[List[LoRARequest], LoRARequest]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        guided_options_request: Optional[Union[LLMGuidedOptions,
                                               GuidedDecodingRequest]] = None,
    ) -> List[RequestOutput]:
        ...

    @overload  # LEGACY: single (token ids + optional prompt)
    @deprecated("'prompt_token_ids' will become part of 'prompts'")
    def generate(
        self,
        prompts: Optional[str] = None,
        sampling_params: Optional[Union[SamplingParams,
                                        List[SamplingParams]]] = None,
        *,
        prompt_token_ids: List[int],
        use_tqdm: bool = True,
        lora_request: Optional[Union[List[LoRARequest], LoRARequest]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        guided_options_request: Optional[Union[LLMGuidedOptions,
                                               GuidedDecodingRequest]] = None,
    ) -> List[RequestOutput]:
        ...

    @overload  # LEGACY: multi (token ids + optional prompt)
    @deprecated("'prompt_token_ids' will become part of 'prompts'")
    def generate(
        self,
        prompts: Optional[List[str]] = None,
        sampling_params: Optional[Union[SamplingParams,
                                        List[SamplingParams]]] = None,
        *,
        prompt_token_ids: List[List[int]],
        use_tqdm: bool = True,
        lora_request: Optional[Union[List[LoRARequest], LoRARequest]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        guided_options_request: Optional[Union[LLMGuidedOptions,
                                               GuidedDecodingRequest]] = None,
    ) -> List[RequestOutput]:
        ...

    @overload  # LEGACY: single or multi token ids [pos-only]
    @deprecated("'prompt_token_ids' will become part of 'prompts'")
    def generate(
        self,
        prompts: None,
        sampling_params: None,
        prompt_token_ids: Union[List[int], List[List[int]]],
        use_tqdm: bool = True,
        lora_request: Optional[Union[List[LoRARequest], LoRARequest]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        guided_options_request: Optional[Union[LLMGuidedOptions,
                                               GuidedDecodingRequest]] = None,
    ) -> List[RequestOutput]:
        ...

    @deprecate_kwargs(
        "prompt_token_ids",
        is_deprecated=lambda: LLM.DEPRECATE_LEGACY,
        additional_message="Please use the 'prompts' parameter instead.",
    )
    def generate(
        self,
        prompts: Union[Union[PromptType, Sequence[PromptType]],
                       Optional[Union[str, List[str]]]] = None,
        sampling_params: Optional[Union[SamplingParams,
                                        Sequence[SamplingParams]]] = None,
        prompt_token_ids: Optional[Union[List[int], List[List[int]]]] = None,
        use_tqdm: bool = True,
        lora_request: Optional[Union[List[LoRARequest], LoRARequest]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        guided_options_request: Optional[Union[LLMGuidedOptions,
                                               GuidedDecodingRequest]] = None,
        priority: Optional[List[int]] = None,
    ) -> List[RequestOutput]:
        """Generates the completions for the input prompts.

        This class automatically batches the given prompts, considering
        the memory constraint. For the best performance, put all of your prompts
        into a single list and pass it to this method.

        Args:
            prompts: The prompts to the LLM. You may pass a sequence of prompts
                for batch inference. See :class:`~vllm.inputs.PromptType`
                for more details about the format of each prompts.
            sampling_params: The sampling parameters for text generation. If
                None, we use the default sampling parameters.
                When it is a single value, it is applied to every prompt.
                When it is a list, the list must have the same length as the
                prompts and it is paired one by one with the prompt.
            use_tqdm: Whether to use tqdm to display the progress bar.
            lora_request: LoRA request to use for generation, if any.
            prompt_adapter_request: Prompt Adapter request to use for
                generation, if any.
            priority: The priority of the requests, if any.
                Only applicable when priority scheduling policy is enabled.

        Returns:
            A list of ``RequestOutput`` objects containing the
            generated completions in the same order as the input prompts.

        Note:
            Using ``prompts`` and ``prompt_token_ids`` as keyword parameters is
            considered legacy and may be deprecated in the future. You should
            instead pass them via the ``inputs`` parameter.
        """
        runner_type = self.llm_engine.model_config.runner_type
        if runner_type != "generate":
            messages = [
                "LLM.generate() is only supported for (conditional) generation "
                "models (XForCausalLM, XForConditionalGeneration).",
            ]

            supported_runner_types = self.llm_engine.model_config \
                .supported_runner_types
            if "generate" in supported_runner_types:
                messages.append(
                    "Your model supports the 'generate' runner, but is "
                    f"currently initialized for the '{runner_type}' runner. "
                    "Please initialize vLLM using `--task generate`.")

            raise ValueError(" ".join(messages))

        if prompt_token_ids is not None:
            parsed_prompts = self._convert_v1_inputs(
                prompts=cast(Optional[Union[str, List[str]]], prompts),
                prompt_token_ids=prompt_token_ids,
            )
        else:
            parsed_prompts = cast(Union[PromptType, Sequence[PromptType]],
                                  prompts)

        if isinstance(guided_options_request, dict):
            if len(guided_options_request) > 1:
                raise ValueError(
                    "You can only use one guided decoding but multiple is "
                    f"specified: {guided_options_request}")
            guided_options_request = GuidedDecodingRequest(
                **guided_options_request)

        if sampling_params is None:
            # Use default sampling params.
            sampling_params = self.get_default_sampling_params()

        self._validate_and_add_requests(
            prompts=parsed_prompts,
            params=sampling_params,
            lora_request=lora_request,
            prompt_adapter_request=prompt_adapter_request,
            guided_options=guided_options_request,
            priority=priority)

        outputs = self._run_engine(use_tqdm=use_tqdm)
        return self.engine_class.validate_outputs(outputs, RequestOutput)

    def collective_rpc(self,
                       method: Union[str, Callable[..., _R]],
                       timeout: Optional[float] = None,
                       args: Tuple = (),
                       kwargs: Optional[Dict[str, Any]] = None) -> List[_R]:
        """
        Execute an RPC call on all workers.

        Args:
            method: Name of the worker method to execute, or a callable that
                is serialized and sent to all workers to execute.

                If the method is a callable, it should accept an additional
                `self` argument, in addition to the arguments passed in `args`
                and `kwargs`. The `self` argument will be the worker object.
            timeout: Maximum time in seconds to wait for execution. Raises a
                :exc:`TimeoutError` on timeout. `None` means wait indefinitely.
            args: Positional arguments to pass to the worker method.
            kwargs: Keyword arguments to pass to the worker method.

        Returns:
            A list containing the results from each worker.
        
        Note:
            It is recommended to use this API to only pass control messages,
            and set up data-plane communication to pass data.
        """
        executor = self.llm_engine.model_executor
        return executor.collective_rpc(method, timeout, args, kwargs)

    def apply_model(self, func: Callable[[nn.Module], _R]) -> list[_R]:
        """
        Run a function directly on the model inside each worker,
        returning the result for each of them.
        """
        executor = self.llm_engine.model_executor
        return executor.apply_model(func)

    def beam_search(
        self,
        prompts: List[Union[TokensPrompt, TextPrompt]],
        params: BeamSearchParams,
    ) -> List[BeamSearchOutput]:
        """
        Generate sequences using beam search.

        Args:
            prompts: A list of prompts. Each prompt can be a string or a list
                of token IDs.
            params: The beam search parameters.

        TODO: how does beam search work together with length penalty, frequency
        penalty, and stopping criteria, etc.?
        """

        beam_width = params.beam_width
        max_tokens = params.max_tokens
        temperature = params.temperature
        ignore_eos = params.ignore_eos
        length_penalty = params.length_penalty

        def sort_beams_key(x: BeamSearchSequence) -> float:
            return get_beam_search_score(x.tokens, x.cum_logprob,
                                         tokenizer.eos_token_id,
                                         length_penalty)

        tokenizer = self.get_tokenizer()
        # generate 2 * beam_width candidates at each step
        # following the huggingface transformers implementation
        # at https://github.com/huggingface/transformers/blob/e15687fffe5c9d20598a19aeab721ae0a7580f8a/src/transformers/generation/beam_search.py#L534 # noqa
        beam_search_params = SamplingParams(logprobs=2 * beam_width,
                                            max_tokens=1,
                                            temperature=temperature)
        instances: List[BeamSearchInstance] = []

        for prompt in prompts:
            if is_token_prompt(prompt):
                prompt_tokens = prompt["prompt_token_ids"]
            else:
                prompt_tokens = tokenizer.encode(prompt["prompt"])
            instances.append(BeamSearchInstance(prompt_tokens))

        for _ in range(max_tokens):
            all_beams: List[BeamSearchSequence] = list(
                sum((instance.beams for instance in instances), []))
            pos = [0] + list(
                itertools.accumulate(
                    len(instance.beams) for instance in instances))
            instance_start_and_end: List[Tuple[int, int]] = list(
                zip(pos[:-1], pos[1:]))

            if len(all_beams) == 0:
                break

            prompts_batch = [
                TokensPrompt(prompt_token_ids=beam.tokens)
                for beam in all_beams
            ]

            # only runs for one step
            # we don't need to use tqdm here
            output = self.generate(prompts_batch,
                                   sampling_params=beam_search_params,
                                   use_tqdm=False)

            for (start, end), instance in zip(instance_start_and_end,
                                              instances):
                instance_new_beams = []
                for i in range(start, end):
                    current_beam = all_beams[i]
                    result = output[i]

                    if result.outputs[0].logprobs is not None:
                        # if `result.outputs[0].logprobs` is None, it means
                        # the sequence is completed because of the max-model-len
                        # or abortion. we don't need to add it to the new beams.
                        logprobs = result.outputs[0].logprobs[0]
                        for token_id, logprob_obj in logprobs.items():
                            new_beam = BeamSearchSequence(
                                tokens=current_beam.tokens + [token_id],
                                logprobs=current_beam.logprobs + [logprobs],
                                cum_logprob=current_beam.cum_logprob +
                                logprob_obj.logprob)

                            if token_id == tokenizer.eos_token_id and \
                                not ignore_eos:
                                instance.completed.append(new_beam)
                            else:
                                instance_new_beams.append(new_beam)
                sorted_beams = sorted(instance_new_beams,
                                      key=sort_beams_key,
                                      reverse=True)
                instance.beams = sorted_beams[:beam_width]

        outputs = []
        for instance in instances:
            instance.completed.extend(instance.beams)
            sorted_completed = sorted(instance.completed,
                                      key=sort_beams_key,
                                      reverse=True)
            best_beams = sorted_completed[:beam_width]

            for beam in best_beams:
                beam.text = tokenizer.decode(beam.tokens)
            outputs.append(BeamSearchOutput(sequences=best_beams))

        return outputs

    def chat(
        self,
        messages: Union[List[ChatCompletionMessageParam],
                        List[List[ChatCompletionMessageParam]]],
        sampling_params: Optional[Union[SamplingParams,
                                        List[SamplingParams]]] = None,
        use_tqdm: bool = True,
        lora_request: Optional[LoRARequest] = None,
        chat_template: Optional[str] = None,
        chat_template_content_format: ChatTemplateContentFormatOption = "auto",
        add_generation_prompt: bool = True,
        continue_final_message: bool = False,
        tools: Optional[List[Dict[str, Any]]] = None,
        mm_processor_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[RequestOutput]:
        """
        Generate responses for a chat conversation.

        The chat conversation is converted into a text prompt using the
        tokenizer and calls the :meth:`generate` method to generate the
        responses.

        Multi-modal inputs can be passed in the same way you would pass them
        to the OpenAI API.

        Args:
            messages: A list of conversations or a single conversation.

              - Each conversation is represented as a list of messages.
              - Each message is a dictionary with 'role' and 'content' keys.

            sampling_params: The sampling parameters for text generation.
                If None, we use the default sampling parameters. When it
                is a single value, it is applied to every prompt. When it
                is a list, the list must have the same length as the
                prompts and it is paired one by one with the prompt.
            use_tqdm: Whether to use tqdm to display the progress bar.
            lora_request: LoRA request to use for generation, if any.
            chat_template: The template to use for structuring the chat.
              If not provided, the model's default chat template will be used.
            chat_template_content_format: The format to render message content.

              - "string" will render the content as a string.
                Example: ``"Who are you?"``
              - "openai" will render the content as a list of dictionaries,
                similar to OpenAI schema.
                Example: ``[{"type": "text", "text": "Who are you?"}]``

            add_generation_prompt: If True, adds a generation template
                to each message.
            continue_final_message: If True, continues the final message in
                the conversation instead of starting a new one. Cannot be
                ``True`` if ``add_generation_prompt`` is also ``True``.
            mm_processor_kwargs: Multimodal processor kwarg overrides for this
                chat request. Only used for offline requests.

        Returns:
            A list of ``RequestOutput`` objects containing the generated
            responses in the same order as the input messages.
        """
        list_of_messages: List[List[ChatCompletionMessageParam]]

        # Handle multi and single conversations
        if is_list_of(messages, list):
            # messages is List[List[...]]
            list_of_messages = cast(List[List[ChatCompletionMessageParam]],
                                    messages)
        else:
            # messages is List[...]
            list_of_messages = [
                cast(List[ChatCompletionMessageParam], messages)
            ]

        tokenizer = self.get_tokenizer()
        model_config = self.llm_engine.get_model_config()
        resolved_content_format = resolve_chat_template_content_format(
            chat_template,
            chat_template_content_format,
            tokenizer,
        )

        prompts: List[Union[TokensPrompt, TextPrompt]] = []

        for msgs in list_of_messages:
            # NOTE: _parse_chat_message_content_parts() currently doesn't
            # handle mm_processor_kwargs, since there is no implementation in
            # the chat message parsing for it.
            conversation, mm_data = parse_chat_messages(
                msgs,
                model_config,
                tokenizer,
                content_format=resolved_content_format,
            )

            prompt_data: Union[str, List[int]]
            if isinstance(tokenizer, MistralTokenizer):
                prompt_data = apply_mistral_chat_template(
                    tokenizer,
                    messages=msgs,
                    chat_template=chat_template,
                    add_generation_prompt=add_generation_prompt,
                    continue_final_message=continue_final_message,
                    tools=tools,
                )
            else:
                prompt_data = apply_hf_chat_template(
                    tokenizer,
                    conversation=conversation,
                    chat_template=chat_template,
                    add_generation_prompt=add_generation_prompt,
                    continue_final_message=continue_final_message,
                    tools=tools,
                )

            prompt: Union[TokensPrompt, TextPrompt]
            if is_list_of(prompt_data, int):
                prompt = TokensPrompt(prompt_token_ids=prompt_data)
            else:
                prompt = TextPrompt(prompt=prompt_data)

            if mm_data is not None:
                prompt["multi_modal_data"] = mm_data

            if mm_processor_kwargs is not None:
                prompt["mm_processor_kwargs"] = mm_processor_kwargs

            prompts.append(prompt)

        return self.generate(
            prompts,
            sampling_params=sampling_params,
            use_tqdm=use_tqdm,
            lora_request=lora_request,
        )

    @overload
    def encode(
        self,
        prompts: Union[PromptType, Sequence[PromptType]],
        /,
        pooling_params: Optional[Union[PoolingParams,
                                       Sequence[PoolingParams]]] = None,
        *,
        use_tqdm: bool = True,
        lora_request: Optional[Union[List[LoRARequest], LoRARequest]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
    ) -> List[PoolingRequestOutput]:
        ...

    @overload  # LEGACY: single (prompt + optional token ids)
    @deprecated("'prompt_token_ids' will become part of 'prompts'")
    def encode(
        self,
        prompts: str,
        pooling_params: Optional[Union[PoolingParams,
                                       Sequence[PoolingParams]]] = None,
        prompt_token_ids: Optional[List[int]] = None,
        use_tqdm: bool = True,
        lora_request: Optional[Union[List[LoRARequest], LoRARequest]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
    ) -> List[PoolingRequestOutput]:
        ...

    @overload  # LEGACY: multi (prompt + optional token ids)
    @deprecated("'prompt_token_ids' will become part of 'prompts'")
    def encode(
        self,
        prompts: List[str],
        pooling_params: Optional[Union[PoolingParams,
                                       Sequence[PoolingParams]]] = None,
        prompt_token_ids: Optional[List[List[int]]] = None,
        use_tqdm: bool = True,
        lora_request: Optional[Union[List[LoRARequest], LoRARequest]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
    ) -> List[PoolingRequestOutput]:
        ...

    @overload  # LEGACY: single (token ids + optional prompt)
    @deprecated("'prompt_token_ids' will become part of 'prompts'")
    def encode(
        self,
        prompts: Optional[str] = None,
        pooling_params: Optional[Union[PoolingParams,
                                       Sequence[PoolingParams]]] = None,
        *,
        prompt_token_ids: List[int],
        use_tqdm: bool = True,
        lora_request: Optional[Union[List[LoRARequest], LoRARequest]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
    ) -> List[PoolingRequestOutput]:
        ...

    @overload  # LEGACY: multi (token ids + optional prompt)
    @deprecated("'prompt_token_ids' will become part of 'prompts'")
    def encode(
        self,
        prompts: Optional[List[str]] = None,
        pooling_params: Optional[Union[PoolingParams,
                                       Sequence[PoolingParams]]] = None,
        *,
        prompt_token_ids: List[List[int]],
        use_tqdm: bool = True,
        lora_request: Optional[Union[List[LoRARequest], LoRARequest]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
    ) -> List[PoolingRequestOutput]:
        ...

    @overload  # LEGACY: single or multi token ids [pos-only]
    @deprecated("'prompt_token_ids' will become part of 'prompts'")
    def encode(
        self,
        prompts: None,
        pooling_params: None,
        prompt_token_ids: Union[List[int], List[List[int]]],
        use_tqdm: bool = True,
        lora_request: Optional[Union[List[LoRARequest], LoRARequest]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
    ) -> List[PoolingRequestOutput]:
        ...

    @deprecate_kwargs(
        "prompt_token_ids",
        is_deprecated=lambda: LLM.DEPRECATE_LEGACY,
        additional_message="Please use the 'prompts' parameter instead.",
    )
    def encode(
        self,
        prompts: Union[Union[PromptType, Sequence[PromptType]],
                       Optional[Union[str, List[str]]]] = None,
        pooling_params: Optional[Union[PoolingParams,
                                       Sequence[PoolingParams]]] = None,
        prompt_token_ids: Optional[Union[List[int], List[List[int]]]] = None,
        use_tqdm: bool = True,
        lora_request: Optional[Union[List[LoRARequest], LoRARequest]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
    ) -> List[PoolingRequestOutput]:
        """Apply pooling to the hidden states corresponding to the input
        prompts.

        This class automatically batches the given prompts, considering
        the memory constraint. For the best performance, put all of your prompts
        into a single list and pass it to this method.

        Args:
            prompts: The prompts to the LLM. You may pass a sequence of prompts
                for batch inference. See :class:`~vllm.inputs.PromptType`
                for more details about the format of each prompts.
            pooling_params: The pooling parameters for pooling. If None, we
                use the default pooling parameters.
            use_tqdm: Whether to use tqdm to display the progress bar.
            lora_request: LoRA request to use for generation, if any.
            prompt_adapter_request: Prompt Adapter request to use for
                generation, if any.

        Returns:
            A list of ``PoolingRequestOutput`` objects containing the
            pooled hidden states in the same order as the input prompts.

        Note:
            Using ``prompts`` and ``prompt_token_ids`` as keyword parameters is
            considered legacy and may be deprecated in the future. You should
            instead pass them via the ``inputs`` parameter.
        """
        runner_type = self.llm_engine.model_config.runner_type
        if runner_type != "pooling":
            messages = ["LLM.encode() is only supported for pooling models."]

            supported_runner_types = self.llm_engine.model_config \
                .supported_runner_types
            if "pooling" in supported_runner_types:
                messages.append(
                    "Your model supports the 'pooling' runner, but is "
                    f"currently initialized for the '{runner_type}' runner. "
                    "Please initialize vLLM using `--task embed`, "
                    "`--task classify`, `--task score` etc.")

            raise ValueError(" ".join(messages))

        if prompt_token_ids is not None:
            parsed_prompts = self._convert_v1_inputs(
                prompts=cast(Optional[Union[str, List[str]]], prompts),
                prompt_token_ids=prompt_token_ids,
            )
        else:
            parsed_prompts = cast(Union[PromptType, Sequence[PromptType]],
                                  prompts)

        if pooling_params is None:
            # Use default pooling params.
            pooling_params = PoolingParams()

        self._validate_and_add_requests(
            prompts=parsed_prompts,
            params=pooling_params,
            lora_request=lora_request,
            prompt_adapter_request=prompt_adapter_request,
        )

        outputs = self._run_engine(use_tqdm=use_tqdm)
        return self.engine_class.validate_outputs(outputs,
                                                  PoolingRequestOutput)

    def embed(
        self,
        prompts: Union[PromptType, Sequence[PromptType]],
        /,
        *,
        use_tqdm: bool = True,
        lora_request: Optional[Union[List[LoRARequest], LoRARequest]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
    ) -> List[EmbeddingRequestOutput]:
        """
        Generate an embedding vector for each prompt.

        This class automatically batches the given prompts, considering
        the memory constraint. For the best performance, put all of your prompts
        into a single list and pass it to this method.

        Args:
            prompts: The prompts to the LLM. You may pass a sequence of prompts
                for batch inference. See :class:`~vllm.inputs.PromptType`
                for more details about the format of each prompts.
            use_tqdm: Whether to use tqdm to display the progress bar.
            lora_request: LoRA request to use for generation, if any.
            prompt_adapter_request: Prompt Adapter request to use for
                generation, if any.

        Returns:
            A list of ``EmbeddingRequestOutput`` objects containing the
            embedding vectors in the same order as the input prompts.
        """
        if self.llm_engine.model_config.task != "embed":
            raise ValueError(
                "Embedding API is only enabled for `--task embed`")

        items = self.encode(prompts,
                            use_tqdm=use_tqdm,
                            lora_request=lora_request,
                            prompt_adapter_request=prompt_adapter_request)

        return [EmbeddingRequestOutput.from_base(item) for item in items]

    def classify(
        self,
        prompts: Union[PromptType, Sequence[PromptType]],
        /,
        *,
        use_tqdm: bool = True,
        lora_request: Optional[Union[List[LoRARequest], LoRARequest]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
    ) -> List[ClassificationRequestOutput]:
        """
        Generate class logits for each prompt.

        This class automatically batches the given prompts, considering
        the memory constraint. For the best performance, put all of your prompts
        into a single list and pass it to this method.

        Args:
            prompts: The prompts to the LLM. You may pass a sequence of prompts
                for batch inference. See :class:`~vllm.inputs.PromptType`
                for more details about the format of each prompts.
            use_tqdm: Whether to use tqdm to display the progress bar.
            lora_request: LoRA request to use for generation, if any.
            prompt_adapter_request: Prompt Adapter request to use for
                generation, if any.

        Returns:
            A list of ``ClassificationRequestOutput`` objects containing the
            embedding vectors in the same order as the input prompts.
        """
        if self.llm_engine.model_config.task != "classify":
            raise ValueError(
                "Classification API is only enabled for `--task classify`")

        items = self.encode(prompts,
                            use_tqdm=use_tqdm,
                            lora_request=lora_request,
                            prompt_adapter_request=prompt_adapter_request)

        return [ClassificationRequestOutput.from_base(item) for item in items]

    def _embedding_score(
        self,
        tokenizer: AnyTokenizer,
        text_1: List[Union[str, TextPrompt, TokensPrompt]],
        text_2: List[Union[str, TextPrompt, TokensPrompt]],
        truncate_prompt_tokens: Optional[int] = None,
        use_tqdm: bool = True,
        lora_request: Optional[Union[List[LoRARequest], LoRARequest]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
    ) -> List[ScoringRequestOutput]:

        encoded_output = self.encode(
            text_1 + text_2,
            use_tqdm=use_tqdm,
            lora_request=lora_request,
            prompt_adapter_request=prompt_adapter_request)
        encoded_output_1 = encoded_output[0:len(text_1)]
        encoded_output_2 = encoded_output[len(text_1):]

        if len(encoded_output_1) == 1:
            encoded_output_1 = encoded_output_1 * len(encoded_output_2)

        output_pairs = [(t1, t2)
                        for t1, t2 in zip(encoded_output_1, encoded_output_2)]

        scores = []
        scorer = torch.nn.CosineSimilarity(0)

        for embed_1, embed_2 in output_pairs:
            pair_score = scorer(embed_1.outputs.data, embed_2.outputs.data)

            if (pad_token_id := getattr(tokenizer, "pad_token_id",
                                        None)) is not None:
                tokens = embed_1.prompt_token_ids + [
                    pad_token_id
                ] + embed_2.prompt_token_ids
            else:
                tokens = embed_1.prompt_token_ids + embed_2.prompt_token_ids

            scores.append(
                PoolingRequestOutput(
                    request_id=f"{embed_1.request_id}_{embed_2.request_id}",
                    outputs=pair_score,
                    prompt_token_ids=tokens,
                    finished=True))

        items = self.engine_class.validate_outputs(scores,
                                                   PoolingRequestOutput)
        return [ScoringRequestOutput.from_base(item) for item in items]

    def _cross_encoding_score(
        self,
        tokenizer: Union[AnyTokenizer],
        text_1: List[Union[str, TextPrompt, TokensPrompt]],
        text_2: List[Union[str, TextPrompt, TokensPrompt]],
        truncate_prompt_tokens: Optional[int] = None,
        use_tqdm: bool = True,
        lora_request: Optional[Union[List[LoRARequest], LoRARequest]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
    ) -> List[ScoringRequestOutput]:

        if isinstance(tokenizer, MistralTokenizer):
            raise ValueError(
                "Score API is only enabled for `--task embed or score`")

        if len(text_1) == 1:
            text_1 = text_1 * len(text_2)

        input_pairs = [(t1, t2) for t1, t2 in zip(text_1, text_2)]

        pooling_params = PoolingParams()

        tokenization_kwargs: Dict[str, Any] = {}
        if truncate_prompt_tokens is not None:
            tokenization_kwargs["truncation"] = True
            tokenization_kwargs["max_length"] = truncate_prompt_tokens

        parsed_prompts = []

        for q, t in input_pairs:
            prompt_inputs = tokenizer(text=q,
                                      text_pair=t,
                                      **tokenization_kwargs)
            engine_prompt = TokensPrompt(
                prompt_token_ids=prompt_inputs["input_ids"],
                token_type_ids=prompt_inputs.get("token_type_ids"))
            parsed_prompts.append(engine_prompt)

        self._validate_and_add_requests(
            prompts=parsed_prompts,
            params=pooling_params,
            lora_request=lora_request,
            prompt_adapter_request=prompt_adapter_request,
        )

        outputs = self._run_engine(use_tqdm=use_tqdm)
        items = self.engine_class.validate_outputs(outputs,
                                                   PoolingRequestOutput)

        return [ScoringRequestOutput.from_base(item) for item in items]

    def score(
        self,
        text_1: Union[SingletonPrompt, Sequence[SingletonPrompt]],
        text_2: Union[SingletonPrompt, Sequence[SingletonPrompt]],
        /,
        *,
        truncate_prompt_tokens: Optional[int] = None,
        use_tqdm: bool = True,
        lora_request: Optional[Union[List[LoRARequest], LoRARequest]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
    ) -> List[ScoringRequestOutput]:
        """Generate similarity scores for all pairs ``<text,text_pair>``.

        The inputs can be ``1 -> 1``, ``1 -> N`` or ``N -> N``.
        In the ``1 - N`` case the ``text_1`` sentence will be replicated ``N``
        times to pair with the ``text_2`` sentences.
        The input pairs are used to build a list of prompts for the
        cross encoder model. This class automatically batches the prompts,
        considering the memory constraint. For the best performance, put all
        of your texts into a single list and pass it to this method.

        Args:
            text_1: can be a single prompt or a list of prompts, in which
                case it has to have the same length as the ``text_2`` list
            text_2: The texts to pair with the query to form the input
                to the LLM. See :class:`~vllm.inputs.PromptType` for
                more details about the format of each prompts.
            use_tqdm: Whether to use tqdm to display the progress bar.
            lora_request: LoRA request to use for generation, if any.
            prompt_adapter_request: Prompt Adapter request to use for
                generation, if any.

        Returns:
            A list of ``ScoringRequestOutput`` objects containing the
            generated scores in the same order as the input prompts.
        """
        runner_type = self.llm_engine.model_config.runner_type
        if runner_type != "pooling":
            messages = ["LLM.score() is only supported for pooling models."]

            supported_runner_types = self.llm_engine.model_config \
                .supported_runner_types
            if "pooling" in supported_runner_types:
                messages.append(
                    "Your model supports the 'pooling' runner, but is "
                    f"currently initialized for the '{runner_type}' runner. "
                    "Please initialize vLLM using `--task embed`, "
                    "`--task classify`, `--task score` etc.")

            raise ValueError(" ".join(messages))

        if self.llm_engine.model_config.task not in ("embed", "score"):
            raise ValueError(
                "Score API is only enabled for `--task embed or --task score`")

        # the tokenizer for models such as
        # "cross-encoder/ms-marco-MiniLM-L-6-v2" doesn't support passing
        # lists of tokens to the `text` and `text_pair` kwargs
        tokenizer = self.llm_engine.get_tokenizer()

        def ensure_str(prompt: SingletonPrompt):
            if isinstance(prompt, dict):
                if "multi_modal_data" in prompt:
                    raise ValueError("Multi-modal prompt is not "
                                     "supported for scoring")
                elif "prompt_token_ids" in prompt:
                    prompt = tokenizer.decode(
                        cast(TokensPrompt, prompt)["prompt_token_ids"])
                elif "prompt" in prompt:
                    prompt = cast(TextPrompt, prompt)["prompt"]
            assert type(prompt) is str
            return prompt

        if isinstance(text_1, (str, dict)):
            # Convert a single prompt to a list.
            text_1 = [text_1]
        text_1 = [ensure_str(t) for t in text_1]

        if isinstance(text_2, (str, dict)):
            # Convert a single prompt to a list.
            text_2 = [text_2]
        text_2 = [ensure_str(t) for t in text_2]

        if len(text_1) > 1 and len(text_1) != len(text_2):
            raise ValueError("Input lengths must be either 1:1, 1:N or N:N")
        if len(text_1) == 0:
            raise ValueError("At least one text element must be given")
        if len(text_2) == 0:
            raise ValueError("At least one text_pair element must be given")

        if self.llm_engine.model_config.is_cross_encoder:
            return self._cross_encoding_score(tokenizer, text_1, text_2,
                                              truncate_prompt_tokens, use_tqdm,
                                              lora_request,
                                              prompt_adapter_request)
        else:
            return self._embedding_score(tokenizer, text_1, text_2,
                                         truncate_prompt_tokens, use_tqdm,
                                         lora_request, prompt_adapter_request)

    def start_profile(self) -> None:
        self.llm_engine.start_profile()

    def stop_profile(self) -> None:
        self.llm_engine.stop_profile()

    def reset_prefix_cache(self) -> bool:
        return self.llm_engine.reset_prefix_cache()

    def sleep(self, level: int = 1):
        """
        Put the engine to sleep. The engine should not process any requests.
        The caller should guarantee that no requests are being processed
        during the sleep period, before `wake_up` is called.

        :param level: The sleep level. Level 1 sleep will offload the model 
            weights and discard the kv cache. The content of kv cache is 
            forgotten. Level 1 sleep is good for sleeping and waking up the 
            engine to run the same model again. The model weights are backed 
            up in CPU memory. Please make sure there's enough CPU memory to 
            store the model weights. Level 2 sleep will discard both the model 
            weights and the kv cache. The content of both the model weights 
            and kv cache is forgotten. Level 2 sleep is good for sleeping and 
            waking up the engine to run a different model or update the model, 
            where previous model weights are not needed. It reduces CPU memory 
            pressure.
        """
        self.reset_prefix_cache()
        self.llm_engine.sleep(level=level)

    def wake_up(self):
        """
        Wake up the engine from sleep mode. See the :meth:`sleep` method
        for more details."""
        self.llm_engine.wake_up()

    # LEGACY
    def _convert_v1_inputs(
        self,
        prompts: Optional[Union[str, List[str]]],
        prompt_token_ids: Optional[Union[List[int], List[List[int]]]],
    ):
        # skip_tokenizer_init is now checked in engine

        if prompts is not None:
            prompts = [p["content"] for p in parse_and_batch_prompt(prompts)]
        if prompt_token_ids is not None:
            prompt_token_ids = [
                p["content"] for p in parse_and_batch_prompt(prompt_token_ids)
            ]

        num_requests = None
        if prompts is not None:
            num_requests = len(prompts)
        if prompt_token_ids is not None:
            if (num_requests is not None
                    and num_requests != len(prompt_token_ids)):
                raise ValueError("The lengths of prompts and prompt_token_ids "
                                 "must be the same.")

            num_requests = len(prompt_token_ids)
        if num_requests is None:
            raise ValueError("Either prompts or prompt_token_ids must be "
                             "provided.")

        parsed_prompts: List[PromptType] = []
        for i in range(num_requests):
            item: PromptType

            if prompts is not None:
                item = TextPrompt(prompt=prompts[i])
            elif prompt_token_ids is not None:
                item = TokensPrompt(prompt_token_ids=prompt_token_ids[i])
            else:
                raise AssertionError

            parsed_prompts.append(item)

        return parsed_prompts

    def _validate_and_add_requests(
        self,
        prompts: Union[PromptType, Sequence[PromptType]],
        params: Union[SamplingParams, Sequence[SamplingParams], PoolingParams,
                      Sequence[PoolingParams]],
        lora_request: Optional[Union[Sequence[LoRARequest], LoRARequest]],
        prompt_adapter_request: Optional[PromptAdapterRequest],
        guided_options: Optional[GuidedDecodingRequest] = None,
        priority: Optional[List[int]] = None,
    ) -> None:
        if guided_options is not None:
            warnings.warn(
                "guided_options_request is deprecated, use "
                "SamplingParams.guided_decoding instead",
                DeprecationWarning,
                stacklevel=2,
            )

        if isinstance(prompts, (str, dict)):
            # Convert a single prompt to a list.
            prompts = [prompts]

        num_requests = len(prompts)
        if isinstance(params, list) and len(params) != num_requests:
            raise ValueError("The lengths of prompts and params "
                             "must be the same.")
        if isinstance(lora_request,
                      list) and len(lora_request) != num_requests:
            raise ValueError("The lengths of prompts and lora_request "
                             "must be the same.")

        for sp in params if isinstance(params, list) else (params, ):
            if isinstance(sp, SamplingParams):
                self._add_guided_params(sp, guided_options)

                # We only care about the final output
                sp.output_kind = RequestOutputKind.FINAL_ONLY

        # Add requests to the engine.
        for i, prompt in enumerate(prompts):
            self._add_request(
                prompt,
                params[i] if isinstance(params, Sequence) else params,
                lora_request=lora_request[i] if isinstance(
                    lora_request, Sequence) else lora_request,
                prompt_adapter_request=prompt_adapter_request,
                priority=priority[i] if priority else 0,
            )

    def _add_request(
        self,
        prompt: PromptType,
        params: Union[SamplingParams, PoolingParams],
        lora_request: Optional[LoRARequest] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        priority: int = 0,
    ) -> None:
        request_id = str(next(self.request_counter))
        self.llm_engine.add_request(
            request_id,
            prompt,
            params,
            lora_request=lora_request,
            prompt_adapter_request=prompt_adapter_request,
            priority=priority,
        )

    def _add_guided_params(
            self,
            params: SamplingParams,
            guided_options: Optional[GuidedDecodingRequest] = None):
        if guided_options is None:
            return params

        if params.guided_decoding is not None:
            raise ValueError("Cannot set both guided_options_request and"
                             "params.guided_decoding.")

        params.guided_decoding = GuidedDecodingParams(
            json=guided_options.guided_json,
            regex=guided_options.guided_regex,
            choice=guided_options.guided_choice,
            grammar=guided_options.guided_grammar,
            json_object=guided_options.guided_json_object,
            backend=guided_options.guided_decoding_backend,
            whitespace_pattern=guided_options.guided_whitespace_pattern)
        return params

    def _extract_boxed_text(self, text: str) -> str:
        pattern = r'oxed{(.*?)}'
        matches = re.findall(pattern, text)
        if not matches:
            return ""
        for match in matches[::-1]:
            if match != "":
                return match
        return ""

    def _has_python_code(self, text: str) -> bool:
        pattern = r"```python(.*?)```"
        return bool(re.search(pattern, text, re.DOTALL))
    
    def _execute_generated_code(self, generated_code: str) -> list[str]:
        code_blocks = re.findall(r"```python(.*?)```", generated_code, re.DOTALL)
        executor = PythonREPL(timeout=10)
        results = []
        for code in code_blocks:
            success, output = executor(code)
            results.append(output)

        return results 
    
    def _get_answer(self, text: str) -> int:
        code_answer = self._execute_generated_code(text)
        try:
            last_element = code_answer[-1]
            num_value = int(round(float(last_element)))
            answer = num_value % 1000
        except:
            try:
                box_answer = self._extract_boxed_text(text)
                num_value = int(round(float(box_answer)))
                answer = num_value % 1000
            except:
                answer = -1

        return answer
    
    def _winner_decided(self, numbers: list[int]) -> bool:
        # -1を除外
        filtered = [num for num in numbers if num != -1]
        # 0 を除外
        filtered = [num for num in filtered if num != 0]
        
        if not filtered:
            # すべて-1 or 0だった場合はwinnerがいないfalseとする
            return False, "all -1 or 0"
        
        # 出現回数をカウント
        counter = _Counter(filtered)
        
        if len(numbers) <= 3:
            return False, "smaller than 3"

        if len(counter) == 1 and len(numbers) >= 4:
            # 1種類だけなので、差はcounter[0]の回数 - 0
            return next(iter(counter.values())) >= 3, "1 type 4 or more"
        
        # gen 6 以上かつ答えが1種類の場合は、差が2以上か判定してboolを返す
        if len(counter) == 1 and len(numbers) >= 6:
            return next(iter(counter.values())) >= 2, "1 type 6 or more"

        # gen 8 以上かつ答えが1種類の場合は、差が1以上か判定してboolを返す
        if len(counter) == 1 and len(numbers) >= 8:
            return next(iter(counter.values())) >= 1, "1 type 8 or more"
        
        # Counterのmost_common()で頻度の高い順に取り出し
        top_two = counter.most_common(2)
        # 1番多いものと2番目に多いものの出現回数を取り出す
        max_count = top_two[0][1]
        second_count = top_two[1][1]
        
        if 4 <= len(numbers) <= 7:
            return (max_count - second_count) >= 3, "4-7"
        elif 8 <= len(numbers) <= 11:
            return (max_count - second_count) >= 2, "8-11"
        elif 12 <= len(numbers) <= 16:
            return (max_count - second_count) >= 1, "12-16"
        else:
            return (max_count - second_count) >= 3, "else"
    
    def _run_engine(
            self, *, use_tqdm: bool
    ) -> List[Union[RequestOutput, PoolingRequestOutput]]:
        # Initialize tqdm.
        if use_tqdm:
            num_requests = self.llm_engine.get_num_unfinished_requests()
            pbar = tqdm(
                total=num_requests,
                desc="Processed prompts",
                dynamic_ncols=True,
                postfix=(f"est. speed input: {0:.2f} toks/s, "
                         f"output: {0:.2f} toks/s"),
            )

        # Run the engine.
        outputs: List[Union[RequestOutput, PoolingRequestOutput]] = []
        outputs_answers: List[int] = []
        total_in_toks = 0
        total_out_toks = 0
        
        # early_stop flag [-1, -1, -1, -1, -1, -1]
        check_too_many_invalid_1 = False
        check_too_many_invalid_2 = False
        check_too_many_invalid_3 = False

        check_divergence = False

        while self.llm_engine.has_unfinished_requests():
            step_outputs = self.llm_engine.step()

            # time out 4h 57min
            if time.time() - self.start_time > (4 * 60 + 57) * 60:
                print("STOP: time out")
                self.llm_engine.abort_all_requests()
                continue

            # early stop -1 or 0
            if not check_too_many_invalid_1 and len(outputs_answers) >= 5:
                check_too_many_invalid_1 = True
                if outputs_answers.count(-1) >= 5:
                    print("STOP: too many -1 gate1")
                    self.llm_engine.abort_all_requests()
                    continue

                if outputs_answers.count(0) >= 5:
                    print("STOP: too many 0 gate1")
                    self.llm_engine.abort_all_requests()
                    continue
                
            if not check_too_many_invalid_2 and len(outputs_answers) >= 8:
                check_too_many_invalid_2 = True
                if outputs_answers.count(-1) >= 4:
                    print("STOP: too many -1 gate2")
                    self.llm_engine.abort_all_requests()
                    continue

                if outputs_answers.count(0) >= 4:
                    print("STOP: too many 0 gate2")
                    self.llm_engine.abort_all_requests()
                    continue
            
            if not check_too_many_invalid_3 and len(outputs_answers) >= 12:
                check_too_many_invalid_3 = True
                if outputs_answers.count(-1) >= 6:
                    print("STOP: too many -1 gate3")
                    self.llm_engine.abort_all_requests()
                    continue

                if outputs_answers.count(0) >= 6:
                    print("STOP: too many 0 gate3")
                    self.llm_engine.abort_all_requests()
                    continue
            
            # early stop divergence
            if not check_divergence and len(outputs_answers) >= 8:
                check_divergence = True
                ans_st = set(outputs_answers)
                ans_st.discard(-1)
                if len(ans_st) >= 6:
                    print("STOP: divergence")
                    self.llm_engine.abort_all_requests()
                    continue
            
            # the winner is decided
            win, reason = self._winner_decided(outputs_answers)
            if win:
                print(f"STOP: winner decided, {reason}")
                self.llm_engine.abort_all_requests()
                continue
            
            for output in step_outputs:
                if output.finished:
                    outputs.append(output)
                    output_ans = self._get_answer(output.outputs[0].text)
                    outputs_answers.append(output_ans)

                    if use_tqdm:
                        if isinstance(output, RequestOutput):
                            # Calculate tokens only for RequestOutput
                            assert output.prompt_token_ids is not None
                            total_in_toks += len(output.prompt_token_ids)
                            in_spd = total_in_toks / pbar.format_dict["elapsed"]
                            total_out_toks += sum(
                                len(stp.token_ids) for stp in output.outputs)
                            out_spd = (total_out_toks /
                                       pbar.format_dict["elapsed"])
                            pbar.postfix = (
                                f"est. speed input: {in_spd:.2f} toks/s, "
                                f"output: {out_spd:.2f} toks/s")
                        pbar.update(1)

           
        if use_tqdm:
            pbar.close()
        # Sort the outputs by request ID.
        # This is necessary because some requests may be finished earlier than
        # its previous requests.
        return outputs_answers


%%writefile /kaggle/working/vllm/engine/llm_engine.py

import copy
import time
from collections import Counter as collectionsCounter
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from typing import (TYPE_CHECKING, Callable, ClassVar, Deque, Dict, Iterable,
                    List, Mapping, NamedTuple, Optional)
from typing import Sequence as GenericSequence
from typing import Set, Type, Union, cast, overload

import torch
from typing_extensions import TypeVar, deprecated

import vllm.envs as envs
from vllm.config import (DecodingConfig, LoRAConfig, ModelConfig,
                         ObservabilityConfig, ParallelConfig, SchedulerConfig,
                         VllmConfig)
from vllm.core.scheduler import (ScheduledSequenceGroup, Scheduler,
                                 SchedulerOutputs)
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.metrics_types import StatLoggerBase, Stats
from vllm.engine.output_processor.interfaces import (
    SequenceGroupOutputProcessor)
from vllm.engine.output_processor.stop_checker import StopChecker
from vllm.engine.output_processor.util import create_output_by_sequence_group
from vllm.entrypoints.openai.logits_processors import (
    get_logits_processors as get_openai_logits_processors)
from vllm.executor.executor_base import ExecutorBase
from vllm.inputs import (INPUT_REGISTRY, InputRegistry, ProcessorInputs,
                         PromptType, SingletonInputsAdapter)
from vllm.inputs.parse import is_encoder_decoder_inputs, is_token_prompt
from vllm.inputs.preprocess import InputPreprocessor
from vllm.logger import init_logger
from vllm.logits_process import get_bad_words_logits_processors
from vllm.lora.request import LoRARequest
from vllm.model_executor.guided_decoding import (
    get_local_guided_decoding_logits_processor)
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.outputs import (PoolingRequestOutput, RequestOutput,
                          RequestOutputFactory)
from vllm.pooling_params import PoolingParams
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.sequence import (ExecuteModelRequest, ParallelSampleSequenceGroup,
                           PoolingSequenceGroupOutput, Sequence, SequenceGroup,
                           SequenceGroupBase, SequenceGroupMetadata,
                           SequenceGroupOutput, SequenceStatus)
from vllm.tracing import (SpanAttributes, SpanKind, extract_trace_context,
                          init_tracer)
from vllm.transformers_utils.detokenizer import Detokenizer
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.transformers_utils.tokenizer_group import (
    BaseTokenizerGroup, init_tokenizer_from_configs)
from vllm.usage.usage_lib import (UsageContext, is_usage_stats_enabled,
                                  usage_message)
from vllm.utils import Counter, Device, deprecate_kwargs, weak_bind
from vllm.version import __version__ as VLLM_VERSION

logger = init_logger(__name__)
_LOCAL_LOGGING_INTERVAL_SEC = 5

_G = TypeVar("_G", bound=BaseTokenizerGroup, default=BaseTokenizerGroup)
_O = TypeVar("_O", RequestOutput, PoolingRequestOutput)


@dataclass
class SchedulerOutputState:
    """Caches the scheduler outputs for a virtual engine. Used for Multi-Step"""
    seq_group_metadata_list: Optional[List[SequenceGroupMetadata]] = None
    scheduler_outputs: Optional[SchedulerOutputs] = None
    allow_async_output_proc: bool = False
    last_output: Optional[SamplerOutput] = None


class OutputData(NamedTuple):
    outputs: List[SamplerOutput]
    seq_group_metadata_list: List[SequenceGroupMetadata]
    scheduler_outputs: SchedulerOutputs
    is_async: bool
    is_last_step: bool
    # Indicates if this output is from the first step of the
    # multi-step. When multi-step is disabled, this is always
    # set to True.
    # is_first_step_output is invalid when `outputs` has
    # outputs from multiple steps.
    is_first_step_output: Optional[bool]
    skip: List[int]


class SchedulerContext:

    def __init__(self, multi_step_stream_outputs: bool = False):
        self.output_queue: Deque[OutputData] = deque()
        self.request_outputs: List[Union[RequestOutput,
                                         PoolingRequestOutput]] = []
        self.seq_group_metadata_list: Optional[
            List[SequenceGroupMetadata]] = None
        self.scheduler_outputs: Optional[SchedulerOutputs] = None

        self.multi_step_stream_outputs: bool = multi_step_stream_outputs

    def append_output(self, outputs: List[SamplerOutput],
                      seq_group_metadata_list: List[SequenceGroupMetadata],
                      scheduler_outputs: SchedulerOutputs, is_async: bool,
                      is_last_step: bool,
                      is_first_step_output: Optional[bool]):
        self.output_queue.append(
            OutputData(outputs=outputs,
                       seq_group_metadata_list=seq_group_metadata_list,
                       scheduler_outputs=scheduler_outputs,
                       is_async=is_async,
                       is_last_step=is_last_step,
                       is_first_step_output=is_first_step_output,
                       skip=[]))


class LLMEngine:
    """An LLM engine that receives requests and generates texts.

    This is the main class for the vLLM engine. It receives requests
    from clients and generates texts from the LLM. It includes a tokenizer, a
    language model (possibly distributed across multiple GPUs), and GPU memory
    space allocated for intermediate states (aka KV cache). This class utilizes
    iteration-level scheduling and efficient memory management to maximize the
    serving throughput.

    The :class:`~vllm.LLM` class wraps this class for offline batched inference
    and the :class:`AsyncLLMEngine` class wraps this class for online serving.

    The config arguments are derived from :class:`~vllm.EngineArgs`. (See
    :ref:`engine-args`)

    Args:
        model_config: The configuration related to the LLM model.
        cache_config: The configuration related to the KV cache memory
            management.
        parallel_config: The configuration related to distributed execution.
        scheduler_config: The configuration related to the request scheduler.
        device_config: The configuration related to the device.
        lora_config (Optional): The configuration related to serving multi-LoRA.
        speculative_config (Optional): The configuration related to speculative
            decoding.
        executor_class: The model executor class for managing distributed
            execution.
        prompt_adapter_config (Optional): The configuration related to serving
            prompt adapters.
        log_stats: Whether to log statistics.
        usage_context: Specified entry point, used for usage info collection.
    """

    DO_VALIDATE_OUTPUT: ClassVar[bool] = False
    """A flag to toggle whether to validate the type of request output."""

    @classmethod
    @contextmanager
    def enable_output_validation(cls):
        cls.DO_VALIDATE_OUTPUT = True

        yield

        cls.DO_VALIDATE_OUTPUT = False

    @classmethod
    def validate_output(
        cls,
        output: object,
        output_type: Type[_O],
    ) -> _O:
        do_validate = cls.DO_VALIDATE_OUTPUT

        if ((TYPE_CHECKING or do_validate)
                and not isinstance(output, output_type)):
            raise TypeError(f"Expected output of type {output_type}, "
                            f"but found type {type(output)}")

        return cast(_O, output)

    @classmethod
    def validate_outputs(
        cls,
        outputs: GenericSequence[object],
        output_type: Type[_O],
    ) -> List[_O]:
        do_validate = cls.DO_VALIDATE_OUTPUT

        outputs_: List[_O]
        if TYPE_CHECKING or do_validate:
            outputs_ = []
            for output in outputs:
                if not isinstance(output, output_type):
                    raise TypeError(f"Expected output of type {output_type}, "
                                    f"but found type {type(output)}")

                outputs_.append(output)
        else:
            outputs_ = outputs

        return outputs_

    tokenizer: Optional[BaseTokenizerGroup]

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: Type[ExecutorBase],
        log_stats: bool,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: Optional[Dict[str, StatLoggerBase]] = None,
        input_registry: InputRegistry = INPUT_REGISTRY,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        use_cached_outputs: bool = False,
    ) -> None:

        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.device_config = vllm_config.device_config
        self.speculative_config = vllm_config.speculative_config  # noqa
        self.load_config = vllm_config.load_config
        self.decoding_config = vllm_config.decoding_config or DecodingConfig(  # noqa
        )
        self.prompt_adapter_config = vllm_config.prompt_adapter_config  # noqa
        self.observability_config = vllm_config.observability_config or ObservabilityConfig(  # noqa
        )

        logger.info(
            "Initializing a V0 LLM engine (v%s) with config: %s, "
            "use_cached_outputs=%s, ",
            VLLM_VERSION,
            vllm_config,
            use_cached_outputs,
        )

        self.log_stats = log_stats
        self.use_cached_outputs = use_cached_outputs

        if not self.model_config.skip_tokenizer_init:
            self.tokenizer = self._init_tokenizer()
            self.detokenizer = Detokenizer(self.tokenizer)
            tokenizer_group = self.get_tokenizer_group()
        else:
            self.tokenizer = None
            self.detokenizer = None
            tokenizer_group = None

        # Ensure that the function doesn't contain a reference to self,
        # to avoid engine GC issues
        def get_tokenizer_for_seq(sequence: Sequence) -> AnyTokenizer:
            assert tokenizer_group, ("tokenizer_group cannot be None, "
                                     "make sure skip_tokenizer_init is False")
            return tokenizer_group.get_lora_tokenizer(sequence.lora_request)

        self.seq_counter = Counter()
        self.generation_config_fields = (
            self.model_config.try_get_generation_config())

        self.input_preprocessor = InputPreprocessor(self.model_config,
                                                    self.tokenizer,
                                                    mm_registry)

        self.input_registry = input_registry
        self.input_processor = input_registry.create_input_processor(
            self.model_config)

        self.model_executor = executor_class(vllm_config=vllm_config, )

        if self.model_config.runner_type != "pooling":
            self._initialize_kv_caches()

        # If usage stat is enabled, collect relevant info.
        if is_usage_stats_enabled():
            from vllm.model_executor.model_loader import (
                get_architecture_class_name)
            usage_message.report_usage(
                get_architecture_class_name(self.model_config),
                usage_context,
                extra_kvs={
                    # Common configuration
                    "dtype":
                    str(self.model_config.dtype),
                    "tensor_parallel_size":
                    self.parallel_config.tensor_parallel_size,
                    "block_size":
                    self.cache_config.block_size,
                    "gpu_memory_utilization":
                    self.cache_config.gpu_memory_utilization,

                    # Quantization
                    "quantization":
                    self.model_config.quantization,
                    "kv_cache_dtype":
                    str(self.cache_config.cache_dtype),

                    # Feature flags
                    "enable_lora":
                    bool(self.lora_config),
                    "enable_prompt_adapter":
                    bool(self.prompt_adapter_config),
                    "enable_prefix_caching":
                    self.cache_config.enable_prefix_caching,
                    "enforce_eager":
                    self.model_config.enforce_eager,
                    "disable_custom_all_reduce":
                    self.parallel_config.disable_custom_all_reduce,
                })

        if self.tokenizer:
            # Ping the tokenizer to ensure liveness if it runs in a
            # different process.
            self.tokenizer.ping()

        self.cached_scheduler_outputs = [
            SchedulerOutputState()
            for _ in range(self.parallel_config.pipeline_parallel_size)
        ]

        self.scheduler_contexts = [
            SchedulerContext(multi_step_stream_outputs=self.scheduler_config.
                             multi_step_stream_outputs)
            for _ in range(self.parallel_config.pipeline_parallel_size)
        ]

        if self.model_config.use_async_output_proc:
            process_model_outputs = weak_bind(self._process_model_outputs)

            self.async_callbacks = [
                partial(process_model_outputs,
                        ctx=self.scheduler_contexts[v_id])
                for v_id in range(self.parallel_config.pipeline_parallel_size)
            ]
        else:
            self.async_callbacks = []

        # Currently used by AsyncLLMEngine to ensure quick append
        # of request outputs to asyncio queues
        self.process_request_outputs_callback: Optional[Callable] = None

        # Create the scheduler.
        # NOTE: the cache_config here have been updated with the numbers of
        # GPU and CPU blocks, which are profiled in the distributed executor.
        self.scheduler = [
            Scheduler(
                self.scheduler_config, self.cache_config, self.lora_config,
                self.parallel_config.pipeline_parallel_size,
                self.async_callbacks[v_id]
                if self.model_config.use_async_output_proc else None)
            for v_id in range(self.parallel_config.pipeline_parallel_size)
        ]

        # Metric Logging.
        if self.log_stats:
            if stat_loggers is not None:
                self.stat_loggers = stat_loggers
            else:
                # Lazy import for prometheus multiprocessing.
                # We need to set PROMETHEUS_MULTIPROC_DIR environment variable
                # before prometheus_client is imported.
                # See https://prometheus.github.io/client_python/multiprocess/
                from vllm.engine.metrics import (LoggingStatLogger,
                                                 PrometheusStatLogger)

                self.stat_loggers = {
                    "logging":
                    LoggingStatLogger(
                        local_interval=_LOCAL_LOGGING_INTERVAL_SEC,
                        vllm_config=vllm_config),
                    "prometheus":
                    PrometheusStatLogger(
                        local_interval=_LOCAL_LOGGING_INTERVAL_SEC,
                        labels=dict(
                            model_name=self.model_config.served_model_name),
                        vllm_config=vllm_config),
                }
                self.stat_loggers["prometheus"].info("cache_config",
                                                     self.cache_config)

        self.tracer = None
        if self.observability_config.otlp_traces_endpoint:
            self.tracer = init_tracer(
                "vllm.llm_engine",
                self.observability_config.otlp_traces_endpoint)

        # Create sequence output processor, e.g. for beam search or
        # speculative decoding.
        self.output_processor = (
            SequenceGroupOutputProcessor.create_output_processor(
                self.scheduler_config,
                self.detokenizer,
                self.scheduler,
                self.seq_counter,
                get_tokenizer_for_seq,
                stop_checker=StopChecker(
                    self.scheduler_config.max_model_len,
                    get_tokenizer_for_seq,
                ),
            ))

        self.seq_id_to_seq_group: Dict[str, SequenceGroupBase] = {}

    def _initialize_kv_caches(self) -> None:
        """Initialize the KV cache in the worker(s).

        The workers will determine the number of blocks in both the GPU cache
        and the swap CPU cache.
        """
        start = time.time()
        num_gpu_blocks, num_cpu_blocks = (
            self.model_executor.determine_num_available_blocks())

        if self.cache_config.num_gpu_blocks_override is not None:
            num_gpu_blocks_override = self.cache_config.num_gpu_blocks_override
            logger.info(
                "Overriding num_gpu_blocks=%d with "
                "num_gpu_blocks_override=%d", num_gpu_blocks,
                num_gpu_blocks_override)
            num_gpu_blocks = num_gpu_blocks_override

        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

        self.model_executor.initialize_cache(num_gpu_blocks, num_cpu_blocks)
        elapsed = time.time() - start
        logger.info(("init engine (profile, create kv cache, "
                     "warmup model) took %.2f seconds"), elapsed)

    @classmethod
    def _get_executor_cls(cls,
                          engine_config: VllmConfig) -> Type[ExecutorBase]:
        distributed_executor_backend = (
            engine_config.parallel_config.distributed_executor_backend)
        # Initialize the cluster and specify the executor class.
        if isinstance(distributed_executor_backend, type):
            if not issubclass(distributed_executor_backend, ExecutorBase):
                raise TypeError(
                    "distributed_executor_backend must be a subclass of "
                    f"ExecutorBase. Got {distributed_executor_backend}.")
            executor_class = distributed_executor_backend
        elif engine_config.parallel_config.world_size > 1:
            if distributed_executor_backend == "ray":
                from vllm.executor.ray_distributed_executor import (
                    RayDistributedExecutor)
                executor_class = RayDistributedExecutor
            elif distributed_executor_backend == "mp":
                from vllm.executor.mp_distributed_executor import (
                    MultiprocessingDistributedExecutor)
                assert not envs.VLLM_USE_RAY_SPMD_WORKER, (
                    "multiprocessing distributed executor backend does not "
                    "support VLLM_USE_RAY_SPMD_WORKER=1")
                executor_class = MultiprocessingDistributedExecutor
            elif distributed_executor_backend == "uni":
                # JAX-style, single-process, multi-device executor.
                from vllm.executor.uniproc_executor import UniProcExecutor
                executor_class = UniProcExecutor
            elif distributed_executor_backend == "external_launcher":
                # executor with external launcher
                from vllm.executor.uniproc_executor import (  # noqa
                    ExecutorWithExternalLauncher)
                executor_class = ExecutorWithExternalLauncher
        else:
            from vllm.executor.uniproc_executor import UniProcExecutor
            executor_class = UniProcExecutor
        return executor_class

    @classmethod
    def from_engine_args(
        cls,
        engine_args: EngineArgs,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: Optional[Dict[str, StatLoggerBase]] = None,
    ) -> "LLMEngine":
        """Creates an LLM engine from the engine arguments."""
        # Create the engine configs.
        engine_config = engine_args.create_engine_config(usage_context)
        executor_class = cls._get_executor_cls(engine_config)
        # Create the LLM engine.
        engine = cls(
            vllm_config=engine_config,
            executor_class=executor_class,
            log_stats=not engine_args.disable_log_stats,
            usage_context=usage_context,
            stat_loggers=stat_loggers,
        )

        return engine

    def __reduce__(self):
        # This is to ensure that the LLMEngine is not referenced in
        # the closure used to initialize Ray worker actors
        raise RuntimeError("LLMEngine should not be pickled!")

    def __del__(self):
        # Shutdown model executor when engine is garbage collected
        # Use getattr since __init__ can fail before the field is set
        if model_executor := getattr(self, "model_executor", None):
            model_executor.shutdown()

    def get_tokenizer_group(
        self,
        group_type: Type[_G] = BaseTokenizerGroup,
    ) -> _G:
        tokenizer_group = self.tokenizer

        if tokenizer_group is None:
            raise ValueError("Unable to get tokenizer because "
                             "skip_tokenizer_init is True")
        if not isinstance(tokenizer_group, group_type):
            raise TypeError("Invalid type of tokenizer group. "
                            f"Expected type: {group_type}, but "
                            f"found type: {type(tokenizer_group)}")

        return tokenizer_group

    def get_tokenizer(
        self,
        lora_request: Optional[LoRARequest] = None,
    ) -> AnyTokenizer:
        return self.get_tokenizer_group().get_lora_tokenizer(lora_request)

    def _init_tokenizer(self) -> BaseTokenizerGroup:
        return init_tokenizer_from_configs(
            model_config=self.model_config,
            scheduler_config=self.scheduler_config,
            parallel_config=self.parallel_config,
            lora_config=self.lora_config)

    def _verify_args(self) -> None:
        self.model_config.verify_with_parallel_config(self.parallel_config)
        self.cache_config.verify_with_parallel_config(self.parallel_config)
        if self.lora_config:
            self.lora_config.verify_with_model_config(self.model_config)
            self.lora_config.verify_with_scheduler_config(
                self.scheduler_config)
        if self.prompt_adapter_config:
            self.prompt_adapter_config.verify_with_model_config(
                self.model_config)

    def _add_processed_request(
        self,
        request_id: str,
        processed_inputs: ProcessorInputs,
        params: Union[SamplingParams, PoolingParams],
        arrival_time: float,
        lora_request: Optional[LoRARequest],
        prompt_adapter_request: Optional[PromptAdapterRequest],
        trace_headers: Optional[Mapping[str, str]] = None,
        priority: int = 0,
    ) -> Optional[SequenceGroup]:
        """Add a processed request to the engine's request pool.
        return the created sequence group.
        """
        if isinstance(params, SamplingParams) and params.n > 1:
            ParallelSampleSequenceGroup.add_request(
                request_id,
                self,
                params,
                processed_inputs=processed_inputs,
                arrival_time=arrival_time,
                lora_request=lora_request,
                trace_headers=trace_headers,
                prompt_adapter_request=prompt_adapter_request,
                priority=priority,
            )
            return None

        self._validate_model_inputs(processed_inputs, lora_request)
        # Create the sequences.
        block_size = self.cache_config.block_size
        seq_id = next(self.seq_counter)
        eos_token_id = self.input_preprocessor.get_eos_token_id(lora_request)

        if is_encoder_decoder_inputs(processed_inputs):
            decoder_inputs = processed_inputs["decoder"]
            encoder_inputs = processed_inputs["encoder"]
        else:
            decoder_inputs = processed_inputs
            encoder_inputs = None

        seq = Sequence(seq_id, decoder_inputs, block_size, eos_token_id,
                       lora_request, prompt_adapter_request)

        encoder_seq = (None if encoder_inputs is None else Sequence(
            seq_id, encoder_inputs, block_size, eos_token_id, lora_request,
            prompt_adapter_request))

        # Create a SequenceGroup based on SamplingParams or PoolingParams
        if isinstance(params, SamplingParams):
            seq_group = self._create_sequence_group_with_sampling(
                request_id,
                seq,
                params,
                arrival_time=arrival_time,
                lora_request=lora_request,
                trace_headers=trace_headers,
                prompt_adapter_request=prompt_adapter_request,
                encoder_seq=encoder_seq,
                priority=priority)
        elif isinstance(params, PoolingParams):
            seq_group = self._create_sequence_group_with_pooling(
                request_id,
                seq,
                params,
                arrival_time=arrival_time,
                lora_request=lora_request,
                prompt_adapter_request=prompt_adapter_request,
                encoder_seq=encoder_seq,
                priority=priority)
        else:
            raise ValueError(
                "Either SamplingParams or PoolingParams must be provided.")

        # Add the sequence group to the scheduler with least unfinished seqs.
        costs = [
            scheduler.get_num_unfinished_seq_groups()
            for scheduler in self.scheduler
        ]
        min_cost_scheduler = self.scheduler[costs.index(min(costs))]
        min_cost_scheduler.add_seq_group(seq_group)

        return seq_group

    def stop_remote_worker_execution_loop(self) -> None:
        self.model_executor.stop_remote_worker_execution_loop()

    @overload
    def add_request(
        self,
        request_id: str,
        prompt: PromptType,
        params: Union[SamplingParams, PoolingParams],
        arrival_time: Optional[float] = None,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        priority: int = 0,
    ) -> None:
        ...

    @overload
    @deprecated("'inputs' will be renamed to 'prompt")
    def add_request(
        self,
        request_id: str,
        *,
        inputs: PromptType,
        params: Union[SamplingParams, PoolingParams],
        arrival_time: Optional[float] = None,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        priority: int = 0,
    ) -> None:
        ...

    @deprecate_kwargs(
        "inputs",
        additional_message="Please use the 'prompt' parameter instead.",
    )
    def add_request(
            self,
            request_id: str,
            prompt: Optional[PromptType] = None,
            params: Optional[Union[SamplingParams, PoolingParams]] = None,
            arrival_time: Optional[float] = None,
            lora_request: Optional[LoRARequest] = None,
            trace_headers: Optional[Mapping[str, str]] = None,
            prompt_adapter_request: Optional[PromptAdapterRequest] = None,
            priority: int = 0,
            *,
            inputs: Optional[PromptType] = None,  # DEPRECATED
    ) -> None:
        """Add a request to the engine's request pool.

        The request is added to the request pool and will be processed by the
        scheduler as `engine.step()` is called. The exact scheduling policy is
        determined by the scheduler.

        Args:
            request_id: The unique ID of the request.
            prompt: The prompt to the LLM. See :class:`~vllm.inputs.PromptType`
                for more details about the format of each input.
            params: Parameters for sampling or pooling.
                :class:`~vllm.SamplingParams` for text generation.
                :class:`~vllm.PoolingParams` for pooling.
            arrival_time: The arrival time of the request. If None, we use
                the current monotonic time.
            lora_request: The LoRA request to add.
            trace_headers: OpenTelemetry trace headers.
            prompt_adapter_request: The prompt adapter request to add.
            priority: The priority of the request.
                Only applicable with priority scheduling.

        Details:
            - Set arrival_time to the current time if it is None.
            - Set prompt_token_ids to the encoded prompt if it is None.
            - Create `n` number of :class:`~vllm.Sequence` objects.
            - Create a :class:`~vllm.SequenceGroup` object
              from the list of :class:`~vllm.Sequence`.
            - Add the :class:`~vllm.SequenceGroup` object to the scheduler.

        Example:
            >>> # initialize engine
            >>> engine = LLMEngine.from_engine_args(engine_args)
            >>> # set request arguments
            >>> example_prompt = "Who is the president of the United States?"
            >>> sampling_params = SamplingParams(temperature=0.0)
            >>> request_id = 0
            >>>
            >>> # add the request to the engine
            >>> engine.add_request(
            >>>    str(request_id),
            >>>    example_prompt,
            >>>    SamplingParams(temperature=0.0))
            >>> # continue the request processing
            >>> ...
        """
        if inputs is not None:
            prompt = inputs
        assert prompt is not None and params is not None

        if lora_request is not None and not self.lora_config:
            raise ValueError(f"Got lora_request {lora_request} but LoRA is "
                             "not enabled!")

        if priority != 0 and not self.scheduler_config.policy == "priority":
            raise ValueError(f"Got priority {priority} but "
                             "Priority scheduling is not enabled.")

        if isinstance(params, SamplingParams) \
            and (params.guided_decoding or params.logits_processors) \
            and self.scheduler_config.num_scheduler_steps > 1:
            raise ValueError(
                "Guided decoding and logits processors are not supported "
                "in multi-step decoding")

        if arrival_time is None:
            arrival_time = time.time()

        if self.tokenizer is not None:
            self._validate_token_prompt(
                prompt,
                tokenizer=self.get_tokenizer(lora_request=lora_request))

        preprocessed_inputs = self.input_preprocessor.preprocess(
            prompt,
            request_id=request_id,
            lora_request=lora_request,
            prompt_adapter_request=prompt_adapter_request,
        )
        processed_inputs = self.input_processor(preprocessed_inputs)

        self._add_processed_request(
            request_id=request_id,
            processed_inputs=processed_inputs,
            params=params,
            arrival_time=arrival_time,
            lora_request=lora_request,
            prompt_adapter_request=prompt_adapter_request,
            trace_headers=trace_headers,
            priority=priority,
        )

    def _validate_token_prompt(self, prompt: PromptType,
                               tokenizer: AnyTokenizer):
        # Guard against out-of-vocab tokens.
        # For some tokenizers, tokenizer.decode will happily return empty text
        # for token ids that are out of vocab, and we don't detect token ids
        # that are greater than the max token id before running the model.
        # However, these token ids will later crash a cuda kernel at runtime
        # with an index out of bounds error. This will crash the entire engine.
        # This needs to happen before multimodal input pre-processing, which
        # may add dummy <image> tokens that aren't part of the tokenizer's
        # vocabulary.
        if is_token_prompt(prompt):
            prompt_ids = prompt["prompt_token_ids"]
            if len(prompt_ids) == 0:
                # Empty prompt check is handled later
                return
            max_input_id = max(prompt_ids)
            if max_input_id > tokenizer.max_token_id:
                raise ValueError(
                    "Token id {} is out of vocabulary".format(max_input_id))

    def _create_sequence_group_with_sampling(
        self,
        request_id: str,
        seq: Sequence,
        sampling_params: SamplingParams,
        arrival_time: float,
        lora_request: Optional[LoRARequest],
        trace_headers: Optional[Mapping[str, str]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        encoder_seq: Optional[Sequence] = None,
        priority: int = 0,
    ) -> SequenceGroup:
        """Creates a SequenceGroup with SamplingParams."""
        max_logprobs = self.get_model_config().max_logprobs
        if (sampling_params.logprobs
                and sampling_params.logprobs > max_logprobs) or (
                    sampling_params.prompt_logprobs
                    and sampling_params.prompt_logprobs > max_logprobs):
            raise ValueError(f"Cannot request more than "
                             f"{max_logprobs} logprobs.")

        sampling_params = self._build_logits_processors(
            sampling_params, lora_request)

        # Defensive copy of SamplingParams, which are used by the sampler,
        # this doesn't deep-copy LogitsProcessor objects
        sampling_params = sampling_params.clone()

        sampling_params.update_from_generation_config(
            self.generation_config_fields, seq.eos_token_id)

        # Create the sequence group.
        seq_group = SequenceGroup(
            request_id=request_id,
            seqs=[seq],
            arrival_time=arrival_time,
            sampling_params=sampling_params,
            lora_request=lora_request,
            trace_headers=trace_headers,
            prompt_adapter_request=prompt_adapter_request,
            encoder_seq=encoder_seq,
            priority=priority)

        return seq_group

    def _create_sequence_group_with_pooling(
        self,
        request_id: str,
        seq: Sequence,
        pooling_params: PoolingParams,
        arrival_time: float,
        lora_request: Optional[LoRARequest],
        prompt_adapter_request: Optional[PromptAdapterRequest],
        encoder_seq: Optional[Sequence] = None,
        priority: int = 0,
    ) -> SequenceGroup:
        """Creates a SequenceGroup with PoolingParams."""
        # Defensive copy of PoolingParams, which are used by the pooler
        pooling_params = pooling_params.clone()
        # Create the sequence group.
        seq_group = SequenceGroup(
            request_id=request_id,
            seqs=[seq],
            arrival_time=arrival_time,
            lora_request=lora_request,
            pooling_params=pooling_params,
            prompt_adapter_request=prompt_adapter_request,
            encoder_seq=encoder_seq,
            priority=priority)
        return seq_group

    def abort_all_requests(self) -> None:
        """
        Abort (forcefully remove) all unfinished requests from all schedulers,
        as well as clear any pending outputs in the output queues.
        """
        # 1) Abort all requests in each Scheduler
        for scheduler in self.scheduler:
            # running / waiting / swapped にいる SequenceGroup を一括で取り出す
            request_ids = []
            for running_req in scheduler.running:
                request_ids.append(running_req.request_id)
            for waiting_req in scheduler.waiting:
                request_ids.append(waiting_req.request_id)
            for swapped_req in scheduler.swapped:
                request_ids.append(swapped_req.request_id)
    
            # 重複IDを除いた上で abort
            unique_ids = set(request_ids)
            for rid in unique_ids:
                scheduler.abort_seq_group(rid)
    
        # 2) scheduler_context の出力キューも一括クリアする
        for ctx in self.scheduler_contexts:
            ctx.output_queue.clear()
            ctx.request_outputs.clear()
    
        # 3) cached_scheduler_outputs も消しておく (multi-step対策)
        for i in range(len(self.cached_scheduler_outputs)):
            self.cached_scheduler_outputs[i] = SchedulerOutputState()

    def abort_request(self, request_id: Union[str, Iterable[str]]) -> None:
        """Aborts a request(s) with the given ID.

        Args:
            request_id: The ID(s) of the request to abort.

        Details:
            - Refer to the
              :meth:`~vllm.core.scheduler.Scheduler.abort_seq_group`
              from class :class:`~vllm.core.scheduler.Scheduler`.

        Example:
            >>> # initialize engine and add a request with request_id
            >>> request_id = str(0)
            >>> # abort the request
            >>> engine.abort_request(request_id)
        """
        for scheduler in self.scheduler:
            scheduler.abort_seq_group(request_id)

    def get_model_config(self) -> ModelConfig:
        """Gets the model configuration."""
        return self.model_config

    def get_parallel_config(self) -> ParallelConfig:
        """Gets the parallel configuration."""
        return self.parallel_config

    def get_decoding_config(self) -> DecodingConfig:
        """Gets the decoding configuration."""
        return self.decoding_config

    def get_scheduler_config(self) -> SchedulerConfig:
        """Gets the scheduler configuration."""
        return self.scheduler_config

    def get_lora_config(self) -> LoRAConfig:
        """Gets the LoRA configuration."""
        return self.lora_config

    def get_num_unfinished_requests(self) -> int:
        """Gets the number of unfinished requests."""
        return sum(scheduler.get_num_unfinished_seq_groups()
                   for scheduler in self.scheduler)

    def has_unfinished_requests(self) -> bool:
        """Returns True if there are unfinished requests."""
        return any(scheduler.has_unfinished_seqs()
                   for scheduler in self.scheduler)

    def has_unfinished_requests_for_virtual_engine(
            self, virtual_engine: int) -> bool:
        """
        Returns True if there are unfinished requests for the virtual engine.
        """
        return self.scheduler[virtual_engine].has_unfinished_seqs()

    def reset_prefix_cache(self) -> bool:
        """Reset prefix cache for all devices."""

        success = True
        for scheduler in self.scheduler:
            success = success and scheduler.reset_prefix_cache()
        return success

    @staticmethod
    def _process_sequence_group_outputs(
        seq_group: SequenceGroup,
        outputs: List[PoolingSequenceGroupOutput],
    ) -> None:
        seq_group.pooled_data = outputs[0].data

        for seq in seq_group.get_seqs():
            seq.status = SequenceStatus.FINISHED_STOPPED

        return

    def _update_num_computed_tokens_for_multi_step_prefill(
            self, seq_group: SequenceGroup,
            seq_group_meta: SequenceGroupMetadata,
            is_first_step_output: Optional[bool]):
        """
        This function updates num_computed_tokens for prompt sequences
        when Multi-Step is enabled.

        seq_group: SequenceGroup to update the num_computed_tokens for.
        seq_group_meta: Metadata of the given SequenceGroup.
        is_first_step_output: Optional[bool] -
            When available, is_first_step_output indicates if the appended
            output token is the output of the first-step in multi-step.
            A value of None indicates that outputs from all steps in
            in multi-step are submitted in a single burst.
        """

        assert self.scheduler_config.is_multi_step

        if not seq_group_meta.is_prompt:
            # num_computed_token updates for multi-step decodes happen after
            # the tokens are appended to the sequence.
            return

        do_update: bool = False
        if self.scheduler_config.chunked_prefill_enabled:
            # In multi-step + chunked-prefill case, the prompt sequences
            # that are scheduled are fully processed in the first step.
            do_update = is_first_step_output is None or is_first_step_output
        else:
            # Normal multi-step decoding case. In this case prompt-sequences
            # are actually single-stepped. Always update in this case.
            assert seq_group.state.num_steps == 1
            do_update = True

        if do_update:
            seq_group.update_num_computed_tokens(
                seq_group_meta.token_chunk_size)

    def _process_model_outputs(self,
                               ctx: SchedulerContext,
                               request_id: Optional[str] = None) -> None:
        """Apply the model output to the sequences in the scheduled seq groups
        and return responses.

        ctx: The virtual engine context to work on
        request_id: If provided, then only this request is going to be processed
        """

        now = time.time()

        if len(ctx.output_queue) == 0:
            return None

        # Get pending async postprocessor
        if request_id:
            # When we process only one request, no pop is required
            # (since later we will process all of the rest)
            (outputs, seq_group_metadata_list, scheduler_outputs, is_async,
             is_last_step, is_first_step_output, skip) = ctx.output_queue[0]
        else:
            (outputs, seq_group_metadata_list, scheduler_outputs, is_async,
             is_last_step, is_first_step_output,
             skip) = ctx.output_queue.popleft()

        # Sanity check
        assert len(seq_group_metadata_list) == len(
            scheduler_outputs.scheduled_seq_groups)

        has_multiple_outputs: bool = len(outputs) > 1
        outputs_by_sequence_group: List[List[SequenceGroupOutput]]
        if has_multiple_outputs:
            assert self.scheduler_config.is_multi_step or \
                     self.speculative_config
            # Organize outputs by [step][sequence group] instead of
            # [sequence group][step].
            if self.scheduler_config.is_multi_step:
                outputs_by_sequence_group = create_output_by_sequence_group(
                    outputs, len(seq_group_metadata_list))
            elif self.speculative_config:
                # Decodes are multi-steps while prefills are not, outputting at
                # most 1 token. Separate them so that we can trigger chunk
                # processing without having to pad or copy over prompts K times
                # to match decodes structure (costly with prompt_logprobs).
                num_prefills = sum(sg.is_prompt
                                   for sg in seq_group_metadata_list)
                prefills, decodes = outputs[:num_prefills], outputs[
                    num_prefills:]
                outputs_by_sequence_group = create_output_by_sequence_group(
                    decodes,
                    num_seq_groups=len(seq_group_metadata_list) - num_prefills)
                outputs_by_sequence_group = [p.outputs for p in prefills
                                             ] + outputs_by_sequence_group
            # We have outputs for multiple steps submitted in a single burst,
            # so invalidate is_first_step_output.
            is_first_step_output = None
        else:
            outputs_by_sequence_group = outputs

        # Determine the requests we need to operate on
        if request_id:
            indices = []
            for i, seq_group_meta in enumerate(seq_group_metadata_list):
                if seq_group_meta.request_id == request_id:
                    assert i not in skip  # Cannot be called twice
                    indices.append(i)
                    break

            # If the request_id was not found, then it means that
            # this is a new request that has no pending async
            # postprocessor
            if not indices:
                return
        else:
            indices = range(len(seq_group_metadata_list))  # type: ignore

        finished_before: List[int] = []
        finished_now: List[int] = []
        for i in indices:
            if i in skip:
                continue

            seq_group_meta = seq_group_metadata_list[i]
            scheduled_seq_group = scheduler_outputs.scheduled_seq_groups[i]

            seq_group: SequenceGroup = scheduled_seq_group.seq_group

            if seq_group.is_finished():
                finished_before.append(i)
                continue

            output: List[SequenceGroupOutput]
            if has_multiple_outputs:
                output = outputs_by_sequence_group[i]
            else:
                output = [outputs_by_sequence_group[0][i]]

            if not is_async:
                if self.scheduler_config.is_multi_step:
                    # Updates happen only if the sequence is prefill
                    self._update_num_computed_tokens_for_multi_step_prefill(
                        seq_group, seq_group_meta, is_first_step_output)
                else:
                    seq_group.update_num_computed_tokens(
                        seq_group_meta.token_chunk_size or 0)

            if outputs:
                for o in outputs:
                    if (isinstance(o, SamplerOutput)
                            and seq_group.metrics is not None):
                        if seq_group.metrics.model_forward_time is not None:
                            seq_group.metrics.model_forward_time += (
                                o.model_forward_time or 0)
                        else:
                            seq_group.metrics.model_forward_time = (
                                o.model_forward_time)
                        if seq_group.metrics.model_execute_time is not None:
                            seq_group.metrics.model_execute_time += (
                                o.model_execute_time or 0)
                        else:
                            seq_group.metrics.model_execute_time = (
                                o.model_execute_time)

            if self.model_config.runner_type == "pooling":
                self._process_sequence_group_outputs(seq_group, output)
            else:
                self.output_processor.process_prompt_logprob(seq_group, output)
                if seq_group_meta.do_sample:
                    self.output_processor.process_outputs(
                        seq_group, output, is_async)

            if seq_group.is_finished():
                finished_now.append(i)

        # Generate outputs for the requests that finished this iteration
        for i in finished_now:
            scheduled_seq_group = scheduler_outputs.scheduled_seq_groups[i]

            seq_group = scheduled_seq_group.seq_group
            seq_group.maybe_set_first_token_time(now)
            if not seq_group.is_prefill():
                seq_group.set_last_token_time(now)
            request_output = RequestOutputFactory.create(
                seq_group,
                self.seq_id_to_seq_group,
                use_cache=self.use_cached_outputs)
            if request_output:
                ctx.request_outputs.append(request_output)

        # When we process a single request, we skip it for the next time,
        # and invoke the request output callback (if there was final output)
        if request_id:
            assert len(indices) == 1
            skip.append(indices[0])

            if (finished_now
                    and self.process_request_outputs_callback is not None):
                self.process_request_outputs_callback(ctx.request_outputs)
                ctx.request_outputs.clear()
            return

        # Free currently finished requests
        if finished_now:
            for scheduler in self.scheduler:
                scheduler.free_finished_seq_groups()

        # For multi-step without streaming, don't create outputs each iteration
        if not is_last_step and not ctx.multi_step_stream_outputs:
            # Immediately process request outputs here (if callback is given)
            if (finished_now
                    and self.process_request_outputs_callback is not None):
                self.process_request_outputs_callback(ctx.request_outputs)
                ctx.request_outputs.clear()
            return

        # Create the outputs
        for i in indices:
            if i in skip or i in finished_before or i in finished_now:
                continue  # Avoids double processing

            scheduled_seq_group = scheduler_outputs.scheduled_seq_groups[i]

            seq_group = scheduled_seq_group.seq_group
            seq_group.maybe_set_first_token_time(now)
            if not seq_group.is_prefill():
                seq_group.set_last_token_time(now)
            request_output = RequestOutputFactory.create(
                seq_group,
                self.seq_id_to_seq_group,
                use_cache=self.use_cached_outputs)
            if request_output:
                ctx.request_outputs.append(request_output)

        # For multi-step with streaming, create outputs each iteration
        if not is_last_step and ctx.multi_step_stream_outputs:
            # Immediately process request outputs here (if callback is given)
            if self.process_request_outputs_callback is not None:
                self.process_request_outputs_callback(ctx.request_outputs)
                ctx.request_outputs.clear()
            return

        for seq_group in scheduler_outputs.ignored_seq_groups:
            params = seq_group.sampling_params
            if params is not None and params.output_kind == (
                    RequestOutputKind.DELTA) and not seq_group.is_finished():
                continue

            request_output = RequestOutputFactory.create(
                seq_group,
                self.seq_id_to_seq_group,
                use_cache=self.use_cached_outputs,
            )
            if request_output:
                ctx.request_outputs.append(request_output)

        # Immediately process request outputs here (if callback is given)
        if (ctx.request_outputs
                and self.process_request_outputs_callback is not None):
            self.process_request_outputs_callback(ctx.request_outputs)
            ctx.request_outputs.clear()

        # For async case, we need to record the stats here.
        # For non-async case, the stats are done in the
        # LLMEngine/AsyncLLMEngine directly
        if is_async:
            # Log stats.
            self.do_log_stats(scheduler_outputs, outputs, finished_before,
                              skip)

            # Tracing
            self.do_tracing(scheduler_outputs, finished_before)

        return None

    def _advance_to_next_step(
            self, output: List[SamplerOutput],
            seq_group_metadata_list: List[SequenceGroupMetadata],
            scheduled_seq_groups: List[ScheduledSequenceGroup]) -> None:
        """Given model output from a single run, append the tokens to the
        sequences. This is normally done inside output processor, but it is
        required if the worker is to perform async forward pass to next step.
        """
        for seq_group_metadata, sequence_group_outputs, scheduled_seq_group in \
            zip(seq_group_metadata_list, output, scheduled_seq_groups):
            seq_group = scheduled_seq_group.seq_group

            if seq_group.is_finished():
                continue

            if self.scheduler_config.is_multi_step:
                # Updates happen only if the sequence is prefill
                self._update_num_computed_tokens_for_multi_step_prefill(
                    seq_group, seq_group_metadata,
                    seq_group.state.num_steps == 1)
            else:
                token_chunk_size = (seq_group_metadata.token_chunk_size
                                    if seq_group_metadata.token_chunk_size
                                    is not None else 0)
                seq_group.update_num_computed_tokens(token_chunk_size)

            if seq_group_metadata.do_sample:
                assert len(sequence_group_outputs.samples) == 1, (
                    "Async output processor expects a single sample"
                    " (i.e sampling_params.n == 1)")
                sample = sequence_group_outputs.samples[0]

                assert len(seq_group.seqs) == 1
                seq = seq_group.seqs[0]

                if self.scheduler_config.is_multi_step:
                    is_prefill_append = seq.data.get_num_uncomputed_tokens(
                    ) == 0
                    seq.append_token_id(sample.output_token, sample.logprobs)
                    if not is_prefill_append:
                        seq_group.update_num_computed_tokens(1)
                else:
                    seq.append_token_id(sample.output_token, sample.logprobs)

    def step(self) -> List[Union[RequestOutput, PoolingRequestOutput]]:
        """Performs one decoding iteration and returns newly generated results.

        .. figure:: https://i.imgur.com/sv2HssD.png
            :alt: Overview of the step function
            :align: center

            Overview of the step function.

        Details:
            - Step 1: Schedules the sequences to be executed in the next
              iteration and the token blocks to be swapped in/out/copy.

                - Depending on the scheduling policy,
                  sequences may be `preempted/reordered`.
                - A Sequence Group (SG) refer to a group of sequences
                  that are generated from the same prompt.

            - Step 2: Calls the distributed executor to execute the model.
            - Step 3: Processes the model output. This mainly includes:

                - Decodes the relevant outputs.
                - Updates the scheduled sequence groups with model outputs
                  based on its `sampling parameters` (`use_beam_search` or not).
                - Frees the finished sequence groups.

            - Finally, it creates and returns the newly generated results.

        Example:
            >>> # Please see the example/ folder for more detailed examples.
            >>>
            >>> # initialize engine and request arguments
            >>> engine = LLMEngine.from_engine_args(engine_args)
            >>> example_inputs = [(0, "What is LLM?",
            >>>    SamplingParams(temperature=0.0))]
            >>>
            >>> # Start the engine with an event loop
            >>> while True:
            >>>     if example_inputs:
            >>>         req_id, prompt, sampling_params = example_inputs.pop(0)
            >>>         engine.add_request(str(req_id),prompt,sampling_params)
            >>>
            >>>     # continue the request processing
            >>>     request_outputs = engine.step()
            >>>     for request_output in request_outputs:
            >>>         if request_output.finished:
            >>>             # return or show the request output
            >>>
            >>>     if not (engine.has_unfinished_requests() or example_inputs):
            >>>         break
        """
        if self.parallel_config.pipeline_parallel_size > 1:
            raise NotImplementedError(
                "Pipeline parallelism is only supported through AsyncLLMEngine "
                "as performance will be severely degraded otherwise.")

        # For llm_engine, there is no pipeline parallel support, so the engine
        # used is always 0.
        virtual_engine = 0

        # These are cached outputs from previous iterations. None if on first
        # iteration
        cached_outputs = self.cached_scheduler_outputs[virtual_engine]
        seq_group_metadata_list = cached_outputs.seq_group_metadata_list
        scheduler_outputs = cached_outputs.scheduler_outputs
        allow_async_output_proc = cached_outputs.allow_async_output_proc

        ctx = self.scheduler_contexts[virtual_engine]

        # Clear outputs for each new scheduler iteration
        ctx.request_outputs.clear()

        # Skip the scheduler if there are any remaining steps in the seq groups.
        # This ensures that the scheduler is only called again when the current
        # batch has completed.
        if not self._has_remaining_steps(seq_group_metadata_list):
            # Schedule iteration
            (seq_group_metadata_list, scheduler_outputs,
             allow_async_output_proc
             ) = self.scheduler[virtual_engine].schedule()

            ctx.seq_group_metadata_list = seq_group_metadata_list
            ctx.scheduler_outputs = scheduler_outputs

            finished_requests_ids = self.scheduler[
                virtual_engine].get_and_reset_finished_requests_ids()

            # Maybe switch from async mode to sync mode
            if not allow_async_output_proc and len(ctx.output_queue) > 0:
                self._process_model_outputs(ctx=ctx)

            if (self.scheduler_config.is_multi_step
                    and scheduler_outputs.num_lookahead_slots > 0):
                # cache the scheduler outputs for the next iteration if we have
                # lookahead slots
                self._cache_scheduler_outputs_for_multi_step(
                    virtual_engine, seq_group_metadata_list, scheduler_outputs,
                    allow_async_output_proc)
        else:
            finished_requests_ids = list()

        assert seq_group_metadata_list is not None
        assert scheduler_outputs is not None

        if not scheduler_outputs.is_empty():

            # Check if we have a cached last_output from the previous iteration.
            # For supporting PP this is probably the best way to pass the
            # sampled_token_ids, as a separate broadcast over all the PP stages
            # will cause one virtual engine's microbatch to block the pipeline.
            last_sampled_token_ids = \
                self._get_last_sampled_token_ids(virtual_engine)

            execute_model_req = ExecuteModelRequest(
                seq_group_metadata_list=seq_group_metadata_list,
                blocks_to_swap_in=scheduler_outputs.blocks_to_swap_in,
                blocks_to_swap_out=scheduler_outputs.blocks_to_swap_out,
                blocks_to_copy=scheduler_outputs.blocks_to_copy,
                num_lookahead_slots=scheduler_outputs.num_lookahead_slots,
                running_queue_size=scheduler_outputs.running_queue_size,
                finished_requests_ids=finished_requests_ids,
                # We use ExecuteModelRequest to pass the last sampled_token_ids
                # to each of the non-last PP stages for in-place prepare_input.
                last_sampled_token_ids=last_sampled_token_ids)

            if allow_async_output_proc:
                execute_model_req.async_callback = self.async_callbacks[
                    virtual_engine]

            outputs = self.model_executor.execute_model(
                execute_model_req=execute_model_req)

            # We need to do this here so that last step's sampled_token_ids can
            # be passed to the next iteration for PP.
            if self.scheduler_config.is_multi_step:
                self._update_cached_scheduler_output(virtual_engine, outputs)
        else:
            # Nothing scheduled => If there is pending async postprocessor,
            # then finish it here.
            if len(ctx.output_queue) > 0:
                self._process_model_outputs(ctx=ctx)
            # No outputs in this case
            outputs = []

        # Finish the current step for all the sequence groups.
        if self.scheduler_config.is_multi_step:
            for seq_group in seq_group_metadata_list:
                seq_group.finish_step()

        if not self._has_remaining_steps(seq_group_metadata_list):
            # clear the cache if we have finished all the steps.
            if self.scheduler_config.is_multi_step:
                self.cached_scheduler_outputs[0] = SchedulerOutputState()

            # is_first_step_output is True only when the num_steps of all
            # the sequences are 1. When the num_steps > 1,
            # multi_step_model_runner does the first-step output append.
            is_first_step_output: bool = False if not seq_group_metadata_list \
                else seq_group_metadata_list[0].state.num_steps == 1

            # Add results to the output_queue
            ctx.append_output(outputs=outputs,
                              seq_group_metadata_list=seq_group_metadata_list,
                              scheduler_outputs=scheduler_outputs,
                              is_async=allow_async_output_proc,
                              is_last_step=True,
                              is_first_step_output=is_first_step_output)

            if outputs and allow_async_output_proc:
                assert len(outputs) == 1, (
                    "Async postprocessor expects only a single output set")

                self._advance_to_next_step(
                    outputs[0], seq_group_metadata_list,
                    scheduler_outputs.scheduled_seq_groups)

            # Check if need to run the usual non-async path
            if not allow_async_output_proc:
                self._process_model_outputs(ctx=ctx)

                # Log stats.
                self.do_log_stats(scheduler_outputs, outputs)

                # Tracing
                self.do_tracing(scheduler_outputs)
        else:
            # Multi-step case
            return ctx.request_outputs

        if not self.has_unfinished_requests():
            # Drain async postprocessor (if exists)
            if len(ctx.output_queue) > 0:
                self._process_model_outputs(ctx=ctx)
            assert len(ctx.output_queue) == 0

            # Stop the execute model loop in parallel workers until there are
            # more requests to process. This avoids waiting indefinitely in
            # torch.distributed ops which may otherwise timeout, and unblocks
            # the RPC thread in the workers so that they can process any other
            # queued control plane messages, such as add/remove lora adapters.
            logger.debug("Stopping remote worker execution loop.")
            self.model_executor.stop_remote_worker_execution_loop()

        return ctx.request_outputs

    def _has_remaining_steps(
        self, seq_group_metadata_list: Optional[List[SequenceGroupMetadata]]
    ) -> bool:
        if (not self.scheduler_config.is_multi_step
                or not seq_group_metadata_list):
            return False

        # TODO(will) this is a sanity check for nowto make sure that all the
        # seqs are on the same steps. Eventually we will want to do some sort of
        # dynamic scheduling when doing multi-step decoding.
        ref_remaining_steps = seq_group_metadata_list[0].state.remaining_steps
        if any([
                seq_group.state.remaining_steps != ref_remaining_steps
                for seq_group in seq_group_metadata_list[1:]
        ]):
            raise AssertionError("All running sequence groups should "
                                 "have the same remaining steps.")

        return ref_remaining_steps > 0

    def _cache_scheduler_outputs_for_multi_step(
            self, virtual_engine: int,
            seq_group_metadata_list: Optional[List[SequenceGroupMetadata]],
            scheduler_outputs: SchedulerOutputs,
            allow_async_output_proc: bool) -> None:
        co = self.cached_scheduler_outputs[virtual_engine]

        co.seq_group_metadata_list = seq_group_metadata_list
        co.scheduler_outputs = scheduler_outputs
        co.allow_async_output_proc = allow_async_output_proc
        co.last_output = None

    def _update_cached_scheduler_output(
            self, virtual_engine: int,
            output: List[Optional[SamplerOutput]]) -> None:
        if (self.parallel_config.pipeline_parallel_size > 1 and len(output) > 0
                and output[0] is not None):
            last_output = output[-1]
            assert last_output is not None
            assert last_output.sampled_token_ids_cpu is not None
            assert last_output.sampled_token_ids is None
            assert last_output.sampled_token_probs is None
            self.cached_scheduler_outputs[
                virtual_engine].last_output = last_output

    def _get_last_sampled_token_ids(
            self, virtual_engine: int) -> Optional[torch.Tensor]:
        cached_last_output = self.cached_scheduler_outputs[
            virtual_engine].last_output
        if (self.scheduler_config.is_multi_step
                and self.parallel_config.pipeline_parallel_size > 1
                and cached_last_output is not None
                and cached_last_output.sampled_token_ids_cpu is not None):
            return cached_last_output.sampled_token_ids_cpu
        return None

    def add_logger(self, logger_name: str, logger: StatLoggerBase) -> None:
        if not self.log_stats:
            raise RuntimeError(
                "Stat logging is disabled. Set `disable_log_stats=False` "
                "argument to enable.")
        if logger_name in self.stat_loggers:
            raise KeyError(f"Logger with name {logger_name} already exists.")
        self.stat_loggers[logger_name] = logger

    def remove_logger(self, logger_name: str) -> None:
        if not self.log_stats:
            raise RuntimeError(
                "Stat logging is disabled. Set `disable_log_stats=False` "
                "argument to enable.")
        if logger_name not in self.stat_loggers:
            raise KeyError(f"Logger with name {logger_name} does not exist.")
        del self.stat_loggers[logger_name]

    def do_log_stats(self,
                     scheduler_outputs: Optional[SchedulerOutputs] = None,
                     model_output: Optional[List[SamplerOutput]] = None,
                     finished_before: Optional[List[int]] = None,
                     skip: Optional[List[int]] = None) -> None:
        """Forced log when no requests active."""
        if self.log_stats:
            stats = self._get_stats(scheduler_outputs, model_output,
                                    finished_before, skip)
            for logger in self.stat_loggers.values():
                logger.log(stats)

    def _get_stats(self,
                   scheduler_outputs: Optional[SchedulerOutputs],
                   model_output: Optional[List[SamplerOutput]] = None,
                   finished_before: Optional[List[int]] = None,
                   skip: Optional[List[int]] = None) -> Stats:
        """Get Stats to be Logged to Prometheus.

        Args:
            scheduler_outputs: Optional, used to populate metrics related to
                the scheduled batch,
            model_output: Optional, used to emit speculative decoding metrics
                which are created by the workers.
            finished_before: Optional, indices of sequences that were finished
                before. These sequences will be ignored.
            skip: Optional, indices of sequences that were preempted. These
                sequences will be ignored.
        """
        now = time.time()

        # System State
        #   Scheduler State
        num_running_sys = sum(
            len(scheduler.running) for scheduler in self.scheduler)
        num_swapped_sys = sum(
            len(scheduler.swapped) for scheduler in self.scheduler)
        num_waiting_sys = sum(
            len(scheduler.waiting) for scheduler in self.scheduler)

        # KV Cache Usage in %
        num_total_gpu = self.cache_config.num_gpu_blocks
        gpu_cache_usage_sys = 0.
        if num_total_gpu:  # Guard against both None and 0
            num_free_gpu = sum(
                scheduler.block_manager.get_num_free_gpu_blocks()
                for scheduler in self.scheduler)
            gpu_cache_usage_sys = 1.0 - (num_free_gpu / num_total_gpu)

        num_total_cpu = self.cache_config.num_cpu_blocks
        cpu_cache_usage_sys = 0.
        if num_total_cpu:  # Guard against both None and 0
            num_free_cpu = sum(
                scheduler.block_manager.get_num_free_cpu_blocks()
                for scheduler in self.scheduler)
            cpu_cache_usage_sys = 1.0 - (num_free_cpu / num_total_cpu)

        # Prefix Cache Hit Rate. Note that we always use
        # the cache hit rate of the first virtual engine.
        cpu_prefix_cache_hit_rate = self.scheduler[
            0].get_prefix_cache_hit_rate(Device.CPU)
        gpu_prefix_cache_hit_rate = self.scheduler[
            0].get_prefix_cache_hit_rate(Device.GPU)

        # Iteration stats
        num_prompt_tokens_iter = 0
        num_generation_tokens_iter = 0
        num_tokens_iter = 0
        time_to_first_tokens_iter: List[float] = []
        time_per_output_tokens_iter: List[float] = []
        num_preemption_iter = (0 if scheduler_outputs is None else
                               scheduler_outputs.preempted)

        # Request stats
        #   Latency
        time_e2e_requests: List[float] = []
        time_queue_requests: List[float] = []
        time_inference_requests: List[float] = []
        time_prefill_requests: List[float] = []
        time_decode_requests: List[float] = []
        time_in_queue_requests: List[float] = []
        model_forward_time_requests: List[float] = []
        model_execute_time_requests: List[float] = []
        #   Metadata
        num_prompt_tokens_requests: List[int] = []
        num_generation_tokens_requests: List[int] = []
        n_requests: List[int] = []
        max_num_generation_tokens_requests: List[int] = []
        max_tokens_requests: List[int] = []
        finished_reason_requests: List[str] = []

        # Lora requests
        running_lora_adapters = dict(
            collectionsCounter([
                running_request.lora_request.lora_name
                for scheduler in self.scheduler
                for running_request in scheduler.running
                if running_request.lora_request
            ]))
        waiting_lora_adapters = dict(
            collectionsCounter([
                waiting_request.lora_request.lora_name
                for scheduler in self.scheduler
                for waiting_request in scheduler.waiting
                if waiting_request.lora_request
            ]))
        max_lora_stat = "0"
        if self.lora_config:
            max_lora_stat = str(self.lora_config.max_loras)

        # NOTE: This loop assumes prefill seq_groups are before
        # decode seq_groups in scheduled_seq_groups.
        if scheduler_outputs is not None:
            # For async postprocessor, already finished sequences need to be
            # not counted (to avoid double counting)
            actual_num_batched_tokens = scheduler_outputs.num_batched_tokens  # type: ignore

            num_generation_tokens_from_prefill_groups = 0
            # NOTE: if scheduler_outputs.num_prefill_groups > 0 and
            # the len of scheduler_outputs.scheduled_seq_groups is !=
            # scheduler_outputs.num_prefill_groups, this means that
            # chunked prefills have been detected.

            for idx, scheduled_seq_group in enumerate(
                    scheduler_outputs.scheduled_seq_groups):
                # Skip double logging when using async output proc
                if finished_before and idx in finished_before:
                    actual_num_batched_tokens -= 1
                    continue

                # Currently, skip == preempted sequences, so we need to skip
                # their log stats
                if skip and idx in skip:
                    continue

                group_was_prefill = idx < scheduler_outputs.num_prefill_groups
                seq_group = scheduled_seq_group.seq_group

                # NOTE: a seq_group that completed all of its prefill tokens
                # in the last iteration will have seq_group.is_prefill() = False
                # with group_was_prefill = True
                if group_was_prefill:
                    # Number of prompt tokens.
                    num_prompt_tokens_iter += (
                        scheduled_seq_group.token_chunk_size)

                    # If the seq_group just finished the prefill state
                    # get TTFT.
                    if not seq_group.is_prefill():
                        latency = seq_group.get_last_token_latency()
                        time_to_first_tokens_iter.append(latency)

                        # One generation token per finished prefill.
                        num_generation_tokens_from_prefill_groups += (
                            seq_group.num_seqs())
                else:
                    # TPOTs.
                    latency = seq_group.get_last_token_latency()
                    time_per_output_tokens_iter.append(latency)
                    if seq_group.state.current_step == 0:
                        # For async_output_proc, the do_log_stats()
                        # is called following init_multi_step(), which
                        # sets the current_step to zero.
                        actual_num_batched_tokens +=\
                            seq_group.state.num_steps - 1
                    else:
                        actual_num_batched_tokens +=\
                            seq_group.state.current_step - 1

                # Because of chunked prefill, we can have a single sequence
                # group that does multiple prompt_runs. To prevent logging
                # the same metadata more than once per request, we standardize
                # on logging request level information for finished requests,
                # which can only happen once.
                if seq_group.is_finished():
                    # Latency timings
                    time_e2e_requests.append(now -
                                             seq_group.metrics.arrival_time)
                    if (seq_group.metrics.first_scheduled_time is not None and
                            seq_group.metrics.first_token_time is not None):
                        time_queue_requests.append(
                            seq_group.metrics.first_scheduled_time -
                            seq_group.metrics.arrival_time)
                        time_prefill_requests.append(
                            seq_group.metrics.first_token_time -
                            seq_group.metrics.first_scheduled_time)
                        time_decode_requests.append(
                            now - seq_group.metrics.first_token_time)
                        time_inference_requests.append(
                            now - seq_group.metrics.first_scheduled_time)
                    if seq_group.metrics.time_in_queue is not None:
                        time_in_queue_requests.append(
                            seq_group.metrics.time_in_queue)
                    if seq_group.metrics.model_forward_time is not None:
                        model_forward_time_requests.append(
                            seq_group.metrics.model_forward_time)
                    if seq_group.metrics.model_execute_time is not None:
                        model_execute_time_requests.append(
                            seq_group.metrics.model_execute_time * 1000)
                    # Metadata
                    num_prompt_tokens_requests.append(
                        len(seq_group.prompt_token_ids))
                    num_generation_tokens_requests.extend([
                        seq.get_output_len()
                        for seq in seq_group.get_finished_seqs()
                    ])
                    max_num_generation_tokens_requests.append(
                        max(seq.get_output_len()
                            for seq in seq_group.get_seqs()))
                    if seq_group.sampling_params is not None:
                        n_requests.append(seq_group.sampling_params.n)
                        max_tokens_requests.append(
                            seq_group.sampling_params.max_tokens)
                    finished_reason_requests.extend([
                        SequenceStatus.get_finished_reason(seq.status)
                        for seq in seq_group.get_finished_seqs()
                    ])

            # Number of generation tokens.
            #   num_batched_tokens equals the number of prompt_tokens plus the
            #   number of decode_tokens in a single iteration. So,
            #   num_generation_tokens = num_batched_tokens - num_prompt_tokens
            #   + num_generation_tokens_from_prefill_groups (since we generate
            #   one token on prefills on iters where the prefill finishes).
            num_generation_tokens_iter = (
                actual_num_batched_tokens - num_prompt_tokens_iter +
                num_generation_tokens_from_prefill_groups)
            num_tokens_iter = (num_generation_tokens_iter +
                               num_prompt_tokens_iter)
        # Spec decode, if enabled, emits specialized metrics from the worker in
        # sampler output.
        if model_output and isinstance(model_output[0], SamplerOutput) and (
                model_output[0].spec_decode_worker_metrics is not None):
            spec_decode_metrics = model_output[0].spec_decode_worker_metrics
        else:
            spec_decode_metrics = None

        return Stats(
            now=now,
            # System stats
            #   Scheduler State
            num_running_sys=num_running_sys,
            num_swapped_sys=num_swapped_sys,
            num_waiting_sys=num_waiting_sys,
            #   KV Cache Usage in %
            gpu_cache_usage_sys=gpu_cache_usage_sys,
            cpu_cache_usage_sys=cpu_cache_usage_sys,
            #   Prefix Cache Hit Rate
            cpu_prefix_cache_hit_rate=cpu_prefix_cache_hit_rate,
            gpu_prefix_cache_hit_rate=gpu_prefix_cache_hit_rate,

            # Iteration stats
            num_prompt_tokens_iter=num_prompt_tokens_iter,
            num_generation_tokens_iter=num_generation_tokens_iter,
            num_tokens_iter=num_tokens_iter,
            time_to_first_tokens_iter=time_to_first_tokens_iter,
            time_per_output_tokens_iter=time_per_output_tokens_iter,
            spec_decode_metrics=spec_decode_metrics,
            num_preemption_iter=num_preemption_iter,

            # Request stats
            #   Latency
            time_e2e_requests=time_e2e_requests,
            time_queue_requests=time_queue_requests,
            time_inference_requests=time_inference_requests,
            time_prefill_requests=time_prefill_requests,
            time_decode_requests=time_decode_requests,
            time_in_queue_requests=time_in_queue_requests,
            model_forward_time_requests=model_forward_time_requests,
            model_execute_time_requests=model_execute_time_requests,
            #   Metadata
            num_prompt_tokens_requests=num_prompt_tokens_requests,
            num_generation_tokens_requests=num_generation_tokens_requests,
            max_num_generation_tokens_requests=
            max_num_generation_tokens_requests,
            n_requests=n_requests,
            max_tokens_requests=max_tokens_requests,
            finished_reason_requests=finished_reason_requests,
            max_lora=str(max_lora_stat),
            waiting_lora_adapters=list(waiting_lora_adapters.keys()),
            running_lora_adapters=list(running_lora_adapters.keys()))

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.model_executor.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.model_executor.remove_lora(lora_id)

    def list_loras(self) -> Set[int]:
        return self.model_executor.list_loras()

    def pin_lora(self, lora_id: int) -> bool:
        return self.model_executor.pin_lora(lora_id)

    def add_prompt_adapter(
            self, prompt_adapter_request: PromptAdapterRequest) -> bool:
        return self.model_executor.add_prompt_adapter(prompt_adapter_request)

    def remove_prompt_adapter(self, prompt_adapter_id: int) -> bool:
        return self.model_executor.remove_prompt_adapter(prompt_adapter_id)

    def list_prompt_adapters(self) -> List[int]:
        return self.model_executor.list_prompt_adapters()

    def start_profile(self) -> None:
        self.model_executor.start_profile()

    def stop_profile(self) -> None:
        self.model_executor.stop_profile()

    def sleep(self, level: int = 1) -> None:
        assert self.vllm_config.model_config.enable_sleep_mode, (
            "Sleep mode is not enabled in the model config")
        self.model_executor.sleep(level=level)

    def wake_up(self) -> None:
        assert self.vllm_config.model_config.enable_sleep_mode, (
            "Sleep mode is not enabled in the model config")
        self.model_executor.wake_up()

    def check_health(self) -> None:
        if self.tokenizer:
            self.tokenizer.check_health()
        self.model_executor.check_health()

    def is_tracing_enabled(self) -> bool:
        return self.tracer is not None

    def do_tracing(self,
                   scheduler_outputs: SchedulerOutputs,
                   finished_before: Optional[List[int]] = None) -> None:
        if self.tracer is None:
            return

        for idx, scheduled_seq_group in enumerate(
                scheduler_outputs.scheduled_seq_groups):
            # Skip double tracing when using async output proc
            if finished_before and idx in finished_before:
                continue

            seq_group = scheduled_seq_group.seq_group
            if seq_group.is_finished():
                self.create_trace_span(seq_group)

    def create_trace_span(self, seq_group: SequenceGroup) -> None:
        if self.tracer is None or seq_group.sampling_params is None:
            return
        arrival_time_nano_seconds = int(seq_group.metrics.arrival_time * 1e9)

        trace_context = extract_trace_context(seq_group.trace_headers)

        with self.tracer.start_as_current_span(
                "llm_request",
                kind=SpanKind.SERVER,
                context=trace_context,
                start_time=arrival_time_nano_seconds) as seq_span:
            metrics = seq_group.metrics
            ttft = metrics.first_token_time - metrics.arrival_time
            e2e_time = metrics.finished_time - metrics.arrival_time
            seq_span.set_attribute(SpanAttributes.GEN_AI_RESPONSE_MODEL,
                                   self.model_config.model)
            seq_span.set_attribute(SpanAttributes.GEN_AI_REQUEST_ID,
                                   seq_group.request_id)
            seq_span.set_attribute(SpanAttributes.GEN_AI_REQUEST_TEMPERATURE,
                                   seq_group.sampling_params.temperature)
            seq_span.set_attribute(SpanAttributes.GEN_AI_REQUEST_TOP_P,
                                   seq_group.sampling_params.top_p)
            seq_span.set_attribute(SpanAttributes.GEN_AI_REQUEST_MAX_TOKENS,
                                   seq_group.sampling_params.max_tokens)
            seq_span.set_attribute(SpanAttributes.GEN_AI_REQUEST_N,
                                   seq_group.sampling_params.n)
            seq_span.set_attribute(SpanAttributes.GEN_AI_USAGE_NUM_SEQUENCES,
                                   seq_group.num_seqs())
            seq_span.set_attribute(SpanAttributes.GEN_AI_USAGE_PROMPT_TOKENS,
                                   len(seq_group.prompt_token_ids))
            seq_span.set_attribute(
                SpanAttributes.GEN_AI_USAGE_COMPLETION_TOKENS,
                sum([
                    seq.get_output_len()
                    for seq in seq_group.get_finished_seqs()
                ]))
            seq_span.set_attribute(SpanAttributes.GEN_AI_LATENCY_TIME_IN_QUEUE,
                                   metrics.time_in_queue)
            seq_span.set_attribute(
                SpanAttributes.GEN_AI_LATENCY_TIME_TO_FIRST_TOKEN, ttft)
            seq_span.set_attribute(SpanAttributes.GEN_AI_LATENCY_E2E, e2e_time)
            if metrics.scheduler_time is not None:
                seq_span.set_attribute(
                    SpanAttributes.GEN_AI_LATENCY_TIME_IN_SCHEDULER,
                    metrics.scheduler_time)
            if metrics.model_forward_time is not None:
                seq_span.set_attribute(
                    SpanAttributes.GEN_AI_LATENCY_TIME_IN_MODEL_FORWARD,
                    metrics.model_forward_time / 1000.0)
            if metrics.model_execute_time is not None:
                seq_span.set_attribute(
                    SpanAttributes.GEN_AI_LATENCY_TIME_IN_MODEL_EXECUTE,
                    metrics.model_execute_time)

    def _validate_model_inputs(self, inputs: ProcessorInputs,
                               lora_request: Optional[LoRARequest]):
        if is_encoder_decoder_inputs(inputs):
            # For encoder-decoder multimodal models, the max_prompt_len
            # restricts the decoder prompt length
            prompt_inputs = inputs["decoder" if self.model_config.
                                   is_multimodal_model else "encoder"]
        else:
            prompt_inputs = inputs

        prompt_ids = SingletonInputsAdapter(prompt_inputs).prompt_token_ids

        if prompt_ids is None or len(prompt_ids) == 0:
            raise ValueError("Prompt cannot be empty")

        if self.model_config.is_multimodal_model:
            max_prompt_len = self.model_config.max_model_len

            if len(prompt_ids) > max_prompt_len:
                raise ValueError(
                    f"The prompt (total length {len(prompt_ids)}) is too long "
                    f"to fit into the model (context length {max_prompt_len}). "
                    "Make sure that `max_model_len` is no smaller than the "
                    "number of text tokens plus multimodal tokens. For image "
                    "inputs, the number of image tokens depends on the number "
                    "of images, and possibly their aspect ratios as well.")

            # TODO: Find out how many placeholder tokens are there so we can
            # check that chunked prefill does not truncate them
            # max_batch_len = self.scheduler_config.max_num_batched_tokens

    def _build_logits_processors(
            self, sampling_params: SamplingParams,
            lora_request: Optional[LoRARequest]) -> SamplingParams:
        """Constructs logits processors based on the guided_decoding,
        logits_bias, and allowed_token_ids fields in sampling_params. Deletes
        those fields and adds the constructed logits processors to the
        logits_processors field. Returns the modified sampling params."""

        logits_processors = []

        if sampling_params.guided_decoding is not None:
            # Defensively copy sampling params since guided decoding logits
            # processors can have different state for each request
            sampling_params = copy.copy(sampling_params)
            guided_decoding = sampling_params.guided_decoding

            logger.debug(
                "Building guided decoding logits processor in "
                "LLMEngine. Params: %s", guided_decoding)

            tokenizer = self.get_tokenizer(lora_request=lora_request)
            guided_decoding.backend = guided_decoding.backend or \
                self.decoding_config.guided_decoding_backend

            processor = get_local_guided_decoding_logits_processor(
                guided_params=guided_decoding,
                tokenizer=tokenizer,
                model_config=self.model_config)
            if processor:
                logits_processors.append(processor)

            # Unset so this doesn't get passed down to the model
            sampling_params.guided_decoding = None

        if (sampling_params.logit_bias or sampling_params.allowed_token_ids):
            tokenizer = self.get_tokenizer(lora_request=lora_request)

            processors = get_openai_logits_processors(
                logit_bias=sampling_params.logit_bias,
                allowed_token_ids=sampling_params.allowed_token_ids,
                tokenizer=tokenizer)
            logits_processors.extend(processors)

            # Unset so these don't get passed down to the model
            sampling_params.logit_bias = None
            sampling_params.allowed_token_ids = None

        if len(sampling_params.bad_words) > 0:
            tokenizer = self.get_tokenizer(lora_request)
            processors = get_bad_words_logits_processors(
                bad_words=sampling_params.bad_words, tokenizer=tokenizer)
            logits_processors.extend(processors)

        if logits_processors:
            if sampling_params.logits_processors is None:
                sampling_params.logits_processors = logits_processors
            else:
                sampling_params.logits_processors.extend(logits_processors)

        return sampling_params



!rm -rf /kaggle/working/ray*

