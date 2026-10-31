"""Model backends. All of them turn chat messages into text."""

import gc
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

import warehouse_planning  # noqa: F401  Sets up the Windows DLL path before torch loads.
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
    set_seed,
)

DEFAULT_HF_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_CLAUDE_MODEL = "claude-opus-5-5"

# Past the GPU's memory Windows moves tensors to system RAM without an error, and generation gets
# several times slower. With this cap PyTorch reuses its memory first and fails loudly if it must.
# Measured on a 4 GB GPU: no spill up to about 3.95 GB reserved.
GPU_MEMORY_SHARE = 0.97

# Called with the text generated so far, returns True when nothing useful can follow.
Verifier = Callable[[str], bool]


def _release_gpu_memory() -> None:
    """Free the caches of finished generations.

    They sit in reference cycles until Python's collector happens to run, which GPU pressure does
    not trigger. Left alone, several of them pile up and a 3 GB run overflows the GPU.
    """
    gc.collect()
    torch.cuda.empty_cache()


class LLM(Protocol):
    name: str

    def generate(
        self,
        batch: list[list[dict]],
        temperature: float,
        seed: int,
        max_new_tokens: int,
        verifiers: list[Verifier] | None = None,
    ) -> list[str]:
        """Answer each conversation in `batch`, in order."""
        ...


class StopWhenVerified(StoppingCriteria):
    """Check each sequence once per finished line, and stop it when its verifier says so.

    Only the steps up to the first error are ever used, so everything generated after it is wasted.
    On a 4 GB GPU, where a batch runs as long as its slowest sequence, that waste dominates run time.
    """

    def __init__(self, tokenizer, prompt_tokens: int, verifiers: list[Verifier]):
        self.tokenizer = tokenizer
        self.prompt_tokens = prompt_tokens
        self.verifiers = verifiers
        self.lines = [0] * len(verifiers)
        self.stopped = [False] * len(verifiers)

    def __call__(self, input_ids: torch.Tensor, scores, **kwargs) -> torch.Tensor:
        tails = self.tokenizer.batch_decode(input_ids[:, self.prompt_tokens :], skip_special_tokens=True)
        for index, text in enumerate(tails):
            lines = text.count("\n")
            if not self.stopped[index] and lines > self.lines[index]:
                self.lines[index] = lines
                self.stopped[index] = self.verifiers[index](text[: text.rfind("\n")])
        return torch.tensor(self.stopped, dtype=torch.bool, device=input_ids.device)


class HuggingFaceLLM:
    """A local causal LM, optionally with a LoRA adapter on top."""

    def __init__(self, model_id: str = DEFAULT_HF_MODEL, adapter: str | None = None, four_bit: bool = False):
        self.name = model_id if adapter is None else f"{model_id}+lora"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_SHARE)
        quantization = None
        if four_bit:
            quantization = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            device_map={"": 0} if torch.cuda.is_available() else None,
            quantization_config=quantization,
        )
        if adapter is not None:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()

    @torch.inference_mode()
    def generate(
        self,
        batch: list[list[dict]],
        temperature: float,
        seed: int,
        max_new_tokens: int,
        verifiers: list[Verifier] | None = None,
    ) -> list[str]:
        set_seed(seed)
        prompts = [
            self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            for messages in batch
        ]
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.model.device)
        prompt_tokens = inputs.input_ids.shape[1]
        sampling = {"do_sample": True, "temperature": temperature} if temperature > 0 else {"do_sample": False}
        stopping = StoppingCriteriaList()
        if verifiers:
            stopping.append(StopWhenVerified(self.tokenizer, prompt_tokens, verifiers))
        _release_gpu_memory()
        output = None
        try:
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                stopping_criteria=stopping,
                **sampling,
            )
        except torch.cuda.OutOfMemoryError:
            pass
        if output is not None:
            return self.tokenizer.batch_decode(output[:, prompt_tokens:], skip_special_tokens=True)

        # Retried here and not in the except block, whose traceback still holds the failed tensors.
        # A single sequence restarts with the same seed and half the budget, so it writes the same
        # text and ends earlier. Only outputs that run far past the plan ever get this far.
        _release_gpu_memory()
        if len(batch) == 1:
            if max_new_tokens < 64:
                raise RuntimeError("the prompt alone does not fit in GPU memory")
            return self.generate(batch, temperature, seed, max_new_tokens // 2, verifiers)
        answers = []
        for index, messages in enumerate(batch):
            verifier = [verifiers[index]] if verifiers else None
            answers += self.generate([messages], temperature, seed + index, max_new_tokens, verifier)
        return answers


class ClaudeLLM:
    """The Claude API, as a stronger comparison point.

    Current Claude models take neither `temperature` nor a seed, so both arguments are ignored;
    replicates differ only through the model's own sampling. `max_new_tokens` and `verifiers` are
    ignored as well: both exist to save local GPU time, and here the same limit would also have to
    cover thinking.
    """

    def __init__(self, model: str = DEFAULT_CLAUDE_MODEL, effort: str = "high", workers: int = 4, client=None):
        import anthropic

        self.name = model
        self.effort = effort
        self.workers = workers
        self.client = client or anthropic.Anthropic()

    def _answer(self, messages: list[dict]) -> str:
        system, *conversation = messages
        with self.client.beta.messages.stream(
            model=self.name,
            max_tokens=16000,
            system=system["content"],
            messages=conversation,
            output_config={"effort": self.effort},
            cache_control={"type": "ephemeral"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        ) as stream:
            response = stream.get_final_message()
        if response.stop_reason == "refusal":
            return ""
        return "".join(block.text for block in response.content if block.type == "text")

    def generate(self, batch, temperature, seed, max_new_tokens, verifiers=None) -> list[str]:
        with ThreadPoolExecutor(self.workers) as pool:
            return list(pool.map(self._answer, batch))
