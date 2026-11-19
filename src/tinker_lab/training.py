from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, datetime

from tinker_lab.data import TrainingExample
from tinker_lab.safety import require_live_run_authorization

DEFAULT_MODEL = "Qwen/Qwen3.5-4B"
DEFAULT_SYSTEM_PROMPT = "You are a concise, accurate assistant."
MAX_STEPS = 10
MAX_EXAMPLES = 32


@dataclass(frozen=True, slots=True)
class SFTConfig:
    model: str = DEFAULT_MODEL
    rank: int = 16
    learning_rate: float = 2e-4
    steps: int = 1
    max_length: int = 512
    max_examples: int = 8
    system_prompt: str = DEFAULT_SYSTEM_PROMPT

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("model must not be blank")
        for name in ("rank", "steps", "max_length", "max_examples"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be greater than zero")
        if self.steps > MAX_STEPS:
            raise ValueError(f"steps must be at most {MAX_STEPS} in this starter")
        if self.max_examples > MAX_EXAMPLES:
            raise ValueError(f"max_examples must be at most {MAX_EXAMPLES} in this starter")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")


@dataclass(frozen=True, slots=True)
class TrainingResult:
    model: str
    examples: int
    steps: int
    sampler_path: str
    losses: tuple[float | None, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "status": "complete",
            "model": self.model,
            "examples": self.examples,
            "steps": self.steps,
            "sampler_path": self.sampler_path,
            "losses": list(self.losses),
        }


def _weighted_mean_loss(outputs: object, training_data: list[object]) -> float | None:
    loss_outputs = getattr(outputs, "loss_fn_outputs", None)
    if loss_outputs is None:
        return None

    weighted_total = 0.0
    weight_total = 0.0
    for output, datum in zip(loss_outputs, training_data, strict=True):
        logprobs = output["logprobs"].tolist()
        weights = datum.loss_fn_inputs["weights"].tolist()
        weighted_total += -sum(
            float(logprob) * float(weight)
            for logprob, weight in zip(logprobs, weights, strict=True)
        )
        weight_total += sum(float(weight) for weight in weights)
    return weighted_total / weight_total if weight_total else None


async def run_sft(
    examples: tuple[TrainingExample, ...],
    config: SFTConfig,
    *,
    execute: bool = False,
    api_key: str | None = None,
) -> TrainingResult:
    resolved_key = api_key if api_key is not None else os.environ.get("TINKER_API_KEY")
    require_live_run_authorization(execute=execute, api_key=resolved_key)
    # Lazy imports keep validation and safety checks local and free.
    import tinker
    from tinker_cookbook import model_info
    from tinker_cookbook.renderers import TrainOnWhat, get_renderer
    from tinker_cookbook.supervised.data import conversation_to_datum

    selected_examples = examples[: config.max_examples]
    service_client = tinker.ServiceClient()
    training_client = await service_client.create_lora_training_client_async(
        base_model=config.model,
        rank=config.rank,
    )
    tokenizer = training_client.get_tokenizer()
    renderer_name = model_info.get_recommended_renderer_name(config.model)
    renderer = get_renderer(renderer_name, tokenizer)
    training_data = [
        conversation_to_datum(
            example.conversation(config.system_prompt),
            renderer,
            max_length=config.max_length,
            train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
        )
        for example in selected_examples
    ]

    losses: list[float | None] = []
    for _ in range(config.steps):
        forward_future = await training_client.forward_backward_async(
            training_data,
            "cross_entropy",
        )
        optimizer_future = await training_client.optim_step_async(
            tinker.AdamParams(learning_rate=config.learning_rate)
        )
        forward_result = await forward_future.result_async()
        await optimizer_future.result_async()
        losses.append(_weighted_mean_loss(forward_result, training_data))

    checkpoint_name = datetime.now(UTC).strftime("starter-%Y%m%dT%H%M%SZ")
    save_future = training_client.save_weights_for_sampler(checkpoint_name)
    save_result = await save_future
    return TrainingResult(
        model=config.model,
        examples=len(selected_examples),
        steps=config.steps,
        sampler_path=save_result.path,
        losses=tuple(losses),
    )
