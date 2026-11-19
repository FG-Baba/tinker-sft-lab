import asyncio
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from tinker_lab.data import TrainingExample
from tinker_lab.training import (
    MAX_EXAMPLES,
    MAX_STEPS,
    SFTConfig,
    TrainingResult,
    _weighted_mean_loss,
    run_sft,
)


class Values:
    def __init__(self, values: list[float]) -> None:
        self._values = list(values)

    def tolist(self) -> list[float]:
        return list(self._values)


class Datum:
    def __init__(self) -> None:
        self.loss_fn_inputs = {"weights": Values([0.0, 1.0])}


class Future:
    def __init__(self, result: object) -> None:
        self._result = result

    async def result_async(self) -> object:
        return self._result

    def __await__(self):
        return self.result_async().__await__()


class TrainingClient:
    def __init__(self) -> None:
        self.forward_calls = 0

    def get_tokenizer(self) -> object:
        return object()

    async def forward_backward_async(self, data: list[object], loss_fn: str) -> Future:
        self.forward_calls += 1
        output = SimpleNamespace(loss_fn_outputs=[{"logprobs": Values([-1.0, -0.5])}])
        return Future(output)

    async def optim_step_async(self, adam_params: object) -> Future:
        return Future(None)

    def save_weights_for_sampler(self, name: str) -> Future:
        return Future(SimpleNamespace(path=f"tinker://run/sampler_weights/{name}"))


class ServiceClient:
    training_client = TrainingClient()

    async def create_lora_training_client_async(
        self, *, base_model: str, rank: int
    ) -> TrainingClient:
        return self.training_client


class TrainingTests(unittest.TestCase):
    def test_config_rejects_invalid_values(self) -> None:
        for arguments, message in (
            ({"model": " "}, "model"),
            ({"rank": 0}, "rank"),
            ({"steps": 0}, "steps"),
            ({"max_length": 0}, "max_length"),
            ({"max_examples": 0}, "max_examples"),
            ({"learning_rate": 0}, "learning_rate"),
            ({"steps": MAX_STEPS + 1}, "steps must be at most"),
            ({"max_examples": MAX_EXAMPLES + 1}, "max_examples must be at most"),
        ):
            with self.subTest(arguments=arguments), self.assertRaisesRegex(ValueError, message):
                SFTConfig(**arguments)

    def test_result_serializes_without_credentials(self) -> None:
        result = TrainingResult("model", 2, 1, "tinker://path", (0.5, None))

        self.assertEqual(result.as_dict()["losses"], [0.5, None])
        self.assertNotIn("key", result.as_dict())

    def test_weighted_loss_and_missing_metrics(self) -> None:
        output = SimpleNamespace(loss_fn_outputs=[{"logprobs": Values([-1.0, -0.5])}])

        self.assertEqual(_weighted_mean_loss(output, [Datum()]), 0.5)
        self.assertIsNone(_weighted_mean_loss(object(), [Datum()]))

    def test_live_sft_adapter_uses_current_sdk_flow_without_network(self) -> None:
        tinker = types.ModuleType("tinker")
        tinker.ServiceClient = ServiceClient
        tinker.AdamParams = lambda **values: values

        cookbook = types.ModuleType("tinker_cookbook")
        cookbook.model_info = SimpleNamespace(get_recommended_renderer_name=lambda model: "qwen3_5")
        renderers = types.ModuleType("tinker_cookbook.renderers")
        renderers.TrainOnWhat = SimpleNamespace(LAST_ASSISTANT_MESSAGE="last")
        renderers.get_renderer = lambda name, tokenizer: object()
        supervised = types.ModuleType("tinker_cookbook.supervised")
        supervised_data = types.ModuleType("tinker_cookbook.supervised.data")
        supervised_data.conversation_to_datum = lambda *args, **kwargs: Datum()

        fake_modules = {
            "tinker": tinker,
            "tinker_cookbook": cookbook,
            "tinker_cookbook.renderers": renderers,
            "tinker_cookbook.supervised": supervised,
            "tinker_cookbook.supervised.data": supervised_data,
        }
        examples = (TrainingExample("Question", "Answer"),)

        with patch.dict(sys.modules, fake_modules):
            with self.assertRaisesRegex(RuntimeError, "--execute"):
                asyncio.run(run_sft(examples, SFTConfig(steps=2)))
            result = asyncio.run(
                run_sft(examples, SFTConfig(steps=2), execute=True, api_key="test-key")
            )

        self.assertEqual(result.losses, (0.5, 0.5))
        self.assertTrue(result.sampler_path.startswith("tinker://run/sampler_weights/starter-"))


if __name__ == "__main__":
    unittest.main()
