import contextlib
import importlib.metadata
import io
import json
import os
import runpy
import unittest
from unittest.mock import AsyncMock, patch

from tinker_lab import cli
from tinker_lab.training import TrainingResult


class CliUnitTests(unittest.TestCase):
    def test_package_version_returns_none_when_distribution_is_missing(self) -> None:
        with patch(
            "tinker_lab.cli.importlib.metadata.version",
            side_effect=importlib.metadata.PackageNotFoundError,
        ):
            self.assertIsNone(cli._package_version("missing"))

    def test_doctor_is_ready_when_dependencies_key_and_python_are_ready(self) -> None:
        with (
            patch.dict(os.environ, {"TINKER_API_KEY": "secret"}, clear=True),
            patch("tinker_lab.cli._package_version", side_effect=["0.23.0", "0.5.2"]),
        ):
            payload, exit_code = cli._doctor()

        self.assertEqual(exit_code, 0)
        self.assertTrue(payload["ready"])

    def test_validate_text_output(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exit_code = cli.main(["validate", "data/example.jsonl"])

        self.assertEqual(exit_code, 0)
        self.assertIn("status: valid", output.getvalue())

    def test_validate_error_is_concise(self) -> None:
        error = io.StringIO()
        with contextlib.redirect_stderr(error):
            exit_code = cli.main(["validate", "missing.jsonl"])

        self.assertEqual(exit_code, 2)
        self.assertIn("not found", error.getvalue())
        self.assertNotIn("Traceback", error.getvalue())

    def test_train_success_prints_machine_readable_result(self) -> None:
        result = TrainingResult(
            model="Qwen/Qwen3.5-4B",
            examples=3,
            steps=1,
            sampler_path="tinker://run/sampler_weights/test",
            losses=(0.5,),
        )
        output = io.StringIO()
        with (
            patch.dict(os.environ, {"TINKER_API_KEY": "secret"}, clear=True),
            patch("tinker_lab.cli.run_sft", new=AsyncMock(return_value=result)),
            contextlib.redirect_stdout(output),
        ):
            exit_code = cli.main(["train-sft", "data/example.jsonl", "--execute", "--steps", "1"])

        self.assertEqual(exit_code, 0)
        self.assertEqual(json.loads(output.getvalue())["sampler_path"], result.sampler_path)

    def test_train_dependency_error_is_actionable(self) -> None:
        error = io.StringIO()
        with (
            patch.dict(os.environ, {"TINKER_API_KEY": "secret"}, clear=True),
            patch("tinker_lab.cli.run_sft", new=AsyncMock(side_effect=ImportError)),
            contextlib.redirect_stderr(error),
        ):
            exit_code = cli.main(["train-sft", "data/example.jsonl", "--execute"])

        self.assertEqual(exit_code, 2)
        self.assertIn("uv sync", error.getvalue())

    def test_train_rejects_an_oversized_run_before_calling_tinker(self) -> None:
        run_sft = AsyncMock()
        error = io.StringIO()
        with (
            patch.dict(os.environ, {"TINKER_API_KEY": "secret"}, clear=True),
            patch("tinker_lab.cli.run_sft", new=run_sft),
            contextlib.redirect_stderr(error),
        ):
            exit_code = cli.main(
                ["train-sft", "data/example.jsonl", "--execute", "--steps", "100000"]
            )

        self.assertEqual(exit_code, 2)
        self.assertIn("steps must be at most", error.getvalue())
        run_sft.assert_not_called()

    def test_train_service_error_does_not_dump_exception_details(self) -> None:
        error = io.StringIO()
        with (
            patch.dict(os.environ, {"TINKER_API_KEY": "secret"}, clear=True),
            patch(
                "tinker_lab.cli.run_sft",
                new=AsyncMock(side_effect=RuntimeError("secret diagnostic")),
            ),
            contextlib.redirect_stderr(error),
        ):
            exit_code = cli.main(["train-sft", "data/example.jsonl", "--execute"])

        self.assertEqual(exit_code, 1)
        self.assertIn("RuntimeError", error.getvalue())
        self.assertNotIn("secret diagnostic", error.getvalue())

    def test_module_entrypoint_returns_main_exit_code(self) -> None:
        with (
            patch("tinker_lab.cli.main", return_value=7),
            self.assertRaisesRegex(SystemExit, "7"),
        ):
            runpy.run_module("tinker_lab", run_name="__main__")


if __name__ == "__main__":
    unittest.main()
