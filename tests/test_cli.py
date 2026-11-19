import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"


def run_cli(
    *arguments: str,
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    process_environment = os.environ.copy()
    process_environment["PYTHONPATH"] = str(SOURCE_ROOT)
    process_environment.pop("TINKER_API_KEY", None)
    if environment:
        process_environment.update(environment)
    return subprocess.run(
        [sys.executable, "-m", "tinker_lab", *arguments],
        cwd=PROJECT_ROOT,
        env=process_environment,
        capture_output=True,
        text=True,
        check=False,
    )


class CliJourneyTests(unittest.TestCase):
    def test_validate_is_local_and_machine_readable(self) -> None:
        result = run_cli("validate", "data/example.jsonl", "--json")

        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["status"], "valid")
        self.assertGreaterEqual(payload["examples"], 2)

    def test_doctor_reports_a_missing_tinker_key_without_revealing_secrets(self) -> None:
        result = run_cli("doctor", "--json")

        self.assertEqual(result.returncode, 2)
        payload = json.loads(result.stdout)
        self.assertFalse(payload["tinker_api_key_present"])
        self.assertNotIn("api_key", result.stdout.lower().replace("tinker_api_key", ""))

    def test_train_stops_before_importing_or_calling_tinker(self) -> None:
        result = run_cli("train-sft", "data/example.jsonl")

        self.assertEqual(result.returncode, 2)
        self.assertIn("--execute", result.stderr)
        self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
