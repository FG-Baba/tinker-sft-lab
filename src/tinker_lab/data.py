from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class DataValidationError(ValueError):
    """Raised when a JSONL training dataset is invalid."""


MAX_DATASET_BYTES = 10 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class TrainingExample:
    prompt: str
    completion: str

    def conversation(self, system_prompt: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": self.prompt},
            {"role": "assistant", "content": self.completion},
        ]


def _parse_example(raw: Any, line_number: int) -> TrainingExample:
    if not isinstance(raw, dict):
        raise DataValidationError(f"line {line_number}: expected a JSON object")

    expected_keys = {"prompt", "completion"}
    if set(raw) != expected_keys:
        raise DataValidationError(
            f"line {line_number}: expected exactly the fields 'prompt' and 'completion'"
        )

    values: dict[str, str] = {}
    for field in ("prompt", "completion"):
        value = raw[field]
        if not isinstance(value, str) or not value.strip():
            raise DataValidationError(f"line {line_number}: {field} must be a non-empty string")
        values[field] = value.strip()

    return TrainingExample(prompt=values["prompt"], completion=values["completion"])


def load_examples(
    path: Path | str,
    max_bytes: int = MAX_DATASET_BYTES,
) -> tuple[TrainingExample, ...]:
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise DataValidationError(f"dataset not found: {dataset_path}")

    size_bytes = dataset_path.stat().st_size
    if size_bytes > max_bytes:
        raise DataValidationError(
            f"dataset exceeds {max_bytes} byte limit ({size_bytes} bytes): {dataset_path}"
        )

    with dataset_path.open("r", encoding="utf-8") as handle:
        raw_text = handle.read(max_bytes + 1)
    if len(raw_text.encode("utf-8")) > max_bytes:
        raise DataValidationError(
            f"dataset exceeds {max_bytes} byte limit: {dataset_path}"
        )

    examples: list[TrainingExample] = []
    for line_number, line in enumerate(raw_text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DataValidationError(f"line {line_number}: invalid JSON") from exc
        examples.append(_parse_example(raw, line_number))

    if not examples:
        raise DataValidationError("dataset must contain at least one training example")
    return tuple(examples)
