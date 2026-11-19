import json
import tempfile
import unittest
from pathlib import Path

from tinker_lab.data import (
    MAX_DATASET_BYTES,
    DataValidationError,
    TrainingExample,
    load_examples,
)


class LoadExamplesTests(unittest.TestCase):
    def write_jsonl(self, rows: list[object]) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "examples.jsonl"
        path.write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n",
            encoding="utf-8",
        )
        return path

    def test_loads_valid_examples_as_an_immutable_tuple(self) -> None:
        sft_def = "Supervised fine-tuning on labeled prompt-completion pairs."
        path = self.write_jsonl(
            [
                {"prompt": "What is SFT?", "completion": sft_def},
                {"prompt": "Be concise.", "completion": "Understood."},
            ]
        )

        examples = load_examples(path)

        self.assertEqual(
            examples,
            (
                TrainingExample("What is SFT?", sft_def),
                TrainingExample("Be concise.", "Understood."),
            ),
        )

    def test_rejects_blank_fields_with_the_line_number(self) -> None:
        path = self.write_jsonl([{"prompt": "Question", "completion": "  "}])

        with self.assertRaisesRegex(DataValidationError, r"line 1.*completion"):
            load_examples(path)

    def test_rejects_malformed_json_with_the_line_number(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "bad.jsonl"
        path.write_text('{"prompt": "missing end"\n', encoding="utf-8")

        with self.assertRaisesRegex(DataValidationError, r"line 1.*JSON"):
            load_examples(path)

    def test_rejects_non_object_rows(self) -> None:
        path = self.write_jsonl([["not", "an", "object"]])

        with self.assertRaisesRegex(DataValidationError, r"line 1.*object"):
            load_examples(path)

    def test_rejects_unknown_or_missing_fields(self) -> None:
        path = self.write_jsonl([{"prompt": "Question", "answer": "Response"}])

        with self.assertRaisesRegex(DataValidationError, r"line 1.*exactly"):
            load_examples(path)

    def test_rejects_a_missing_file(self) -> None:
        with self.assertRaisesRegex(DataValidationError, "not found"):
            load_examples("does-not-exist.jsonl")

    def test_rejects_an_empty_dataset(self) -> None:
        path = self.write_jsonl([])

        with self.assertRaisesRegex(DataValidationError, "at least one"):
            load_examples(path)

    def test_accepts_dataset_at_exact_byte_limit_and_rejects_above_it(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        row = json.dumps({"prompt": "Hi", "completion": "Hello"}) + "\n"
        exact_path = Path(directory.name) / "exact.jsonl"
        exact_path.write_text(row, encoding="utf-8")
        exact_len = exact_path.stat().st_size

        self.assertEqual(len(load_examples(exact_path, max_bytes=exact_len)), 1)
        with self.assertRaisesRegex(DataValidationError, "byte limit"):
            load_examples(exact_path, max_bytes=exact_len - 1)
        self.assertEqual(MAX_DATASET_BYTES, 10 * 1024 * 1024)


if __name__ == "__main__":
    unittest.main()
