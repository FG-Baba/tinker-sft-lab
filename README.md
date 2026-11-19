# Tinker SFT Lab

A small, cost-guarded supervised fine-tuning (SFT) CLI for [Thinking Machines' Tinker API](https://tinker-docs.thinkingmachines.ai/).

When coding agents have access to a training CLI, a single bad flag or malformed JSONL file can kick off an unintended GPU run before anyone checks the data. This CLI puts three hard gates in front of paid training:

1. **Local dataset validation first (`validate`):** Verifies non-empty `prompt` and `completion` strings on every JSONL line (with a 10 MB file-size cap) before touching the network.
2. **Two-key execution guard (`--execute` + `TINKER_API_KEY`):** Both the `train-sft` CLI command and `run_sft` check that `TINKER_API_KEY` and explicit execution authorization are present before creating a Tinker client.
3. **Bounded run ceilings:** Defaults to `Qwen/Qwen3.5-4B` (truncating training examples to `512` tokens by default), `1` step, and `8` examples, and blocks runs above `10` steps or `32` examples unless you intentionally raise the ceiling in code. Step and example caps bound run size rather than acting as a dollar billing cap, so always check current Tinker per-token and LoRA pricing before running live training.

## Quick start

Requires Python `3.11+` and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env
```

Add `TINKER_API_KEY` to `.env`, then run the two free local checks:

```bash
uv run --env-file .env tinker-lab doctor
uv run tinker-lab validate data/example.jsonl
```

When you are ready to run a real billed SFT step:

```bash
uv run --env-file .env tinker-lab train-sft data/example.jsonl --execute
```

## Dataset format

One JSON object per line:

```json
{"prompt": "User input or instruction", "completion": "Expected model response"}
```

## Run tests

```bash
uv run pytest -q
uv run ruff check .
```

All tests (`27` unit and CLI tests) run offline and verify dataset validation, the 10 MB file-size boundary, CLI exit codes, step and example bounds, and the two-key `--execute` gate without making network calls.

## License

MIT (`LICENSE`)
