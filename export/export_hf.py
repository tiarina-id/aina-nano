import argparse
import json
import shutil
from pathlib import Path

from transformers import LlamaForCausalLM, LlamaTokenizer


def load_json(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_model_card(model_name: str, metadata: dict) -> str:
    intended_use = metadata.get("intended_use", "Educational LLM pipeline artifact for local experimentation.")
    limitations = metadata.get(
        "limitations",
        "This is a tiny model trained for pipeline validation. Outputs may be inaccurate, repetitive, or nonsensical.",
    )
    dataset_summary = metadata.get("dataset_summary", "See the internal experiment manifest for dataset details.")
    training_summary = metadata.get("training_summary", "See trainer_state.json and the release manifest for training details.")

    return f"""# {model_name}

## Summary

{model_name} is an Aina Nano model artifact exported in Hugging Face-compatible format.

## Intended Use

{intended_use}

## Dataset

{dataset_summary}

## Training

{training_summary}

## Limitations

{limitations}

## Usage

```python
from transformers import LlamaForCausalLM, LlamaTokenizer

tokenizer = LlamaTokenizer.from_pretrained("{model_name}")
model = LlamaForCausalLM.from_pretrained("{model_name}")
```

## Notes

This artifact is part of an internal production-learning pipeline. It should pass reload, generation, and export smoke tests before being promoted.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Export checkpoint to Hugging Face model directory.")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--metadata", type=Path, default=None)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = LlamaForCausalLM.from_pretrained(args.checkpoint_dir)
    tokenizer = LlamaTokenizer.from_pretrained(args.checkpoint_dir)
    model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)

    generation_config = {
        "max_new_tokens": 128,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
    (args.output_dir / "generation_config.json").write_text(json.dumps(generation_config, indent=2), encoding="utf-8")

    metadata = load_json(args.metadata)
    model_name = args.model_name or args.output_dir.name
    (args.output_dir / "README.md").write_text(build_model_card(model_name, metadata), encoding="utf-8")

    if metadata:
        (args.output_dir / "release_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    state_file = args.checkpoint_dir / "trainer_state.json"
    if state_file.exists():
        shutil.copy2(state_file, args.output_dir / "trainer_state.json")

    print(f"exported Hugging Face model to {args.output_dir}")


if __name__ == "__main__":
    main()
