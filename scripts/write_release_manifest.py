import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Write release manifest for an Aina Nano artifact.")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--train-config", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--dataset-report", type=Path, default=None)
    parser.add_argument("--param-report", type=Path, default=None)
    parser.add_argument("--eval-report", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = {
        "model_name": args.model_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "model_config_path": str(args.model_config),
        "train_config_path": str(args.train_config),
        "checkpoint_dir": str(args.checkpoint_dir),
        "tokenizer_dir": str(args.tokenizer_dir),
        "model_config": read_json(args.model_config),
        "trainer_state": read_json(args.checkpoint_dir / "trainer_state.json"),
        "parameter_report": read_json(args.param_report) if args.param_report else {},
        "dataset_report": read_json(args.dataset_report) if args.dataset_report else {},
        "eval_report": str(args.eval_report) if args.eval_report else None,
        "intended_use": "Educational base language model artifact for internal production-pipeline learning.",
        "limitations": "Tiny base model; not instruction-tuned by default; outputs can be inaccurate, repetitive, unsafe, or nonsensical.",
        "dataset_summary": "See dataset_report in this manifest.",
        "training_summary": "See trainer_state, model_config, train_config_path, and parameter_report in this manifest.",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote release manifest to {args.output}")


if __name__ == "__main__":
    main()
