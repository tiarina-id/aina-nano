import argparse
import json
from pathlib import Path

from transformers import LlamaConfig, LlamaForCausalLM


def main() -> None:
    parser = argparse.ArgumentParser(description="Count model parameters from a LLaMA config.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config = LlamaConfig.from_pretrained(args.config)
    model = LlamaForCausalLM(config)
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    result = {
        "config": str(args.config),
        "total_parameters": total,
        "trainable_parameters": trainable,
        "billions": round(total / 1_000_000_000, 6),
        "millions": round(total / 1_000_000, 3),
    }

    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
