import argparse
import json
from pathlib import Path

from transformers import LlamaTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Count approximate tokenizer tokens in JSONL text data.")
    parser.add_argument("--tokenizer-dir", type=Path, default=Path("tokenizer"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_dir)
    files = sorted(args.input.glob("*.jsonl")) if args.input.is_dir() else [args.input]
    total_records = 0
    total_chars = 0
    total_tokens = 0
    per_file = []

    for path in files:
        records = 0
        chars = 0
        tokens = 0
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                text = payload.get("text", "")
                if not isinstance(text, str):
                    raise ValueError(f"text must be string at {path}:{line_number}")
                token_ids = tokenizer.encode(text, add_special_tokens=False)
                records += 1
                chars += len(text)
                tokens += len(token_ids)
        per_file.append({"path": str(path), "records": records, "chars": chars, "tokens": tokens})
        total_records += records
        total_chars += chars
        total_tokens += tokens

    result = {
        "tokenizer_dir": str(args.tokenizer_dir),
        "input": str(args.input),
        "records": total_records,
        "chars": total_chars,
        "tokens": total_tokens,
        "files": per_file,
    }
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
