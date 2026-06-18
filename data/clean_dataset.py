import argparse
import json
import random
import re
from pathlib import Path


def normalize_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_records(raw_dir: Path, min_chars: int) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    seen: set[str] = set()

    for path in sorted(raw_dir.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc

                text = normalize_text(str(payload.get("text", "")))
                if len(text) < min_chars or text in seen:
                    continue
                seen.add(text)
                records.append({"text": text})

    return records


def write_jsonl(path: Path, records: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean local raw JSONL text data.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/clean"))
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--min-chars", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = load_records(args.raw_dir, args.min_chars)
    if len(records) < 2:
        raise SystemExit("Need at least two cleaned records to create train/val splits.")

    random.Random(args.seed).shuffle(records)
    val_size = max(1, int(len(records) * args.val_ratio))
    val_records = records[:val_size]
    train_records = records[val_size:]

    write_jsonl(args.output_dir / "train.jsonl", train_records)
    write_jsonl(args.output_dir / "val.jsonl", val_records)
    print(f"cleaned={len(records)} train={len(train_records)} val={len(val_records)}")


if __name__ == "__main__":
    main()
