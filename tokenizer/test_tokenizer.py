import argparse
from pathlib import Path

from transformers import LlamaTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Aina Nano tokenizer encode/decode.")
    parser.add_argument("--tokenizer-dir", type=Path, default=Path("tokenizer"))
    parser.add_argument("--text", type=str, default="Halo, saya Aina Nano. Apa itu API?")
    args = parser.parse_args()

    tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_dir)
    token_ids = tokenizer.encode(args.text)
    decoded = tokenizer.decode(token_ids)
    print(f"tokenizer_dir: {args.tokenizer_dir}")
    print(f"input: {args.text}")
    print(f"ids: {token_ids}")
    print(f"decoded: {decoded}")


if __name__ == "__main__":
    main()
