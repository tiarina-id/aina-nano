from pathlib import Path

from transformers import LlamaTokenizer

TOKENIZER_DIR = Path("tokenizer")


def main() -> None:
    tokenizer = LlamaTokenizer.from_pretrained(TOKENIZER_DIR)
    text = "Halo, saya Aina Nano. Apa itu API?"
    token_ids = tokenizer.encode(text)
    decoded = tokenizer.decode(token_ids)
    print(f"input: {text}")
    print(f"ids: {token_ids}")
    print(f"decoded: {decoded}")


if __name__ == "__main__":
    main()
