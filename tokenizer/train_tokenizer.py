import argparse
import json
import tempfile
from pathlib import Path

import sentencepiece as spm
from transformers import LlamaTokenizerFast

USER_DEFINED_TOKENS = ["<|system|>", "<|user|>", "<|assistant|>"]


def iter_texts(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)["text"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Aina Nano SentencePiece BPE tokenizer.")
    parser.add_argument("--train-file", type=Path, default=Path("data/clean/train.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("tokenizer"))
    parser.add_argument("--vocab-size", type=int, default=512)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as corpus:
        for text in iter_texts(args.train_file):
            corpus.write(text + "\n")
        corpus_path = corpus.name

    model_prefix = str(args.output_dir / "aina_spm")
    spm.SentencePieceTrainer.Train(
        input=corpus_path,
        model_prefix=model_prefix,
        vocab_size=args.vocab_size,
        model_type="bpe",
        character_coverage=1.0,
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3,
        unk_piece="<unk>",
        bos_piece="<bos>",
        eos_piece="<eos>",
        pad_piece="<pad>",
        user_defined_symbols=USER_DEFINED_TOKENS,
        hard_vocab_limit=False,
    )

    tokenizer_model = args.output_dir / "tokenizer.model"
    tokenizer_vocab = args.output_dir / "tokenizer.vocab"
    Path(model_prefix + ".model").replace(tokenizer_model)
    Path(model_prefix + ".vocab").replace(tokenizer_vocab)

    tokenizer = LlamaTokenizerFast(
        vocab_file=str(tokenizer_model),
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
        pad_token="<pad>",
        additional_special_tokens=USER_DEFINED_TOKENS,
        legacy=False,
    )
    tokenizer.save_pretrained(args.output_dir)
    config_path = args.output_dir / "tokenizer_config.json"
    tokenizer_config = json.loads(config_path.read_text(encoding="utf-8"))
    tokenizer_config.pop("extra_special_tokens", None)
    tokenizer_config["tokenizer_class"] = "LlamaTokenizer"
    tokenizer_config["model_max_length"] = 1024
    config_path.write_text(json.dumps(tokenizer_config, indent=2), encoding="utf-8")
    (args.output_dir / "tokenizer.json").unlink(missing_ok=True)
    print(f"saved LLaMA-compatible tokenizer to {args.output_dir}")


if __name__ == "__main__":
    main()
