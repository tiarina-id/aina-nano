import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer

try:
    from train.path_utils import expand_path
except ModuleNotFoundError:
    from path_utils import expand_path


def load_simple_yaml(path: Path) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, value = line.split(":", 1)
        value = value.strip()
        if value.lower() in {"true", "false"}:
            parsed: Any = value.lower() == "true"
        else:
            try:
                parsed = int(value)
            except ValueError:
                try:
                    parsed = float(value)
                except ValueError:
                    parsed = value
        config[key.strip()] = parsed
    return config


def load_texts(path: Path) -> list[str]:
    texts: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                texts.append(json.loads(line)["text"])
    return texts


class CausalTextDataset(Dataset):
    def __init__(self, texts: list[str], tokenizer, context_length: int):
        self.examples: list[torch.Tensor] = []
        eos = tokenizer.eos_token or "<eos>"
        for text in texts:
            token_ids = tokenizer.encode(text + eos, add_special_tokens=False)
            for start in range(0, max(1, len(token_ids)), context_length):
                chunk = token_ids[start : start + context_length]
                if len(chunk) < 2:
                    continue
                self.examples.append(torch.tensor(chunk, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.examples[index]


def collate_batch(batch: list[torch.Tensor], pad_token_id: int) -> dict[str, torch.Tensor]:
    max_length = max(item.numel() for item in batch)
    input_ids = torch.full((len(batch), max_length), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_length), dtype=torch.long)
    for row, item in enumerate(batch):
        input_ids[row, : item.numel()] = item
        attention_mask[row, : item.numel()] = 1
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


@torch.no_grad()
def evaluate(model: LlamaForCausalLM, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    losses: list[float] = []
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        output = model(**batch)
        losses.append(float(output.loss.detach().cpu()))
    model.train()
    return sum(losses) / max(1, len(losses))


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain tiny Aina Nano causal LM.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_simple_yaml(args.config)
    random.seed(int(cfg["seed"]))
    torch.manual_seed(int(cfg["seed"]))

    tokenizer = LlamaTokenizer.from_pretrained(str(expand_path(cfg["tokenizer_dir"])))
    model_config = LlamaConfig.from_pretrained(str(cfg["model_config"]))
    model_config.vocab_size = len(tokenizer)
    model_config.pad_token_id = tokenizer.pad_token_id
    model_config.bos_token_id = tokenizer.bos_token_id
    model_config.eos_token_id = tokenizer.eos_token_id

    train_dataset = CausalTextDataset(load_texts(expand_path(cfg["train_file"])), tokenizer, int(cfg["context_length"]))
    val_dataset = CausalTextDataset(load_texts(expand_path(cfg["val_file"])), tokenizer, int(cfg["context_length"]))
    if not train_dataset:
        raise SystemExit("No training examples found. Run data/clean_dataset.py first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LlamaForCausalLM(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        collate_fn=lambda batch: collate_batch(batch, tokenizer.pad_token_id),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        collate_fn=lambda batch: collate_batch(batch, tokenizer.pad_token_id),
    )

    output_dir = expand_path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    max_steps = int(cfg["max_steps"])
    global_step = 0
    model.train()

    progress = tqdm(total=max_steps, desc="pretraining")
    for _epoch in range(int(cfg["epochs"])):
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**batch)
            loss = output.loss
            if not math.isfinite(float(loss.detach().cpu())):
                raise RuntimeError(f"Non-finite loss at step {global_step}: {loss}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            progress.update(1)
            progress.set_postfix(loss=f"{float(loss.detach().cpu()):.4f}")

            if global_step % int(cfg["eval_every_steps"]) == 0 and len(val_dataset) > 0:
                val_loss = evaluate(model, val_loader, device)
                print(f"step={global_step} val_loss={val_loss:.4f}")
            if global_step % int(cfg["save_every_steps"]) == 0:
                model.save_pretrained(output_dir, safe_serialization=True)
                tokenizer.save_pretrained(output_dir)
            if global_step >= max_steps:
                break
        if global_step >= max_steps:
            break

    progress.close()
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    (output_dir / "trainer_state.json").write_text(json.dumps({"global_step": global_step}, indent=2), encoding="utf-8")
    print(f"saved checkpoint to {output_dir}")


if __name__ == "__main__":
    main()
