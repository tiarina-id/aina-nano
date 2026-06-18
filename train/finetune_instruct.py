import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
from train.path_utils import expand_path
from transformers import LlamaForCausalLM, LlamaTokenizer

IGNORE_INDEX = -100


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


def render_prompt(messages: list[dict[str, str]]) -> tuple[str, str]:
    system = messages[0]["content"].strip()
    user = messages[1]["content"].strip()
    assistant = messages[2]["content"].strip()
    prompt = f"System:\n{system}\nUser:\n{user}\nAssistant:\n"
    answer = f"{assistant}<eos>"
    return prompt, answer


class InstructionDataset(Dataset):
    def __init__(self, path: Path, tokenizer: LlamaTokenizer, context_length: int):
        self.examples: list[dict[str, torch.Tensor]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                prompt, answer = render_prompt(payload["messages"])
                prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
                answer_ids = tokenizer.encode(answer, add_special_tokens=False)
                input_ids = (prompt_ids + answer_ids)[:context_length]
                labels = [IGNORE_INDEX] * min(len(prompt_ids), len(input_ids))
                labels += input_ids[len(labels) :]
                if len(input_ids) < 2 or all(label == IGNORE_INDEX for label in labels):
                    continue
                self.examples.append(
                    {
                        "input_ids": torch.tensor(input_ids, dtype=torch.long),
                        "labels": torch.tensor(labels, dtype=torch.long),
                    }
                )
        if not self.examples:
            raise ValueError(f"No usable instruction examples found in {path}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.examples[index]


def collate_batch(batch: list[dict[str, torch.Tensor]], pad_token_id: int) -> dict[str, torch.Tensor]:
    max_length = max(item["input_ids"].numel() for item in batch)
    input_ids = torch.full((len(batch), max_length), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_length), dtype=torch.long)
    labels = torch.full((len(batch), max_length), IGNORE_INDEX, dtype=torch.long)
    for row, item in enumerate(batch):
        length = item["input_ids"].numel()
        input_ids[row, :length] = item["input_ids"]
        attention_mask[row, :length] = 1
        labels[row, :length] = item["labels"]
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
    parser = argparse.ArgumentParser(description="Fine-tune Aina Nano on instruction JSONL.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_simple_yaml(args.config)
    random.seed(int(cfg["seed"]))
    torch.manual_seed(int(cfg["seed"]))

    tokenizer = LlamaTokenizer.from_pretrained(str(expand_path(cfg["tokenizer_dir"])))
    model = LlamaForCausalLM.from_pretrained(str(expand_path(cfg["base_model_dir"])))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataset = InstructionDataset(expand_path(cfg["train_file"]), tokenizer, int(cfg["context_length"]))
    val_size = max(1, int(len(dataset) * 0.1))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(int(cfg["seed"])),
    )

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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
    )

    output_dir = expand_path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    max_steps = int(cfg["max_steps"])
    global_step = 0
    model.train()

    progress = tqdm(total=max_steps, desc="instruction-tuning")
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

            if global_step % int(cfg["eval_every_steps"]) == 0:
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
    print(f"saved instruct checkpoint to {output_dir}")


if __name__ == "__main__":
    main()
