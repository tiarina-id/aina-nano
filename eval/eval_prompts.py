import argparse
import json
from pathlib import Path

import torch
from transformers import LlamaForCausalLM, LlamaTokenizer


def load_prompts(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run prompt evaluation and save JSONL report.")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, default=Path("eval/prompts.txt"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()

    tokenizer = LlamaTokenizer.from_pretrained(args.model_dir)
    model = LlamaForCausalLM.from_pretrained(args.model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for prompt in load_prompts(args.prompts):
            input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(device)
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
            handle.write(json.dumps({"prompt": prompt, "output": text}, ensure_ascii=False) + "\n")
            print(f"PROMPT: {prompt}\nOUTPUT: {text}\n")


if __name__ == "__main__":
    main()
