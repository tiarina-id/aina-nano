import argparse
from pathlib import Path

import torch
from transformers import LlamaForCausalLM, LlamaTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text with an Aina Nano checkpoint.")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--prompt", type=str, default="Apa itu API?")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    tokenizer = LlamaTokenizer.from_pretrained(args.model_dir)
    model = LlamaForCausalLM.from_pretrained(args.model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    input_ids = tokenizer.encode(args.prompt, return_tensors="pt", add_special_tokens=False).to(device)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    print(tokenizer.decode(output_ids[0], skip_special_tokens=False))


if __name__ == "__main__":
    main()
