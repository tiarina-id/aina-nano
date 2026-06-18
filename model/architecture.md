# Aina Nano Architecture

Aina Nano uses a tiny LLaMA-compatible decoder-only Transformer so the learning pipeline stays close to production LLM tooling.

## Why LLaMA-compatible

- Hugging Face can load and save the model with standard `LlamaForCausalLM` classes.
- GGUF conversion is easier than with a fully custom architecture.
- Ollama and llama.cpp are designed around LLaMA-style model artifacts.

## Initial sizes

- `config_1m.json`: sanity-check model for local pipeline validation.
- `config_10m.json`: next target after the 1M pipeline works end-to-end.
