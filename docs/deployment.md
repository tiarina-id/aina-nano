# Deployment and Export

Deployment awal Aina Nano berarti mengubah checkpoint training menjadi artifact inference.

## Hugging Face Export

```bash
python export/export_hf.py \
  --checkpoint-dir checkpoints/aina-nano-10m/pretrain \
  --output-dir models/aina-nano-10m-base
```

Output umum:

```text
config.json
model.safetensors
tokenizer.model
tokenizer_config.json
special_tokens_map.json
generation_config.json
README.md
```

## GGUF Conversion

GGUF dipakai oleh runtime seperti llama.cpp/Ollama.

```bash
./export/convert_gguf.sh \
  models/aina-nano-10m-base \
  models/aina-nano-10m-base.gguf
```

## Local Runtime Smoke Test

Gunakan prompt sederhana dulu:

```text
API adalah
Tokenizer adalah
```

Tujuan smoke test:

- model bisa dimuat
- tokenizer runtime tidak error
- output tidak kosong
- stop token bekerja

## Important

Jangan menyimpulkan kualitas model dari smoke test. Smoke test hanya membuktikan artifact bisa dijalankan.
