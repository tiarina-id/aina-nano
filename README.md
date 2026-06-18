# Aina Nano Internal LLM Pipeline

Aina Nano adalah repo internal untuk belajar lifecycle LLM secara end-to-end dalam skala kecil. Tujuannya bukan membuat model yang langsung pintar, tetapi memahami alur production-style yang juga dipakai pada model besar: data preparation, tokenizer, pretraining, checkpointing, evaluation, export, inference, dan serving.

Repo ini sengaja memakai model kecil 1M–10M parameter agar seluruh pipeline bisa dipelajari, dijalankan, dan di-debug tanpa infrastruktur besar.

## Tujuan Belajar

- Memahami alur pembuatan model bahasa dari dataset mentah sampai inference.
- Melatih tokenizer sendiri dan memahami konsekuensi perubahan tokenizer.
- Melatih model decoder-only Transformer kecil yang LLaMA-compatible.
- Menyimpan checkpoint dalam format Hugging Face / `safetensors`.
- Mengonversi model ke GGUF untuk runtime lokal seperti llama.cpp/Ollama.
- Membiasakan workflow production: smoke test, guardrail, artifact management, dan dokumentasi runbook.
- Mengetahui batas model kecil sebelum naik ke training yang lebih mahal.

## Mental Model

Pipeline production LLM besar secara konsep:

```text
raw data
→ data cleaning/filtering/dedup
→ train tokenizer
→ tokenize or stream dataset
→ pretraining base model
→ evaluate base model
→ instruction tuning / alignment
→ evaluate chat model
→ export model artifacts
→ inference runtime
→ monitoring and feedback loop
```

Aina Nano meniru alur itu dalam bentuk kecil:

```text
data/raw/*.jsonl
→ data/clean/train.jsonl + data/clean/val.jsonl
→ tokenizer/tokenizer.model
→ checkpoints/aina-nano-*/pretrain
→ models/aina-nano-*
→ GGUF
→ local inference smoke test
```

## Scope Saat Ini

Termasuk:

- JSONL text dataset untuk pretraining.
- Dataset cleaning sederhana.
- SentencePiece BPE tokenizer LLaMA-compatible.
- LLaMA-compatible causal language model config 1M dan 10M.
- Training loop PyTorch sederhana.
- Evaluation via generation prompt.
- Hugging Face export.
- GGUF conversion helper.
- Dokumentasi production-learning dan runbook.

Belum termasuk production penuh:

- Distributed training multi-GPU.
- Optimized tokenized binary dataset streaming.
- Mixed precision training policy yang matang.
- Robust instruction tuning final.
- RLHF/DPO/alignment.
- Automated benchmark suite.
- Model registry/artifact store.
- Monitoring server production.

## Struktur Repo

```text
data/
  raw/              # dataset mentah JSONL, tidak masuk Git
  clean/            # hasil cleaning train/val, tidak masuk Git
  tokenized/        # tokenized artifacts opsional, tidak masuk Git
  instruction/      # dataset instruct opsional, tidak masuk Git
tokenizer/          # train/test tokenizer
model/              # config model dan catatan arsitektur
train/              # pretraining dan fine-tuning scripts
train/configs/      # config 1M dan 10M
eval/               # generate/eval prompt helpers
export/             # export HF dan convert GGUF
serving/            # Modelfile dan catatan serving lokal
docs/               # dokumentasi detail pipeline
scripts/            # utility scripts
checkpoints/        # output training, tidak masuk Git
models/             # exported models/GGUF, tidak masuk Git
```

## Dataset Format

Pretraining dataset memakai JSONL:

```json
{"text":"API adalah antarmuka yang memungkinkan aplikasi saling berkomunikasi."}
```

Simpan di:

```text
data/raw/*.jsonl
```

Instruction dataset memakai format chat JSONL:

```json
{"messages":[{"role":"system","content":"Kamu adalah Aina Nano."},{"role":"user","content":"Apa itu API?"},{"role":"assistant","content":"API adalah antarmuka antar aplikasi."}]}
```

Simpan di:

```text
data/instruction/*.jsonl
```

Catatan: untuk target GGUF/Ollama, chat template harus diuji lewat smoke test sebelum training lama. Jangan melakukan training instruct panjang sebelum template serving terbukti aman.

## Config Model

Config training yang dipakai:

- `train/configs/aina_nano_1m.yaml` untuk sanity/smoke test.
- `train/configs/aina_nano_10m.yaml` untuk eksperimen 10M.

Config arsitektur:

- `model/config_1m.json`
- `model/config_10m.json`

## Quickstart Smoke Test

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu

python scripts/check_device.py
python data/clean_dataset.py
python tokenizer/train_tokenizer.py --vocab-size 1024
python tokenizer/test_tokenizer.py
python train/pretrain.py --config train/configs/aina_nano_1m.yaml
python eval/generate.py --model-dir checkpoints/aina-nano-1m/pretrain --prompt "API adalah" --max-new-tokens 80
```

## 10M Training Run

Setelah smoke test 1M berhasil dan dataset cukup besar:

```bash
python data/clean_dataset.py
python tokenizer/train_tokenizer.py --vocab-size 8000
python tokenizer/test_tokenizer.py
python train/pretrain.py --config train/configs/aina_nano_10m.yaml
```

## Export Artifact

```bash
python export/export_hf.py \
  --checkpoint-dir checkpoints/aina-nano-10m/pretrain \
  --output-dir models/aina-nano-10m-base

./export/convert_gguf.sh \
  models/aina-nano-10m-base \
  models/aina-nano-10m-base.gguf
```

## Guardrail Internal

Sebelum menjalankan training mahal:

1. Dataset raw valid JSONL.
2. Cleaning menghasilkan train/val yang masuk akal.
3. Tokenizer encode/decode normal.
4. 1M smoke training berhasil.
5. HF export berhasil.
6. GGUF conversion berhasil.
7. Inference smoke test berhasil.
8. Baru jalankan config 10M.

## Dokumentasi Detail

- `docs/production_pipeline.md` — gambaran lengkap lifecycle production LLM.
- `docs/runbook.md` — urutan operasional dari dataset sampai export.
- `docs/guardrails.md` — aturan mencegah biaya/waktu terbuang.
- `docs/dataset.md` — format dan cleaning dataset.
- `docs/tokenizer.md` — tokenizer dan special tokens.
- `docs/training.md` — training configs dan output.
- `docs/deployment.md` — export, GGUF, dan serving lokal.
- `docs/scaling.md` — catatan scaling eksperimen.
- `docs/release.md` — checklist dan command release artifact.
