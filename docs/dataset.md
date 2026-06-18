# Dataset

Aina Nano memakai dua jenis dataset:

1. Pretraining dataset untuk base model.
2. Instruction dataset untuk fine-tuning chat, jika sudah siap.

## Pretraining Format

Lokasi:

```text
data/raw/*.jsonl
```

Format per baris:

```json
{"text":"API adalah antarmuka antar aplikasi."}
```

Aturan:

- satu baris = satu JSON object
- field wajib `text`
- jangan memakai array JSON
- jangan multiline JSON
- teks harus legal, aman, dan tidak berisi secret

## Cleaning

Run:

```bash
python data/clean_dataset.py
```

Output:

```text
data/clean/train.jsonl
data/clean/val.jsonl
```

Cleaning saat ini:

- normalisasi whitespace
- hapus HTML tag sederhana
- hapus baris kosong
- hapus teks terlalu pendek
- dedup exact match
- shuffle dan split train/val

## Instruction Format

Lokasi:

```text
data/instruction/*.jsonl
```

Format:

```json
{"messages":[{"role":"system","content":"Kamu adalah Aina Nano."},{"role":"user","content":"Apa itu API?"},{"role":"assistant","content":"API adalah antarmuka antar aplikasi."}]}
```

Instruction tuning harus menunggu template chat terbukti aman lewat smoke test export/inference.
