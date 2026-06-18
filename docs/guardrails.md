# Guardrails

Dokumen ini berisi aturan internal untuk menghindari training yang membuang waktu/biaya.

## 1. Never Scale Before Smoke Test

Sebelum training panjang, wajib lolos:

```text
clean dataset
→ train tokenizer
→ 1M train
→ generate
→ HF export
→ optional GGUF export
```

Jika salah satu gagal, jangan lanjut ke 10M.

## 2. Tokenizer Is a Contract

Setelah model dilatih:

- jangan retrain tokenizer untuk checkpoint yang sama
- jangan ubah special tokens
- jangan ubah vocab size
- jangan mix checkpoint dengan tokenizer lain

Jika tokenizer berubah, training harus dianggap eksperimen baru.

## 3. Instruct Template Must Be Proven First

Instruction tuning rawan gagal jika template tidak cocok dengan runtime target.

Aturan:

- pilih chat template sebelum training
- train SFT hanya 10–50 step dulu
- export
- convert
- run inference
- baru lakukan SFT lebih panjang

Jangan melakukan SFT mahal sebelum template inference terbukti aman.

## 4. Dataset Quality Beats Dataset Size

Lebih baik dataset kecil tapi unik dan bersih daripada besar tapi duplikat.

Check:

- jumlah baris
- jumlah unique exact text
- sample manual
- top duplicate
- train/val split

## 5. Watch for Overfitting

Tanda overfit:

- train loss turun tajam
- val loss tidak turun atau naik
- output mengulang kalimat dataset
- jawaban terlalu mirip contoh training

Untuk model kecil, overfit bisa berguna untuk belajar, tetapi jangan disangka kualitas general model.

## 6. Keep Artifacts Out of Git

Jangan commit:

- raw dataset
- clean dataset
- tokenizer hasil eksperimen jika masih berubah
- checkpoints
- exported models
- GGUF
- virtualenv
- cache

Artifact harus dipindahkan lewat storage/artifact registry terpisah jika diperlukan.

## 7. Stop Conditions

Hentikan training jika:

- loss menjadi NaN
- disk hampir penuh
- checkpoint tidak bisa disimpan
- tokenizer decode rusak
- validation loss tidak masuk akal sejak awal
- model tidak bisa reload dari checkpoint

## 8. Versioning Rule

Setiap eksperimen penting harus punya nama versi yang jelas:

```text
aina-nano-1m-smoke
aina-nano-10m-base-v0
aina-nano-10m-base-v1
aina-nano-10m-instruct-v0
```

Jangan overwrite artifact penting tanpa catatan.

## 9. Release Requires Manifest

Artifact tidak dianggap release-ready jika belum punya:

```text
reports/<model-name>/manifest.json
reports/<model-name>/params.json
reports/<model-name>/train_tokens.json
reports/<model-name>/eval_prompts.jsonl
models/<artifact>/README.md
```

Gunakan:

```bash
./scripts/release_base_model.sh 10m aina-nano-0.01b-base
```
