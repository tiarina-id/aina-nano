# Production LLM Pipeline Notes

Dokumen ini menjelaskan alur production LLM yang ingin ditiru Aina Nano dalam skala kecil.

## 1. Data Layer

Pada model besar, data adalah bagian paling penting dan paling mahal. Pipeline umum:

```text
raw sources
→ extraction
→ normalization
→ filtering
→ deduplication
→ safety/PII filtering
→ quality scoring
→ train/validation split
```

Aina Nano versi kecil:

```text
data/raw/*.jsonl
→ data/clean_dataset.py
→ data/clean/train.jsonl
→ data/clean/val.jsonl
```

Tujuan tahap ini:

- memastikan JSONL valid
- menghapus baris kosong
- menghapus teks terlalu pendek
- normalisasi whitespace
- dedup exact match
- membuat validation split

## 2. Tokenizer Layer

Tokenizer adalah kontrak permanen antara teks dan model. Setelah model dilatih, tokenizer tidak boleh diganti sembarangan.

Pipeline:

```text
clean text
→ train SentencePiece/BPE tokenizer
→ save tokenizer.model + tokenizer_config
→ validate encode/decode
```

Aina Nano memakai SentencePiece BPE agar lebih mudah dikonversi ke format LLaMA/GGUF.

Risiko yang harus dihindari:

- tokenizer berubah setelah checkpoint dibuat
- special token berbeda antara training dan inference
- tokenizer bisa encode tetapi decode rusak
- tokenizer tidak kompatibel dengan runtime target

## 3. Model Architecture Layer

Aina Nano memakai arsitektur decoder-only LLaMA-compatible:

- causal self-attention
- RoPE position embedding
- RMSNorm
- SwiGLU/SILU feed-forward
- next-token prediction objective

Kenapa LLaMA-compatible:

- mudah dipakai dengan Hugging Face Transformers
- lebih mudah dikonversi ke GGUF
- lebih dekat dengan runtime lokal seperti llama.cpp/Ollama

## 4. Pretraining Layer

Pretraining melatih model memprediksi token berikutnya.

```text
input tokens:  Indonesia adalah negara di
label:         Indonesia adalah negara di Asia
```

Di code:

```text
train/pretrain.py
```

Output:

```text
checkpoints/aina-nano-*/pretrain/
```

Yang dipantau:

- train loss turun
- val loss turun atau stabil
- tidak NaN
- checkpoint tersimpan
- model bisa reload
- generate tidak crash

## 5. Evaluation Layer

Evaluasi awal bukan untuk membuktikan model pintar, tetapi membuktikan pipeline sehat.

Minimal checks:

- model load dari checkpoint
- prompt sederhana menghasilkan teks
- output tidak kosong
- tidak infinite loop
- loss tidak NaN
- tokenizer decode masih normal

Prompt contoh:

```text
API adalah
Tokenizer adalah
Dalam pipeline LLM, checkpoint
```

## 6. Export Layer

Training checkpoint harus bisa dipindahkan ke artifact inference.

```text
checkpoint
→ Hugging Face folder
→ GGUF
→ runtime lokal
```

Hugging Face artifact biasanya berisi:

```text
config.json
model.safetensors
tokenizer.model
tokenizer_config.json
special_tokens_map.json
generation_config.json
README.md
```

## 7. Inference Layer

Inference berbeda dari training.

Training fokus pada:

- gradient
- optimizer
- checkpoint
- throughput batch

Inference fokus pada:

- prompt format
- latency
- memory
- stop token
- sampling
- compatibility runtime

Untuk Aina Nano, inference lokal diuji via GGUF/runtime lokal.

## 8. Instruction Tuning Layer

Instruction tuning hanya dilakukan setelah base model cukup stabil.

Pipeline:

```text
base checkpoint
+ instruction dataset
+ fixed chat template
→ instruct checkpoint
→ export
→ inference smoke test
```

Peringatan penting: chat template harus dipilih berdasarkan runtime target dan diuji kecil terlebih dahulu. Jangan menjalankan SFT panjang sebelum template terbukti bisa dipakai di export/runtime.

## 9. Model Iteration Loop

Production model bukan satu kali train. Siklusnya:

```text
evaluate failures
→ improve data
→ retrain/fine-tune
→ export
→ test inference
→ compare versions
```

Untuk Aina Nano, versi eksperimen harus dicatat minimal:

- dataset yang dipakai
- vocab size tokenizer
- config model
- max steps
- loss akhir
- sample output
- export status
- inference status
