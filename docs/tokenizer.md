# Tokenizer

Tokenizer mengubah teks menjadi token ID. Model tidak membaca teks mentah secara langsung.

Aina Nano memakai SentencePiece BPE tokenizer agar kompatibel dengan workflow LLaMA/Hugging Face/GGUF.

## Train

Smoke/local:

```bash
python tokenizer/train_tokenizer.py --vocab-size 1024
```

Larger run:

```bash
python tokenizer/train_tokenizer.py --vocab-size 8000
```

Output:

```text
tokenizer/tokenizer.model
tokenizer/tokenizer.vocab
tokenizer/tokenizer_config.json
tokenizer/special_tokens_map.json
```

## Special Tokens

```text
<unk>
<bos>
<eos>
<pad>
<|system|>
<|user|>
<|assistant|>
```

Catatan: special token untuk instruction tuning harus diuji dengan runtime target sebelum training panjang.

## Validate

```bash
python tokenizer/test_tokenizer.py
```

Decoded text harus mirip input. Jika kosong atau banyak `<unk>` pada teks umum, tokenizer perlu diperiksa.

## Rule

Tokenizer adalah kontrak. Jangan mengganti tokenizer untuk checkpoint yang sudah dilatih.
