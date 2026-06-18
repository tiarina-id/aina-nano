# Local Pipeline

Local pipeline dipakai untuk membuktikan end-to-end flow sebelum eksperimen lebih besar.

```text
raw JSONL
→ clean train/val JSONL
→ train tokenizer
→ train 1M model
→ generate sample
→ export Hugging Face
→ optional GGUF conversion
```

## Minimal Flow

```bash
python data/clean_dataset.py
python tokenizer/train_tokenizer.py --vocab-size 1024
python tokenizer/test_tokenizer.py
python train/pretrain.py --config train/configs/aina_nano_1m.yaml
python eval/generate.py --model-dir checkpoints/aina-nano-1m/pretrain --prompt "API adalah" --max-new-tokens 80
```

## Success Criteria

- cleaning menghasilkan train/val
- tokenizer encode/decode normal
- training tidak NaN
- checkpoint tersimpan
- model bisa reload
- generate menghasilkan teks

Kualitas teks belum menjadi target utama local smoke test.
