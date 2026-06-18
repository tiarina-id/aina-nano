# Training

Aina Nano provides two main training configs:

- `train/configs/aina_nano_1m.yaml` for a small sanity-check model.
- `train/configs/aina_nano_10m.yaml` for a larger toy model.

## Local Sanity Run

Use the 1M config to verify that the full training loop works before attempting a longer run.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu

python data/clean_dataset.py
python tokenizer/train_tokenizer.py --vocab-size 1024
python tokenizer/test_tokenizer.py
python train/pretrain.py --config train/configs/aina_nano_1m.yaml
```

Expected output:

```txt
checkpoints/aina-nano-1m/pretrain/
```

## Larger Toy Run

After the 1M sanity run succeeds, use the 10M config for a longer experiment:

```bash
python data/clean_dataset.py
python tokenizer/train_tokenizer.py --vocab-size 8000
python tokenizer/test_tokenizer.py
python train/pretrain.py --config train/configs/aina_nano_10m.yaml
```

Expected output:

```txt
checkpoints/aina-nano-10m/pretrain/
```

## Export

```bash
python export/export_hf.py \
  --checkpoint-dir checkpoints/aina-nano-10m/pretrain \
  --output-dir models/aina-nano-10m-base
```
