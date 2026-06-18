# Change Plan: Output Path for Training Artifacts

## Summary

Simpan output training dan artifact model di folder user otomatis, bukan di dalam repo source code. Default root output memakai:

```bash
/home/data/aina-runs
```

Jika ada mounted disk khusus, path bisa dioverride dengan:

```bash
export AINA_RUN_ROOT=/mnt/aina-data/aina-runs
```

Tujuan perubahan ini adalah memisahkan:

```text
repo code      = source code, config, docs
training output = dataset hasil clean, tokenizer, checkpoint, model, report, GGUF, log
```

## Proposed Output Structure

```text
$AINA_RUN_ROOT/
├── data/
│   ├── raw/
│   └── clean/
├── tokenizers/
│   └── aina-nano-10m/
├── checkpoints/
│   └── aina-nano-10m/
│       └── pretrain/
├── models/
│   └── aina-nano-10m-base/
├── gguf/
│   └── aina-nano-10m-base.gguf
├── reports/
│   └── aina-nano-0.01b-base/
└── logs/
```

## Planned Script Changes

Update path handling in training/export/release scripts so paths can use:

- `${AINA_RUN_ROOT}`
- `${HOME}`
- `~`
- normal relative paths

Scripts that may need updates:

```text
train/pretrain.py
train/finetune_instruct.py
export/export_hf.py
scripts/release_base_model.sh
scripts/count_dataset_tokens.py
scripts/write_release_manifest.py
```

## Planned Config Changes

Keep only the two existing training config files:

```text
train/configs/aina_nano_1m.yaml
train/configs/aina_nano_10m.yaml
```

Update output/data paths to support artifact root.

Example target for 10M:

```yaml
tokenizer_dir: ${AINA_RUN_ROOT}/tokenizers/aina-nano-10m
train_file: ${AINA_RUN_ROOT}/data/clean/train.jsonl
val_file: ${AINA_RUN_ROOT}/data/clean/val.jsonl
output_dir: ${AINA_RUN_ROOT}/checkpoints/aina-nano-10m/pretrain
```

Keep source-controlled model config as repo-relative path:

```yaml
model_config: model/config_10m.json
```

## Planned Helper Script

Add:

```text
scripts/prepare_run_dirs.sh
```

Expected behavior:

```bash
export AINA_RUN_ROOT="${AINA_RUN_ROOT:-/home/data/aina-runs}"
mkdir -p \
  "$AINA_RUN_ROOT/data/raw" \
  "$AINA_RUN_ROOT/data/clean" \
  "$AINA_RUN_ROOT/tokenizers" \
  "$AINA_RUN_ROOT/checkpoints" \
  "$AINA_RUN_ROOT/models" \
  "$AINA_RUN_ROOT/gguf" \
  "$AINA_RUN_ROOT/reports" \
  "$AINA_RUN_ROOT/logs"
```

## Intended AWS Workflow

```bash
export AINA_RUN_ROOT="/home/data/aina-runs"
./scripts/prepare_run_dirs.sh
```

Upload dataset to:

```text
$AINA_RUN_ROOT/data/raw/*.jsonl
```

Run cleaning:

```bash
python data/clean_dataset.py \
  --raw-dir "$AINA_RUN_ROOT/data/raw" \
  --output-dir "$AINA_RUN_ROOT/data/clean"
```

Train tokenizer:

```bash
python tokenizer/train_tokenizer.py \
  --train-file "$AINA_RUN_ROOT/data/clean/train.jsonl" \
  --output-dir "$AINA_RUN_ROOT/tokenizers/aina-nano-10m" \
  --vocab-size 8000
```

Run training:

```bash
python train/pretrain.py --config train/configs/aina_nano_10m.yaml
```

Expected checkpoint:

```text
$AINA_RUN_ROOT/checkpoints/aina-nano-10m/pretrain/
```

## Test Plan

Set test root:

```bash
export AINA_RUN_ROOT="/home/data/aina-runs-test"
```

Prepare folders:

```bash
./scripts/prepare_run_dirs.sh
```

Use a small JSONL dataset in:

```text
$AINA_RUN_ROOT/data/raw/test.jsonl
```

Run:

```bash
python data/clean_dataset.py \
  --raw-dir "$AINA_RUN_ROOT/data/raw" \
  --output-dir "$AINA_RUN_ROOT/data/clean"

python tokenizer/train_tokenizer.py \
  --train-file "$AINA_RUN_ROOT/data/clean/train.jsonl" \
  --output-dir "$AINA_RUN_ROOT/tokenizers/aina-nano-1m" \
  --vocab-size 1024

python train/pretrain.py --config train/configs/aina_nano_1m.yaml
```

Verify outputs are outside repo:

```text
$AINA_RUN_ROOT/data/clean/
$AINA_RUN_ROOT/tokenizers/
$AINA_RUN_ROOT/checkpoints/
$AINA_RUN_ROOT/models/
$AINA_RUN_ROOT/reports/
```

Verify repo remains clean from artifacts:

```bash
git status --short
```

## Assumptions

- Repo stays source-only.
- Training artifacts are not committed to Git.
- Default output root is `/home/data/aina-runs`.
- Mounted disk can be used by setting `AINA_RUN_ROOT`.
- `$HOME` handles `/home/{username}` automatically on Linux.
- Config count remains minimal: only 1M and 10M training configs.
