# Aina Nano Runbook

Runbook ini adalah urutan operasional internal dari dataset sampai model artifact.

## 0. Clean Workspace Check

Dataset dan artifact tidak masuk Git. Pastikan folder berikut ada:

```text
data/raw/
data/clean/
data/tokenized/
tokenizer/
checkpoints/
models/
```

## 1. Add Raw Dataset

Masukkan file JSONL ke:

```text
data/raw/*.jsonl
```

Format:

```json
{"text":"..."}
```

Validasi cepat:

```bash
python - <<'PY'
import json
from pathlib import Path
for path in Path('data/raw').glob('*.jsonl'):
    count = 0
    for i, line in enumerate(path.open(encoding='utf-8'), 1):
        obj = json.loads(line)
        assert isinstance(obj.get('text'), str) and obj['text'].strip(), f'{path}:{i}'
        count += 1
    print(path, count)
PY
```

## 2. Clean Dataset

```bash
python data/clean_dataset.py
```

Check output:

```bash
wc -l data/clean/train.jsonl data/clean/val.jsonl
ls -lh data/clean
```

## 3. Train Tokenizer

Small smoke:

```bash
python tokenizer/train_tokenizer.py --vocab-size 1024
```

Larger run:

```bash
python tokenizer/train_tokenizer.py --vocab-size 8000
```

Validate:

```bash
python tokenizer/test_tokenizer.py
```

Expected: decoded text resembles input text.

## 4. Run 1M Smoke Training

```bash
python train/pretrain.py --config train/configs/aina_nano_1m.yaml
```

Validate generation:

```bash
python eval/generate.py \
  --model-dir checkpoints/aina-nano-1m/pretrain \
  --prompt "API adalah" \
  --max-new-tokens 80
```

## 5. Export 1M Smoke Artifact

```bash
python export/export_hf.py \
  --checkpoint-dir checkpoints/aina-nano-1m/pretrain \
  --output-dir models/aina-nano-1m
```

Optional GGUF:

```bash
./export/convert_gguf.sh models/aina-nano-1m models/aina-nano-1m.gguf
```

If this fails, do not run the 10M config yet.

## 6. Run 10M Training

```bash
python train/pretrain.py --config train/configs/aina_nano_10m.yaml
```

Monitor:

- train loss
- val loss every `eval_every_steps`
- checkpoint save every `save_every_steps`
- disk usage
- runtime stability

## 7. Export 10M Artifact

```bash
python export/export_hf.py \
  --checkpoint-dir checkpoints/aina-nano-10m/pretrain \
  --output-dir models/aina-nano-10m-base
```

GGUF:

```bash
./export/convert_gguf.sh \
  models/aina-nano-10m-base \
  models/aina-nano-10m-base.gguf
```

## 8. Record Experiment

Catat minimal:

```text
dataset name/size:
tokenizer vocab size:
model config:
training config:
final train loss:
final val loss:
sample outputs:
HF export status:
GGUF status:
notes:
```

## 9. Release Base Artifact

Setelah checkpoint final tersedia, buat release bundle:

```bash
./scripts/release_base_model.sh 10m aina-nano-0.01b-base
```

Output:

```text
reports/aina-nano-0.01b-base/
models/aina-nano-10m-base/
```

Optional GGUF:

```bash
./export/convert_gguf.sh \
  models/aina-nano-10m-base \
  models/aina-nano-10m-base.gguf
```
