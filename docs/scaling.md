# Scaling Notes

Aina Nano is intentionally small, but the same pipeline can be run on stronger hardware or larger datasets.

## Recommended Order

Always scale in this order:

```text
small dataset + 1M config
→ full pipeline smoke test
→ larger dataset + 10M config
→ export and inference test
```

Do not start a long run until a small end-to-end run has succeeded.

## Hardware

The 1M config is intended for local sanity checks. The 10M config benefits from accelerator hardware, but it can still be adjusted for smaller machines by lowering:

- `context_length`
- `batch_size`
- `max_steps`

## Dataset Size

For learning purposes:

- small smoke test: MB-scale text
- larger toy run: tens of millions of tokens

More data is useful only if it is clean, diverse, and relevant.

## Guardrail

Before increasing training time or dataset size, verify:

```bash
python train/pretrain.py --config train/configs/aina_nano_1m.yaml
python eval/generate.py --model-dir $AINA_RUN_ROOT/checkpoints/aina-nano-1m/pretrain --prompt "API adalah" --max-new-tokens 80
python export/export_hf.py --checkpoint-dir $AINA_RUN_ROOT/checkpoints/aina-nano-1m/pretrain --output-dir $AINA_RUN_ROOT/models/aina-nano-1m
```

If this fails, fix the pipeline before scaling up.
