# Release Checklist

This checklist defines when an Aina Nano base artifact can be called release-ready internally.

## Target Names

Recommended names:

```text
aina-nano-0.001b-base  # 1M sanity artifact
aina-nano-0.01b-base   # 10M base artifact
```

## Required Before Release

- Dataset is valid JSONL.
- Clean train/val split exists.
- Tokenizer is trained and encode/decode validated.
- Parameter report exists.
- Training completed without NaN.
- Checkpoint reload succeeds.
- Prompt eval report exists.
- Hugging Face export succeeds.
- Model card is generated.
- Optional GGUF conversion succeeds.
- Optional runtime smoke test succeeds.

## Release Command

For 1M sanity artifact:

```bash
./scripts/release_base_model.sh 1m aina-nano-0.001b-base
```

For 10M base artifact:

```bash
./scripts/release_base_model.sh 10m aina-nano-0.01b-base
```

This writes:

```text
reports/<model-name>/params.json
reports/<model-name>/train_tokens.json
reports/<model-name>/eval_prompts.jsonl
reports/<model-name>/manifest.json
models/<artifact-dir>/
```

## GGUF

After release export:

```bash
./export/convert_gguf.sh \
  $AINA_RUN_ROOT/models/aina-nano-10m-base \
  $AINA_RUN_ROOT/models/aina-nano-10m-base.gguf
```

## Promotion Rule

An artifact can be promoted only if:

- `reports/<model-name>/manifest.json` exists
- `models/<artifact-dir>/README.md` exists
- `model.safetensors` exists
- tokenizer files exist
- at least one eval report has been reviewed manually

## Notes

A release-ready artifact is not necessarily a high-quality model. It means the artifact is reproducible, documented, reloadable, and exportable.
