# Run Aina Nano with Ollama

Ollama needs a GGUF file. Keep Hugging Face format as the main artifact, then convert it to GGUF.

## 1. Export Hugging Face model

```bash
python export/export_hf.py \
  --checkpoint-dir checkpoints/aina-nano-1m/pretrain \
  --output-dir models/aina-nano-1m
```

## 2. Install or clone llama.cpp

```bash
git clone https://github.com/ggml-org/llama.cpp ../llama.cpp
python -m pip install -r ../llama.cpp/requirements.txt
```

## 3. Convert to GGUF

```bash
./export/convert_gguf.sh models/aina-nano-1m models/aina-nano-1m.gguf
```

If your llama.cpp version requires a different converter path, run the equivalent `convert_hf_to_gguf.py` command from that checkout.

## 4. Create Ollama model

```bash
ollama create aina-nano -f serving/Modelfile
ollama run aina-nano "Apa itu API?"
```

The first model is only a sanity check. Bad or repetitive answers are expected.
