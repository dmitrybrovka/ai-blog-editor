## LoRA/QLoRA fine-tuning guide (cloud GPU → local use)

This project can generate a JSONL dataset from your posts:

```bash
aiblog dataset lora --out out/lora_dataset.jsonl
```

Then you can fine-tune with QLoRA on a **cloud GPU**, and use the adapter locally.

### 0) Important compatibility note (Ollama adapters)
Ollama supports loading adapters via `ADAPTER` in a `Modelfile`, but **not for every architecture**.
According to Ollama docs, supported Safetensors adapters include:
- Llama (Llama 2/3/3.1/3.2)
- Mistral (Mistral 1/2, Mixtral)
- Gemma (Gemma 1/2)

If you want to run your tuned model via **Ollama + ADAPTER**, choose a base model from those families.
If you fine-tune Qwen/Qwen2.x adapters, you will likely need a different runtime/tooling.

References:
- `https://docs.ollama.com/modelfile` (ADAPTER)
- `https://docs.ollama.com/import` (importing fine-tuned adapters)

Also make sure your local config uses the same base family you tuned:
- update `config.yaml` → `ollama.chat_model` to the tuned base (or your derived model name in Ollama)
- do **not** keep `qwen2.5:*` as `chat_model` if you tuned a Llama/Mistral/Gemma adapter for Ollama

### 1) Pick a base model
Good defaults for style adapters:
- `meta-llama/Meta-Llama-3.1-8B-Instruct`
- `mistralai/Mistral-7B-Instruct-v0.3`
- `google/gemma-2-9b-it`

Note: bigger models need more VRAM.

### 2) Prepare a cloud environment
Recommended: RunPod / Lambda / Colab with an NVIDIA GPU.

Suggested environment (example):
- Python 3.10+
- CUDA matching the runtime image
- packages: `unsloth`, `transformers`, `datasets`, `trl`, `peft`, `accelerate`

### 3) Training with Unsloth (example script)
Below is a minimal example that trains a LoRA adapter from `out/lora_dataset.jsonl`.
Run this on the cloud box.

Create `train_unsloth_lora.py`:

```python
import json
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
DATA_PATH = "lora_dataset.jsonl"
OUT_DIR = "adapter_out"

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            rows.append({
                "text": f\"\"\"### Instruction
{obj['instruction']}

### Input
{obj['input']}

### Response
{obj['output']}
\"\"\"
            })
    return Dataset.from_list(rows)

dataset = load_jsonl(DATA_PATH)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=4096,
    dtype=None,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=4096,
    args=TrainingArguments(
        output_dir=OUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        num_train_epochs=2,
        logging_steps=10,
        save_steps=200,
        fp16=True,
        optim="adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        report_to="none",
    ),
)

trainer.train()

# Save adapter in Safetensors format (PEFT adapter)
model.save_pretrained(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)
print("Saved adapter to", OUT_DIR)
```

Run:

```bash
pip install -U unsloth transformers datasets trl peft accelerate
python train_unsloth_lora.py
```

Output: `adapter_out/` directory with adapter weights.

### 4) Use adapter locally with Ollama (ADAPTER)
1) Ensure you have the base model in Ollama (example):

```bash
ollama pull llama3.1
```

2) Create a `Modelfile` near your adapter directory:

```dockerfile
FROM llama3.1
ADAPTER ./adapter_out
```

3) Update `config.yaml` to use the new Ollama model name:

```yaml
ollama:
  chat_model: "my-style"
```

4) Create a new Ollama model:

```bash
ollama create my-style -f Modelfile
ollama run my-style
```

### 5) If you need GGUF adapters
Ollama also supports GGUF adapters (`ADAPTER ./adapter.gguf`). One way is to convert LoRA to GGUF using
`llama.cpp` tooling (`convert_lora_to_gguf.py`). Keep base model compatibility strict.

### 6) Practical tips to avoid overfitting
- Start with 1–2 epochs; stop if outputs become too “copy-paste”.
- Keep a small held-out set of posts and evaluate before/after.
- Use smaller LoRA rank (`r=8..16`) first.
- Prefer diverse topics; remove duplicates.

