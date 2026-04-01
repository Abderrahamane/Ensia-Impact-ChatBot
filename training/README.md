# Training Scripts (LoRA SFT)

This folder contains optional scripts to fine-tune a Hugging Face instruct model for ENSIA assistant response style.

## 1) Install dependencies

```cmd
python -m pip install -r requirements-train.txt
```

## 2) Prepare SFT data

```cmd
python training/prepare_sft_data.py --top-k 5 --max-sources 3
```

This writes `data/processed/sft_train.jsonl`.

## 3) Train a LoRA adapter

```cmd
python training/train_sft_lora.py --train-file data/processed/sft_train.jsonl --base-model Qwen/Qwen2.5-7B-Instruct --output-dir artifacts/lora-ensia-assistant --epochs 1 --batch-size 1 --grad-accum 8
```

## 4) Smoke run on tiny subset

```cmd
python training/prepare_sft_data.py --max-examples 4 --output data/processed/sft_train_smoke.jsonl
python training/train_sft_lora.py --train-file data/processed/sft_train_smoke.jsonl --max-train-samples 4 --epochs 0.1 --output-dir artifacts/lora-ensia-smoke
```

## 5) Use the adapter in RAG runtime

```cmd
set "ENSIA_GENERATION_BACKEND=hf_lora"
set "ENSIA_HF_BASE_MODEL=Qwen/Qwen2.5-7B-Instruct"
set "ENSIA_HF_LORA_ADAPTER_DIR=artifacts\lora-ensia-assistant"
python pipeline/rag_query.py --query "Quels sont les partenariats actuels de l'ecole ?"
```

