import os
import sys
import yaml
import json
from pathlib import Path

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

from modules import (
    compute_metrics,
    load_and_prepare_datasets,
    DataCollatorSpeechSeq2SeqWithPadding,
    load_model
)

# get length of training data
train_manifest = Path("/data/processed_data/train_manifest_HF.json")
with open(train_manifest, "r", encoding="utf-8") as f:
    train_data = json.load(f)["data"]

train_data_len = len(train_data) 

def main():
    # --- 1. Load config ---
    with open("configs/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    
    # --- 2. Load model + processor ---
    model, processor, device = load_model(
        model_name=model_cfg["name"],
        language=model_cfg.get("language", "english"),
        task=model_cfg.get("task", "transcribe"),
    )

    model.config.use_cache = False

    # --- 3. Load and prepare datasets ---
    train_ds, dev_ds = load_and_prepare_datasets(
        "/data/processed_data/train_manifest_HF.json",
        "/data/processed_data/dev_manifest_HF.json",
        processor
    )

    # --- 4. data collator ---
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # --- 5. Training args from YAML ---
    epochs = train_cfg.get("num_train_epochs", 5)
    batch_size = train_cfg.get("per_device_train_batch_size", 16)
    gradient_accumulation_steps = train_cfg.get("gradient_accumulation_steps", 1)
    max_steps = train_data_len * (epochs + 1 ) // (batch_size * gradient_accumulation_steps)
    train_cfg["max_steps"] = max_steps

    training_args = Seq2SeqTrainingArguments(**train_cfg)

    # --- 6. Trainer ---
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=data_collator,
        tokenizer=processor.feature_extractor,
        compute_metrics=lambda pred: compute_metrics(pred, processor),
    )

    # --- 7. Train ---
    trainer.train()
    trainer.save_model(train_cfg["output_dir"])
    processor.save_pretrained(train_cfg["output_dir"])

    print(f"Model saved to {train_cfg['output_dir']}")


if __name__ == "__main__":
    main()
