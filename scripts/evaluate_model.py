import os
import sys
import yaml
import json
import numpy as np  
from pathlib import Path
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

# make sure modules/ is importable
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

from modules import (
    load_model,
    load_and_prepare_testset,
    DataCollatorSpeechSeq2SeqWithPadding,
    compute_metrics,
)

# ---- Config Path ----
DATA_DIR = Path("/data")      
EVAL_DIR = DATA_DIR / "evaluation" / "whisper-large-v2-2"  # change manually before each run
EVAL_DIR.mkdir(parents=True, exist_ok=True)

def main():
    # 1) config
    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg["model"]
    eval_cfg  = cfg["eval"]

    # 2) load model
    model, processor, device = load_model(
        model_name=eval_cfg.get("model_dir", cfg["train"]["output_dir"]),
        language=model_cfg.get("language", "english"),
        task=model_cfg.get("task", "transcribe"),
    )

    # 3) load dataset 
    test_manifest = eval_cfg["test_manifest"]   
    test_ds = load_and_prepare_testset(test_manifest, processor)

    # 3b) also load the same JSON to retrieve refs & intents later
    with open(test_manifest, "r", encoding="utf-8") as f:
        items = json.load(f)["data"]   

    # 4) collator / args / trainer
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    training_args = Seq2SeqTrainingArguments(
        output_dir=eval_cfg.get("output_dir", "eval_results"),
        per_device_eval_batch_size=eval_cfg.get("per_device_eval_batch_size", 16),
        predict_with_generate=True,
        eval_accumulation_steps=16,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        eval_dataset=test_ds,
        data_collator=data_collator,
        tokenizer=processor.feature_extractor,
        compute_metrics=lambda pred: compute_metrics(pred, processor),
    )

    # 5) evaluate
    preds = trainer.predict(test_ds)
    results = preds.metrics
    print("Evaluation results:", results)

    # 6) predict and decode
    gen_ids, label_ids = preds.predictions, preds.label_ids
    label_ids = np.where(label_ids == -100, processor.tokenizer.pad_token_id, label_ids)

    pred_texts  = processor.tokenizer.batch_decode(gen_ids,   skip_special_tokens=True) # whisper text predictions
    label_texts = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True) # ground truth texts 
    pred_texts  = [p.strip() for p in pred_texts]
    label_texts = [y.strip() for y in label_texts]

    # 7) align with original items 
    # keeps everything same length -> to attach audio path  and intent 
    n = min(len(pred_texts), len(label_texts), len(items))
    pred_texts, label_texts, items = pred_texts[:n], label_texts[:n], items[:n]

    # audio path getter
    def get_audio_path(it):
        return (it.get("audio") or {}).get("path") or it.get("file")

    # utt_id getter
    def get_utt_id(it):
        p = get_audio_path(it)
        return Path(p).stem if p else None
    
    # 8) write outputs
    asr_rows = []
    t5_rows  = []

    for p, y, it in zip(pred_texts, label_texts, items):
        asr_rows.append({
            "utt_id": get_utt_id(it),
            "audio_filepath": get_audio_path(it),
            "asr_pred": p, # whisper predicted text
            "asr_ref": y, # ground truth transcription
            "intent_ref": it.get("dialog_act"),
        })

    for p, it in zip(pred_texts, items):
        t5_rows.append({
            "text": f"{p}",   
            "dialog_act": it.get("dialog_act"),
        })

    # write in json files
    def write_json(path, rows):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)


    # Output 1: whisper pred vs ground truth text + ground truth intent
    asr_json = EVAL_DIR / "asr_text_vs_ref_and_intent.json"
    write_json(asr_json, asr_rows)
    print(f"Saved → {asr_json}")

    # Output 2 (input to T5): predicted text -> intent
    t5_json = EVAL_DIR / "t5_eval_input_from_asr.json"
    write_json(t5_json, t5_rows)
    print(f"Saved → {t5_json}")

    # Output 3: evaluation metrics results
    metrics_path = EVAL_DIR / "eval_results.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved → {metrics_path}")

if __name__ == "__main__":
    main()