import evaluate
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

# Load metric + normalizer once
wer_metric = evaluate.load("wer")
normalizer = BasicTextNormalizer()


def compute_metrics(pred, processor):

    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id so we can decode
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # Decode predictions & labels
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    # --- Orthographic WER ---
    wer_ortho = 100 * wer_metric.compute(predictions=pred_str, references=label_str)

    # --- Normalized WER ---
    pred_str_norm = [normalizer(p) for p in pred_str]
    label_str_norm = [normalizer(l) for l in label_str]

    # filter out empty references (avoid divide-by-zero)
    pred_str_norm = [
        pred_str_norm[i] for i in range(len(pred_str_norm)) if len(label_str_norm[i]) > 0
    ]
    label_str_norm = [
        label_str_norm[i] for i in range(len(label_str_norm)) if len(label_str_norm[i]) > 0
    ]

    wer = 100 * wer_metric.compute(predictions=pred_str_norm, references=label_str_norm)

    return {"wer_ortho": wer_ortho, "wer": wer}
