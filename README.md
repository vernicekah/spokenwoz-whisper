# Whisper ASR Finetuning Pipeline

A HuggingFace-based **Whisper Automatic Speech Recognition (ASR)** pipeline for fine-tuning and evaluating pretrained Whisper models on the **SpokenWOZ** dataset.

This repository supports:
- Audio preprocessing and manifest creation  
- Whisper fine-tuning  
- ASR evaluation (WER)  
- Preparing ASR outputs for downstream SLU / Intent Classification

---

## Setup the Environment (via Docker)

### Requirements
- Docker  
- Docker Compose  
- NVIDIA GPU + NVIDIA Container Toolkit (for GPU training)

Before building, check `build/docker-compose.yml` and update the mounted volumes to match:
- Dataset location
- Whisper pretrained models (if stored offline)
- Model checkpoints / TensorBoard logs

---

### Build the Docker Image
```bash
docker compose -f build/docker-compose.yml build
```

## Run the Containers

```bash
docker compose -f build/docker-compose.yml up -d
```

This will start two containers:

* `whisper-finetuning` – training & evaluation
* `tensorboard` – training logs

Enter the training container:

```bash
docker exec -it whisper-finetuning bash
```

You are now inside the configured environment.

---

## Code Structure

```
.
├─ build/                  # Docker setup
│  ├─ docker-compose.yml
│  └─ Dockerfile
├─ configs/
│  └─ config.yaml          # model, training, evaluation configs
├─ modules/                # reusable logic
│  ├─ prepare_dataset.py
│  ├─ data_collator.py
│  ├─ load_model.py
│  └─ metrics.py
├─ scripts/                # main pipeline steps
│  ├─ 01_make_manifest.py
│  ├─ 02_filter_and_convert.py
│  ├─ 03_data_split.py
│  ├─ finetuning.py
│  └─ evaluate_model.py
├─ test_set_prep/          # test set preparation steps
│  ├─ 01_make_manifest.py
│  ├─ 02_filter_and_convert.py
│  └─ prepare_dataset_test.py
└─ eval_results/           # evaluation outputs
```

---

## Writing Your Config File

All experiments parameters are controlled via:

```
configs/config.yaml
```

This includes:

* Whisper model name / checkpoint
* Training hyperparameters
* Dataset manifest paths
* Evaluation settings
* Output directories

Modify this file before running training or evaluation.

---

## Pipeline Execution

### 1. Create NeMo Manifest from Raw Data

Segments raw audio into utterance-level clips and creates a NeMo-style manifest.

```bash
python scripts/01_make_manifest.py
```
Modify the config (of dataset location) before running the script

---

### 2. Convert to HuggingFace Manifest

* Converts NeMo manifest → HuggingFace JSON format
* Resamples audio to **16 kHz mono**

```bash
python scripts/02_filter_and_convert.py
```

---

### 3. Train / Dev Split

Splits the dataset into training and development sets.

```bash
python scripts/03_data_split.py
```

---

### 4. Fine-tune Whisper

Fine-tunes a pretrained Whisper model using HuggingFace `Seq2SeqTrainer`.

```bash
python scripts/finetuning.py
```

Outputs:

* Fine-tuned Whisper model
* Processor/tokenizer
* TensorBoard logs

---

### 5. Evaluate the Model

Evaluates Whisper on the test set and generates predictions.

```bash
python scripts/evaluate_model.py
```

Outputs:

* ASR predictions vs ground truth
* Word Error Rate (WER)
* JSON files for downstream SLU models (e.g. T5)

---

## Metrics

The evaluation computes:

* **Orthographic WER**
* **Normalized WER** (with text normalization)

Metrics are saved as JSON for easy analysis.

---

## Data Format

### NeMo Manifest (Input)

```json
{"audio_filepath": ".../audio.wav", "duration": 3.2, "text": "transcription"}
```

### HuggingFace Manifest (Output)

```json
{
  "data": [
    {
      "audio": { "path": "...", "sampling_rate": 16000 },
      "text": "transcription",
      "duration": 3.2
    }
  ]
}
```

---

## Other Notes

* Audio longer than **30 seconds** is filtered automatically
* Padding tokens are masked correctly during training
* ASR outputs are formatted for easy reuse in SLU pipelines

