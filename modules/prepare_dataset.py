from datasets import load_dataset, Audio
from transformers import WhisperProcessor

def prepare_dataset(dataset, processor, max_input_length=30.0):
    """
    Prepares dataset entries with WhisperProcessor (audio -> tensors).
    Filters out clips longer than max_input_length seconds.
    """
    def _prepare(example):
        audio = example["audio"]
        processed = processor(
            audio=audio["array"],
            sampling_rate=audio["sampling_rate"],
            text=example["text"]
        )
        processed["input_length"] = len(audio["array"]) / audio["sampling_rate"]
        return processed

    dataset = dataset.map(_prepare, remove_columns=dataset.column_names, num_proc=4, load_from_cache_file=True)
    dataset = dataset.filter(
        lambda length: length < max_input_length,
        input_columns=["input_length"],
        num_proc=4
    )
    return dataset


def load_and_prepare_datasets(train_json, dev_json, processor):
    """Load HF-style manifests and process with WhisperProcessor."""
    train_ds = load_dataset("json", data_files=train_json, field="data")["train"]
    dev_ds   = load_dataset("json", data_files=dev_json, field="data")["train"]

    train_ds = train_ds.cast_column("audio", Audio(sampling_rate=16000))
    dev_ds   = dev_ds.cast_column("audio", Audio(sampling_rate=16000))

    train_ds = prepare_dataset(train_ds, processor)
    dev_ds   = prepare_dataset(dev_ds, processor)

    return train_ds, dev_ds


def load_and_prepare_testset(test_json, processor):
    """Load HF-style test manifest and process with WhisperProcessor."""
    test_ds = load_dataset("json", data_files=test_json, field="data")["train"]

    # ensure audio is decoded + resampled to 16kHz
    test_ds = test_ds.cast_column("audio", Audio(sampling_rate=16000))

    # prepare features using same pipeline
    test_ds = prepare_dataset(test_ds, processor)

    return test_ds