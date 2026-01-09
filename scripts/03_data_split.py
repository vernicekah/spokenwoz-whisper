from datasets import load_dataset
from pathlib import Path
import json

def split_manifest_hf(input_json="data/SpokenWOZ/root_manifest_hf.json",
                      test_size=0.1, seed=42):
    # Load dataset
    dataset = load_dataset("json", data_files=input_json, field="data")["train"]

    # Split
    splits = dataset.train_test_split(test_size=test_size, seed=seed)

    # Save into the same folder as input_json
    input_path = Path(input_json)
    save_dir = input_path.parent

    train_out = save_dir / "train_manifest_hf.json"
    dev_out = save_dir / "dev_manifest_hf.json"

    # Convert datasets to list of dicts
    train_data = splits["train"].to_list()
    dev_data = splits["test"].to_list()

    # Wrap with "data": [ ]
    with open(train_out, "w", encoding="utf-8") as f:
        json.dump({"data": train_data}, f, ensure_ascii=False, indent=2)

    with open(dev_out, "w", encoding="utf-8") as f:
        json.dump({"data": dev_data}, f, ensure_ascii=False, indent=2)

    print(f"Train: {len(train_data)} → {train_out}")
    print(f"Dev: {len(dev_data)} → {dev_out}")

    return splits

if __name__ == "__main__":
    split_manifest_hf("data/SpokenWOZ/root_manifest_hf.json", test_size=0.1, seed=42)

# Total: 167386
# Train: 150647 
# Dev: 16739