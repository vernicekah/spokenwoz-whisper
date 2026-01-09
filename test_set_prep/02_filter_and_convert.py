import json
from pathlib import Path
from tqdm import tqdm

# --- Config ---
ROOT_DIR = Path("data/SpokenWOZ")
INPUT_MANIFEST = ROOT_DIR / "test_root_manifest.json"       
OUTPUT_MANIFEST = ROOT_DIR / "test_root_manifest_hf.json"
LANG = "en"
SUBSET = "spokenwoz"

def main():
    # load NeMo manifest (line-delimited JSON)
    with open(INPUT_MANIFEST, "r", encoding="utf-8") as f:
        nemo_entries = [json.loads(line) for line in f]

    hf_data = []

    for entry in tqdm(nemo_entries, desc="Converting"):
        audio_path = Path(entry["audio_filepath"])
        text = entry["text"]
        duration = entry["duration"]

        hf_entry = {
            "file": str(audio_path),
            "audio": {
                "path": str(audio_path),
                "sampling_rate": 16000   # already resampled earlier
            },
            "language": LANG,
            "text": text,
            "duration": duration,
            "subset": SUBSET
        }
        hf_data.append(hf_entry)

    # wrap in dict for HuggingFace style
    output = {"data": hf_data}

    with open(OUTPUT_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"{len(hf_data)} samples written to {OUTPUT_MANIFEST}")

if __name__ == "__main__":
    main()
