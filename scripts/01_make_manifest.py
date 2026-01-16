import json
import soundfile as sf
import torchaudio
from pathlib import Path
from tqdm import tqdm

# --- Config ---
ROOT_DIR = Path("data/raw_data")  # root directory for raw data
AUDIO_DIR = ROOT_DIR / "audio_5700_train_dev"
TEXT_JSON = ROOT_DIR / "text_5700_train_dev" / "data.json"
OUTPUT_MANIFEST = ROOT_DIR / "processed_data" / "root_manifest.json"
SEGMENTS_DIR = ROOT_DIR / "processed_audio" / "audio_segments"

def main():
    """Combines audio and text into JSON NeMo manifest format
    Segments audio based on word-level timestamps"""

    with open(TEXT_JSON, "r", encoding="utf-8") as f:
        text_data = json.load(f)

    SEGMENTS_DIR.mkdir(parents=True, exist_ok=True) # directory for audio segments

    manifest = []

    # iterate over all utterance IDs from the JSON
    for utt_id, record in tqdm(text_data.items()):
        audio_path = AUDIO_DIR / f"{utt_id}.wav"
        if not audio_path.exists():
            continue

        waveform, sr = torchaudio.load(audio_path)

        # iterate over dialogue turns
        for i, turn in enumerate(record.get("log", [])):
            if "words" not in turn or len(turn["words"]) == 0:
                continue

            # start and end time from the first and last word
            start = turn["words"][0]["BeginTime"] / 1000.0
            end = turn["words"][-1]["EndTime"] / 1000.0

            # convert times to audio sample indices
            start_frame = int(start * sr)
            end_frame = int(end * sr)

            chunk = waveform[:, start_frame:end_frame] # extract slice from waveform
            duration = (end_frame - start_frame) / sr # each segment duration

            if chunk.shape[1] == 0: # skip empty segments
                continue

            # If stereo → downmix to mono
            if chunk.shape[0] > 1:
                chunk = chunk.mean(dim=0, keepdim=True)

            # save chunk as a new wav -> save using soudfile
            out_name = f"{utt_id}_turn{i+1}.wav" # each segment file name 
            out_path = SEGMENTS_DIR / out_name
            sf.write(out_path, chunk.squeeze().numpy().astype("float32"), sr)

            # manifest entry
            entry = {
                "audio_filepath": str(out_path.resolve()),
                "duration": duration,
                "text": turn["text"].strip()
            }
            manifest.append(entry)

    with open(OUTPUT_MANIFEST, "w", encoding="utf-8") as f:
        for entry in manifest:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"{len(manifest)} entries written to {OUTPUT_MANIFEST}")

if __name__ == "__main__":
    main()
