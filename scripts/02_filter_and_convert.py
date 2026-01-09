import json
import soundfile as sf
import torchaudio
from pathlib import Path
from tqdm import tqdm
import shutil

"""Convert NeMo manifest to HuggingFace style JSON.
Resample audio to 16khz"""

# --- Config ---
ROOT_DIR = Path("data/SpokenWOZ")
INPUT_MANIFEST = ROOT_DIR / "root_manifest.json"
OUTPUT_MANIFEST = ROOT_DIR / "root_manifest_hf.json"
OUTPUT_AUDIO_DIR = ROOT_DIR / "audio_16k"

TARGET_SR = 16000
LANG = "en"
SUBSET = "spokenwoz"


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def process_audio(audio_path: Path):
    """Return path, sr, duration. Resample only if needed."""
    # read header only (faster than full load)
    info = sf.info(audio_path)
    sr = info.samplerate
    channels = info.channels
    duration = info.frames / sr

    # if already 16k mono, no need to resample
    if sr == TARGET_SR and channels == 1:
        return audio_path, sr, duration

    # else: resample and save to OUTPUT_AUDIO_DIR
    waveform, sr_loaded = torchaudio.load(audio_path)

    # downmix stereo → mono if needed
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr_loaded != TARGET_SR:
        waveform = torchaudio.functional.resample(waveform, sr_loaded, TARGET_SR)

    out_path = OUTPUT_AUDIO_DIR / audio_path.name
    audio_np = waveform.squeeze().numpy().astype("float32")
    sf.write(out_path, audio_np, TARGET_SR)

    duration = audio_np.shape[0] / TARGET_SR
    return out_path, TARGET_SR, duration


def main():
    ensure_dir(OUTPUT_AUDIO_DIR)

    # load NeMo manifest
    with open(INPUT_MANIFEST, "r", encoding="utf-8") as f:
        nemo_entries = [json.loads(line) for line in f]

    hf_data = []
    resampled_count = 0
    reused_count = 0

    for entry in tqdm(nemo_entries, desc="Processing"):
        audio_path = Path(entry["audio_filepath"])
        text = entry["text"]

        out_path, sr, duration = process_audio(audio_path)

        if out_path != audio_path:
            resampled_count += 1
        else:
            reused_count += 1

        hf_entry = {
            "file": str(out_path),
            "audio": {
                "path": str(out_path),
                "sampling_rate": sr
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
    print(f"Resampled: {resampled_count}, Reused (already 16kHz mono): {reused_count}")


if __name__ == "__main__":
    main()
