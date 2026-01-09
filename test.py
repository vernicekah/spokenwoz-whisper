# from pathlib import Path
# import torchaudio

# AUDIO_DIR = Path("data/SpokenWOZ/audio_5700_train_dev")

# # Pick the first wav file in the folder
# first_wav = next(AUDIO_DIR.glob("*.wav"))

# waveform, sr = torchaudio.load(first_wav)

# print(f"File: {first_wav.name}")
# print(f"Shape: {waveform.shape} (channels x samples)")
# print(f"Sampling rate: {sr} Hz")
# print(f"Duration: {waveform.shape[1] / sr:.2f} seconds")

import json

# Open and load the JSON file
with open('data/SpokenWOZ/text_5700_train_dev/data.json', 'r') as file:
    data = json.load(file)

print (data)