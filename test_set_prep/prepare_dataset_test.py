from transformers import WhisperProcessor
import os
import sys
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)
from modules import load_and_prepare_testset

processor = WhisperProcessor.from_pretrained("models/whisper-spokenwoz")
test_ds = load_and_prepare_testset("data/SpokenWOZ/test_root_manifest_hf.json", processor)

print(test_ds)