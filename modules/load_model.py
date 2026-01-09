import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration


def load_model(model_name: str = "openai/whisper-small",
               language: str = "english",
               task: str = "transcribe",
               base_processor_name: str = "openai/whisper-large-v2"
               ):
    
    # Load processor (feature extractor + tokenizer)
    processor = WhisperProcessor.from_pretrained(
        base_processor_name, language=language, task=task
    )

    # Load model
    model = WhisperForConditionalGeneration.from_pretrained(model_name)

    # Choose device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    return model, processor, device

    # Quick test
if __name__ == "__main__":
    model, processor, device = load_model()
    print(f"Loaded model on {device}")
    print("Processor vocab size:", len(processor.tokenizer))
