import torch
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
import os

# Load model only once at startup
model_path = os.path.join("backend", "models", "roberta-sentiment")
tokenizer = RobertaTokenizerFast.from_pretrained(model_path)
model = RobertaForSequenceClassification.from_pretrained(model_path)
model.eval()

id2label = model.config.id2label  # e.g., {'0': '0', '1': '1', ...}

@torch.no_grad()
def predict_emotion(text: str) -> int:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
    logits = model(**inputs).logits
    pred_id = torch.argmax(logits, dim=1).item()
    return pred_id  # ✅ No mapping from id2label