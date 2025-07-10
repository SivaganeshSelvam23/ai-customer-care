from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
import torch

# Load model and tokenizer
model_path = "backend/models/roberta-sentiment"

# Load tokenizer from local if saved, else fallback to base
try:
    tokenizer = RobertaTokenizerFast.from_pretrained(model_path)
except:
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

# Load the fine-tuned model from your local path
model = RobertaForSequenceClassification.from_pretrained(model_path)
model.eval()


# Class label mapping
label_map = {
    0: "no emotion",
    1: "anger",
    2: "disgust",
    3: "fear",
    4: "happiness",
    5: "sadness",
    6: "surprise"
}


def predict_sentiment(text: str) -> str:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
        print("Logits:", outputs.logits)
        prediction = torch.argmax(outputs.logits, dim=1).item()
        print("Predicted label index:", prediction)
    return label_map[prediction]
