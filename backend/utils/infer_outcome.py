from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
import torch
import joblib

# Load model and tokenizer
model_path = "backend/models/roberta-outcome"

# Load tokenizer
try:
    tokenizer = RobertaTokenizerFast.from_pretrained(model_path)
except:
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

# Load model
model = RobertaForSequenceClassification.from_pretrained(model_path)
model.eval()

# Load label encoder
label_encoder = joblib.load(f"{model_path}/label_encoder.pkl")


def predict_outcome(dialog: list[str]) -> str:
    """
    Predict outcome of a full multi-turn dialog.
    Input: list of strings ["Customer: ...", "Agent: ...", ...]
    """
    joined_dialog = " ".join(dialog)
    inputs = tokenizer(joined_dialog, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()
    return label_encoder.inverse_transform([prediction])[0]


# Example usage
if __name__ == "__main__":
    sample_dialog = [
        "Customer: My electricity bill spiked to \u00a3320 this month without prior notice.",
        "Agent: That’s a serious issue—I’m reviewing your account details now.",
        "Customer: I normally pay around \u00a3100, but this surge is unexpected.",
        "Agent: The meter shows high usage on 17/09/2025; I’ll schedule a technician visit.",
        "Customer: Please send them tomorrow between 08:00 and 10:00 AM.",
        "Agent: Scheduled for 08:00–10:00 AM tomorrow; you will receive confirmation shortly.",
        "Customer: Thank you, I appreciate the prompt response.",
        "Agent: Happy to help."
    ]

    predicted = predict_outcome(sample_dialog)
    print(f"Predicted outcome: {predicted}")
