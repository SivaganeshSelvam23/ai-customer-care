import json
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from torch.nn import CrossEntropyLoss

# Load JSON data directly
json_path = "data/chat_datasets/customer_care_dataset.json"
with open(json_path, "r", encoding="utf-8") as f:
    raw_data = json.load(f)["data"]

# Flatten data to dialog + outcome
records = []
for item in raw_data:
    dialog = " ".join(item["dialog"])
    outcome = item["outcome"]
    records.append({"dialog": dialog, "outcome": outcome})

# Encode labels
texts = [r["dialog"] for r in records]
labels = [r["outcome"] for r in records]
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Train/test split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, encoded_labels, test_size=0.2, stratify=encoded_labels, random_state=42
)

# Tokenize
model_name = "roberta-base"
tokenizer = RobertaTokenizerFast.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True, max_length=256)

train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels}).map(tokenize, batched=True)
val_dataset = Dataset.from_dict({"text": val_texts, "label": val_labels}).map(tokenize, batched=True)

# Class weights
class_counts = np.bincount(train_labels)
weights = 1.0 / class_counts
class_weights = torch.tensor(weights / weights.sum(), dtype=torch.float)

# Custom model with weighted loss
class RobertaForOutcomeClassification(RobertaForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.loss_fct = CrossEntropyLoss(weight=class_weights)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=None)
        logits = outputs.logits
        loss = None
        if labels is not None:
            loss = self.loss_fct(logits, labels)
        return {"loss": loss, "logits": logits}

# Load model
model = RobertaForOutcomeClassification.from_pretrained(model_name, num_labels=len(label_encoder.classes_))

# Metrics

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }

# Trainer setup
training_args = TrainingArguments(
    output_dir="backend/models/roberta-outcome",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="logs",
    logging_steps=10,
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Train
trainer.train()

# Save model and tokenizer properly for inference
model_path = "backend/models/roberta-outcome"
tokenizer.save_pretrained(model_path)
model.save_pretrained(model_path)

# Save label encoder
import joblib
joblib.dump(label_encoder, f"{model_path}/label_encoder.pkl")
