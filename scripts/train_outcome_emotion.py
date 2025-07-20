import json
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import (
    RobertaTokenizerFast,
    RobertaModel,
    TrainingArguments,
    Trainer
)
from torch import nn
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score

# Load data
json_path = "data/chat_datasets/customer_care_dataset.json"
with open(json_path, "r", encoding="utf-8") as f:
    raw_data = json.load(f)["data"]

# Prepare data
records = []
for item in raw_data:
    dialog = " ".join(item["dialog"])
    outcome = item["outcome"]
    emotion_vector = [0] * 7
    for e in item["emotion"]:
        emotion_vector[e] += 1
    emotion_vector = [x / len(item["emotion"]) for x in emotion_vector]  # normalize
    records.append({"text": dialog, "emotion_vec": emotion_vector, "label": outcome})

# Encode labels
texts = [r["text"] for r in records]
emotion_vecs = [r["emotion_vec"] for r in records]
labels = [r["label"] for r in records]

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Train/val split
X_train, X_val, emo_train, emo_val, y_train, y_val = train_test_split(
    texts, emotion_vecs, encoded_labels, test_size=0.2, stratify=encoded_labels, random_state=42
)

# Tokenizer
model_name = "roberta-base"
tokenizer = RobertaTokenizerFast.from_pretrained(model_name)

def tokenize(batch):
    enc = tokenizer(batch["text"], padding=True, truncation=True, max_length=256)
    enc["emotion"] = batch["emotion_vec"]
    return enc

train_hf_dataset = Dataset.from_dict({
    "text": X_train,
    "emotion_vec": emo_train,
    "label": y_train
}).map(tokenize, batched=True)

val_hf_dataset = Dataset.from_dict({
    "text": X_val,
    "emotion_vec": emo_val,
    "label": y_val
}).map(tokenize, batched=True)

train_encodings = {
    "input_ids": train_hf_dataset["input_ids"],
    "attention_mask": train_hf_dataset["attention_mask"]
}
val_encodings = {
    "input_ids": val_hf_dataset["input_ids"],
    "attention_mask": val_hf_dataset["attention_mask"]
}

# Custom model
class RobertaWithEmotion(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.emotion_proj = nn.Linear(7, 64)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(self.roberta.config.hidden_size + 64, num_labels)

    def forward(self, input_ids=None, attention_mask=None, emotion=None, labels=None):
        r_out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = r_out.last_hidden_state[:, 0]  # [CLS] token
        emo_emb = self.emotion_proj(emotion)
        fused = torch.cat([cls_emb, emo_emb], dim=1)
        logits = self.classifier(self.dropout(fused))
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return {"loss": loss, "logits": logits}

model = RobertaWithEmotion(model_name, num_labels=len(label_encoder.classes_))

# Trainer wrapper
class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, emotions, labels):
        self.encodings = encodings
        self.emotions = torch.tensor(emotions, dtype=torch.float)
        self.labels = torch.tensor(labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["emotion"] = self.emotions[idx]
        item["labels"] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

train_dataset_torch = EmotionDataset(train_encodings, emo_train, y_train)
val_dataset_torch = EmotionDataset(val_encodings, emo_val, y_val)

# Metrics

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }

# Training setup
training_args = TrainingArguments(
    output_dir="backend/models/roberta-outcome-emotion",
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
    train_dataset=train_dataset_torch,
    eval_dataset=val_dataset_torch,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Train
trainer.train()

# Save model + encoder
tokenizer.save_pretrained("backend/models/roberta-outcome-emotion")
model_path = "backend/models/roberta-outcome-emotion"
import joblib
joblib.dump(label_encoder, f"{model_path}/label_encoder.pkl")
