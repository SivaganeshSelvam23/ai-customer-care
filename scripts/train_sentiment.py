import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score

# Step 1: Load flattened dataset
df = pd.read_csv("data/chat_datasets/customer_care_dataset_flat.csv")
df = df[["utterance", "emotion"]]

# Step 2: Fix label mappings (ensure string keys)
unique_labels = sorted(df["emotion"].unique())
label2id = {str(label): i for i, label in enumerate(unique_labels)}
id2label = {i: str(label) for label, i in label2id.items()}

# Step 3: Map emotion column to numeric labels
df["label"] = df["emotion"].map(lambda x: label2id[str(x)])

# Step 4: Tokenizer
tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

def tokenize(example):
    return tokenizer(example["utterance"], truncation=True, padding="max_length", max_length=64)

dataset = Dataset.from_pandas(df[["utterance", "label"]])
dataset = dataset.map(tokenize)

# Step 5: Train/Test split
split = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = split["train"]
eval_dataset = split["test"]

# Step 6: Compute class weights
labels = df["label"].values
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(labels), y=labels)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

# Step 7: Load model
model = RobertaForSequenceClassification.from_pretrained(
    "roberta-base",
    num_labels=len(label2id),
    id2label={str(k): v for k, v in id2label.items()},
    label2id={v: k for k, v in id2label.items()}
)

# Step 8: Custom Trainer with class-weighted loss
def custom_loss(outputs, labels):
    loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights_tensor.to(outputs.logits.device))
    return loss_fct(outputs.logits, labels)

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = custom_loss(outputs, labels)
        return (loss, outputs) if return_outputs else loss

# Step 9: Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted")
    }

# Step 10: TrainingArguments
training_args = TrainingArguments(
    output_dir="./models/roberta-sentiment",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="./models/logs",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True
)

# Step 11: Train
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer
)

trainer.train()
trainer.save_model("backend/models/roberta-sentiment")
print("✅ Training complete. Model saved to backend/models/roberta-sentiment")
