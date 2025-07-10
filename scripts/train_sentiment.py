import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score

# Step 1: Load your cleaned dataset
df = pd.read_csv("data/chat_datasets/customer_care_dataset_flat.csv")

# Step 2: Keep only the necessary columns
df = df[["utterance", "emotion"]]  # We're only training emotion classifier for now

# Step 3: Map emotion labels to numeric (if not already numeric)
label2id = {int(label): int(i) for i, label in enumerate(sorted(df["emotion"].unique()))}
id2label = {int(v): str(k) for k, v in label2id.items()}
df["label"] = df["emotion"].map(label2id)

# Step 4: Load tokenizer
tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

# Step 5: Tokenize utterances
def tokenize(example):
    return tokenizer(example["utterance"], truncation=True, padding="max_length", max_length=64)

dataset = Dataset.from_pandas(df[["utterance", "label"]])
dataset = dataset.map(tokenize)

# Step 6: Split into train and test
train_test = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = train_test["train"]
eval_dataset = train_test["test"]

# Step 7: Load model
model = RobertaForSequenceClassification.from_pretrained(
    "roberta-base", num_labels=len(label2id), id2label=id2label, label2id=label2id
)

# Step 8: Define training arguments
training_args = TrainingArguments(
    output_dir="./models/roberta-sentiment",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./models/logs",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)

# Step 9: Evaluation metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted")
    }

# Step 10: Set up trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

# Step 11: Train
trainer.train()

# Step 12: Save the model
trainer.save_model("backend/models/roberta-sentiment")
print("✅ Model training complete and saved.")
