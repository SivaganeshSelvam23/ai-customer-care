import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, Trainer, TrainingArguments
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


# Load the cleaned dataset
df = pd.read_csv("data/chat_datasets/emotionlines.csv")

# Convert the emotion string into a flat list (same as in EDA)
import ast
df["emotion"] = df["emotion"].dropna().apply(ast.literal_eval)
df = df.explode("emotion").dropna().reset_index(drop=True)

# Keep only necessary columns
df = df[["dialog", "emotion"]]

# Load the RoBERTa tokenizer
tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

# Tokenization function
def tokenize(example):
    return tokenizer(example["dialog"], truncation=True, padding="max_length", max_length=64)

# Convert pandas DataFrame to HuggingFace Dataset
dataset = Dataset.from_pandas(df)

# Apply tokenizer to the dataset
tokenized_dataset = dataset.map(tokenize)


# Rename 'emotion' to 'label'
tokenized_dataset = tokenized_dataset.rename_column("emotion", "label")

# Split into train and test
train_test = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = train_test["train"]
eval_dataset = train_test["test"]

# Load the RoBERTa model for classification
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=7)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./models/roberta-sentiment",   # model checkpoints
    evaluation_strategy="epoch",               # ✅ supported in 4.40.0
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./models/logs",               # for logs
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)


# Metrics function

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted")
    }

# Set up the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Save the final model
trainer.save_model("backend/models/roberta-sentiment")
print("✅ Model training complete and saved.")
