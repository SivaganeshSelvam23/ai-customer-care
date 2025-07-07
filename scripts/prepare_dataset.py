from datasets import load_dataset
import pandas as pd
import os

# Load dataset from HuggingFace
dataset = load_dataset("daily_dialog", split="train", trust_remote_code=True)


# Convert to pandas DataFrame
df = pd.DataFrame(dataset)

# Show sample structure
print(df.head())

# Create output path
output_dir = "data/chat_datasets"
os.makedirs(output_dir, exist_ok=True)

# Save as CSV
df.to_csv(f"{output_dir}/emotionlines.csv", index=False)
print("✅ Dataset saved to:", f"{output_dir}/emotionlines.csv")
