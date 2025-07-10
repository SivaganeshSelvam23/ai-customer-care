import json
import pandas as pd

# Paths
input_path = "data/chat_datasets/customer_care_dataset.json"
output_path = "data/chat_datasets/customer_care_dataset_flat.csv"

# Load JSON
with open(input_path, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

flattened_rows = []

# Flatten each chat into individual utterances with emotion and outcome
for chat in raw_data["data"]:
    dialog = chat["dialog"]
    emotions = chat["emotion"]
    outcome = chat["outcome"]

    for turn_index, (utterance, emotion) in enumerate(zip(dialog, emotions)):
        flattened_rows.append({
            "turn": turn_index,
            "utterance": utterance,
            "emotion": emotion,
            "outcome": outcome
        })

# Convert to DataFrame
df = pd.DataFrame(flattened_rows)

# ⛔ Limit no-emotion (label 0) to 230 rows
max_no_emotion = 230
no_emotion_df = df[df["emotion"] == 0].sample(n=max_no_emotion, random_state=42)

# ✅ Keep all other emotion rows
other_emotions_df = df[df["emotion"] != 0]

# 🔀 Combine and shuffle
balanced_df = pd.concat([no_emotion_df, other_emotions_df]).reset_index(drop=True)
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to CSV
balanced_df.to_csv(output_path, index=False, encoding="utf-8")
print(f"✅ Flattened and balanced dataset saved to: {output_path}")
