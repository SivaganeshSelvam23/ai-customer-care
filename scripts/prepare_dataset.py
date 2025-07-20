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

# ❌ Limit overrepresented classes
cap_0 = 280   # no emotion
cap_4 = 310   # happiness
cap_1 = 310   # anger
cap_3 = 310   # fear
cap_2 = 295   # disgust
cap_6 = 295   # surprise

no_emotion_df = df[df["emotion"] == 0].sample(n=cap_0, random_state=42)
happiness_df = df[df["emotion"] == 4].sample(n=cap_4, random_state=42)
anger_df = df[df["emotion"] == 1].sample(n=cap_1, random_state=42)
fear_df = df[df["emotion"] == 3].sample(n=cap_3, random_state=42)
disgust_df = df[df["emotion"] == 2].sample(n=cap_2, random_state=42)
surprise_df = df[df["emotion"] == 6].sample(n=cap_6, random_state=42)

# ✅ Keep all other emotion rows
other_emotions_df = df[~df["emotion"].isin([0, 1, 2, 3, 4, 6])]

# 🔀 Combine and shuffle
balanced_df = pd.concat([
    no_emotion_df,
    happiness_df,
    anger_df,
    fear_df,
    disgust_df,
    surprise_df,
    other_emotions_df
]).reset_index(drop=True)

balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to CSV
balanced_df.to_csv(output_path, index=False, encoding="utf-8")
print(f"✅ Flattened and balanced dataset saved to: {output_path}")
