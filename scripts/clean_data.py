import pandas as pd
import re

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def clean_dataset(input_path, output_path):
    df = pd.read_csv(input_path)
    df.dropna(subset=['text'], inplace=True)
    df['clean_text'] = df['text'].apply(clean_text)
    df.to_csv(output_path, index=False)
    print(f"✅ Cleaned dataset saved to: {output_path}")

if __name__ == "__main__":
    clean_dataset('data/raw/twcs.csv', 'data/cleaned/cleaned_twcs.csv')
