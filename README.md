# 🤖 AI-Powered Customer Care System

This repository contains the code and assets for my **MSc Data Science Dissertation** at Kingston University. The goal is to develop a real-time AI assistant that empowers customer care teams using:

- 💬 **Sentiment Analysis** – to understand customer emotions
- 🔍 **Named Entity Recognition (NER)** – to extract context
- 📚 **Knowledge Embeddings (SBERT + FAISS)** – to suggest dynamic responses and insights

---

## 🎯 Project Objectives

- 🧠 Detect sentiment from customer queries
- 🧾 Extract relevant entities (product names, issues, etc.)
- 🔎 Retrieve real-time knowledge snippets using semantic search
- 🤝 Assist human agents with intelligent, in-the-moment guidance

---

## 🗂️ Project Structure

```text
📁 ai-customer-care/
│
├── 📂 data/                            → All data-related files
│   ├── 📄 raw/                         → Raw original datasets (ignored by Git)
│   └── 📄 cleaned/                     → Preprocessed datasets (ignored by Git)
│
├── 📓 notebooks/                      → Jupyter notebooks for EDA & modeling
│   └── 📊 eda_sentiment.ipynb         → Exploratory Data Analysis notebook
│
├── 🛠 scripts/                         → Python scripts for automation
│   └── 🧹 clean_data.py                → Script to clean raw Twitter dataset
│
├── 🧠 models/                          → (Optional) Trained models and weights
│
├── 📤 outputs/                        → (Optional) Visualizations, logs, and exports
│
├── 📄 .gitignore                      → Git exclusions (e.g., venv, CSVs)
├── 📦 requirements.txt                → Python dependencies
└── 📝 README.md                        → Project overview and documentation

🧰 Tech Stack
Python (3.12+)

.🐼 pandas, matplotlib, seaborn

.🤗 HuggingFace Transformers (BERT, RoBERTa)

.🧠 SpaCy, Sentence-BERT, FAISS

.⚡ FastAPI / Flask (planned for real-time API design)

Git, GitHub, VS Code
```
