# 🤖 AI-Powered Customer Care Chat System

This repository is part of my **MSc Data Science Dissertation** at Kingston University.  
The project aims to build a real-time **AI assistant for customer care**, where agents and customers communicate through a live chat interface. The system performs:

- 💬 **Sentiment Analysis**: Understand user emotions in real-time
- 🏷️ **Named Entity Recognition (NER)**: Extract key entities from the conversation
- 🧠 **Knowledge Embeddings (SBERT + FAISS)**: Suggest relevant responses and resources

---

## 🎯 Project Goals

- 👥 Create a realistic **agent-customer chat interface** using Streamlit
- 🧠 Perform real-time **sentiment analysis** on incoming messages
- 🔍 Identify relevant entities like products or issues using **NER**
- 📚 Retrieve helpful responses using **knowledge embeddings**
- ⚙️ Simulate an intelligent assistant supporting live customer care

---

## 🗂️ Project Structure

```text
📁 ai-customer-care/
│
├── 📂 data/
│   └── conversation_datasets/      → Chat datasets (DailyDialog, EmotionLines, etc.)
│
├── 📂 app/
│   ├── streamlit_chat.py           → Chat interface (Streamlit)
│   └── sentiment_engine.py         → Real-time sentiment classifier
│
├── 📂 models/                      → (Optional) Trained sentiment/NER models
│
├── 📂 scripts/
│   └── prepare_dataset.py          → Preprocessing script for chat datasets
│
├── 📂 notebooks/
│   └── eda_chat_dataset.ipynb      → Exploratory analysis on chat data
│
├── 📄 .gitignore
├── 📦 requirements.txt
└── 📝 README.md

🧰 Tech Stack
Python (3.12+)

.🐼 pandas, matplotlib, seaborn

.🤗 HuggingFace Transformers (BERT, RoBERTa)

.🧠 Sentence-BERT (SBERT), FAISS for knowledge search

.⚡ Streamlit for frontend chat UI

FastAPI or Flask for backend integration

Git & GitHub for version control

🚫 Note on Datasets
To keep the repo lightweight, datasets are excluded from version control using .gitignore.
To work with data, download from sources like HuggingFace:

from datasets import load_dataset
dataset = load_dataset("daily_dialog")

🙌 Contributions
This is an academic project and not intended for production use.
Feel free to fork, star, or use for learning purposes.

📄 License
This project is for educational and non-commercial purposes only.
Third-party models and datasets are used under their respective licenses.

```
