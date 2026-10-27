# 📱 Hinglish WhatsApp Fake News Detector & Dashboard

This project is an interactive, machine learning-powered dashboard designed to analyze and detect fake or misleading WhatsApp forward messages written in **Hinglish** (a blend of Hindi and English written in the Roman script). 

We utilize a **Dual-Model System** (a lightweight baseline model and a state-of-the-art multilingual Transformer) and serve them using a user-friendly **Streamlit** dashboard.

---

## 🚀 Features
- **Dual-Model Predictions:** Compare a fast TF-IDF Baseline with a Deep Learning XLM-RoBERTa model.
- **Real-Time Analysis:** Paste any suspicious WhatsApp forward text and get immediate probability scores on whether it is "Fake/Misleading" or "Genuine".
- **Dataset Explorer:** Interactively view and visualize the synthetic and real-world evaluation datasets.
- **Model Benchmark Matrix:** Easy comparison of both architectures based on performance, speed, and hardware footprints.

---

## 🛠️ Tech Stack & Libraries
- **Language:** Python
- **Frameworks:** Streamlit, Gradio
- **Machine Learning & NLP:**
  - Hugging Face `transformers` (XLM-RoBERTa)
  - `scikit-learn` (TfidfVectorizer & Logistic Regression)
  - `datasets` (Hugging Face Datasets management)
  - `joblib` (Model serialization)
  - `PyTorch` (Backend for Deep Learning)

---

## 📁 File Structure
```text
├── dashboard.py               # Main Streamlit Dashboard Application
├── data_synthetic.csv         # Synthetic Hinglish dataset for model training
├── real_test.csv              # Benchmark dataset containing real-world test cases
├── baseline_model.pkl         # Saved Logistic Regression Model
├── baseline_vectorizer.pkl    # Saved TF-IDF Vectorizer
├── fakenews_xlmr/             # Fine-tuned XLM-RoBERTa Model & Tokenizer
├── README.md                  # Project documentation
└── .gitignore                 # Files excluded from Git tracking
```

---

## 📊 Models Overview

### 1. Baseline Model (TF-IDF + Logistic Regression)
- **Algorithm:** Character n-grams (range 2-5) with TF-IDF Vectorization followed by a Logistic Regression classifier.
- **Pros:** Instant inference speed (<5ms), extremely low memory footprint (a few Kilobytes).
- **Use Case:** Pattern matching for classic phishing phrases, aggressive calls-to-action (e.g., "forward this to 15 people!"), or specific currency/monetary scams.

### 2. Transformer Model (XLM-RoBERTa-Base)
- **Algorithm:** Fine-tuned `xlm-roberta-base` for sequence classification.
- **Pros:** Deep contextual understanding of multilingual/code-switched texts, captures semantics over simple keyword matching.
- **Use Case:** Detecting complex misinformation, structural manipulation, and emotionally manipulative vocabulary in Hinglish text.

---

## 🖥️ How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/MukundRana-2212/fake_news_detector.git
cd fake_news_detector
```

### 2. Install dependencies
```bash
pip install transformers datasets scikit-learn pandas streamlit torch joblib
```

### 3. Run the Streamlit Dashboard
```bash
streamlit run dashboard.py
```

---

## ⚠️ Disclaimer
This tool is primarily designed to analyze the **style and patterns** of suspicious online messages (e.g., creating panic, false incentives, forwarding chain-letters). It **does not cross-verify real-world facts or database logs**. Always use official fact-checking platforms to verify news and critical information.
