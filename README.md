# 🎓 YouTube RAG Assistant

> **Ask questions about any YouTube video and get instant answers powered by AI.**

YouTube RAG Assistant is a RAG (Retrieval-Augmented Generation) based chatbot that extracts transcripts from YouTube videos and lets you ask questions about the content. It uses LangChain, FAISS/Chroma vector stores, and Google Gemini to deliver accurate, context-aware answers.

---

## ✨ Features

- 🎥 **YouTube Transcript Extraction** — Automatically fetches transcripts (English & Hindi) from any YouTube video.
- 🧠 **RAG Pipeline** — Retrieval-Augmented Generation ensures answers are grounded in the actual video content.
- 🔍 **Multi-Query Retrieval** — Generates multiple reformulations of your question to improve retrieval accuracy.
- 📦 **Vector Store Support** — Uses **FAISS** (Version 2) and **Chroma** (Version 1) for fast similarity search over transcript chunks.
- 🤖 **Multiple LLM Support** — Works with Google Gemini, OpenAI, HuggingFace (TinyLlama), and Anthropic models.
- 🖥️ **Streamlit Web UI** — Clean, interactive web interface to load videos and ask questions (Version 2).
- 📊 **RAG Evaluation** — Built-in evaluation pipeline using **DeepEval** with metrics like Faithfulness, Answer Relevancy, Contextual Precision, and Contextual Recall (Version 1).

---

## 🏗️ Architecture

```
YouTube URL
    │
    ▼
┌──────────────────┐
│  Transcript Fetch │  (youtube-transcript-api + pytube)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Text Splitting   │  (RecursiveCharacterTextSplitter)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Embeddings       │  (HuggingFace: all-MiniLM-L6-v2)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Vector Store     │  (FAISS / Chroma)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Multi-Query      │  Generate 5 query variations
│  Retriever        │  Retrieve & deduplicate docs
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  LLM Generation   │  (Google Gemini 2.5 Flash)
└────────┬─────────┘
         │
         ▼
      Answer
```

---

## 📁 Project Structure

```
LangChain/
├── app_version2.py              # Streamlit web app (main app)
├── YoutubeChatBot/
│   └── app_version1.py          # CLI version with RAG evaluation
├── req.txt                      # Python dependencies
├── .env                         # API keys (not committed)
└── README.md
```

| File | Description |
|------|-------------|
| `app_version2.py` | **Version 2** — Streamlit-based web UI with multi-query retrieval and FAISS vector store. |
| `YoutubeChatBot/app_version1.py` | **Version 1** — CLI-based pipeline with Chroma vector store, TinyLlama LLM, and DeepEval evaluation suite. |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 – 3.12 (recommended)
- API keys for the LLM providers you want to use

### 1. Clone the Repository

```bash
git clone https://github.com/raykundan655/YouTube-RAG-Assistant.git
cd YouTube-RAG-Assistant
```

### 2. Create a Virtual Environment

```bash
python -m venv myenv

# Windows
myenv\Scripts\activate

# macOS/Linux
source myenv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r req.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_api_key
OPENAI_API_KEY=your_openai_api_key          # optional
ANTHROPIC_API_KEY=your_anthropic_api_key    # optional
HUGGINGFACEHUB_API_TOKEN=your_hf_token      # optional
```

### 5. Run the App

**Streamlit Web App (Version 2):**

```bash
streamlit run app_version2.py
```

**CLI with Evaluation (Version 1):**

```bash
python YoutubeChatBot/app_version1.py
```

---

## 🖥️ Usage

1. Open the Streamlit app in your browser.
2. Paste a YouTube video URL and click **Load Video**.
3. Wait for the transcript to be fetched and indexed.
4. Type your question about the video and click **Ask**.
5. Get an AI-generated answer based on the video content!

---

## 🧪 Evaluation (Version 1)

Version 1 includes a built-in evaluation pipeline using [DeepEval](https://github.com/confident-ai/deepeval) with the following metrics:

| Metric | Threshold | Description |
|--------|-----------|-------------|
| **Answer Relevancy** | 0.7 | How relevant the answer is to the question |
| **Faithfulness** | 0.7 | How faithful the answer is to the retrieved context |
| **Contextual Precision** | 0.7 | How precise the retrieved context is |
| **Contextual Recall** | 0.7 | How well the retriever recalls relevant context |

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|------------|
| **Framework** | [LangChain](https://github.com/langchain-ai/langchain) |
| **LLMs** | Google Gemini, OpenAI, HuggingFace (TinyLlama) |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` |
| **Vector Stores** | FAISS, ChromaDB |
| **Transcript** | `youtube-transcript-api`, `pytube` |
| **Web UI** | [Streamlit](https://streamlit.io/) |
| **Evaluation** | [DeepEval](https://github.com/confident-ai/deepeval) |

---

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to open an issue or submit a pull request.

---

<p align="center">
  Made with ❤️  by <strong>YouTube RAG Assistant</strong>
</p>
