# Chat-with-PDF-using-LLMs
Chat with any PDF using LLMs powered by GROQ and Hugging Face
# 🧠 Chat with PDF using LLMs

An AI-powered Streamlit web application that allows users to upload PDF files and interact with their content using natural language. It uses GROQ’s LLMs and Hugging Face embeddings to generate intelligent, context-aware responses from documents.

---

## 🚀 Features

- 📄 Upload and process one or multiple PDF files
- 💬 Chat with the content of your PDF
- 🔍 Context-aware Q&A using Retrieval-Augmented Generation (RAG)
- 🧠 Uses **GROQ LLM** (Gemma2-9b-It) and **HuggingFace Embeddings**
- 🔐 Secure API integration using `.env` file
- 🧾 Maintains session-based chat history
- 🎯 Built with **Streamlit** for an interactive user interface

---

## 📦 Tech Stack

- **Python 3.10+**
- **Streamlit**
- **LangChain**
- **GROQ LLM**
- **HuggingFace Transformers**
- **Chroma DB**
- **PyPDFLoader**

---

## 📁 Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
Create virtual environment

bash
Copy
Edit
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Set up .env file

ini
Copy
Edit
GROQ_API_KEY=your_groq_api_key
HF_TOKEN=your_huggingface_token
Run the Streamlit app

bash
Copy
Edit
streamlit run app.py
