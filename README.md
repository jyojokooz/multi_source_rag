# 📚 RAG Textbook Assistant
An AI-powered web application that allows users to "chat" with a massive textbook. 

## 🚀 Live Demo
[[Click here to view the live app!](put-your-streamlit-url-here)](https://multisourcerag-dypf8vhybzrixpbgpxxtbq.streamlit.app/)

## 🛠️ Architecture & Tech Stack
This project uses Retrieval-Augmented Generation (RAG) to prevent AI hallucinations by forcing the LLM to only answer based on retrieved vector context.
* **Frontend:** Streamlit
* **LLM:** Meta Llama-3.1-8B (via Groq Cloud for ultra-fast inference)
* **Vector Database:** Pinecone
* **Embeddings:** HuggingFace (`BAAI/bge-small-en-v1.5`)
* **Orchestration:** LangChain

## ✨ Features
* Semantic search through 400+ pages of PDF data.
* Chat history memory.
* Automated source citation and page number retrieval.
