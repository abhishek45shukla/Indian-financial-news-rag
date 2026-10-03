# Indian Financial News RAG Assistant 📰

A simple AI-powered application that lets you ask questions about Indian financial news and get answers based on relevant news articles from the dataset.

I built this project to explore how **Retrieval-Augmented Generation (RAG)** works in a practical application using LangChain, vector databases, embeddings, and an LLM.

## What it does

You can enter a question about the financial news in the dataset, and the application:

- Finds the most relevant news articles
- Uses those articles as context
- Generates an answer using Gemini
- Shows the articles used to generate the answer

This helps make the responses more connected to the actual data instead of relying only on the LLM's general knowledge.

## Tech Stack

- Python
- LangChain
- ChromaDB
- Hugging Face Embeddings
- Google Gemini
- Streamlit
- Pandas

## How it works

```text
User Question
     ↓
Semantic Search
     ↓
Relevant News Articles
     ↓
Context + Question
     ↓
Gemini
     ↓
Answer + Sources
