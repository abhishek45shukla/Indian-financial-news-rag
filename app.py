import os
import streamlit as st

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")


st.set_page_config(
    page_title="Indian Financial News Assistant",
    page_icon="📰",
    layout="wide"
)


st.title("📰 Indian Financial News Assistant")
st.write(
    "A RAG-based assistant for searching and understanding "
    "Indian financial news."
)

with st.sidebar:
    st.header("About the Project")

    st.write(
        "This application uses Retrieval-Augmented Generation "
        "(RAG) to answer questions from an Indian financial "
        "news dataset."
    )

    st.divider()

    st.subheader("Technologies")

    st.write("""
    - Python
    - LangChain
    - ChromaDB
    - Hugging Face Embeddings
    - Gemini 3.8 Flash
    - Streamlit
    """)

    st.divider()

    st.caption("50,000+ financial news articles")


@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


@st.cache_resource
def load_vectorstore():
    embeddings = load_embeddings()

    return Chroma(
        persist_directory="./vectorstore/chroma_db",
        collection_name="indian_financial_news",
        embedding_function=embeddings
    )


@st.cache_resource
def load_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        google_api_key=api_key
    )


prompt = ChatPromptTemplate.from_template("""
You are an AI financial news assistant.

Answer the user's question using only the information
provided in the context.

If the information needed to answer the question is not
available in the context, clearly say so.

Do not invent facts, dates, companies, events, or numbers.

Context:
{context}

Question:
{question}

Answer:
""")


def rag_query(question):
    vectorstore = load_vectorstore()
    llm = load_llm()

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    documents = retriever.invoke(question)

    context = "\n\n".join(
        f"Title: {doc.metadata.get('title')}\n"
        f"Date: {doc.metadata.get('date')}\n"
        f"Content: {doc.page_content}"
        for doc in documents
    )

    final_prompt = prompt.invoke({
        "context": context,
        "question": question
    })

    response = llm.invoke(final_prompt)

    return response.text, documents


question = st.text_input(
    "Ask a question about Indian financial news",
    placeholder="Example: What are the major challenges faced by Indian banks?"
)


if st.button("🔍 Search News", type="primary"):

    if not question.strip():
        st.warning("Please enter a question.")

    else:
        with st.spinner("Searching the news database..."):
            answer, documents = rag_query(question)

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Retrieved Sources")

        st.write(
            f"The answer was generated using {len(documents)} relevant news articles."
        )

        for i, doc in enumerate(documents, 1):

            title = doc.metadata.get("title", "Unknown title")
            date = doc.metadata.get("date", "Unknown")

            with st.expander(f"Source {i}: {title}"):

                st.write(f"**Date:** {date}")

                st.write(doc.page_content)