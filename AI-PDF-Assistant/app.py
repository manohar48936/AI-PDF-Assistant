import os 
import requests
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

# Load API Key from environment variable
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"

if not DEEPSEEK_API_KEY:
    st.error("⚠️ DeepSeek API Key not found! Please set it as an environment variable.")

# Function to extract text from PDFs
def extract_text_from_pdfs(pdf_files):
    """Extracts text from uploaded PDF files."""
    all_text = ""
    for pdf in pdf_files:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                all_text += text + "\n"
    return all_text

# Function to create vector store
def get_vectorstore(text):
    """Splits text into chunks and creates a vector store using FAISS."""
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    text_chunks = text_splitter.split_text(text)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_texts(text_chunks, embeddings)

# Function to call DeepSeek API
def call_deepseek_api(query, context):
    """Sends a query to the DeepSeek API with the provided context."""
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "You are an AI assistant answering questions based on context."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}\nAnswer:"}
        ],
        "temperature": 0.7
    }
    response = requests.post(DEEPSEEK_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    return f"Error: {response.json()}"

# Function to process query
def answer_query(query, vectorstore):
    """Searches for relevant context in vectorstore and calls the DeepSeek API."""
    docs = vectorstore.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in docs])
    return call_deepseek_api(query, context)

# Streamlit UI
st.set_page_config(page_title="📚 AI PDF Assistant", layout="wide")

# Sidebar: Upload & History Section
with st.sidebar:
    st.title("📂 Upload & Chat History")
    uploaded_files = st.file_uploader("📎 Upload PDFs", type=["pdf"], accept_multiple_files=True)

    if uploaded_files and st.button("📑 Process PDFs", use_container_width=True):
        text = extract_text_from_pdfs(uploaded_files)
        st.session_state["vectorstore"] = get_vectorstore(text)
        st.success("✅ PDFs Processed Successfully!")

    # Display question history
    st.markdown("---")
    st.subheader("⏳ Chat History")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    for i, chat in enumerate(reversed(st.session_state.chat_history)):
        with st.expander(f"🔍 {chat['question']}"):
            st.write(f"💡 **Answer:** {chat['answer']}")

# Main Section
st.markdown("""
    <h1 style='text-align: center;'>📘 AI-Powered PDF Chatbot</h1>
    <p style='text-align: center; font-size:18px;'>Upload PDFs, process them, and start asking questions! The AI will fetch the most relevant answers for you.</p>
    <hr>
""", unsafe_allow_html=True)

if "vectorstore" in st.session_state:
    st.markdown("### 💬 Ask Your Question")
    query = st.text_input("❓ Type your question:")
    if st.button("🚀 Get Answer", use_container_width=True):
        if query:
            with st.spinner("🤖 Thinking..."):
                answer = answer_query(query, st.session_state["vectorstore"])
                st.session_state.chat_history.append({"question": query, "answer": answer})
                st.success("✅ Answer Generated!")
                st.markdown("### 💡 Answer:")
                st.write(answer)
        else:
            st.warning("⚠️ Please enter a question!")
