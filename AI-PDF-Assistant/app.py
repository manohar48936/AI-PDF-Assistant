import streamlit as st  # ✅ Import Streamlit first

# ✅ Must be the first Streamlit command
st.set_page_config(page_title="📚 AI PDF Assistant", layout="wide")

import os
import requests
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

# ✅ Load environment variables
load_dotenv()

# ✅ Retrieve DeepSeek API key
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"

# ✅ Ensure API key is set
if not DEEPSEEK_API_KEY:
    st.error("⚠️ DeepSeek API Key not found! Please set it as an environment variable.")
    st.stop()

# ✅ Function to extract text from PDFs
def extract_text_from_pdfs(pdf_files):
    """Extracts text from uploaded PDF files."""
    all_text = ""
    for pdf in pdf_files:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                all_text += text + "\n"
    return all_text.strip()

# ✅ Function to create a vector store
def get_vectorstore(text):
    """Splits text into chunks and creates a vector store using FAISS."""
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    text_chunks = text_splitter.split_text(text)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_texts(text_chunks, embeddings)

# ✅ Function to call DeepSeek API
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
    try:
        response = requests.post(DEEPSEEK_URL, headers=headers, json=payload)
        response.raise_for_status()  # ✅ Raise an error for bad responses (4xx, 5xx)
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        return f"❌ API Error: {str(e)}"

# ✅ Function to process query
def answer_query(query, vectorstore):
    """Searches for relevant context in vectorstore and calls the DeepSeek API."""
    docs = vectorstore.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in docs])
    return call_deepseek_api(query, context)

# ✅ Sidebar: Upload & History Section
with st.sidebar:
    st.title("📂 Upload & Chat History")
    uploaded_files = st.file_uploader("📎 Upload PDFs", type=["pdf"], accept_multiple_files=True)

    if uploaded_files and st.button("📑 Process PDFs", use_container_width=True):
        text = extract_text_from_pdfs(uploaded_files)
        if text:
            st.session_state["vectorstore"] = get_vectorstore(text)
            st.success("✅ PDFs Processed Successfully!")
        else:
            st.warning("⚠️ No text extracted from PDFs. Please try different files.")

    # ✅ Display chat history
    st.markdown("---")
    st.subheader("⏳ Chat History")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    for i, chat in enumerate(reversed(st.session_state.chat_history)):
        with st.expander(f"🔍 {chat['question']}"):
            st.write(f"💡 **Answer:** {chat['answer']}")

# ✅ Main Section UI
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
