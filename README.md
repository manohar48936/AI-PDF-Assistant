# 📚 Smart PDF AI Chatbot  

An AI-powered chatbot that allows users to upload PDFs, process them, and ask questions. The AI extracts text, converts it into vector embeddings, and retrieves the most relevant answers using FAISS and DeepSeek API.

---

## 🚀 Features  
- 📂 Upload multiple PDF files  
- 🤖 AI-powered search and Q&A  
- ⚡ Fast and efficient document processing  
- 📝 Stores chat history for reference  

---

## 🛠️ Installation  

### 1⃣ Clone the Repository  
```sh
git clone https://github.com/YOUR_GITHUB_USERNAME/Smart-PDF-AI.git
cd Smart-PDF-AI
```

### 2⃣ Install Dependencies  
```sh
pip install -r requirements.txt
```

### 3⃣ Set Up DeepSeek API Key  
Create a `.env` file in the project folder and add:  
```env
DEEPSEEK_API_KEY=your_deepseek_api_key_here
```
Or manually set it in the script:
```sh
export DEEPSEEK_API_KEY="your_deepseek_api_key_here"
```

---

## ▶️ Usage  

Run the Streamlit app:  
```sh
streamlit run app.py
```

1⃣ Upload your **PDF files**  
2⃣ Ask a **question**  
3⃣ Get **instant AI-powered answers!**  

---

## 🐜 Dependencies  

Below are the required dependencies (also included in `requirements.txt`):  
```sh
langchain==0.0.184
PyPDF2==3.0.1
python-dotenv==1.0.0
streamlit==1.18.1
openai==0.27.6
faiss-cpu==1.7.4
altair==4
tiktoken==0.4.0
requests==2.31.0
sentence-transformers==2.2.2
```

---

## 🔧 Technologies Used  
- **Python**  
- **Streamlit**  
- **LangChain**  
- **FAISS (Facebook AI Similarity Search)**  
- **DeepSeek API**  
- **PyPDF2**  
- **HuggingFace Sentence Transformers**  

---

## 🤝 Contributing  
Pull requests are welcome! Feel free to **fork** the project, improve it, and submit a PR.  

---

## 📝 License  
This project is **MIT Licensed** – Free to use and modify!  
