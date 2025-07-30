import os
import hashlib
import streamlit as st
from pypdf import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Constants
VECTORSTORE_PARENT_DIR = "dbfaiss"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Load embedding model and LLM once
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)

# Prompt template
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful and knowledgeable AI assistant.
Use only the information from the context below to answer the user's question.
If the answer is not found in the context, reply: "Sorry, I couldn't find relevant information."

Context:
{context}

Question:
{question}
""")

def extract_text_from_pdf(pdf_file):
    pdf = PdfReader(pdf_file)
    return "".join(page.extract_text() or "" for page in pdf.pages)

def split_text_into_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    return splitter.split_text(text)

def get_file_hash(uploaded_file):
    content = uploaded_file.read()
    uploaded_file.seek(0)  # Reset stream pointer
    return hashlib.md5(content).hexdigest()

def create_and_save_vectorstore(chunks, save_dir):
    vectorstore = FAISS.from_texts(chunks, embedding=embedding_model)
    vectorstore.save_local(folder_path=save_dir)
    return vectorstore

def load_vectorstore(save_dir):
    return FAISS.load_local(folder_path=save_dir, embeddings=embedding_model)

def run_qa_chain(vectorstore, query):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # Only top 3 chunks
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    return chain(query)

def main():
    st.set_page_config(page_title="Retriever - Document QA", layout="centered")
    st.title("📄🔍 Retriever: Ask Any Question from Your PDF")

    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_file:
        file_hash = get_file_hash(uploaded_file)
        save_path = os.path.join(VECTORSTORE_PARENT_DIR, file_hash)

        if os.path.exists(save_path):
            st.info("🔄 Loading cached vectorstore...")
            vectorstore = load_vectorstore(save_path)
        else:
            with st.spinner("🔍 Extracting text and creating vectorstore..."):
                raw_text = extract_text_from_pdf(uploaded_file)
                chunks = split_text_into_chunks(raw_text)
                os.makedirs(VECTORSTORE_PARENT_DIR, exist_ok=True)
                vectorstore = create_and_save_vectorstore(chunks, save_path)

        query = st.text_input("Ask a question based on the PDF:")

        if query:
            with st.spinner("🤖 Thinking..."):
                result = run_qa_chain(vectorstore, query)

                st.markdown("### 💡 Answer")
                st.write(result["result"])

                with st.expander("📚 Source Chunks"):
                    for i, doc in enumerate(result["source_documents"]):
                        st.markdown(f"Chunk {i+1}:")
                        st.write(doc.page_content)

if __name__ == "__main__":
    main()
