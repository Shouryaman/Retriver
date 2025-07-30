import os
import streamlit as st
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Load Gemini key
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def extract_text_from_pdf(pdf_file):
    pdf = PdfReader(pdf_file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text() or ""
    return text

def split_text_into_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)

def create_vectorstore(chunks):
    return FAISS.from_texts(chunks, embedding=embedding_model)

def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY)

def get_prompt():
    return PromptTemplate(
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

def run_qa_chain(llm, vectorstore, prompt, query):
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    return chain(query)

def main():
    st.set_page_config(page_title="Retriever - Document QA", layout="centered")
    st.title("📄🔍 Retriever: Ask Any Question from Your PDF")

    uploaded_file = st.file_uploader("Upload any PDF file", type="pdf")

    if uploaded_file:
        with st.spinner("Processing your PDF..."):
            raw_text = extract_text_from_pdf(uploaded_file)
            chunks = split_text_into_chunks(raw_text)
            vectorstore = create_vectorstore(chunks)

        query = st.text_input("Ask something related to your document:")

        if query:
            with st.spinner("Thinking..."):
                llm = get_llm()
                prompt = get_prompt()
                result = run_qa_chain(llm, vectorstore, prompt, query)

                st.markdown("### 💡 Answer")
                st.write(result["result"])

                with st.expander("📚 Source Chunks"):
                    for i, doc in enumerate(result["source_documents"]):
                        st.markdown(f"*Chunk {i+1}:*")
                        st.write(doc.page_content)

if __name__ == "__main__":
    main()