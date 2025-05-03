import os
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredExcelLoader

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.embeddings = OpenAIEmbeddings(openai_api_key="api-key")

    def load_document(self, file_path):
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        elif file_path.endswith(".xlsx"):
            loader = UnstructuredExcelLoader(file_path)
        else:
            raise ValueError("Unsupported file type")
        
        documents = loader.load()
        return self.text_splitter.split_documents(documents)

    def create_vector_store(self, docs, save_path="faiss_index"):
        db = FAISS.from_documents(docs, self.embeddings)
        db.save_local(save_path)
        return db
    
