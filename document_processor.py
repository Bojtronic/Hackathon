import os
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredExcelLoader
from embeddings_model import RenewableEnergyEmbedder  # Nuestro modelo personalizado

class DocumentProcessor:
    def __init__(self, model_path="models/embedder_weights.pth"):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=self._calculate_tokens
        )
        self.embedder = RenewableEnergyEmbedder(model_path)
        
    def _calculate_tokens(self, text: str) -> int:
        return len(text) // 4  # Aproximación simple
        
    def load_document(self, file_path):
        file_ext = os.path.splitext(file_path)[1].lower()
        
        loaders = {
            '.pdf': PyPDFLoader,
            '.docx': Docx2txtLoader,
            '.xlsx': UnstructuredExcelLoader
        }
        
        if file_ext not in loaders:
            raise ValueError(f"Formato no soportado: {file_ext}")
            
        loader = loaders[file_ext](file_path)
        documents = loader.load()
        
        # Añadir metadatos de página y archivo
        for doc in documents:
            doc.metadata['full_path'] = file_path
            doc.metadata['extension'] = file_ext
            
        return self.text_splitter.split_documents(documents)
    
    def create_vector_store(self, docs, save_path="faiss_index"):
        texts = [doc.page_content for doc in docs]
        metadatas = [doc.metadata for doc in docs]
        
        embeddings = self.embedder.embed_documents(texts)
        db = FAISS.from_texts(
            texts=texts,
            embedding=self.embedder,
            metadatas=metadatas
        )
        db.save_local(save_path)
        return db


