#from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
#from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

class RAGSystem:
    def __init__(self, db_path="faiss_index", openai_api_key="api-key"):
        self.llm = ChatOpenAI(
            model="gpt-4-turbo",
            temperature=0,
            openai_api_key=openai_api_key
        )
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.db = FAISS.load_local(db_path, self.embeddings)
        self.qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.db.as_retriever(),
            return_source_documents=True
        )

    def query(self, question):
        result = self.qa({"query": question})
        return {
            "answer": result["result"],
            "sources": result["source_documents"] 
        }
    
