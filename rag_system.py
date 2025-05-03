from generation_model import RenewableEnergyGenerator
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
import os

class RenewableEnergyRAG:
    def __init__(self, db_path="faiss_index", model_path="models/generator_weights.pth"):
        self.generator = RenewableEnergyGenerator(model_path)
        self.db = FAISS.load_local(db_path)
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self._custom_langchain_adapter(),
            chain_type="stuff",
            retriever=self.db.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )
    
    def _custom_langchain_adapter(self):
        # Adaptador para conectar nuestro modelo a LangChain
        from langchain.llms.base import LLM
        from typing import Optional, List, Mapping, Any
        
        class CustomLLMWrapper(LLM):
            @property
            def _llm_type(self) -> str:
                return "renewable-energy-custom"
            
            def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
                return self.generator.generate(prompt)
                
        return CustomLLMWrapper()
    
    def query(self, question):
        result = self.qa_chain({"query": question})
        
        # Formatear respuesta con referencias
        answer = result["result"]
        sources = []
        
        for doc in result["source_documents"]:
            source_info = {
                "content": doc.page_content[:200] + "...",
                "page": doc.metadata.get("page", "N/A"),
                "file": os.path.basename(doc.metadata.get("source", "unknown"))
            }
            sources.append(source_info)
        
        return {
            "answer": answer,
            "sources": sources
        }


