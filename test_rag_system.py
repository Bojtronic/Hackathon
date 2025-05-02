import streamlit as st
from rag_system import RAGSystem

rag = RAGSystem(openai_api_key=st.secrets["OPENAI_API_KEY"])
response = rag.query("What are the terms of the land lease agreement?")

print("Answer:", response["answer"])
print("\nSources:")
for doc in response["sources"]:
    print(f"- Page {doc.metadata.get('page', 'N/A')} from {doc.metadata.get('source', 'unknown')}")