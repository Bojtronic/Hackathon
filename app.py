import streamlit as st
from document_processor import DocumentProcessor
from rag_system import RenewableEnergyRAG
import pandas as pd

# Configuración de la UI
st.set_page_config(page_title="AI Copilot - Energías Renovables", layout="wide")
st.title("🌱 AI Copilot para Documentación de Energías Renovables")

# Procesamiento de documentos
with st.expander("📤 Subir Documentos", expanded=True):
    uploaded_files = st.file_uploader(
        "Arrastra tus documentos (PDF, DOCX, XLSX)",
        type=["pdf", "docx", "xlsx"],
        accept_multiple_files=True
    )

if uploaded_files:
    processor = DocumentProcessor()
    with st.spinner("Procesando documentos..."):
        docs = []
        for file in uploaded_files:
            temp_path = f"/tmp/{file.name}"
            with open(temp_path, "wb") as f:
                f.write(file.getbuffer())
            docs.extend(processor.load_document(temp_path))
        
        db = processor.create_vector_store(docs)
        st.success(f"✅ {len(docs)} fragmentos procesados de {len(uploaded_files)} archivos")

# Consultas
query = st.text_input("🔍 Haz una pregunta sobre los documentos")
if query:
    rag = RenewableEnergyRAG()
    with st.spinner("Buscando en los documentos..."):
        response = rag.query(query)
        
    st.subheader("Respuesta")
    st.write(response["answer"])
    
    with st.expander("📚 Referencias"):
        for i, source in enumerate(response["sources"], 1):
            st.markdown(f"""
            **Referencia {i}**  
            📄 Archivo: `{source['file']}`  
            📑 Página: {source['page']}  
            📝 Extracto:  
            > {source['content']}
            """)

# Checklist automation
with st.expander("📋 Automatización de Checklist", expanded=False):
    checklist_file = st.file_uploader("Sube tu checklist (Excel)", type=["xlsx"])
    
    if checklist_file:
        df = pd.read_excel(checklist_file)
        if st.button("Generar Checklist Automatizado"):
            rag = RenewableEnergyRAG()
            results = []
            
            for _, row in df.iterrows():
                response = rag.query(f"¿El documento menciona algo sobre: {row['Requisito']}?")
                results.append({
                    "Requisito": row["Requisito"],
                    "Cumplido": "Sí" if "sí" in response["answer"].lower() else "No",
                    "Referencias": "; ".join([f"{s['file']} (p. {s['page']})" for s in response["sources"]])
                })
            
            st.dataframe(pd.DataFrame(results))

