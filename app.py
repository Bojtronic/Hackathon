import streamlit as st
import os
from tempfile import NamedTemporaryFile
from document_processor import DocumentProcessor
from rag_system import RAGSystem
import pandas as pd
from io import BytesIO

st.set_page_config(page_title="AI Copilot for Renewable Energy Data Rooms", layout="wide")

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")

st.title("AI Copilot for Renewable Energy Data Rooms")

with st.expander("Upload Documents", expanded=True):
    uploaded_files = st.file_uploader(
        "Upload Data Room Files (PDF, DOCX, XLSX)", 
        accept_multiple_files=True, 
        type=["pdf", "docx", "xlsx"]
    )

if uploaded_files:
    processor = DocumentProcessor()
    all_docs = []
    
    with st.spinner("Processing documents..."):
        for file in uploaded_files:
            try:
                with NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as temp_file:
                    temp_file.write(file.getbuffer())
                    temp_path = temp_file.name
                
                docs = processor.load_document(temp_path)
                all_docs.extend(docs)
                os.unlink(temp_path) 
                
            except Exception as e:
                st.error(f"Error processing {file.name}: {str(e)}")
                continue
    
    if all_docs:
        db = processor.create_vector_store(all_docs)
        st.success(f"Processed {len(all_docs)} document chunks from {len(uploaded_files)} files!")
        
        st.divider()
        query = st.text_input("Ask a question about the documents:", placeholder="e.g. What are the warranty terms?")
        
        if query:
            with st.spinner("Searching documents..."):
                try:
                    rag = RAGSystem(openai_api_key=OPENAI_API_KEY)
                    response = rag.query(query)
                    
                    st.subheader("Answer")
                    st.write(response["answer"])
                    
                    with st.expander("Document References"):
                        for doc in response["sources"]:
                            st.write(f"{doc.metadata.get('source', 'Unknown')} (Page {doc.metadata.get('page', 'N/A')})")
                            st.caption(doc.page_content[:200] + "...")
                            
                except Exception as e:
                    st.error(f"Query failed: {str(e)}")

st.divider()
with st.expander("Checklist Automation", expanded=False):
    if st.checkbox("Upload a checklist template (Excel)"):
        checklist_file = st.file_uploader("Upload Checklist Template", type=["xlsx"])
        
        if checklist_file:
            with st.spinner("Processing checklist..."):
                try:                  
                    excel_data = BytesIO(checklist_file.read())
                    df = pd.read_excel(excel_data)
                    
                    required_columns = ['Item', 'Requirement', 'Verified']
                    if not all(col in df.columns for col in required_columns):
                        st.warning("The checklist must contain the columns: 'Item', 'Requirement', 'Verified'")
                    else:
                        if 'rag' not in locals():
                            rag = RAGSystem(openai_api_key=OPENAI_API_KEY)
                        
                        progress_bar = st.progress(0)
                        results = []
                        
                        for i, row in df.iterrows():
                            query = f"Does the document contain information about: {row['Requirement']}?"
                            response = rag.query(query)
                            
                            verification_status = "Yes" if "yes" in response["answer"].lower() else "No"
                            
                            sources = []
                            for doc in response["sources"]:
                                source_info = {
                                    "document": doc.metadata.get('source', 'Unknown'),
                                    "page": doc.metadata.get('page', 'N/A'),
                                    "excerpt": doc.page_content[:150] + "..."
                                }
                                sources.append(source_info)
                            
                            results.append({
                                "Item": row['Item'],
                                "Requirement": row['Requirement'],
                                "Verified": verification_status,
                                "Sources": sources
                            })
                            
                            progress_bar.progress((i + 1) / len(df))
                        
                        st.success("Checklist processed successfully!")
                        
                        tab1, tab2 = st.tabs(["Summary", "Full Detail"])
                        
                        with tab1:
                            summary_df = pd.DataFrame([{
                                'Item': r['Item'],
                                'Requirement': r['Requirement'],
                                'Verified': r['Verified']
                            } for r in results])
                            st.dataframe(summary_df, use_container_width=True)
                            
                            verified_count = sum(1 for r in results if r['Verified'] == "Yes")
                            st.metric("Verified requirements", f"{verified_count}/{len(results)}")
                        
                        with tab2:
                            for result in results:
                                with st.expander(f"{result['Item']}: {result['Requirement']} - {result['Verified']}"):
                                    st.write(f"**Status:** {result['Verified']}")
                                    
                                    if result['Sources']:
                                        st.write("**Found sources:**")
                                        for source in result['Sources']:
                                            st.write(f"--- {source['document']} (Page {source['page']})")
                                            st.caption(f"Excerpt: {source['excerpt']}")
                                    else:
                                        st.write("No references were found in the documents")
                        
                        output = BytesIO()
                        completed_df = pd.DataFrame(results)
                        completed_df.to_excel(output, index=False)
                        st.download_button(
                            label="Download Completed Checklist",
                            data=output.getvalue(),
                            file_name="completed_checklist.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                        
                except Exception as e:
                    st.error(f"Checklist processing failed: {str(e)}")
                    st.exception(e)