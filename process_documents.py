import os
from document_processor import DocumentProcessor

def process_folder(input_dir="data/documents/", output_dir="data/vector_store/"):
    processor = DocumentProcessor(model_path="models/embedder.pth")
    all_docs = []
    
    # Procesar todos los archivos en la carpeta
    for filename in os.listdir(input_dir):
        filepath = os.path.join(input_dir, filename)
        if os.path.isfile(filepath):
            try:
                docs = processor.load_document(filepath)
                all_docs.extend(docs)
                print(f"Procesado: {filename} ({len(docs)} chunks)")
            except Exception as e:
                print(f"Error en {filename}: {str(e)}")
    
    # Crear vectorstore
    if all_docs:
        db = processor.create_vector_store(all_docs, save_path=output_dir)
        print(f"✅ Base de datos FAISS guardada en {output_dir}")

if __name__ == "__main__":
    process_folder()

