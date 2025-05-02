from document_processor import DocumentProcessor

processor = DocumentProcessor()
docs = processor.load_document("sample.pdf") 
db = processor.create_vector_store(docs)
print("Vector database created successfully!")