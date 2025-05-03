# 🌱 AI Copilot for Renewable Energy Data Rooms

**Autonomous document analysis system** for renewable energy projects using fully local AI models. No API dependencies.

## 🚀 Key Features

- 🔍 **Natural Language Querying** - Ask questions in natural language about contracts, permits, and reports
- 📑 **Multi-Format Support** - Process PDFs, Word docs, Excel files (no external APIs)
- 🏷️ **Precise Source Referencing** - Answers include:
  - Exact document excerpts
  - Page numbers 
  - File names with clickable links
- ✅ **Smart Checklist Automation** - Upload Excel templates to auto-populate from documents
- 🏗️ **Custom AI Models** - Domain-specific models for renewable energy documentation

## 🛠️ Technical Stack

| Component               | Technology |
|-------------------------|------------|
| Document Processing     | PyPDF, pdfplumber, python-docx |
| Text Embeddings         | Custom Transformer Models (PyTorch) |
| Vector Database         | FAISS (local) |
| Language Model          | Custom Renewable Energy LLM |
| UI Framework            | Streamlit |
