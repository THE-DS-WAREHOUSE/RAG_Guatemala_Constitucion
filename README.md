# RAG System: Constitution of Guatemala

This project implements a Retrieval-Augmented Generation (RAG) pipeline designed specifically for querying legal documents, specifically the Constitution of Guatemala. It extracts text from local files, chunks them logically by legal articles, creates a vector database using multilingual embeddings, and provides an interactive chat interface using OpenAI's LLMs.

## Project Structure

* `my_legal_documents/` - The input directory where your raw PDF or TXT legal documents should be placed.
* `LegalDocumentLoader.py` - Extracts text from PDFs and TXTs, reduces noise (removes page numbers/excess empty lines), and saves the structured output as a JSON file.
* `build_vector_db.py` - Reads the cleaned JSON, chunks the text smartly by "Artículo" (Article), generates embeddings using a multilingual HuggingFace model, and stores everything in a local Chroma vector database.
* `rag_system.py` - The main application file. It connects the Chroma database (as a retriever) to an OpenAI Chat model, applying a strict system prompt to ensure factual, context-based answers in Spanish. 
* `requirements.txt` - Lists all necessary Python dependencies.

## Prerequisites

Before running the project, you need Python installed on your system and an OpenAI API Key.

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
