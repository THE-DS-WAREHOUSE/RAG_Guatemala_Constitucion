import json
import re
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
#%%
class LegalVectorDB:
    def __init__(self, json_path):
        self.json_path = json_path
        # We use a multilingual model because the Constitution is in Spanish
        print("Loading embedding model (this might take a minute the first time)...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        self.db_directory = "./chroma_db"

    def load_json(self):
        """Loads the cleaned JSON data."""
        with open(self.json_path, 'r', encoding='utf-8') as file:
            return json.load(file)

    def chunk_by_article(self, text, metadata):
        """
        Custom chunker that splits the text exactly where a new 'Artículo' begins.
        This ensures laws are kept whole and context is preserved.
        """
        # Regex explanation: Split string but KEEP the matched "Artículo X" text attached to the chunk
        # Lookahead assertion: (?=Artículo\s+\d+)
        chunks = re.split(r'(?=Artículo\s+\d+-?)', text)

        documents = []
        for chunk in chunks:
            chunk = chunk.strip()
            if len(chunk) > 10:  # Ignore tiny or empty chunks
                # Package it into a LangChain Document object, passing down the metadata
                doc = Document(page_content=chunk, metadata=metadata)
                documents.append(doc)

        return documents

    def build_database(self):
        """Processes the JSON, chunks the text, and stores it in ChromaDB."""
        raw_data = self.load_json()
        all_documents = []

        print("Chunking documents by Article...")
        for item in raw_data:
            text = item["page_content"]
            metadata = item["metadata"]

            # Create our smart, article-based chunks
            chunks = self.chunk_by_article(text, metadata)
            all_documents.extend(chunks)

        print(f"Created {len(all_documents)} separate chunks (Articles).")

        print("Building the Vector Database... (this converts text to numbers)")
        # This creates the database and saves it to a local folder
        vector_db = Chroma.from_documents(
            documents=all_documents,
            embedding=self.embeddings,
            persist_directory=self.db_directory
        )
        print(f"Success! Database saved to '{self.db_directory}' folder.")
        return vector_db

#%%
if __name__ == "__main__":
    # 1. Initialize our builder with the JSON file you just created
    db_builder = LegalVectorDB("cleaned_legal_data.json")

    # 2. Build and save the database
    vector_db = db_builder.build_database()
#%%

    # Let's ask a question in Spanish about the Constitution
    query = "¿Cuáles son los deberes del Estado?"
    print(f"Question: {query}\n")

    # Search the database for the top 3 most relevant articles
    results = vector_db.similarity_search(query, k=3)

    for i, result in enumerate(results):
        print(f"--- MATCH {i + 1} ---")
        print(f"Source: {result.metadata.get('source_file')}")
        print(f"Text: {result.page_content}\n")