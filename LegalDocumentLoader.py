import os
import re
import fitz
import json
#%%
class LegalDocumentLoader:
    def __init__(self, doc_type="legal_document"):
        """
        Initializes the loader with a default document type for metadata.
        """
        self.doc_type = doc_type

    def load_txt(self, file_path):
        """Extracts text from a .txt file."""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

    def load_pdf(self, file_path):
        """Extracts text from a .pdf file using PyMuPDF."""
        text = ""
        try:
            with fitz.open(file_path) as pdf_doc:
                for page in pdf_doc:
                    text += page.get_text() + "\n"
        except Exception as e:
            print(f"Error reading PDF {file_path}: {e}")
        return text

    def clean_text(self, text):
        """
        Applies Regex to reduce noise (page numbers, headers, excessive newlines).
        """
        # 1. Remove "--- PAGE X ---" markers
        text = re.sub(r"---\s*PAGE\s+\d+\s*---", "", text, flags=re.IGNORECASE)

        # 2. Remove specific source tags if they exist (e.g., "")
        text = re.sub(r"\\", "", text)

        # 3. Normalize whitespace (replace 3 or more consecutive newlines with exactly 2)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # 4. Strip leading/trailing whitespace
        return text.strip()

    def process_file(self, file_path):
        """
        Routes the file to the correct loader, cleans it, and attaches metadata.
        """
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()

        # Text Extraction
        if ext == '.txt':
            raw_text = self.load_txt(file_path)
        elif ext == '.pdf':
            raw_text = self.load_pdf(file_path)
        else:
            print(f"Skipping unsupported file format: {ext}")
            return None

        # Noise Reduction
        cleaned_text = self.clean_text(raw_text)

        # Metadata Tagging
        metadata = {
            "source_file": os.path.basename(file_path),
            "doc_type": self.doc_type,
            "extension": ext
        }

        # Return a structured document
        return {
            "page_content": cleaned_text,
            "metadata": metadata
        }

    def process_directory(self, directory_path):
        """
        Loops through a directory and processes all supported files.
        """
        processed_documents = []
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)

            if os.path.isfile(file_path):
                doc = self.process_file(file_path)
                if doc:
                    processed_documents.append(doc)

        return processed_documents

    def save_to_json(self, documents, output_filename="processed_documents.json"):
        """
        Saves the list of processed document dictionaries to a JSON file.
        """
        with open(output_filename, 'w', encoding='utf-8') as file:
            # ensure_ascii=False keeps Spanish characters like á, é, ñ intact
            json.dump(documents, file, ensure_ascii=False, indent=4)
        print(f"DEBUG: Saved all data to {output_filename}")
#%%
if __name__ == "__main__":
    loader = LegalDocumentLoader(doc_type="constitution_guatemala")

    # We will use the subfolder since you moved the file there
    folder_path = "my_legal_documents"

    if os.path.exists(folder_path):
        all_docs = loader.process_directory(folder_path)
        print(f"Successfully processed {len(all_docs)} documents.")

        if all_docs:
            # ---> ADD THIS LINE TO SAVE THE FILES <---
            loader.save_to_json(all_docs, "cleaned_legal_data.json")

    else:
        print(f"Please create a folder named '{folder_path}' and add some files to test.")