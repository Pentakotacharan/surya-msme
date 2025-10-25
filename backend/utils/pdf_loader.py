import os
from langchain_community.document_loaders import PyPDFLoader


def load_pdfs_from_folder(folder_path):
    docs = []
    print(f"📂 Loading PDFs from: {folder_path}")
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_path = os.path.join(root, file)
                try:
                    loader = PyPDFLoader(pdf_path)
                    docs.extend(loader.load())
                except Exception as e:
                    print(f"⚠️ Error reading {pdf_path}: {e}")
    return docs
