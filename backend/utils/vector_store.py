from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def create_vector_store(docs):
    """Create a FAISS vector store from loaded PDF documents."""
    texts = [doc.page_content for doc in docs]

    print("🧠 Creating embeddings using HuggingFace (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print(f"📊 Creating FAISS index for {len(texts)} documents...")
    db = FAISS.from_texts(texts, embeddings)

    print("✅ Vector store created successfully!")
    return db
