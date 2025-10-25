# import os
# import pickle
# import shutil
# from pathlib import Path
# from fastapi import FastAPI, File, UploadFile, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from dotenv import load_dotenv
# from groq import Groq

# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_groq import ChatGroq
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser

# # === Load environment variables ===
# load_dotenv()

# # === Initialize FastAPI ===
# app = FastAPI(title="MSME RAG Chatbot with Voice Support")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # === Get Groq API key ===
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# if not GROQ_API_KEY:
#     raise ValueError("⚠️ GROQ_API_KEY not found in .env file.")
# os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# # Initialize Groq client for speech-to-text
# groq_client = Groq(api_key=GROQ_API_KEY)

# # === Define paths ===
# BASE_DIR = Path(__file__).parent
# DATA_DIR = BASE_DIR.parent / "data"
# CACHE_DIR = BASE_DIR / "cache"

# VECTOR_STORE_PATH = CACHE_DIR / "faiss_index"
# PROCESSED_DOCS_PATH = CACHE_DIR / "processed_docs.pkl"

# CACHE_DIR.mkdir(exist_ok=True)

# print(f"📁 Base Directory: {BASE_DIR}")
# print(f"📁 Data Directory: {DATA_DIR}")
# print(f"📁 Cache Directory: {CACHE_DIR}")

# # === Step 1: Load all PDFs ===
# def load_all_pdfs(base_path):
#     """Load all PDFs from the given path recursively."""
#     docs = []
#     if not os.path.exists(base_path):
#         print(f"⚠️ Data directory not found: {base_path}")
#         return docs
    
#     for root, _, files in os.walk(base_path):
#         for file in files:
#             if file.endswith(".pdf"):
#                 pdf_path = os.path.join(root, file)
#                 print(f"📘 Loading: {pdf_path}")
#                 try:
#                     loader = PyPDFLoader(pdf_path)
#                     docs.extend(loader.load())
#                 except Exception as e:
#                     print(f"⚠️ Error reading {pdf_path}: {e}")
#     return docs

# # === Step 2: Load or process documents (with caching) ===
# print("\n" + "="*60)
# print("📂 DOCUMENT LOADING & CACHING SYSTEM")
# print("="*60)

# if os.path.exists(PROCESSED_DOCS_PATH):
#     print(f"📦 Loading cached documents from '{PROCESSED_DOCS_PATH}'...")
#     try:
#         with open(PROCESSED_DOCS_PATH, "rb") as f:
#             split_docs = pickle.load(f)
#         print(f"✅ Loaded {len(split_docs)} pre-processed chunks from cache.")
#     except Exception as e:
#         print(f"⚠️ Error loading cache: {e}. Reprocessing documents...")
#         split_docs = None
# else:
#     split_docs = None

# # If cache doesn't exist or failed, process documents
# if split_docs is None:
#     print(f"📂 Loading all PDFs from '{DATA_DIR}'...")
#     docs = load_all_pdfs(str(DATA_DIR))
    
#     if not docs:
#         print("❌ No documents found. Please check your data folder.")
#         exit(1)
    
#     print(f"✅ Loaded {len(docs)} documents from PDF folders.")
    
#     # Split into chunks
#     print("✂️ Splitting documents into chunks...")
#     splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
#     split_docs = splitter.split_documents(docs)
#     print(f"✅ Split into {len(split_docs)} chunks.")
    
#     # Save processed documents
#     print(f"💾 Saving processed documents to '{PROCESSED_DOCS_PATH}'...")
#     with open(PROCESSED_DOCS_PATH, "wb") as f:
#         pickle.dump(split_docs, f)
#     print(f"✅ Saved {len(split_docs)} chunks to cache.")

# # Check if we have documents
# if not split_docs:
#     print("❌ No documents available. Cannot proceed.")
#     exit(1)

# # === Step 3: Create or Load Vector Store ===
# print("\n" + "="*60)
# print("🧠 VECTOR STORE INITIALIZATION")
# print("="*60)

# print("🔄 Initializing HuggingFace embeddings (all-MiniLM-L6-v2)...")
# embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# if os.path.exists(VECTOR_STORE_PATH):
#     print(f"📦 Loading existing vector store from '{VECTOR_STORE_PATH}'...")
#     try:
#         vectorstore = FAISS.load_local(
#             str(VECTOR_STORE_PATH), 
#             embedding_model, 
#             allow_dangerous_deserialization=True
#         )
#         print("✅ Vector store loaded successfully.")
#     except Exception as e:
#         print(f"⚠️ Error loading vector store: {e}. Rebuilding...")
#         print("🧠 Creating new embeddings...")
#         vectorstore = FAISS.from_documents(split_docs, embedding_model)
#         vectorstore.save_local(str(VECTOR_STORE_PATH))
#         print(f"✅ Vector store created and saved.")
# else:
#     print("🧠 Creating new embeddings (this may take a few minutes)...")
#     vectorstore = FAISS.from_documents(split_docs, embedding_model)
#     vectorstore.save_local(str(VECTOR_STORE_PATH))
#     print(f"✅ Vector store created and saved to '{VECTOR_STORE_PATH}'.")

# # === CRITICAL: Create retriever BEFORE language-specific chains ===
# retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
# print("✅ Retriever initialized.")

# # === Step 4: Setup LLM (Groq / Llama3.3) ===
# print("\n" + "="*60)
# print("🤖 GROQ LLM INITIALIZATION")
# print("="*60)
# print("Initializing Groq LLM (llama-3.3-70b-versatile)...")
# llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)
# print("✅ Groq LLM initialized.")

# # === Language-specific prompts with stronger instructions ===
# ENGLISH_TEMPLATE = """You are an expert assistant for MSME schemes, guidelines, and policies in India. 
# You MUST respond ONLY in English. Your response must contain NO Telugu, Hindi, or any other language.

# Provide clear, accurate, and detailed answers based on the context provided.

# IMPORTANT RULES:
# - RESPOND ONLY IN ENGLISH
# - DO NOT include any Telugu translations
# - DO NOT include any Hindi translations
# - DO NOT include any other language
# - ONLY English text in your response

# Context from documents:
# {context}

# User Question:
# {question}

# Your Response (ENGLISH ONLY):"""

# TELUGU_TEMPLATE = """మీరు భారతదేశంలో MSME పథకాలు, మార్గదర్శకాలు మరియు విధానాల గురించి ఒక నిపుణ సహాయక.
# మీరు తెలుగులో ఖచ్చితమైన సమాధానాలను ఇవ్వాలి. మీ సమాధానం తెలుగులో మాత్రమే ఉండాలి - ఇంగ్లీషు లేదా ఇతర భాషలు కూడా ఉండకూడదు.

# సందర్భం నుండి స్పష్టమైన, ఖచ్చితమైన మరియు వివరణాత్మక సమాధానాలను అందించండి.

# ముఖ్యమైన నియమాలు:
# - తెలుగులో మాత్రమే సమాధానం ఇవ్వండి
# - ఇంగ్లీషు అనువాదం చేయవద్దు
# - హిందీ అనువాదం చేయవద్దు
# - ఇతర భాషలు ఉపయోగించవద్దు
# - మీ సమాధానం పూర్తిగా తెలుగు టెక్స్‌ట్‌గా ఉండాలి

# సందర్భం:
# {context}

# ఉపయోగకర్త ప్రశ్న:
# {question}

# మీ సమాధానం (తెలుగులో మాత్రమే):"""

# # === Document formatter ===
# def format_docs(docs):
#     """Format retrieved documents for the prompt"""
#     return "\n\n".join(doc.page_content for doc in docs)

# # === Create language-specific chains ===
# def create_language_chain(language: str):
#     """Create a language-specific RAG chain"""
#     print(f"🔗 Creating {language.upper()} chain...")
    
#     template = TELUGU_TEMPLATE if language == "te" else ENGLISH_TEMPLATE
#     prompt = ChatPromptTemplate.from_template(template)
    
#     chain = (
#         {"context": retriever | format_docs, "question": RunnablePassthrough()}
#         | prompt
#         | llm
#         | StrOutputParser()
#     )
#     return chain

# print("\n" + "="*60)
# print("🔗 RAG CHAIN SETUP")
# print("="*60)
# print("✅ Language-specific RAG chains configured (English & Telugu).")

# print("\n" + "="*60)
# print("🚀 SERVER READY")
# print("="*60 + "\n")

# # === Request Models ===
# class Message(BaseModel):
#     message: str
#     language: str = "en"

# # === API Endpoints ===

# @app.get("/")
# async def root():
#     """Health check endpoint."""
#     return {
#         "message": "MSME RAG Chatbot with Voice Support",
#         "status": "ready",
#         "version": "2.0",
#         "endpoints": {
#             "chat": "/chat (POST)",
#             "transcribe": "/transcribe (POST - multipart/form-data)",
#             "cache-status": "/cache-status (GET)",
#             "rebuild-cache": "/rebuild-cache (POST)"
#         }
#     }

# @app.post("/chat")
# async def chat(msg: Message):
#     """Chat endpoint for querying the MSME knowledge base."""
#     query = msg.message.strip()
#     language = msg.language or "en"
    
#     print(f"\n🔍 Query ({language}): {query[:50]}...")
#     print(f"📝 Full language setting: {language}")
    
#     try:
#         # Create language-specific chain
#         chain = create_language_chain(language)
        
#         # Invoke chain
#         result = chain.invoke(query)
        
#         # Debug: Check language in response
#         has_telugu = any(ord(char) > 0x0C00 and ord(char) < 0x0C7F for char in result)
#         has_english = any(char.isascii() and char.isalpha() for char in result)
        
#         if language == "te":
#             print(f"✅ Response generated in Telugu (Has Telugu: {has_telugu}, Has English: {has_english})")
#         else:
#             print(f"✅ Response generated in English (Has Telugu: {has_telugu}, Has English: {has_english})")
        
#         return {
#             "reply": result,
#             "status": "success",
#             "language": language
#         }
#     except Exception as e:
#         print(f"❌ Error: {str(e)}")
#         return {
#             "reply": f"❌ Error: {str(e)}",
#             "status": "error",
#             "language": language
#         }

# @app.post("/transcribe")
# async def transcribe_audio(file: UploadFile = File(...), language: str = "en"):
#     """
#     Transcribe audio file to text using Groq Whisper API.
#     Supports English (en) and Telugu (te).
#     """
#     temp_file_path = None
#     try:
#         print(f"\n🎤 Transcribing audio in language: {language}")
        
#         # Save uploaded file temporarily
#         temp_file_path = f"temp_audio_{file.filename}"
#         with open(temp_file_path, "wb") as temp_file:
#             content = await file.read()
#             temp_file.write(content)
        
#         # Transcribe using Groq Whisper API
#         with open(temp_file_path, "rb") as audio_file:
#             # For Telugu, use language code "te"
#             transcription = groq_client.audio.transcriptions.create(
#                 file=(file.filename, audio_file.read()),
#                 model="whisper-large-v3-turbo",
#                 language=language,  # "en" for English, "te" for Telugu
#                 response_format="text",
#                 temperature=0.0
#             )
        
#         # Clean up temp file
#         if os.path.exists(temp_file_path):
#             os.remove(temp_file_path)
        
#         # Extract text
#         text = transcription.strip() if isinstance(transcription, str) else transcription.get("text", "").strip()
        
#         print(f"✅ Transcription ({language}): {text[:60]}...")
        
#         return {
#             "transcription": text,
#             "status": "success",
#             "language": language
#         }
    
#     except Exception as e:
#         print(f"❌ Transcription error: {str(e)}")
#         if temp_file_path and os.path.exists(temp_file_path):
#             os.remove(temp_file_path)
        
#         return {
#             "transcription": "",
#             "status": "error",
#             "message": str(e),
#             "language": language
#         }

# @app.get("/cache-status")
# async def cache_status():
#     """Check cache status."""
#     return {
#         "cached_documents": os.path.exists(PROCESSED_DOCS_PATH),
#         "cached_vector_store": os.path.exists(VECTOR_STORE_PATH),
#         "total_chunks": len(split_docs),
#         "cache_directory": str(CACHE_DIR)
#     }

# @app.post("/rebuild-cache")
# async def rebuild_cache():
#     """Delete cached files to force rebuild on next restart."""
#     try:
#         if os.path.exists(PROCESSED_DOCS_PATH):
#             os.remove(PROCESSED_DOCS_PATH)
#             print(f"🗑️ Deleted {PROCESSED_DOCS_PATH}")
        
#         if os.path.exists(VECTOR_STORE_PATH):
#             shutil.rmtree(VECTOR_STORE_PATH)
#             print(f"🗑️ Deleted {VECTOR_STORE_PATH}")
        
#         return {
#             "message": "✅ Cache cleared successfully. Restart server to rebuild.",
#             "status": "success"
#         }
#     except Exception as e:
#         return {
#             "message": f"❌ Error clearing cache: {str(e)}",
#             "status": "error"
#         }

import os
import pickle
import shutil
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq
import time

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# === Load environment variables ===
load_dotenv()

# === Initialize FastAPI ===
app = FastAPI(title="MSME RAG Chatbot with Voice Support")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Get Groq API key ===
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("⚠️ GROQ_API_KEY not found in .env file.")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Initialize Groq client for speech-to-text
groq_client = Groq(api_key=GROQ_API_KEY)

# === Define paths ===
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR.parent / "data"
CACHE_DIR = BASE_DIR / "cache"

VECTOR_STORE_PATH = CACHE_DIR / "faiss_index"
PROCESSED_DOCS_PATH = CACHE_DIR / "processed_docs.pkl"

CACHE_DIR.mkdir(exist_ok=True)

print(f"📁 Base Directory: {BASE_DIR}")
print(f"📁 Data Directory: {DATA_DIR}")
print(f"📁 Cache Directory: {CACHE_DIR}")

# === Step 1: Load all PDFs ===
def load_all_pdfs(base_path):
    """Load all PDFs from the given path recursively."""
    docs = []
    if not os.path.exists(base_path):
        print(f"⚠️ Data directory not found: {base_path}")
        return docs
    
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith(".pdf"):
                pdf_path = os.path.join(root, file)
                print(f"📘 Loading: {pdf_path}")
                try:
                    loader = PyPDFLoader(pdf_path)
                    docs.extend(loader.load())
                except Exception as e:
                    print(f"⚠️ Error reading {pdf_path}: {e}")
    return docs

# === Step 2: Load or process documents (with caching) ===
print("\n" + "="*60)
print("📂 DOCUMENT LOADING & CACHING SYSTEM")
print("="*60)

if os.path.exists(PROCESSED_DOCS_PATH):
    print(f"📦 Loading cached documents...")
    try:
        with open(PROCESSED_DOCS_PATH, "rb") as f:
            split_docs = pickle.load(f)
        print(f"✅ Loaded {len(split_docs)} pre-processed chunks from cache.")
    except Exception as e:
        print(f"⚠️ Error loading cache: {e}")
        split_docs = None
else:
    split_docs = None

if split_docs is None:
    print(f"📂 Loading all PDFs from '{DATA_DIR}'...")
    docs = load_all_pdfs(str(DATA_DIR))
    
    if not docs:
        print("❌ No documents found.")
        exit(1)
    
    print(f"✅ Loaded {len(docs)} documents.")
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    split_docs = splitter.split_documents(docs)
    print(f"✅ Split into {len(split_docs)} chunks.")
    
    with open(PROCESSED_DOCS_PATH, "wb") as f:
        pickle.dump(split_docs, f)
    print(f"✅ Saved processed documents to cache.")

if not split_docs:
    print("❌ No documents available.")
    exit(1)

# === Step 3: Create or Load Vector Store ===
print("\n" + "="*60)
print("🧠 VECTOR STORE INITIALIZATION")
print("="*60)

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

if os.path.exists(VECTOR_STORE_PATH):
    print(f"📦 Loading existing vector store...")
    try:
        vectorstore = FAISS.load_local(
            str(VECTOR_STORE_PATH), 
            embedding_model, 
            allow_dangerous_deserialization=True
        )
        print("✅ Vector store loaded.")
    except Exception as e:
        print(f"⚠️ Error: {e}. Rebuilding...")
        vectorstore = FAISS.from_documents(split_docs, embedding_model)
        vectorstore.save_local(str(VECTOR_STORE_PATH))
        print(f"✅ Vector store created.")
else:
    print("🧠 Creating embeddings...")
    vectorstore = FAISS.from_documents(split_docs, embedding_model)
    vectorstore.save_local(str(VECTOR_STORE_PATH))
    print(f"✅ Vector store saved.")

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
print("✅ Retriever initialized.")

# === Step 4: Setup LLM ===
print("\n" + "="*60)
print("🤖 GROQ LLM INITIALIZATION")
print("="*60)

# Initialize LLM with lower temperature for more consistent responses
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2,
    timeout=30,  # 30 second timeout
    max_retries=2
)
print("✅ Groq LLM initialized.")

# === Language-specific prompts ===
ENGLISH_TEMPLATE = """You are an MSME expert. Answer ONLY in English. No other languages.

Context:
{context}

Question: {question}

Answer (ENGLISH ONLY):"""

TELUGU_TEMPLATE = """మీరు MSME నిపుణుడు. తెలుగులో మాత్రమే సమాధానం ఇవ్వండి. ఇతర భాషలు ఉపయోగించవద్దు.

సందర్భం:
{context}

ప్రశ్న:
{question}

సమాధానం (తెలుగులో మాత్రమే):"""

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def create_language_chain(language: str):
    template = TELUGU_TEMPLATE if language == "te" else ENGLISH_TEMPLATE
    prompt = ChatPromptTemplate.from_template(template)
    
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

print("\n" + "="*60)
print("✅ RAG CHAIN READY")
print("="*60 + "\n")

# === Models ===
class Message(BaseModel):
    message: str
    language: str = "en"

# === Endpoints ===

@app.get("/")
async def root():
    return {
        "status": "ready",
        "message": "MSME Chatbot API",
        "version": "3.0"
    }

@app.post("/chat")
async def chat(msg: Message):
    """Chat endpoint - with retry logic"""
    query = msg.message.strip()
    language = msg.language or "en"
    
    if not query:
        return {"reply": "❌ Empty message", "status": "error"}
    
    print(f"\n🔍 Query ({language}): {query[:50]}...")
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            chain = create_language_chain(language)
            result = chain.invoke(query)
            
            if not result or len(result.strip()) == 0:
                return {"reply": "❌ Empty response from LLM", "status": "error"}
            
            print(f"✅ Response generated on attempt {attempt + 1}")
            return {
                "reply": result,
                "status": "success",
                "language": language
            }
        
        except Exception as e:
            print(f"❌ Attempt {attempt + 1} failed: {str(e)}")
            
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"⏳ Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                return {
                    "reply": f"❌ Error after {max_retries} attempts: {str(e)}",
                    "status": "error"
                }

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...), language: str = "en"):
    """Transcribe audio with retry logic"""
    temp_file_path = None
    
    try:
        print(f"\n🎤 Transcribing ({language})...")
        
        temp_file_path = f"temp_{int(time.time())}_{file.filename}"
        with open(temp_file_path, "wb") as f:
            f.write(await file.read())
        
        with open(temp_file_path, "rb") as f:
            transcription = groq_client.audio.transcriptions.create(
                file=(file.filename, f.read()),
                model="whisper-large-v3-turbo",
                language=language,
                response_format="text",
                temperature=0.0
            )
        
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        
        text = transcription.strip() if isinstance(transcription, str) else ""
        
        if not text:
            return {
                "transcription": "",
                "status": "error",
                "message": "No speech detected"
            }
        
        print(f"✅ Transcription: {text[:50]}...")
        
        return {
            "transcription": text,
            "status": "success",
            "language": language
        }
    
    except Exception as e:
        print(f"❌ Transcription error: {str(e)}")
        return {
            "transcription": "",
            "status": "error",
            "message": str(e)
        }
    
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.get("/cache-status")
async def cache_status():
    return {
        "cached_documents": os.path.exists(PROCESSED_DOCS_PATH),
        "cached_vector_store": os.path.exists(VECTOR_STORE_PATH),
        "total_chunks": len(split_docs)
    }

@app.post("/rebuild-cache")
async def rebuild_cache():
    try:
        for path in [PROCESSED_DOCS_PATH, VECTOR_STORE_PATH]:
            if os.path.exists(path):
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
        
        return {"message": "✅ Cache cleared", "status": "success"}
    except Exception as e:
        return {"message": f"❌ Error: {str(e)}", "status": "error"}
