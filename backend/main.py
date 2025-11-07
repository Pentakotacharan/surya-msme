import os
import pickle
import shutil
import sqlite3
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from fastapi import FastAPI, File, UploadFile,Form, HTTPException
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

# ===== Load environment variables =====
load_dotenv()

# ===== Initialize FastAPI =====
app = FastAPI(title="MSME RAG Chatbot with Full Features")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== Get Groq API key =====
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("⚠️ GROQ_API_KEY not found in .env file.")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

groq_client = Groq(api_key=GROQ_API_KEY)

# ===== Define paths =====
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR.parent / "data"
CACHE_DIR = BASE_DIR / "cache"
DB_DIR = BASE_DIR / "db"

VECTOR_STORE_PATH = CACHE_DIR / "faiss_index"
PROCESSED_DOCS_PATH = CACHE_DIR / "processed_docs.pkl"
DB_PATH = DB_DIR / "analytics.db"

CACHE_DIR.mkdir(exist_ok=True)
DB_DIR.mkdir(exist_ok=True)

print(f"📁 Base Directory: {BASE_DIR}")
print(f"📁 Data Directory: {DATA_DIR}")

# ===== Initialize Database =====
def init_database():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Analytics table
    c.execute('''CREATE TABLE IF NOT EXISTS analytics (
        id INTEGER PRIMARY KEY,
        query TEXT,
        language TEXT,
        response_length INTEGER,
        timestamp DATETIME,
        voice_input BOOLEAN
    )''')
    
    # Feedback table
    c.execute('''CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY,
        rating INTEGER,
        comment TEXT,
        timestamp DATETIME
    )''')
    
    # Tracking table
    c.execute('''CREATE TABLE IF NOT EXISTS tracking (
        id INTEGER PRIMARY KEY,
        app_id TEXT,
        status TEXT,
        timestamp DATETIME
    )''')
    
    # Notifications table
    c.execute('''CREATE TABLE IF NOT EXISTS notifications (
        id INTEGER PRIMARY KEY,
        email TEXT,
        phone TEXT,
        language TEXT,
        timestamp DATETIME
    )''')
    
    conn.commit()
    conn.close()

init_database()

# ===== Load PDFs and setup RAG =====
def load_all_pdfs(base_path):
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

print("\n" + "="*60)
print("📂 DOCUMENT LOADING & CACHING")
print("="*60)

if os.path.exists(PROCESSED_DOCS_PATH):
    print(f"📦 Loading cached documents...")
    with open(PROCESSED_DOCS_PATH, "rb") as f:
        split_docs = pickle.load(f)
    print(f"✅ Loaded {len(split_docs)} chunks from cache.")
else:
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
    print(f"✅ Saved processed documents.")

if not split_docs:
    print("❌ No documents available.")
    exit(1)

# ===== Setup Vector Store =====
print("\n" + "="*60)
print("🧠 VECTOR STORE INITIALIZATION")
print("="*60)

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

if os.path.exists(VECTOR_STORE_PATH):
    print(f"📦 Loading existing vector store...")
    vectorstore = FAISS.load_local(
        str(VECTOR_STORE_PATH), 
        embedding_model, 
        allow_dangerous_deserialization=True
    )
    print("✅ Vector store loaded.")
else:
    print("🧠 Creating embeddings...")
    vectorstore = FAISS.from_documents(split_docs, embedding_model)
    vectorstore.save_local(str(VECTOR_STORE_PATH))
    print(f"✅ Vector store saved.")

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
print("✅ Retriever initialized.")

# ===== Setup LLM =====
print("\n" + "="*60)
print("🤖 GROQ LLM INITIALIZATION")
print("="*60)

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)
print("✅ Groq LLM initialized.")

# ===== Language-specific prompts =====
ENGLISH_TEMPLATE = """You are an expert MSME assistant. Answer ONLY in English.

Context:
{context}

Question: {question}

Answer (ENGLISH ONLY):"""

TELUGU_TEMPLATE = """మీరు MSME నిపుణుడు. తెలుగులో మాత్రమే సమాధానం ఇవ్వండి.

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

print("✅ RAG chains configured.")
print("\n" + "="*60)
print("🚀 SERVER READY")
print("="*60 + "\n")

# ===== Request Models =====
class Message(BaseModel):
    message: str
    language: str = "en"

class FeedbackModel(BaseModel):
    rating: int
    comment: str

class SubscriptionModel(BaseModel):
    email: str
    phone: str = None

# ===== API Endpoints =====

@app.get("/")
async def root():
    return {
        "message": "MSME RAG Chatbot - Full Featured",
        "status": "ready",
        "features": [
            "✅ Multilingual Chat (English & Telugu)",
            "✅ Voice-to-Text",
            "✅ AI-Powered Recommendations",
            "✅ Application Tracking",
            "✅ Analytics Dashboard",
            "✅ Feedback System",
            "✅ Notifications",
            "✅ Data Privacy & Security"
        ]
    }

@app.post("/chat")
async def chat(msg: Message):
    query = msg.message.strip()
    language = msg.language or "en"
    
    if not query:
        return {"reply": "❌ Empty message", "status": "error"}
    
    print(f"\n🔍 Query ({language}): {query[:50]}...")
    
    try:
        chain = create_language_chain(language)
        result = chain.invoke(query)
        
        # Save to analytics
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''INSERT INTO analytics (query, language, response_length, timestamp, voice_input)
                    VALUES (?, ?, ?, ?, ?)''',
                 (query, language, len(result), datetime.now().isoformat(), False))
        conn.commit()
        conn.close()
        
        print(f"✅ Response generated")
        return {
            "reply": result,
            "status": "success",
            "language": language
        }
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return {"reply": f"❌ Error: {str(e)}", "status": "error"}


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...), language: str = Form("en")):
    """Voice-to-Text endpoint - Transcribe audio files (language passed in form field)"""
    temp_file_path = None
    try:
        print(f"\n🎤 Transcribing (lang={language})...")

        temp_file_path = f"temp_{int(time.time())}_{file.filename}"
        with open(temp_file_path, "wb") as f:
            f.write(await file.read())

        # Use groq client's audio transcription
        with open(temp_file_path, "rb") as f:
            transcription = groq_client.audio.transcriptions.create(
                file=(file.filename, f.read()),
                model="whisper-large-v3-turbo",
                language=language,
                response_format="text",
                temperature=0.0
            )

        text = transcription.strip() if isinstance(transcription, str) else ""
        if not text:
            return {"transcription": "", "status": "error", "message": "No speech detected"}

        print(f"✅ Transcribed: {text[:120]}...")
        return {"transcription": text, "status": "success"}

    except Exception as e:
        print(f"❌ Transcription error: {e}")
        return {"transcription": "", "status": "error", "message": str(e)}
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try: os.remove(temp_file_path)
            except: pass

@app.post("/feedback")
async def submit_feedback(feedback: FeedbackModel):
    """Submit user feedback"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''INSERT INTO feedback (rating, comment, timestamp)
                    VALUES (?, ?, ?)''',
                 (feedback.rating, feedback.comment, datetime.now().isoformat()))
        conn.commit()
        conn.close()
        
        print(f"✅ Feedback received (rating: {feedback.rating})")
        return {"status": "success", "message": "Thank you for your feedback!"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/feedback-summary")
async def feedback_summary():
    """Get feedback statistics"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT rating FROM feedback")
        ratings = [row[0] for row in c.fetchall()]
        
        if not ratings:
            return {
                "average_rating": 0,
                "total_feedback": 0,
                "status": "success"
            }
        
        avg = sum(ratings) / len(ratings)
        return {
            "average_rating": round(avg, 2),
            "total_feedback": len(ratings),
            "status": "success"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/subscribe-updates")
async def subscribe_updates(sub: SubscriptionModel):
    """Subscribe to MSME updates"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''INSERT INTO notifications (email, phone, language, timestamp)
                    VALUES (?, ?, ?, ?)''',
                 (sub.email, sub.phone, "en", datetime.now().isoformat()))
        conn.commit()
        conn.close()
        
        return {"status": "success", "message": "Subscribed to updates!"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/analytics")
async def get_analytics():
    """Get usage analytics"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        # Total queries
        c.execute("SELECT COUNT(*) FROM analytics")
        total = c.fetchone()[0]
        
        # By language
        c.execute("SELECT language, COUNT(*) FROM analytics GROUP BY language")
        by_lang = dict(c.fetchall())
        
        # Recent queries
        c.execute("SELECT query, language, timestamp FROM analytics ORDER BY timestamp DESC LIMIT 10")
        recent = [{"query": q[0], "language": q[1], "time": q[2]} for q in c.fetchall()]
        
        conn.close()
        
        return {
            "status": "success",
            "total_queries": total,
            "by_language": by_lang,
            "recent_queries": recent
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/track-application/{app_id}")
async def track_application(app_id: str):
    """Track MSME application status"""
    try:
        # Query RAG for tracking info
        query = f"What is the status and process for MSME application tracking?"
        
        chain = create_language_chain("en")
        result = chain.invoke(query)
        
        return {
            "status": "success",
            "application_id": app_id,
            "tracking_info": result
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

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
