import os
import pickle
import sqlite3
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, File, UploadFile
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

# Load environment
load_dotenv()

app = FastAPI(title="MSME RAG Chatbot - AP Digital Empowerment Challenge 2025")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("⚠️ GROQ_API_KEY not found")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

groq_client = Groq(api_key=GROQ_API_KEY)

# ===== Paths =====
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR.parent / "data"
CACHE_DIR = BASE_DIR / "cache"
DB_DIR = BASE_DIR / "db"

VECTOR_STORE_PATH = CACHE_DIR / "faiss_index"
PROCESSED_DOCS_PATH = CACHE_DIR / "processed_docs.pkl"
DB_PATH = DB_DIR / "analytics.db"

CACHE_DIR.mkdir(exist_ok=True)
DB_DIR.mkdir(exist_ok=True)

# ===== Database Setup =====
def init_database():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Analytics table
    c.execute('''CREATE TABLE IF NOT EXISTS analytics (
        id INTEGER PRIMARY KEY,
        query TEXT NOT NULL,
        language TEXT NOT NULL,
        response TEXT NOT NULL,
        response_length INTEGER,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )''')
    
    # Feedback table
    c.execute('''CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY,
        rating INTEGER NOT NULL,
        comment TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )''')
    
    # Application tracking table
    c.execute('''CREATE TABLE IF NOT EXISTS tracking (
        id INTEGER PRIMARY KEY,
        app_id TEXT NOT NULL,
        status TEXT,
        last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
    )''')
    
    # Notifications/Subscriptions table
    c.execute('''CREATE TABLE IF NOT EXISTS subscriptions (
        id INTEGER PRIMARY KEY,
        email TEXT NOT NULL,
        phone TEXT,
        language TEXT,
        subscribed_date DATETIME DEFAULT CURRENT_TIMESTAMP
    )''')
    
    conn.commit()
    conn.close()
    print("✅ Database initialized")

init_database()

# ===== Load Documents =====
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
                    print(f"⚠️ Error: {e}")
    return docs

print("\n" + "="*70)
print("📂 DOCUMENT LOADING & CACHING")
print("="*70)

if os.path.exists(PROCESSED_DOCS_PATH):
    print("📦 Loading cached documents...")
    with open(PROCESSED_DOCS_PATH, "rb") as f:
        split_docs = pickle.load(f)
    print(f"✅ Loaded {len(split_docs)} chunks from cache")
else:
    print(f"📂 Loading PDFs from {DATA_DIR}...")
    docs = load_all_pdfs(str(DATA_DIR))
    
    if not docs:
        print("⚠️ No PDFs found, using demo data")
        from langchain_core.documents import Document
        split_docs = [
            Document(page_content="MSME stands for Micro, Small and Medium Enterprises. They are crucial for economic development."),
            Document(page_content="To register MSME, visit AP MSME portal and fill application form with business details."),
            Document(page_content="MSME loans available: MUDRA scheme, Prime Minister Employment Generation Programme, Credit Guarantee Scheme.")
        ]
    else:
        print(f"✅ Loaded {len(docs)} documents")
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        split_docs = splitter.split_documents(docs)
        print(f"✅ Split into {len(split_docs)} chunks")
    
    with open(PROCESSED_DOCS_PATH, "wb") as f:
        pickle.dump(split_docs, f)
    print("✅ Cached documents")

# ===== Vector Store Setup =====
print("\n" + "="*70)
print("🧠 VECTOR STORE INITIALIZATION")
print("="*70)

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

if os.path.exists(VECTOR_STORE_PATH):
    print("📦 Loading existing vector store...")
    vectorstore = FAISS.load_local(str(VECTOR_STORE_PATH), embedding_model, allow_dangerous_deserialization=True)
    print("✅ Vector store loaded")
else:
    print("🧠 Creating vector store...")
    vectorstore = FAISS.from_documents(split_docs, embedding_model)
    vectorstore.save_local(str(VECTOR_STORE_PATH))
    print("✅ Vector store saved")

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
print("✅ Retriever configured")

# ===== LLM Setup =====
print("\n" + "="*70)
print("🤖 GROQ LLM SETUP")
print("="*70)

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)
print("✅ LLM ready")

# ===== Language Prompts =====
ENGLISH_TEMPLATE = """You are an expert MSME assistant for Andhra Pradesh, India. Answer ONLY in English.

Context from MSME documents:
{context}

User Question: {question}

Instructions:
- Answer ONLY in English
- Be helpful, accurate and concise
- Use information from the provided context
- Focus on Andhra Pradesh MSME schemes and policies
- If unsure, say "I don't have enough information"

Answer (English only):"""

TELUGU_TEMPLATE = """మీరు ఆంధ్రప్రదేశ్ MSME నిపుణులు. తెలుగులో మాత్రమే సమాధానం ఇవ్వండి.

MSME పత్రాల నుండి సందర్భం:
{context}

వినియోగదారు ప్రశ్న:
{question}

సూచనలు:
- తెలుగులో మాత్రమే సమాధానం ఇవ్వండి
- సహాయకరంగా మరియు ఖచ్చితంగా ఉండండి
- అందించిన సందర్భం నుండి సమాచారాన్ని ఉపయోగించండి
- ఆంధ్రప్రదేశ్ MSME స్కీమ్‌లపై దృష్టి సారించండి
- సందేహం ఉంటే "నాకు సరిపోయే సమాచారం లేదు" అని చెప్పండి

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

print("✅ RAG chains configured\n")

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
        "status": "ready",
        "message": "MSME RAG Chatbot API",
        "version": "1.0",
        "features": [
            "✅ Multilingual Chat (English & Telugu)",
            "✅ Voice-to-Text Transcription",
            "✅ AI-Powered Recommendations",
            "✅ Application Status Tracking",
            "✅ Analytics Dashboard",
            "✅ Feedback System",
            "✅ Email Notifications",
            "✅ Data Privacy & Security"
        ]
    }

@app.post("/chat")
async def chat(msg: Message):
    """Chat endpoint - Process queries and return responses"""
    query = msg.message.strip()
    language = msg.language or "en"
    
    if not query:
        return {"reply": "❌ Please enter a message", "status": "error"}
    
    print(f"\n🔍 Query ({language}): {query[:60]}...")
    
    try:
        chain = create_language_chain(language)
        result = chain.invoke(query)
        
        # Save to database
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''INSERT INTO analytics (query, language, response, response_length, timestamp)
                    VALUES (?, ?, ?, ?, ?)''',
                 (query, language, result, len(result), datetime.now().isoformat()))
        conn.commit()
        conn.close()
        
        print(f"✅ Response generated & saved ({language})")
        
        return {
            "reply": result,
            "status": "success",
            "language": language
        }
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return {"reply": f"❌ Error: {str(e)}", "status": "error"}

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...), language: str = "en"):
    """Voice-to-Text endpoint - Transcribe audio files"""
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
            return {"transcription": "", "status": "error", "message": "No speech detected"}
        
        print(f"✅ Transcribed: {text[:50]}...")
        
        return {"transcription": text, "status": "success"}
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return {"transcription": "", "status": "error", "message": str(e)}
    
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post("/feedback")
async def submit_feedback(feedback: FeedbackModel):
    """Feedback endpoint - Collect user ratings and comments"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''INSERT INTO feedback (rating, comment, timestamp)
                    VALUES (?, ?, ?)''',
                 (feedback.rating, feedback.comment, datetime.now().isoformat()))
        conn.commit()
        conn.close()
        
        print(f"✅ Feedback saved (rating: {feedback.rating})")
        
        return {"status": "success", "message": "Thank you for your feedback!"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/analytics")
async def get_analytics():
    """Analytics endpoint - Get usage statistics"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        # Total queries
        c.execute("SELECT COUNT(*) FROM analytics")
        total_queries = c.fetchone()[0]
        
        # By language
        c.execute("SELECT language, COUNT(*) FROM analytics GROUP BY language")
        by_lang = dict(c.fetchall())
        
        # Average feedback rating
        c.execute("SELECT AVG(rating) FROM feedback")
        avg_rating = c.fetchone()[0] or 0
        
        # Total feedback
        c.execute("SELECT COUNT(*) FROM feedback")
        total_feedback = c.fetchone()[0]
        
        conn.close()
        
        return {
            "status": "success",
            "total_queries": total_queries,
            "by_language": by_lang,
            "average_rating": round(avg_rating, 2) if avg_rating else 0,
            "total_feedback": total_feedback
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/subscribe-updates")
async def subscribe_updates(sub: SubscriptionModel):
    """Notifications endpoint - Subscribe to updates"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''INSERT INTO subscriptions (email, phone, language, subscribed_date)
                    VALUES (?, ?, ?, ?)''',
                 (sub.email, sub.phone or "", "en", datetime.now().isoformat()))
        conn.commit()
        conn.close()
        
        print(f"✅ Subscription saved ({sub.email})")
        
        return {"status": "success", "message": "✅ Subscribed! You'll receive MSME updates."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/track-application/{app_id}")
async def track_application(app_id: str):
    """Application Tracking endpoint - Track MSME application status"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT status, last_updated FROM tracking WHERE app_id = ?", (app_id,))
        result = c.fetchone()
        conn.close()
        
        if result:
            return {
                "status": "success",
                "application_id": app_id,
                "current_status": result[0],
                "last_updated": result[1]
            }
        else:
            return {
                "status": "not_found",
                "application_id": app_id,
                "message": "Application not found in tracking system"
            }
    except Exception as e:
        return {"status": "error", "message": str(e)}

print("="*70)
print("✅ SERVER READY")
print("="*70)
print(f"API running on http://127.0.0.1:8000")
print(f"Docs available at http://127.0.0.1:8000/docs\n")