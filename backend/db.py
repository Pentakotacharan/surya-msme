import sqlite3
from datetime import datetime

DB_FILE = "chat_history.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_msg TEXT,
            bot_msg TEXT,
            lang TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_chat(user_msg, bot_msg, lang):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO chats (user_msg, bot_msg, lang, timestamp) VALUES (?, ?, ?, ?)",
              (user_msg, bot_msg, lang, datetime.now().isoformat()))
    conn.commit()
    conn.close()

def fetch_chats():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT * FROM chats ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()
    return rows
