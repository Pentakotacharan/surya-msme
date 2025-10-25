import os
from groq import Groq
from utils.pdf_loader import load_pdfs_from_folder
from utils.vector_store import create_vector_store
from dotenv import load_dotenv
load_dotenv()

# ======================================================
# 1️⃣ Load environment key
# ======================================================
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError("❌ GROQ_API_KEY not found in environment variables. Please add it to your .env file.")

# Initialize Groq client
client = Groq(api_key=groq_api_key)

# ======================================================
# 2️⃣ Load and index PDFs (from data/guidelines, data/policies, data/ramp)
# ======================================================
pdf_folder = "../data"  # parent folder
print(f"📂 Loading PDFs from: {pdf_folder}")

docs = load_pdfs_from_folder(pdf_folder)
if not docs:
    raise ValueError("❌ No PDFs found in the data folder or its subfolders.")

print(f"✅ Loaded {len(docs)} documents from PDF folders.")

vector_store = create_vector_store(docs)
print("✅ Vector store created successfully.")

# ======================================================
# 3️⃣ Chat function
# ======================================================
def chat_response(user_query):
    """
    Generate an intelligent response based on user query and loaded PDFs.
    """
    try:
        # Step 1: Retrieve relevant docs
        relevant_docs = vector_store.similarity_search(user_query, k=3)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        # Step 2: Combine context + user query for LLM
        prompt = f"""
        You are an AI assistant for MSME Knowledge Retrieval.
        Answer the following query based on the context from government policies, guidelines, or ramp documents.

        Context:
        {context}

        Question:
        {user_query}

        Answer clearly and concisely.
        """

        # Step 3: Generate answer via Groq API (mixtral model recommended)
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are an MSME chatbot helping users with official policy and scheme information."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=512
        )

        answer = completion.choices[0].message["content"]
        return answer.strip()

    except Exception as e:
        print(f"⚠️ Error in chat_response: {e}")
        return "⚠️ Sorry, I ran into an issue while processing your question."

# ======================================================
# 4️⃣ Debug test (optional)
# ======================================================
if __name__ == "__main__":
    while True:
        query = input("🧠 Ask MSME Chatbot: ")
        if query.lower() in ["exit", "quit"]:
            print("👋 Exiting chatbot.")
            break
        print("🤖", chat_response(query))
