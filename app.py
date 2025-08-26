import base64
import json
import os
import re
import sqlite3
import sys
import time
import requests
import uuid
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask, render_template, request, jsonify, session
from datetime import datetime
import threading

app = Flask(__name__)
app.secret_key = 'secret_key'  # Change this to a random secret key

API_KEYS = [
    "api-key1",
    "api-key2",
    "api-key3",
    "api-key4"
]
ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

RELEVANCE_THRESHOLD = 0.25
PDF_PATHS = ['legal.pdf', 
             '1408278839.pdf', 
             '120240_99Z_Zariski_2014-Legal_Literacy.pdf']
FILE_PATHS = ["content/" + name for name in PDF_PATHS]
DB_PATH = "chatbot_interactions.db"
MAX_CONTEXT_ENTRIES = 5

# Global variables to match base code exactly
model = None
docs = []
processed = []

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            question TEXT,
            answer TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    return conn

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def save_interaction(conn, session_id, question, answer):
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO interactions (session_id, question, answer) VALUES (?, ?, ?)",
        (session_id, question, answer)
    )
    conn.commit()

def get_sessions_ordered(conn):
    cur = conn.cursor()
    cur.execute(
        "SELECT session_id, MAX(timestamp) as last_time FROM interactions GROUP BY session_id ORDER BY last_time DESC"
    )
    return [row[0] for row in cur.fetchall()]

def get_last_n_from_session(conn, session_id, n):
    cur = conn.cursor()
    cur.execute(
        "SELECT question, answer FROM interactions WHERE session_id=? ORDER BY timestamp DESC LIMIT ?",
        (session_id, n)
    )
    return list(reversed(cur.fetchall()))

def get_last_n_cascading(conn, total_n=MAX_CONTEXT_ENTRIES):
    sessions = get_sessions_ordered(conn)
    collected = []
    for session_id in sessions:
        needed = total_n - len(collected)
        if needed <= 0:
            break
        qas = get_last_n_from_session(conn, session_id, needed)
        collected = qas + collected
    return collected[-total_n:]

def update_context_memory(conn, session_id, new_qas, total_n=MAX_CONTEXT_ENTRIES):
    if len(new_qas) >= total_n:
        return new_qas[-total_n:]
    older_qas = []
    sessions = get_sessions_ordered(conn)
    sessions = [s for s in sessions if s != session_id]

    for s in sessions:
        needed = total_n - len(new_qas) - len(older_qas)
        if needed <= 0:
            break
        older_part = get_last_n_from_session(conn, s, needed)
        older_qas = older_part + older_qas

    combined = older_qas + new_qas
    return combined[-total_n:]

def encode_file_to_base64(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def generate_payload(file_paths):
    parts = [{"text": "there are multiple files, provide a summary for each file with its filename."}]
    for fp in file_paths:
        ext = os.path.splitext(fp)[1].lower()
        if ext == ".pdf":
            data = encode_file_to_base64(fp)
            parts.append({"inline_data": {"mime_type": "application/pdf", "data": data}})
        elif ext in [".jpg", ".jpeg", ".png"]:
            data = encode_file_to_base64(fp)
            parts.append({"inline_data": {"mime_type": f"image/{ext.lstrip('.')}", "data": data}})
        else:
            print(f"Skipping unsupported: {fp}")
    return {"contents": [{"parts": parts}]}

def post_with_key_rotation_and_retry(payload, max_keys_retries=2):
    for key in API_KEYS:
        for attempt in range(max_keys_retries):
            print(f"Using key {key[:8]}..., attempt {attempt+1}")
            resp = requests.post(ENDPOINT,
                                 headers={"Content-Type": "application/json", "X-goog-api-key": key},
                                 json=payload)
            if resp.status_code == 200:
                return resp
            elif resp.status_code in (503, 429):
                wait = 2 ** attempt
                print(f"Error {resp.status_code}. Retrying in {wait}s.")
                time.sleep(wait)
                continue
            else:
                print(f"Non-retryable error {resp.status_code}: {resp.text}")
                break
    print("All keys exhausted or failed.")
    return None

def parse_summaries(resp_json):
    text = resp_json['candidates'][0]['content']['parts'][0]['text']
    entries, current = [], None
    for line in text.splitlines():
        m = re.match(r"\*\*\d+\.\s+(.+?)\*\*", line.strip())
        if m:
            if current: entries.append(current)
            current = {"fileName": m.group(1), "text": ""}
        elif current:
            current["text"] += line.strip() + " "
    if current: entries.append(current)
    return entries

def filter_relevant(docs, q):
    if not docs:
        return []
    texts = [doc["summary"] for doc in docs]
    vec = TfidfVectorizer().fit(texts + [q])
    sim = cosine_similarity(vec.transform([q]), vec.transform(texts))[0]
    return [doc for doc, score in zip(docs, sim) if score >= RELEVANCE_THRESHOLD]

def initialize_system():
    global model, docs, processed
    
    print("Initializing system...")
    
    # Generate payload and get summaries (exactly like base code)
    payload = generate_payload(FILE_PATHS)
    resp = post_with_key_rotation_and_retry(payload)
    
    if not resp or resp.status_code != 200:
        print("Failed to initialize system")
        return False
        
    resp_json = resp.json()
    
    if "candidates" not in resp_json:
        print("No candidates in response:", json.dumps(resp_json, indent=2))
        return False
    
    # Load model and parse summaries (exactly like base code)
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    docs = parse_summaries(resp_json)
    
    # Process documents with embeddings (exactly like base code)
    query = "legal reasoning and terms"
    processed = []
    for d in docs:
        doc_emb = model.encode([d["text"]])
        query_emb = model.encode([query])
        sim = cosine_similarity(doc_emb, query_emb)[0][0]
        summary = d["text"] if sim >= 0.5 else d["text"][:150] + "..."
        processed.append({
            "fileName": d["fileName"],
            "summary": summary,
            "similarity": sim
        })
    
    processed.sort(key=lambda x: x["similarity"], reverse=True)
    
    for d in processed:
        print(f"{d['fileName']}: {d['summary']}")
    
    print("System initialized successfully!")
    return True

def ask(conn, session_id, q, follow_up=False):
    global processed
    
    # EXACTLY like base code: filter relevant documents for each question
    relevant = filter_relevant(processed, q)
    
    # EXACTLY like base code: build PDF parts from relevant documents  
    pdf_parts = []
    for path in PDF_PATHS:
        if path in [d["fileName"] for d in relevant]:
            pdf_parts.append({
                "inline_data": {
                    "mime_type": "application/pdf",
                    "data": encode_file_to_base64("content/" + path)
                }
            })
    
    # Get current session Q&As and update context memory (exactly like base code)
    cur = conn.cursor()
    cur.execute("SELECT question, answer FROM interactions WHERE session_id=? ORDER BY timestamp ASC", (session_id,))
    current_session_qas = cur.fetchall()
    rolling_qas = update_context_memory(conn, session_id, list(current_session_qas), MAX_CONTEXT_ENTRIES)
    
    # Build context exactly like base code
    context = "\n\n".join(f"Q: {q}\nA: {a}" for q, a in rolling_qas)
    context += "\n\n" + "\n\n".join(f"Document: {d['fileName']}\nSummary: {d['summary']}" for d in relevant)
    
    # Create payload exactly like base code (note: variable name 'jira' matches base code)
    jira = {"contents": [{"parts": [{"text": context + "\n\nAnswer using context:\n" + q}] + pdf_parts}]}
    r2 = post_with_key_rotation_and_retry(jira)

    if r2 and r2.status_code == 200:
        answer = r2.json()['candidates'][0]['content']['parts'][0]['text']
        print(f"--- Answer ---\n{answer}\n")
        save_interaction(conn, session_id, q, answer)
        
        # EXACTLY like base code: update context after saving interaction
        cur.execute("SELECT question, answer FROM interactions WHERE session_id=? ORDER BY timestamp ASC", (session_id,))
        current_session_qas = cur.fetchall()
        rolling_qas = update_context_memory(conn, session_id, list(current_session_qas), MAX_CONTEXT_ENTRIES)
        context_base = "\n\n".join(f"Q: {q}\nA: {a}" for q, a in rolling_qas)
        context = context_base + "\n\n" + "\n\n".join(f"Document: {d['fileName']}\nSummary: {d['summary']}" for d in relevant)
        
        return answer
    else:
        error_msg = f"Failed to get answer (status {r2.status_code if r2 else 'No response'}): {r2.text if r2 else 'No response'}"
        print(error_msg)
        return None

@app.route('/')
def index():
    # Initialize session ID if not exists
    if 'session_id' not in session:
        session['session_id'] = "session_" + str(uuid.uuid4())[:8]
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask_question():
    conn = None
    try:
        conn = get_db_connection()
        
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({'error': 'Please provide a question'}), 400
        
        session_id = session.get('session_id')
        if not session_id:
            session['session_id'] = "session_" + str(uuid.uuid4())[:8]
            session_id = session['session_id']
        
        # Use the exact same ask function as base code
        answer = ask(conn, session_id, question)
        
        if answer is None:
            return jsonify({'error': 'Sorry, I couldn\'t process your question at the moment. Please try again.'}), 500
        
        return jsonify({
            'answer': answer,
            'session_id': session_id
        })
        
    except Exception as e:
        print(f"Error processing question: {e}")
        return jsonify({'error': 'An error occurred while processing your question'}), 500
    finally:
        if conn:
            conn.close()

@app.route('/history')
def get_history():
    conn = None
    try:
        session_id = session.get('session_id')
        if not session_id:
            return jsonify({'history': []})
        
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "SELECT question, answer, timestamp FROM interactions WHERE session_id=? ORDER BY timestamp ASC",
            (session_id,)
        )
        history = [{'question': row[0], 'answer': row[1], 'timestamp': row[2]} for row in cur.fetchall()]
        
        return jsonify({'history': history})
    except Exception as e:
        print(f"Error getting history: {e}")
        return jsonify({'history': []})
    finally:
        if conn:
            conn.close()

@app.route('/new_session', methods=['POST'])
def new_session():
    session['session_id'] = "session_" + str(uuid.uuid4())[:8]
    return jsonify({'session_id': session['session_id']})

if __name__ == '__main__':
    # EXACTLY like base code: initialize database and get previous context first
    conn = init_db()
    previous_context_qas = get_last_n_cascading(conn, MAX_CONTEXT_ENTRIES)
    conn.close()
    
    # Initialize system exactly like base code
    if initialize_system():
        print("Starting Flask server...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to initialize system. Exiting.")
        sys.exit(1)