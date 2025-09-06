from flask import Flask, render_template, jsonify, request
import pandas as pd
from pathlib import Path
import sqlite3, os, re, json
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Optional OpenAI use
import os
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
try:
    import openai
    if OPENAI_API_KEY:
        openai.api_key = OPENAI_API_KEY
except Exception:
    openai = None

DB_PATH = "emails.db"
# Path to the CSV dataset
CSV_PATH = r"C:\Users\HP\Downloads\emails.csv"

  # provided in the environment

app = Flask(__name__)

### Simple helper heuristics
PRIORITY_KEYWORDS = ["immediately", "urgent", "asap", "cannot access", "can't access", "critical", "down", "unable to", "payment failed", "payment issue"]
NEGATIVE_WORDS = ["not working","error","fail","frustrat","angry","disappoint","problem","can't","cannot","unable","issue"]

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS emails
                 (id INTEGER PRIMARY KEY, sender TEXT, subject TEXT, body TEXT, received_at TEXT, sentiment TEXT, priority TEXT, extracted TEXT, reply TEXT, resolved INTEGER DEFAULT 0)''')
    conn.commit()
    conn.close()

def simple_sentiment(text):
    t = text.lower()
    neg = sum(1 for w in NEGATIVE_WORDS if w in t)
    if neg>1:
        return "Negative"
    if neg==1:
        return "Neutral"
    return "Positive"

def detect_priority(text):
    t = text.lower()
    for k in PRIORITY_KEYWORDS:
        if k in t:
            return "Urgent"
    return "Not urgent"

def extract_contacts(text):
    emails = re.findall(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)
    phones = re.findall(r"\+?\d[\d\-\s]{7,}\d", text)
    return {"emails": list(set(emails)), "phones": list(set(phones))}

### Simple RAG: load local KB, compute TF-IDF
KB_PATH = Path("sample_kb.md")
if KB_PATH.exists():
    kb_docs = [KB_PATH.read_text()]
else:
    kb_docs = ["Our product FAQ: billing, login, subscriptions, refunds. Contact support at support@example.com"]

vectorizer = TfidfVectorizer().fit(kb_docs)
kb_tfidf = vectorizer.transform(kb_docs)

def kb_retrieve(query, top_k=1):
    q_tfidf = vectorizer.transform([query])
    sims = linear_kernel(q_tfidf, kb_tfidf).flatten()
    idxs = sims.argsort()[::-1][:top_k]
    return [kb_docs[i] for i in idxs if sims[i]>0]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/import_csv", methods=["POST"])
def import_csv():
    init_db()
    if not Path(CSV_PATH).exists():
        return jsonify({"error":"CSV not found at: "+CSV_PATH}), 400
    df = pd.read_csv(CSV_PATH)
    # Normalize column names to lowercase
    df.columns = [c.strip().lower() for c in df.columns]
    # filter subjects with keywords
    if 'subject' not in df.columns:
        return jsonify({"error":"CSV missing 'subject' column"}),400
    mask = df['subject'].fillna("").str.contains("support|query|request|help", case=False, regex=True)
    df = df[mask].copy()
    conn = sqlite3.connect(DB_PATH)
    for _, row in df.iterrows():
        sender = str(row.get('sender',''))
        subject = str(row.get('subject',''))
        body = str(row.get('body','') or row.get('message','') )
        received_at = str(row.get('date', datetime.utcnow().isoformat()))
        sentiment = simple_sentiment(subject+" "+body)
        priority = detect_priority(subject+" "+body)
        extracted = json.dumps(extract_contacts(subject+" "+body))
        c = conn.cursor()
        c.execute("INSERT INTO emails (sender,subject,body,received_at,sentiment,priority,extracted) VALUES (?,?,?,?,?,?,?)",
                  (sender,subject,body,received_at,sentiment,priority,extracted))
    conn.commit()
    conn.close()
    return jsonify({"imported": int(df.shape[0])})

@app.route("/emails", methods=["GET"])
def get_emails():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    rows = c.execute("SELECT * FROM emails ORDER BY CASE WHEN priority='Urgent' THEN 0 ELSE 1 END, received_at DESC").fetchall()
    emails = [dict(ix) for ix in rows]
    conn.close()
    return jsonify(emails)

@app.route("/generate_reply", methods=["POST"])
def generate_reply():
    data = request.json or {}
    email_id = data.get("id")
    if not email_id:
        return jsonify({"error":"id required"}),400
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    row = c.execute("SELECT * FROM emails WHERE id=?",(email_id,)).fetchone()
    if not row:
        return jsonify({"error":"email not found"}),404
    sender, subject, body = row[1], row[2], row[3]
    # retrieve KB
    kb = kb_retrieve(subject+" "+body, top_k=1)
    kb_text = kb[0] if kb else ""
    # If OpenAI available, generate using it
    if openai and OPENAI_API_KEY:
        prompt = ("You are an empathetic, professional customer support assistant.\n"
                  "Read the customer email below and draft a concise, friendly, professional reply.\n"
                  "Include acknowledgement of frustration if present, reference product/issues in the email,\n"
                  "and propose next actionable steps. Keep under 200 words.\n"
                  "Knowledge base:\n" + kb_text + "\n\nCustomer email:\nFrom: " + sender + "\nSubject: " + subject + "\n\n" + body + "\n\nReply:")
        try:
            resp = openai.Completion.create(model="text-davinci-003", prompt=prompt, max_tokens=250, temperature=0.2)
            reply = resp.choices[0].text.strip()
        except Exception as e:
            reply = f"Auto-reply generation failed ({e}). Fallback: Thanks for contacting us. We received your request about '{subject}'. We'll look into it and get back to you shortly."
    else:
        # Fallback template
        sentiment = simple_sentiment(subject+" "+body)
        reply = f"Hello,\n\nThanks for reaching out about \"{subject}\". "
        if sentiment=="Negative":
            reply += "I'm sorry you're experiencing this — I understand how frustrating that must be. "
        reply += "We have reviewed your message and will investigate immediately. Could you please provide any additional details (screenshots, steps to reproduce, or order id) if available? \n\nBest regards,\nSupport Team"
    # save reply
    c.execute("UPDATE emails SET reply=? WHERE id=?",(reply,email_id))
    conn.commit()
    conn.close()
    return jsonify({"reply":reply})

@app.route("/update_reply", methods=["POST"])
def update_reply():
    data = request.json or {}
    email_id = data.get("id"); reply = data.get("reply","")
    conn = sqlite3.connect(DB_PATH); c = conn.cursor()
    c.execute("UPDATE emails SET reply=? WHERE id=?",(reply,email_id))
    conn.commit(); conn.close()
    return jsonify({"ok":True})

@app.route("/mark_resolved", methods=["POST"])
def mark_resolved():
    data = request.json or {}
    email_id = data.get("id")
    conn = sqlite3.connect(DB_PATH); c = conn.cursor()
    c.execute("UPDATE emails SET resolved=1 WHERE id=?",(email_id,))
    conn.commit(); conn.close()
    return jsonify({"ok":True})

@app.route("/stats", methods=["GET"])
def stats():
    conn = sqlite3.connect(DB_PATH); c = conn.cursor()
    total = c.execute("SELECT count(*) FROM emails").fetchone()[0]
    resolved = c.execute("SELECT count(*) FROM emails WHERE resolved=1").fetchone()[0]
    pending = total - resolved
    by_sentiment = dict((row[0], row[1]) for row in c.execute("SELECT sentiment,count(*) FROM emails GROUP BY sentiment"))
    by_priority = dict((row[0], row[1]) for row in c.execute("SELECT priority,count(*) FROM emails GROUP BY priority"))
    conn.close()
    return jsonify({"total":total,"resolved":resolved,"pending":pending,"by_sentiment":by_sentiment,"by_priority":by_priority})

if __name__ == "__main__":
    init_db()
    app.run(debug=True)