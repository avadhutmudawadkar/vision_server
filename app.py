import os
import math
import time
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
import requests as http

app = Flask(__name__)
CORS(app)

MONGODB_URI = os.environ.get("MONGODB_URI")
client = MongoClient(MONGODB_URI)
db = client["vision_db"]
memories = db["memories"]

FINETUNED_URL = os.environ.get("FINETUNED_URL")

STOPWORDS = set("the a an is are was were be been being to of and or but in on at for with as by from this that it its i you your my me he she they we".split())

def tokenize(text):
    words = re.findall(r"[a-zA-Z0-9']+", text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 1]

def get_all_memory_docs():
    return list(memories.find({}))

def score_and_retrieve(query, top_k=5):
    docs = get_all_memory_docs()
    if not docs:
        return []

    query_tokens = set(tokenize(query))
    if not query_tokens:
        return []

    total_docs = len(docs)
    doc_freq = {}
    doc_token_lists = []
    for doc in docs:
        tokens = tokenize(doc.get("text", ""))
        doc_token_lists.append(tokens)
        for t in set(tokens):
            doc_freq[t] = doc_freq.get(t, 0) + 1

    scored = []
    for doc, tokens in zip(docs, doc_token_lists):
        common = query_tokens & set(tokens)
        if not common:
            continue
        score = sum(math.log(total_docs / (1 + doc_freq.get(t, 0))) + 1 for t in common)
        scored.append((score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for score, doc in scored[:top_k]]

@app.route("/remember", methods=["POST"])
def remember():
    data = request.get_json()
    text = (data or {}).get("text", "").strip()
    if not text:
        return jsonify({"ok": False, "error": "No text provided"}), 400

    memories.insert_one({
        "text": text,
        "created_at": time.time(),
        "source": data.get("source", "manual"),
    })
    return jsonify({"ok": True})

@app.route("/memories", methods=["GET"])
def list_memories():
    docs = list(memories.find({}).sort("created_at", -1).limit(200))
    result = [{"id": str(d["_id"]), "text": d["text"], "created_at": d["created_at"], "source": d.get("source", "manual")} for d in docs]
    return jsonify({"memories": result})

@app.route("/memories/<mem_id>", methods=["DELETE"])
def delete_memory(mem_id):
    from bson.objectid import ObjectId
    try:
        memories.delete_one({"_id": ObjectId(mem_id)})
        return jsonify({"ok": True})
    except Exception:
        return jsonify({"ok": False}), 400

@app.route("/memories/clear", methods=["POST"])
def clear_memories():
    memories.delete_many({})
    return jsonify({"ok": True})

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    message = (data or {}).get("message", "").strip()
    groq_api_key = (data or {}).get("groq_api_key", "").strip()
    history = (data or {}).get("history", [])
    auto_remember = (data or {}).get("auto_remember", True)

    if not message:
        return jsonify({"error": "No message provided"}), 400
    if not groq_api_key:
        return jsonify({"error": "No Groq API key provided"}), 400

    relevant = score_and_retrieve(message, top_k=5)
    memory_context = ""
    if relevant:
        memory_context = "Relevant things you know about the user from past interactions:\n"
        memory_context += "\n".join(f"- {m['text']}" for m in relevant)
        memory_context += "\n\n"

    system_prompt = (
        "You are VISION, a helpful AI assistant capable of answering questions and writing "
        "complex, correct code. Use the following remembered context naturally if relevant, "
        "without explicitly saying 'according to my memory'.\n\n" + memory_context
    )

    messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": message}]

    try:
        res = http.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {groq_api_key}"},
            json={"model": "openai/gpt-oss-120b", "messages": messages, "temperature": 0.7, "max_tokens": 1500},
            timeout=60,
        )
        result = res.json()
        if "error" in result:
            return jsonify({"error": result["error"].get("message", "Groq API error")}), 400
        reply = result["choices"][0]["message"]["content"]
    except Exception as e:
        return jsonify({"error": f"Request to Groq failed: {str(e)}"}), 500

    if auto_remember:
        memories.insert_one({
            "text": f"User asked: {message} | VISION replied: {reply[:300]}",
            "created_at": time.time(),
            "source": "conversation",
        })

    return jsonify({"reply": reply, "used_memories": [m["text"] for m in relevant]})

@app.route("/chat-finetuned", methods=["POST"])
def chat_finetuned():
    data = request.get_json()
    message = (data or {}).get("message", "").strip()

    if not message:
        return jsonify({"error": "No message provided"}), 400
    if not FINETUNED_URL:
        return jsonify({"error": "FINETUNED_URL not set on the server"}), 500

    try:
        res = http.post(
            f"{FINETUNED_URL}/chat",
            json={"message": message, "auto_remember": True},
            timeout=120,
        )
        result = res.json()
    except Exception as e:
        return jsonify({"error": f"Request to fine-tuned model failed: {str(e)}"}), 500

    return jsonify(result)

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "VISION server running"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
