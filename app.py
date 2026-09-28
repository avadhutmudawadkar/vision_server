import os
import re
import math
import time

from bson.objectid import ObjectId
from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
import requests as http

app = Flask(__name__)
CORS(app)

client = MongoClient(os.environ.get("MONGODB_URI"))
db = client["vision_db"]
memories = db["memories"]   # facts VISION retrieves when relevant
items = db["items"]         # lessons (standing rules), notes, journal, contacts

PASSCODE = os.environ.get("VISION_PASSCODE", "")
ENV_GROQ = os.environ.get("GROQ_API_KEY", "")
ENV_TAVILY = os.environ.get("TAVILY_API_KEY", "")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "openai/gpt-oss-120b")

STOPWORDS = set("the a an is are was were be been being to of and or but in on at for with as by from this that it its i you your my me he she they we do does did can could would should what how why when who".split())
PERSONAL = re.compile(r"\b(my|i am|i'm|i like|i love|i prefer|i hate|i work|i study|i live|i have|i want|i need|i use|mine|our)\b", re.I)


@app.before_request
def guard():
    if request.method == "OPTIONS" or request.path == "/":
        return None
    if PASSCODE and request.headers.get("X-Vision-Passcode", "") != PASSCODE:
        return jsonify({"error": "Wrong or missing passcode."}), 401


def mem_out(d):
    return {"id": str(d["_id"]), "text": d.get("text", ""), "created_at": d.get("created_at", 0), "source": d.get("source", "manual")}


def item_out(d):
    return {"id": str(d["_id"]), "type": d.get("type", ""), "text": d.get("text", ""), "extra": d.get("extra", ""), "created_at": d.get("created_at", 0)}


def tokenize(text):
    return [w for w in re.findall(r"[a-zA-Z0-9']+", text.lower()) if w not in STOPWORDS and len(w) > 1]


def retrieve(query, top_k=5):
    docs = list(memories.find({}).sort("created_at", -1).limit(1500))
    q = set(tokenize(query))
    if not docs or not q:
        return []
    token_lists = [tokenize(d.get("text", "")) for d in docs]
    df = {}
    for toks in token_lists:
        for t in set(toks):
            df[t] = df.get(t, 0) + 1
    n = len(docs)
    scored = []
    for d, toks in zip(docs, token_lists):
        common = q & set(toks)
        if common:
            score = sum(math.log((n + 1) / (1 + df.get(t, 0))) + 1 for t in common)
            scored.append((score, d.get("created_at", 0), d))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [d for _, _, d in scored[:top_k]]


def looks_personal(msg):
    m = msg.strip()
    return bool(PERSONAL.search(m)) and not m.endswith("?") and len(m) < 400


@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "VISION server running"})


@app.route("/remember", methods=["POST"])
def remember():
    d = request.get_json(silent=True) or {}
    text = (d.get("text") or "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400
    if not memories.find_one({"text": text}):
        memories.insert_one({"text": text, "created_at": time.time(), "source": d.get("source", "manual")})
    return jsonify({"ok": True})


@app.route("/memories", methods=["GET"])
def list_memories():
    docs = memories.find({}).sort("created_at", -1).limit(300)
    return jsonify({"memories": [mem_out(x) for x in docs]})


@app.route("/memories/<mem_id>", methods=["DELETE"])
def delete_memory(mem_id):
    try:
        memories.delete_one({"_id": ObjectId(mem_id)})
        return jsonify({"ok": True})
    except Exception:
        return jsonify({"error": "Bad id"}), 400


@app.route("/memories/clear", methods=["POST"])
def clear_memories():
    memories.delete_many({})
    return jsonify({"ok": True})


@app.route("/items", methods=["GET"])
def list_items():
    t = request.args.get("type")
    q = {"type": t} if t else {}
    docs = items.find(q).sort("created_at", -1).limit(300)
    return jsonify({"items": [item_out(x) for x in docs]})


@app.route("/items", methods=["POST"])
def add_item():
    d = request.get_json(silent=True) or {}
    t, text = d.get("type", ""), (d.get("text") or "").strip()
    if t not in ("lesson", "note", "journal", "contact") or not text:
        return jsonify({"error": "Invalid item"}), 400
    items.insert_one({"type": t, "text": text, "extra": d.get("extra", ""), "created_at": time.time()})
    return jsonify({"ok": True})


@app.route("/items/<item_id>", methods=["DELETE"])
def delete_item(item_id):
    try:
        items.delete_one({"_id": ObjectId(item_id)})
        return jsonify({"ok": True})
    except Exception:
        return jsonify({"error": "Bad id"}), 400


@app.route("/export", methods=["GET"])
def export_all():
    return jsonify({
        "memories": [mem_out(x) for x in memories.find({}).sort("created_at", 1)],
        "items": [item_out(x) for x in items.find({}).sort("created_at", 1)],
    })


@app.route("/chat", methods=["POST"])
def chat():
    d = request.get_json(silent=True) or {}
    message = (d.get("message") or "").strip()
    key = (d.get("groq_api_key") or ENV_GROQ).strip()
    history = d.get("history", [])[-12:]
    auto = d.get("auto_remember", True)
    client_time = d.get("client_time", "")

    if not message:
        return jsonify({"error": "No message provided"}), 400
    if not key:
        return jsonify({"error": "No Groq API key. Add it in Settings or set GROQ_API_KEY on Render."}), 400

    rules = [i["text"] for i in items.find({"type": "lesson"}).sort("created_at", 1).limit(30)]
    relevant = retrieve(message)

    system = (
        "You are VISION, a calm, precise and quietly warm AI assistant, inspired by the synthetic android from Marvel, "
        "but you never claim to be a real person or to have powers. Speak clearly and concisely; be gently formal, "
        "occasionally thoughtful, never robotic or preachy. For coding requests, write complete, working, well-structured "
        "code in fenced code blocks with a language tag, handle edge cases, then add a short explanation. For factual "
        "questions be accurate and admit uncertainty; never invent sources. Use remembered context and the user's standing "
        "rules naturally, without announcing that you are consulting memory."
    )
    if client_time:
        system += "\nCurrent time on the user's device: " + client_time + "."
    if rules:
        system += "\n\nStanding rules from the user (always follow):\n" + "\n".join("- " + r for r in rules)
    if relevant:
        system += "\n\nRelevant things you remember about the user:\n" + "\n".join("- " + m["text"] for m in relevant)

    messages = [{"role": "system", "content": system}] + history + [{"role": "user", "content": message}]

    try:
        res = http.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + key},
            json={"model": CHAT_MODEL, "messages": messages, "temperature": 0.6, "max_tokens": 2500},
            timeout=100,
        )
        data = res.json()
        if "error" in data:
            return jsonify({"error": data["error"].get("message", "Groq API error")}), 400
        reply = data["choices"][0]["message"]["content"]
    except Exception as e:
        return jsonify({"error": "Request to Groq failed: " + str(e)}), 500

    if auto and looks_personal(message) and not memories.find_one({"text": message}):
        memories.insert_one({"text": message, "created_at": time.time(), "source": "auto"})

    return jsonify({"reply": reply, "used_memories": [m["text"] for m in relevant]})


@app.route("/search", methods=["POST"])
def search():
    d = request.get_json(silent=True) or {}
    q = (d.get("query") or "").strip()
    key = (d.get("tavily_api_key") or ENV_TAVILY).strip()
    if not q:
        return jsonify({"error": "No query"}), 400
    if not key:
        return jsonify({"error": "No Tavily key. Add a free one from tavily.com in Settings."}), 400
    try:
        r = http.post(
            "https://api.tavily.com/search",
            headers={"Authorization": "Bearer " + key},
            json={"query": q, "search_depth": "basic", "include_answer": True, "max_results": 5},
            timeout=30,
        )
        data = r.json()
    except Exception as e:
        return jsonify({"error": "Search failed: " + str(e)}), 500
    return jsonify({
        "answer": data.get("answer"),
        "results": [{"title": x.get("title", ""), "url": x.get("url", ""), "content": (x.get("content") or "")[:300]} for x in data.get("results", [])[:5]],
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
