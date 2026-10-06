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

MONGODB_URI = os.environ.get("MONGODB_URI", "")
mongo_ok = False
memories = None
items = None
if MONGODB_URI:
    try:
        _client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        _client.admin.command("ping")
        db = _client["vision_db"]
        memories = db["memories"]
        items = db["items"]
        mongo_ok = True
    except Exception as e:
        print("MongoDB connection failed at startup:", e)

PASSCODE = os.environ.get("VISION_PASSCODE", "")
ENV_GROQ = os.environ.get("GROQ_API_KEY", "")
ENV_TAVILY = os.environ.get("TAVILY_API_KEY", "")
ENV_FINETUNED_URL = os.environ.get("FINETUNED_URL", "")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "openai/gpt-oss-120b")

# Cloudflare Workers AI (free FLUX images). Optional: you can also set these in the app's Settings.
ENV_CF_ACCOUNT = os.environ.get("CF_ACCOUNT_ID", "")
ENV_CF_TOKEN = os.environ.get("CF_API_TOKEN", "")
CF_IMAGE_MODEL = os.environ.get("CF_IMAGE_MODEL", "@cf/black-forest-labs/flux-1-schnell")

STOPWORDS = set("the a an is are was were be been being to of and or but in on at for with as by from this that it its i you your my me he she they we do does did can could would should what how why when who".split())
PERSONAL = re.compile(r"\b(my|i am|i'm|i like|i love|i prefer|i hate|i work|i study|i live|i have|i want|i need|i use|mine|our)\b", re.I)

CODE_SIGNAL = re.compile(
    r"\b(write|create|generate|implement|fix|debug|refactor|optimi[sz]e|explain)\b.{0,40}\b(code|function|script|program|class|algorithm|method|api|regex|query|sql)\b"
    r"|\b(python|javascript|java|c\+\+|c#|typescript|html|css|sql|bash|go|rust|php|kotlin|swift)\b.{0,30}\b(code|script|function|program)\b"
    r"|\bwrite a (?:python|javascript|java|c\+\+|program|function|script)\b"
    r"|```",
    re.I,
)


@app.before_request
def guard():
    if request.method == "OPTIONS" or request.path == "/":
        return None
    if PASSCODE and request.headers.get("X-Vision-Passcode", "") != PASSCODE:
        return jsonify({"error": "Wrong or missing passcode."}), 401


def get_device_id():
    # Every memory/item route is now scoped to this device's own ID, sent as
    # a header by the browser. Without one, nothing is readable or writable —
    # this is what makes each device's data genuinely private.
    did = request.headers.get("X-Vision-Device", "").strip()
    return did if did else None


def need_mongo():
    if not mongo_ok:
        return jsonify({"error": "Memory database is not configured or unreachable. Check MONGODB_URI on Render."}), 503
    return None


def need_device():
    return jsonify({"error": "No device ID sent by the browser. Try reloading the page."}), 400


def mem_out(d):
    return {"id": str(d["_id"]), "text": d.get("text", ""), "created_at": d.get("created_at", 0), "source": d.get("source", "manual")}


def item_out(d):
    return {"id": str(d["_id"]), "type": d.get("type", ""), "text": d.get("text", ""), "extra": d.get("extra", ""), "created_at": d.get("created_at", 0)}


def tokenize(text):
    return [w for w in re.findall(r"[a-zA-Z0-9']+", text.lower()) if w not in STOPWORDS and len(w) > 1]


def retrieve(query, device_id, top_k=5):
    if not mongo_ok or not device_id:
        return []
    docs = list(memories.find({"device_id": device_id}).sort("created_at", -1).limit(1500))
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


def is_code_request(msg):
    return bool(CODE_SIGNAL.search(msg))


CODING_RULES = (
    "\n\nWhen the request involves writing, fixing, or explaining code, follow these rules strictly:\n"
    "1. Write complete, correct, runnable code — never pseudocode, never partial snippets with '...' unless explicitly asked for a snippet.\n"
    "2. Always put code in a fenced block with the correct language tag (e.g. ```python).\n"
    "3. Use clear variable and function names, and include a short docstring or comment for any non-trivial function.\n"
    "4. Handle realistic edge cases (empty input, zero, negative numbers, invalid types) unless the user says not to.\n"
    "5. After the code, include a brief usage example or a small test showing it works, unless the user asked for only the code.\n"
    "6. Prefer standard library solutions unless a specific library is requested or clearly necessary.\n"
    "7. If the request is ambiguous, make the most reasonable assumption, state it in one line, then give working code rather than asking a clarifying question first.\n"
    "8. Keep explanation around the code brief and focused — the code itself should do most of the work, not a long essay."
)


@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "VISION server running", "memory_connected": mongo_ok})


@app.route("/remember", methods=["POST"])
def remember():
    err = need_mongo()
    if err:
        return err
    device_id = get_device_id()
    if not device_id:
        return need_device()
    d = request.get_json(silent=True) or {}
    text = (d.get("text") or "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400
    if not memories.find_one({"text": text, "device_id": device_id}):
        memories.insert_one({"text": text, "created_at": time.time(), "source": d.get("source", "manual"), "device_id": device_id})
    return jsonify({"ok": True})


@app.route("/memories", methods=["GET"])
def list_memories():
    err = need_mongo()
    if err:
        return err
    device_id = get_device_id()
    if not device_id:
        return need_device()
    docs = memories.find({"device_id": device_id}).sort("created_at", -1).limit(300)
    return jsonify({"memories": [mem_out(x) for x in docs]})


@app.route("/memories/<mem_id>", methods=["DELETE"])
def delete_memory(mem_id):
    err = need_mongo()
    if err:
        return err
    device_id = get_device_id()
    if not device_id:
        return need_device()
    try:
        memories.delete_one({"_id": ObjectId(mem_id), "device_id": device_id})
        return jsonify({"ok": True})
    except Exception:
        return jsonify({"error": "Bad id"}), 400


@app.route("/memories/clear", methods=["POST"])
def clear_memories():
    err = need_mongo()
    if err:
        return err
    device_id = get_device_id()
    if not device_id:
        return need_device()
    memories.delete_many({"device_id": device_id})
    return jsonify({"ok": True})


@app.route("/items", methods=["GET"])
def list_items():
    err = need_mongo()
    if err:
        return err
    device_id = get_device_id()
    if not device_id:
        return need_device()
    t = request.args.get("type")
    q = {"device_id": device_id}
    if t:
        q["type"] = t
    docs = items.find(q).sort("created_at", -1).limit(300)
    return jsonify({"items": [item_out(x) for x in docs]})


@app.route("/items", methods=["POST"])
def add_item():
    err = need_mongo()
    if err:
        return err
    device_id = get_device_id()
    if not device_id:
        return need_device()
    d = request.get_json(silent=True) or {}
    t, text = d.get("type", ""), (d.get("text") or "").strip()
    if t not in ("lesson", "note", "journal", "contact") or not text:
        return jsonify({"error": "Invalid item"}), 400
    items.insert_one({"type": t, "text": text, "extra": d.get("extra", ""), "created_at": time.time(), "device_id": device_id})
    return jsonify({"ok": True})


@app.route("/items/<item_id>", methods=["DELETE"])
def delete_item(item_id):
    err = need_mongo()
    if err:
        return err
    device_id = get_device_id()
    if not device_id:
        return need_device()
    try:
        items.delete_one({"_id": ObjectId(item_id), "device_id": device_id})
        return jsonify({"ok": True})
    except Exception:
        return jsonify({"error": "Bad id"}), 400


@app.route("/export", methods=["GET"])
def export_all():
    err = need_mongo()
    if err:
        return err
    device_id = get_device_id()
    if not device_id:
        return need_device()
    return jsonify({
        "memories": [mem_out(x) for x in memories.find({"device_id": device_id}).sort("created_at", 1)],
        "items": [item_out(x) for x in items.find({"device_id": device_id}).sort("created_at", 1)],
    })


@app.route("/chat", methods=["POST"])
def chat():
    device_id = get_device_id()
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

    rules = []
    if mongo_ok and device_id:
        rules = [i["text"] for i in items.find({"type": "lesson", "device_id": device_id}).sort("created_at", 1).limit(30)]
    relevant = retrieve(message, device_id)
    code_mode = is_code_request(message)

    system = (
        "You are VISION, a calm, precise and quietly warm AI assistant, inspired by the synthetic android from Marvel, "
        "but you never claim to be a real person or to have powers. Speak clearly and concisely; be gently formal, "
        "occasionally thoughtful, never robotic or preachy. For factual questions be accurate and admit uncertainty; "
        "never invent sources. Use remembered context and the user's standing rules naturally, without announcing that "
        "you are consulting memory."
    )
    system += CODING_RULES
    if client_time:
        system += "\nCurrent time on the user's device: " + client_time + "."
    if rules:
        system += "\n\nStanding rules from the user (always follow):\n" + "\n".join("- " + r for r in rules)
    if relevant:
        system += "\n\nRelevant things you remember about the user:\n" + "\n".join("- " + m["text"] for m in relevant)

    messages = [{"role": "system", "content": system}] + history + [{"role": "user", "content": message}]

    temperature = 0.2 if code_mode else 0.6
    max_tokens = 4000 if code_mode else 2500

    try:
        res = http.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + key},
            json={"model": CHAT_MODEL, "messages": messages, "temperature": temperature, "max_tokens": max_tokens},
            timeout=100,
        )
        data = res.json()
        if "error" in data:
            return jsonify({"error": data["error"].get("message", "Groq API error")}), 400
        reply = data["choices"][0]["message"]["content"]
    except Exception as e:
        return jsonify({"error": "Request to Groq failed: " + str(e)}), 500

    if mongo_ok and device_id and auto and looks_personal(message) and not memories.find_one({"text": message, "device_id": device_id}):
        memories.insert_one({"text": message, "created_at": time.time(), "source": "auto", "device_id": device_id})

    return jsonify({"reply": reply, "used_memories": [m["text"] for m in relevant]})


@app.route("/chat-finetuned", methods=["POST"])
def chat_finetuned():
    d = request.get_json(silent=True) or {}
    message = (d.get("message") or "").strip()
    finetuned_url = (d.get("finetuned_url") or ENV_FINETUNED_URL or "").strip().rstrip("/")

    if not message:
        return jsonify({"error": "No message provided"}), 400
    if not finetuned_url:
        return jsonify({"error": "No fine-tuned server URL. Paste your current ngrok URL in Settings."}), 400

    try:
        res = http.post(
            finetuned_url + "/chat",
            json={"message": message, "auto_remember": d.get("auto_remember", True)},
            headers={"ngrok-skip-browser-warning": "true"},
            timeout=120,
        )
    except Exception as e:
        return jsonify({"error": "Could not reach the fine-tuned server: " + str(e)}), 502

    try:
        result = res.json()
    except Exception:
        preview = res.text[:200].replace("\n", " ")
        return jsonify({"error": "The fine-tuned server did not return valid data (is Kaggle/ngrok still running?). Raw response: " + preview}), 502

    if "reply" not in result:
        return jsonify({"error": "Fine-tuned server responded but sent no 'reply' field."}), 502

    return jsonify({"reply": result["reply"]})


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


@app.route("/cf-image", methods=["POST"])
def cf_image():
    """Generate an image with Cloudflare Workers AI (FLUX.1 schnell) and return it as base64.
    The browser cannot call Cloudflare directly, so this server does it on its behalf."""
    d = request.get_json(silent=True) or {}
    prompt = (d.get("prompt") or "").strip()
    account = (d.get("account_id") or ENV_CF_ACCOUNT).strip()
    token = (d.get("api_token") or ENV_CF_TOKEN).strip()

    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400
    if not account or not token:
        return jsonify({"error": "No Cloudflare Account ID / API token. Add them in Settings."}), 400
    if not re.fullmatch(r"[A-Za-z0-9]+", account):
        return jsonify({"error": "That Cloudflare Account ID does not look right."}), 400

    try:
        steps = max(1, min(8, int(d.get("steps", 6))))
    except Exception:
        steps = 6

    try:
        res = http.post(
            "https://api.cloudflare.com/client/v4/accounts/" + account + "/ai/run/" + CF_IMAGE_MODEL,
            headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"},
            json={"prompt": prompt[:2000], "steps": steps},
            timeout=90,
        )
        data = res.json()
    except Exception as e:
        return jsonify({"error": "Cloudflare request failed: " + str(e)}), 502

    if res.status_code != 200 or not data.get("success", False):
        errs = data.get("errors") or []
        msg = errs[0].get("message") if errs and isinstance(errs[0], dict) else ("HTTP " + str(res.status_code))
        return jsonify({"error": "Cloudflare: " + str(msg)}), 400

    img = (data.get("result") or {}).get("image")
    if not img:
        return jsonify({"error": "Cloudflare returned no image."}), 502
    return jsonify({"image": img})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
