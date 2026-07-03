from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from dotenv import load_dotenv
import requests
import time
import json
import jwt
import os
import base64
from datetime import datetime

load_dotenv()

app = Flask(__name__)
CORS(app)

# ── CONFIG ─────────────────────────────────────────────────────────────
PDF_API_URL = "https://us1.pdfgeneratorapi.com/api/v4/documents/generate"
PDF_TEMPLATE_ID = os.getenv("PDF_TEMPLATE_ID", "1120770")
API_KEY = os.getenv("PDF_API_KEY", "")
SECRET_KEY = os.getenv("PDF_SECRET_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/chat")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "myWorksheetMaker:v4")
OLLAMA_TIMEOUT_SECONDS = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "30"))


def ensure_configured():
    missing = []
    if not API_KEY:
        missing.append("PDF_API_KEY")
    if not SECRET_KEY:
        missing.append("PDF_SECRET_KEY")
    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

# ── JWT ────────────────────────────────────────────────────────────────
def generate_jwt():
    ensure_configured()
    payload = {
        "iss": API_KEY,
        "sub": API_KEY,
        "exp": int(time.time()) + 60
    }
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")


# ── IMAGE GENERATION ───────────────────────────────────────────────────
def generate_image(image_prompt):
    ensure_configured()
    url = "https://api.openai.com/v1/images/generations"

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "gpt-image-1",
        "prompt": image_prompt,
        "size": "1024x1024"
    }

    response = requests.post(url, headers=headers, json=payload, timeout=60)
    data = response.json()

    if "data" not in data:
        raise ValueError(f"Image generation failed: {data}")

    img = data["data"][0]

    # handle both formats safely
    if "b64_json" in img:
        return img["b64_json"]

    if "url" in img:
        img_bytes = requests.get(img["url"], timeout=60).content
        return base64.b64encode(img_bytes).decode()

    raise ValueError("No valid image returned")


def create_image_prompt(title, story):
    return f"A children's educational illustration titled '{title}'. Scene: {story[:200]}"


# ── OLLAMA TEXT GENERATION ─────────────────────────────────────────────
def generate_text(prompt):
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False
    }

    try:
        r = requests.post(OLLAMA_API_URL, json=payload, timeout=OLLAMA_TIMEOUT_SECONDS)
        r.raise_for_status()
        data = r.json()
        return data["message"]["content"]
    except requests.exceptions.Timeout as exc:
        raise RuntimeError(
            f"Ollama timed out after {OLLAMA_TIMEOUT_SECONDS} seconds. Make sure Ollama is running and the model '{OLLAMA_MODEL}' is available."
        ) from exc
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            f"Could not reach Ollama at {OLLAMA_API_URL}. Start it with 'ollama serve' and confirm the model exists."
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Ollama request failed: {exc}") from exc
    except (KeyError, ValueError, TypeError) as exc:
        raise RuntimeError("Ollama returned an unexpected response format.") from exc


# ── VALIDATION ─────────────────────────────────────────────────────────
def validate_structure(data):
    required = ["title", "story", "questions"]

    for r in required:
        if r not in data:
            raise ValueError(f"Missing field: {r}")

    q = data["questions"]
    for key in ["q1", "q2", "q3", "q4", "q5"]:
        if key not in q:
            raise ValueError(f"Missing question: {key}")

    return True


# ── NORMALIZATION ──────────────────────────────────────────────────────
def normalize_model_output(data):
    return {
        "Title": data["title"],
        "Story": data["story"],
        "Who": data["questions"]["q1"],
        "What": data["questions"]["q2"],
        "When": data["questions"]["q3"],
        "Where": data["questions"]["q4"],
        "Why": data["questions"]["q5"]
    }


# ── PDF API ────────────────────────────────────────────────────────────
def send_to_pdf_api(parsed_json):
    jwt_token = generate_jwt()

    payload = {
        "template": {
            "id": PDF_TEMPLATE_ID,
            "data": [parsed_json]
        },
        "format": "pdf",
        "output": "url"
    }

    headers = {
        "Authorization": f"Bearer {jwt_token}",
        "Content-Type": "application/json"
    }

    response = requests.post(PDF_API_URL, headers=headers, json=payload, timeout=60)
    json_response = response.json()

    pdf_url = json_response.get("response")
    if not pdf_url:
        raise ValueError(f"No PDF URL returned: {json_response}")

    return requests.get(pdf_url, timeout=60).content


# ── ROUTE ──────────────────────────────────────────────────────────────
@app.route("/generate", methods=["POST"])
def generate():
    body = request.get_json() or {}

    grade = body.get("grade", "")
    reading_level = body.get("reading_level", "")
    interests = body.get("interests", "")

    prompt = (
        f"I have a student who is in {grade} grade and reads at a "
        f"{reading_level} grade level. {interests}"
    )

    try:
        raw_text = generate_text(prompt)
        model_data = json.loads(raw_text)
        validate_structure(model_data)

        return jsonify({
            "content": model_data
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "raw_model_output": raw_text if "raw_text" in locals() else None
        }), 500


@app.route("/create-pdf", methods=["POST"])
def create_pdf():
    body = request.get_json() or {}

    try:
        validate_structure(body)
        pdf_json = normalize_model_output(body)

        image_prompt = create_image_prompt(pdf_json["Title"], pdf_json["Story"])
        image_base64 = generate_image(image_prompt)
        pdf_json["image"] = f"data:image/png;base64,{image_base64}"

        pdf_bytes = send_to_pdf_api(pdf_json)

        tmp_path = f"worksheet_{int(time.time())}.pdf"
        with open(tmp_path, "wb") as f:
            f.write(pdf_bytes)

        return send_file(
            tmp_path,
            mimetype="application/pdf",
            as_attachment=True,
            download_name=f"{pdf_json['Title']}.pdf"
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)