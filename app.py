from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from dotenv import load_dotenv
import requests
import time
import json
import jwt
import re
import os
import base64
from datetime import datetime

load_dotenv()

app = Flask(__name__)
CORS(app)

# ── PDF Generator API credentials ──────────────────────────────────────────────
PDF_API_URL = "https://us1.pdfgeneratorapi.com/api/v4/documents/generate"
API_KEY     = os.environ["PDF_API_KEY"]
SECRET_KEY  = os.environ["PDF_SECRET_KEY"]

# ── OpenAI key ──────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]


# ── Helpers (same logic as your original script) ───────────────────────────────

def generate_jwt():
    payload = {
        "iss": API_KEY,
        "sub": API_KEY,
        "exp": int(time.time()) + 60
    }
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")


def generate_image(image_prompt):
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
    response = requests.post(url, headers=headers, json=payload)
    data = response.json()
    if "data" not in data:
        raise ValueError(f"Image generation failed: {json.dumps(data)}")
    return data["data"][0].get("b64_json")


def create_image_prompt(title, story):
    return f"A children's illustration of: {title}. Scene: {story[:200]}"


def parse_worksheet(output_text):
    output_text = output_text.replace("*", "")
    lines = [line.strip() for line in output_text.splitlines() if line.strip()]
    if len(lines) < 7:
        raise ValueError("Output too short to parse")
    title = lines[0]
    question_indices = [i for i, line in enumerate(lines) if line.endswith("?")]
    if len(question_indices) < 5:
        raise ValueError("Not enough question lines detected")
    last_five_q_indices = question_indices[-5:]
    questions = [lines[i] for i in last_five_q_indices]
    story_start = 1
    story_end = last_five_q_indices[0]
    story = "\n\n".join(lines[story_start:story_end])
    return {
        "Title": title,
        "Story": story,
        "Who":   questions[0],
        "What":  questions[1],
        "When":  questions[2],
        "Where": questions[3],
        "Why":   questions[4]
    }


def generate_text(prompt):
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": "myWorksheetMaker:v4",
        "messages": [{"role": "user", "content": prompt}],
        "stream": False
    }
    r = requests.post(url, json=payload)
    return r.json()["message"]["content"]


def send_to_pdf_api(parsed_json):
    jwt_token = generate_jwt()
    payload = {
        "template": {
            "id": "1120770",
            "data": [parsed_json]
        },
        "format": "pdf",
        "output": "url"
    }
    headers = {
        "Authorization": f"Bearer {jwt_token}",
        "Content-Type": "application/json"
    }
    response = requests.post(PDF_API_URL, headers=headers, json=payload)
    json_response = response.json()
    pdf_url = json_response.get("response")
    if not pdf_url:
        raise ValueError(f"No PDF URL returned: {json_response}")
    pdf_bytes = requests.get(pdf_url).content
    return pdf_bytes


# ── Route ───────────────────────────────────────────────────────────────────────

@app.route("/generate", methods=["POST"])
def generate():
    body = request.get_json()
    grade         = body.get("grade", "")
    reading_level = body.get("reading_level", "")
    interests     = body.get("interests", "")

    prompt = (
        f"I have a student who is in {grade} grade and reads at a "
        f"{reading_level} grade level. {interests}"
    )

    try:
        # 1. Generate text via local Ollama model
        raw_text = generate_text(prompt)

        # 2. Parse into structured fields
        pdf_json = parse_worksheet(raw_text)

        # 3. Generate illustration
        image_prompt  = create_image_prompt(pdf_json["Title"], pdf_json["Story"])
        image_base64  = generate_image(image_prompt)
        pdf_json["image"] = f"data:image/png;base64,{image_base64}"

        # 4. Generate PDF and return it directly as a download
        pdf_bytes = send_to_pdf_api(pdf_json)

        tmp_path = f"/tmp/worksheet_{int(time.time())}.pdf"
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
    # Run on port 5000 by default; open http://localhost:5000 for health check
    app.run(debug=True, port=5000)
