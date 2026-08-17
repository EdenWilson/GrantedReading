import os
import re
import time

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, send_file

import ai
import file

load_dotenv()

app = Flask(__name__)

OUTPUT_DIR =  "worksheets"


def save_worksheet_pdf(pdf_bytes, title):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    safe_title = re.sub(r"[^\w\s-]", "", title).strip().replace(" ", "_") or "worksheet"
    filename = f"{safe_title}_{int(time.time())}.pdf"
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "wb") as f:
        f.write(pdf_bytes)
    return path


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    body = request.get_json() or {}

    grade = body.get("grade", "")
    reading_level = body.get("reading_level", "")
    interests = body.get("interests", "")

    try:
        model_data, raw_text = ai.generate_worksheet_content(grade, reading_level, interests)
        return jsonify({"content": model_data})
    except Exception as e:
        return jsonify({
            "error": str(e),
            "raw_model_output": raw_text if "raw_text" in locals() else None,
        }), 500


@app.route("/create-pdf", methods=["POST"])
def create_pdf():
    body = request.get_json() or {}

    try:
        ai.validate_structure(body)
        pdf_data = ai.build_pdf_data(body)
        pdf_bytes = file.generate_worksheet_pdf(pdf_data)
        pdf_path = save_worksheet_pdf(pdf_bytes, pdf_data["Title"])

        return send_file(
            pdf_path,
            mimetype="application/pdf",
            as_attachment=False,
            download_name=f"{pdf_data['Title']}.pdf",
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "true").lower() == "true"
    app.run(host="0.0.0.0", debug=debug, port=port)
