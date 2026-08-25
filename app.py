import base64
import logging
import os
import re
import time

from dotenv import load_dotenv
from flask import Flask, jsonify, redirect, render_template, request, send_file

import ai
import file
from reading_level import ReadingLevelError

load_dotenv()

app = Flask(__name__)

OUTPUT_DIR = "worksheets"

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
    return render_template("index.html", active_page="worksheet")


@app.route("/reading-level-corrector")
def reading_level_corrector():
    return render_template(
        "reading_level_corrector.html", active_page="reading_level"
    )


@app.route("/lexile-corrector")
def lexile_corrector_redirect():
    return redirect("/reading-level-corrector", code=301)


@app.route("/correct-reading-level", methods=["POST"])
def correct_reading_level():
    """Measure and repair pasted text.

    Deterministic by default: no model call unless `allow_rewrite` is set, in
    which case a failing passage escalates to the LLM rewrite pass. Unlike the
    generation route this never returns an error for a passage it could not fix
    -- a teacher is reading the output here, and a partial improvement plus an
    honest "still above band" flag is more useful than a refusal.
    """
    body = request.get_json() or {}

    text = (body.get("text") or "").strip()
    target_grade = body.get("target_grade")
    allow_rewrite = bool(body.get("allow_rewrite"))
    # Left as None when the form does not say. correct_with_rewrite treats that
    # as "assume the strict tier and warn" rather than picking the permissive
    # one, since nothing about pasted text reveals whether it is a story.
    passage_type = body.get("passage_type") or None
    protected_terms = [
        term.strip()
        for term in (body.get("protected_terms") or "").split(",")
        if term.strip()
    ]

    if not text:
        return jsonify({"error": "Enter some text to correct."}), 400
    if target_grade in (None, ""):
        return jsonify({"error": "Choose a target grade level."}), 400

    try:
        outcome = ai.correct_with_rewrite(
            text,
            target_grade,
            protected_terms=protected_terms,
            allow_rewrite=allow_rewrite,
            passage_type=passage_type,
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except ReadingLevelError as exc:
        app.logger.exception("Reading level analysis is unavailable")
        return jsonify({"error": str(exc)}), 500

    result = outcome.correction

    return jsonify({
        "corrected_text": outcome.text,
        "reading_level": {
            "estimated_grade": round(outcome.final_score.estimated_grade, 2),
            "target_band": result.target_band.display,
            "in_band": outcome.gate_passed,
            "confidence": outcome.final_score.confidence,
        },
        "initial_reading_level": {
            "estimated_grade": round(result.initial_score.estimated_grade, 2),
            "confidence": result.initial_score.confidence,
        },
        "gate_passed": outcome.gate_passed,
        "failure_reason": outcome.failure_reason,
        "llm_rewrite_applied": outcome.llm_rewrite_applied,
    })


@app.route("/generate-leveled-passage", methods=["POST"])
def generate_leveled_passage():
    """Generate a passage and gate it on measured reading level."""
    body = request.get_json() or {}

    topic = (body.get("topic") or "").strip()
    target_grade = body.get("target_grade")
    student_interest = (body.get("student_interest") or "").strip() or None
    protected_terms = [
        term.strip()
        for term in (body.get("protected_terms") or "").split(",")
        if term.strip()
    ]

    if not topic:
        return jsonify({"error": "Enter a topic."}), 400
    if target_grade in (None, ""):
        return jsonify({"error": "Choose a target grade level."}), 400

    try:
        result = ai.generate_leveled_passage(
            topic,
            target_grade,
            protected_terms=protected_terms,
            student_interest=student_interest,
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except ReadingLevelError as exc:
        # A content-quality failure, not a server fault.
        status = 422 if exc.reason else 500
        return jsonify({"error": str(exc), "reason": exc.reason}), status

    return jsonify({
        "title": result["title"],
        "passage": result["passage"],
        "questions": result["questions"],
        "question_levels": result["question_levels"],
        "reading_level": result["reading_level"],
        "corrections_applied": result["corrections_applied"],
        "protected_terms_preserved": result["protected_terms_preserved"],
        "llm_rewrite_applied": result.get("llm_rewrite_applied", False),
        "naturalness": result.get("naturalness"),
    })


@app.route("/generate", methods=["POST"])
def generate():
    body = request.get_json() or {}

    grade = body.get("grade", "")
    reading_level = body.get("reading_level", "")
    interests = body.get("interests", "")

    try:
        model_data, pdf_bytes, image_base64, _, leveling = ai.generate_full_worksheet(
            grade, reading_level, interests
        )
        save_worksheet_pdf(pdf_bytes, model_data["title"])

        return jsonify({
            "content": model_data,
            "image_base64": image_base64,
            "pdf_base64": base64.b64encode(pdf_bytes).decode(),
            "reading_level": leveling,
        })
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
        pdf_bytes, _image = ai.build_pdf_bytes(body)
        pdf_data = ai.normalize_model_output(body)
        pdf_path = save_worksheet_pdf(pdf_bytes, pdf_data["Title"])

        return send_file(
            pdf_path,
            mimetype="application/pdf",
            as_attachment=False,
            download_name=f"{pdf_data['Title']}.pdf",
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/update-pdf", methods=["POST"])
def update_pdf():
    body = request.get_json() or {}

    try:
        ai.validate_structure(body)
        image_feedback = (body.get("image_feedback") or "").strip()
        existing_image = (body.get("image_base64") or "").strip()

        pdf_bytes, image_base64 = ai.build_pdf_bytes(
            body,
            image_feedback=image_feedback or None,
            existing_image_base64=existing_image or None,
        )
        save_worksheet_pdf(pdf_bytes, body["title"])

        return jsonify({
            "content": body,
            "image_base64": image_base64,
            "pdf_base64": base64.b64encode(pdf_bytes).decode(),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "true").lower() == "true"
    app.run(host="0.0.0.0", debug=debug, port=port)
