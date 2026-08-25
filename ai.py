import base64
import json
import logging
import os
import re
import threading

import file
import requests
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

import ai_rewrite
from reading_level import ReadingLevelError, correct_text, score_text
from reading_level import bands as rl_bands
from reading_level import config as rl_config

load_dotenv()

logger = logging.getLogger(__name__)

LLM_ENDPOINT = os.getenv("LLM_ENDPOINT", "")
LLM_AGENT_NAME = os.getenv("LLM_AGENT_NAME", "")
FLUX_API_KEY = os.getenv("FLUX_API_KEY", "")
FLUX_API_URL = os.getenv("FLUX_API_URL", "")
FLUX_MODEL = os.getenv("FLUX_MODEL", "")

WORKSHEET_SYSTEM_PROMPT = """You are a reading comprehension worksheet generator for special education teachers.

Given information about a student (grade, reading level, and interests), write a short fiction story and five reading comprehension questions.

Rules:
- Write the story at the student's READING LEVEL (not their grade level).
- Incorporate the student's interests naturally into the story.
- Keep the story engaging and age-appropriate for the student's grade.
- Write exactly 5 comprehension questions using Who, What, When, Where, and Why.
- Each question must end with a question mark and be answerable from the story.
- The story must be one paragraph with no line breaks.

Respond with ONLY valid JSON (no markdown, no extra text) in this exact shape:
{
  "title": "Story title",
  "story": "The full story text",
  "questions": {
    "q1": "Who ...?",
    "q2": "What ...?",
    "q3": "When ...?",
    "q4": "Where ...?",
    "q5": "Why ...?"
  }
}"""


def ensure_ai_configured():
    missing = []
    if not FLUX_API_KEY:
        missing.append("FLUX_API_KEY")
    if not LLM_ENDPOINT:
        missing.append("LLM_ENDPOINT")
    if not LLM_AGENT_NAME:
        missing.append("LLM_AGENT_NAME")
    if not FLUX_API_URL:
        missing.append("FLUX_API_URL")
    if not FLUX_MODEL:
        missing.append("FLUX_MODEL")
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")


def build_worksheet_prompt(grade, reading_level, interests):
    user_prompt = (
        f"I have a student who is in {grade} grade and reads at a "
        f"{reading_level} grade level. {interests}"
    )
    return f"{WORKSHEET_SYSTEM_PROMPT}\n\nUser request:\n{user_prompt}"


def create_image_prompt(title, story, feedback=None):
    prompt = f"A children's educational illustration titled '{title}'. Scene: {story[:200]}"
    if feedback:
        prompt += f" Additional instructions: {feedback.strip()}"
    return prompt


def validate_structure(data):
    required = ["title", "story", "questions"]

    for field in required:
        if field not in data:
            raise ValueError(f"Missing field: {field}")

    questions = data["questions"]
    for key in ["q1", "q2", "q3", "q4", "q5"]:
        if key not in questions:
            raise ValueError(f"Missing question: {key}")

    return True


def normalize_model_output(data):
    return {
        "Title": data["title"],
        "Story": data["story"],
        "Who": data["questions"]["q1"],
        "What": data["questions"]["q2"],
        "When": data["questions"]["q3"],
        "Where": data["questions"]["q4"],
        "Why": data["questions"]["q5"],
    }


def _running_on_azure() -> bool:
    """Managed identity only exists in Azure. Asking IMDS on a laptop just waits."""
    return bool(
        os.getenv("IDENTITY_ENDPOINT")
        or os.getenv("MSI_ENDPOINT")
        or os.getenv("IDENTITY_HEADER")
    )


_client_lock = threading.Lock()
_project_client = None
_openai_client = None


def _get_openai_client():
    """Reuse one project client. Building DefaultAzureCredential per call
    used to spend ~8s on a dead IMDS probe before falling through to `az login`.
    """
    global _project_client, _openai_client
    if _openai_client is not None:
        return _project_client, _openai_client

    with _client_lock:
        if _openai_client is not None:
            return _project_client, _openai_client

        credential = DefaultAzureCredential(
            exclude_managed_identity_credential=not _running_on_azure(),
        )
        _project_client = AIProjectClient(
            endpoint=LLM_ENDPOINT,
            credential=credential,
            allow_preview=True,
        )
        _openai_client = _project_client.get_openai_client(
            agent_name=LLM_AGENT_NAME
        )
        return _project_client, _openai_client


def generate_text(prompt):
    ensure_ai_configured()
    _, openai_client = _get_openai_client()

    response = openai_client.responses.create(input=prompt)
    text = response.output_text
    if text is None or not str(text).strip():
        raise RuntimeError(f"Azure agent returned empty text output: {response}")
    return text


def generate_image(image_prompt):
    ensure_ai_configured()
    url = f"{FLUX_API_URL.rstrip('/')}?api-version=preview"

    headers = {
        "Authorization": f"Bearer {FLUX_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": FLUX_MODEL,
        "prompt": image_prompt,
        "width": 1024,
        "height": 1024,
        "output_format": "jpeg",
        "num_images": 1,
    }

    response = requests.post(url, headers=headers, json=payload, timeout=120)
    data = response.json()

    if response.status_code >= 400:
        raise ValueError(f"Image generation failed: {data}")

    if "data" not in data:
        raise ValueError(f"Image generation failed: {data}")

    img = data["data"][0]

    if "b64_json" in img:
        return img["b64_json"]

    if "url" in img:
        img_bytes = requests.get(img["url"], timeout=60).content
        return base64.b64encode(img_bytes).decode()

    raise ValueError("No valid image returned")


def parse_worksheet_output(raw_text):
    text = raw_text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        extracted = _extract_json_fields(text)
        if extracted:
            return extracted
        return _parse_worksheet_text(text)


def _extract_json_fields(text):
    title_match = re.search(r'"title"\s*:\s*"((?:\\.|[^"\\])*)"', text, re.DOTALL)
    story_match = re.search(r'"story"\s*:\s*"((?:\\.|[^"\\])*)"', text, re.DOTALL)
    if not title_match or not story_match:
        return None

    questions = {}
    for key in ["q1", "q2", "q3", "q4", "q5"]:
        match = re.search(rf'"{key}"\s*:\s*"((?:\\.|[^"\\])*)"', text, re.DOTALL)
        if not match:
            return None
        questions[key] = json.loads(f'"{match.group(1)}"')

    return {
        "title": json.loads(f'"{title_match.group(1)}"'),
        "story": json.loads(f'"{story_match.group(1)}"'),
        "questions": questions,
    }


def _parse_worksheet_text(output_text):
    """Fallback parser preserved from the former backend."""
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
    story_lines = lines[1:last_five_q_indices[0]]
    story = "\n\n".join(story_lines)

    return {
        "title": title,
        "story": story,
        "questions": {
            "q1": questions[0],
            "q2": questions[1],
            "q3": questions[2],
            "q4": questions[3],
            "q5": questions[4],
        },
    }


def generate_worksheet_content(grade, reading_level, interests):
    prompt = build_worksheet_prompt(grade, reading_level, interests)
    raw_text = generate_text(prompt)
    model_data = parse_worksheet_output(raw_text)
    validate_structure(model_data)
    return model_data, raw_text


def build_pdf_bytes(worksheet_content, image_feedback=None, existing_image_base64=None):
    pdf_json = normalize_model_output(worksheet_content)

    if image_feedback:
        image_prompt = create_image_prompt(
            pdf_json["Title"], pdf_json["Story"], feedback=image_feedback
        )
        image_base64 = generate_image(image_prompt)
    elif existing_image_base64:
        image_base64 = existing_image_base64
    else:
        image_prompt = create_image_prompt(pdf_json["Title"], pdf_json["Story"])
        image_base64 = generate_image(image_prompt)

    pdf_json["image"] = f"data:image/jpeg;base64,{image_base64}"
    pdf_bytes = file.generate_worksheet_pdf(pdf_json)
    return pdf_bytes, image_base64


def _leveling_report(original_story, first_score, outcome):
    """Before/after numbers for the worksheet page comparison."""
    band = outcome.target_band
    return {
        "target_band": band.display,
        "first_draft": {
            "estimated_grade": round(first_score.estimated_grade, 2),
            "confidence": first_score.confidence,
            "in_band": rl_bands.in_band(first_score, band),
            "below_target": first_score.estimated_grade < band.low,
            "story": original_story,
        },
        "final": {
            "estimated_grade": round(outcome.final_score.estimated_grade, 2),
            "confidence": outcome.final_score.confidence,
            "in_band": outcome.gate_passed,
        },
        "improved": original_story.strip() != outcome.text.strip(),
        "llm_rewrite_applied": outcome.llm_rewrite_applied,
    }


def generate_full_worksheet(grade, reading_level, interests):
    """Generate a worksheet, then run the reading-level pipeline on the story.

    The first model draft is kept so the review panel can show what the
    pipeline changed. The PDF uses the leveled story. A leveling miss does
    not abort the worksheet -- the closest result still ships, flagged.
    """
    model_data, raw_text = generate_worksheet_content(grade, reading_level, interests)
    original_story = model_data["story"]
    first_score = score_text(original_story)

    try:
        outcome = correct_with_rewrite(
            original_story,
            reading_level,
            allow_rewrite=True,
            passage_type=rl_config.PASSAGE_TYPE_NARRATIVE,
        )
        model_data["story"] = outcome.text
        leveling = _leveling_report(original_story, first_score, outcome)
    except (ReadingLevelError, ValueError):
        logger.exception("Reading-level pipeline failed; shipping the first draft")
        try:
            fallback_band = rl_bands.target_band(reading_level)
            target_display = fallback_band.display
            below_target = first_score.estimated_grade < fallback_band.low
        except ValueError:
            target_display = str(reading_level)
            below_target = False
        leveling = {
            "target_band": target_display,
            "first_draft": {
                "estimated_grade": round(first_score.estimated_grade, 2),
                "confidence": first_score.confidence,
                "in_band": False,
                "below_target": below_target,
                "story": original_story,
            },
            "final": {
                "estimated_grade": round(first_score.estimated_grade, 2),
                "confidence": first_score.confidence,
                "in_band": False,
            },
            "improved": False,
            "llm_rewrite_applied": False,
        }

    pdf_bytes, image_base64 = build_pdf_bytes(model_data)
    return model_data, pdf_bytes, image_base64, raw_text, leveling


def _log_stage(stage, **fields):
    """Emit a parseable log line.

    These records are the dataset for future calibration and fine-tuning:
    every passage that needed several iterations is a training example, so the
    payload is JSON rather than prose.
    """
    logger.info(json.dumps({"stage": stage, **fields}, default=str))


def _score_summary(score):
    return {
        "estimated_grade": round(score.estimated_grade, 2),
        "confidence": score.confidence,
        "flags": list(score.flags),
    }


def build_leveled_prompt(topic, band, student_interest=None, feedback=None):
    """Prompt for a passage aimed at a band.

    The band is included because it improves first-draft quality, but it is
    only a hint: the scorer, not the model, decides whether the draft counts.
    """
    request = (
        f"Write the story about: {topic}. "
        f"Target reading level: {band.display} "
        f"(roughly grade {band.center:.0f})."
    )
    if student_interest:
        request += f" The student is interested in {student_interest}."
    if feedback:
        request += f" The previous attempt was rejected because {feedback}."

    return f"{WORKSHEET_SYSTEM_PROMPT}\n\nUser request:\n{request}"


def _difficulty_feedback(result):
    """Name the specific failure so the retry prompt can act on it."""
    if result.failure_reason == "below_target_band":
        return (
            "the passage read below the target grade band "
            "(use more specific words and slightly longer sentences)"
        )

    features = result.final_score.features
    limit = rl_bands.max_sentence_length(result.target_band)

    problems = []
    if features.max_sentence_length > limit:
        problems.append(
            f"the sentences were too long (up to {features.max_sentence_length} "
            f"words; keep them under {limit})"
        )
    if features.hard_word_ratio > 0:
        problems.append(
            "the vocabulary was too advanced (use shorter, more common words)"
        )
    if not problems:
        problems.append("the passage read above the target grade band")

    return " and ".join(problems)


def _level_questions(questions, band):
    """Score question stems against a widened band.

    A single stem is far below the length at which readability statistics mean
    anything, so these are measured and logged but never hard-gated.
    """
    tolerant_band = rl_bands.widen(band, rl_config.QUESTION_BAND_TOLERANCE)
    report = {}

    for key, stem in questions.items():
        score = score_text(stem)
        report[key] = {
            **_score_summary(score),
            "in_tolerant_band": rl_bands.in_band(score, tolerant_band),
        }

    _log_stage(
        "questions_scored",
        band=band.label,
        tolerance=rl_config.QUESTION_BAND_TOLERANCE,
        questions=report,
    )
    return report


def _leveled_response(model_data, band, result, protected_terms, rewritten=False):
    """Shape the API payload for an accepted passage.

    Raw feature values and internal coefficients never cross this boundary.
    """
    model_data["story"] = result.text
    questions = _level_questions(model_data["questions"], band)
    lowered = result.text.lower()

    naturalness = ai_rewrite.check_naturalness(
        result.text, band, generate=generate_text
    )

    return {
        "title": model_data["title"],
        "passage": result.text,
        "questions": model_data["questions"],
        "question_levels": questions,
        "reading_level": {
            "estimated_grade": round(result.final_score.estimated_grade, 2),
            "target_band": band.display,
            "in_band": result.gate_passed,
            "confidence": result.final_score.confidence,
        },
        "corrections_applied": len(result.edits),
        "protected_terms_preserved": [
            term for term in protected_terms if term.lower() in lowered
        ],
        "llm_rewrite_applied": rewritten,
        "naturalness": naturalness,
    }


def _log_correction(result, attempt):
    for edit in result.edits:
        _log_stage(
            "edit_applied",
            attempt=attempt,
            operator=edit.operator,
            sentence_index=edit.sentence_index,
            before=edit.before,
            after=edit.after,
            sense_disambiguated=edit.sense_disambiguated,
        )


def correct_with_rewrite(
    text,
    target_grade,
    *,
    protected_terms=None,
    allow_rewrite=False,
    passage_type=None,
):
    """Repair `text`, optionally escalating to an LLM rewrite when repair fails.

    Returns an ai_rewrite.RewriteOutcome and never raises on a content-quality
    failure: this backs the corrector page, where a teacher is reading the
    output and a partial improvement plus an honest "still above band" flag is
    more useful than an error. The generation path takes the opposite line --
    see generate_leveled_passage.

    `allow_rewrite` defaults to False so the deterministic promise of this path
    holds unless a caller explicitly asks to spend a model call.

    `passage_type` has no default here on purpose. This path takes arbitrary
    pasted text, so nothing in the request tells us whether it is a story or a
    science paragraph -- only the teacher knows. Omitting it warns and takes
    the stricter tier rather than handing narrative freedom to a passage whose
    facts a question set may depend on.
    """
    protected_terms = list(protected_terms or [])
    result = correct_text(text, target_grade, protected_terms=protected_terms)
    _log_correction(result, attempt=0)

    outcome = ai_rewrite.RewriteOutcome(
        text=result.text,
        correction=result,
        final_score=result.final_score,
        gate_passed=result.gate_passed,
        failure_reason=result.failure_reason,
    )

    # Deterministic operators only simplify, so a below-band draft is
    # unchanged until this rewrite pass raises it.
    if not allow_rewrite or result.gate_passed:
        return outcome

    return ai_rewrite.escalate_to_rewrite(
        result,
        target_grade,
        generate=generate_text,
        protected_terms=protected_terms,
        passage_type=passage_type,
        keep_closest=True,
    )


def generate_leveled_passage(
    topic,
    target_grade,
    *,
    protected_terms=None,
    student_interest=None,
    passage_type=rl_config.PASSAGE_TYPE_NARRATIVE,
):
    """Generate a passage and guarantee it lands in the target reading band.

    Returns a dict with the passage, its measured level, and correction
    metadata.     Raises ReadingLevelError if the passage cannot be brought into
    band -- too hard or too easy. Shipping mis-leveled material to a special
    educator silently is a worse failure than a visible error. That is the
    opposite of what correct_with_rewrite does, deliberately -- this output
    goes to a student, that one goes to a teacher who is reviewing it.

    Deterministic repair runs first. Only when it fails does Operator D spend a
    model call on a diagnostic-driven rewrite.

    `passage_type` defaults to narrative because that is what this path asks
    for: WORKSHEET_SYSTEM_PROMPT requests "a short fiction story", so a rewrite
    here is free to restructure the plot. If this flow ever grows a mode that
    generates factual passages, that mode must pass "informational" instead --
    the default is read off the generation prompt, not assumed.
    """
    band = rl_bands.target_band(target_grade)
    protected_terms = list(protected_terms or [])
    feedback = None
    last_result = None
    last_outcome = None

    # One shared budget across generation AND rewrite calls, so adding Operator
    # D cannot multiply the number of model calls a request makes. With the
    # default of 2 this buys one draft plus one targeted rewrite, in place of
    # two blind drafts -- a rewrite steered by diagnostics is a better use of
    # the second call than regenerating and hoping.
    calls_remaining = rl_config.MAX_GENERATION_ATTEMPTS

    for attempt in range(1, rl_config.MAX_GENERATION_ATTEMPTS + 1):
        if calls_remaining <= 0:
            break

        prompt = build_leveled_prompt(topic, band, student_interest, feedback)
        raw_text = generate_text(prompt)
        calls_remaining -= 1
        model_data = parse_worksheet_output(raw_text)
        validate_structure(model_data)

        passage = model_data["story"]
        initial = score_text(passage)
        _log_stage(
            "generated",
            attempt=attempt,
            band=band.label,
            initial_score=_score_summary(initial),
        )

        result = correct_text(
            passage, target_grade, protected_terms=protected_terms
        )
        last_result = result
        _log_correction(result, attempt)

        _log_stage(
            "corrected",
            attempt=attempt,
            band=band.label,
            iterations=result.iterations,
            edits=len(result.edits),
            score_trajectory=[round(grade, 2) for grade in result.score_trajectory],
            in_band=result.in_band,
            failure_reason=result.failure_reason,
            final_score=_score_summary(result.final_score),
            protected_terms_preserved=result.protected_terms_preserved,
        )

        if result.in_band:
            return _leveled_response(model_data, band, result, protected_terms)

        # Operator D: deterministic repair could not close the gap, so spend a
        # call on a rewrite steered by what the scorer actually measured.
        if calls_remaining > 0:
            outcome = ai_rewrite.escalate_to_rewrite(
                result,
                target_grade,
                generate=generate_text,
                protected_terms=protected_terms,
                max_attempts=calls_remaining,
                passage_type=passage_type,
            )
            calls_remaining -= outcome.llm_attempts
            last_outcome = outcome

            if outcome.gate_passed:
                model_data["story"] = outcome.text
                return _leveled_response(
                    model_data, band, outcome, protected_terms, rewritten=True
                )

        feedback = _difficulty_feedback(result)

    failure_reason = (
        last_outcome.failure_reason
        if last_outcome is not None
        else (last_result.failure_reason if last_result else None)
    )
    _log_stage(
        "gate_failed",
        band=band.label,
        attempts=rl_config.MAX_GENERATION_ATTEMPTS,
        failure_reason=failure_reason,
        llm_rewrite_attempted=last_outcome is not None,
    )
    raise ReadingLevelError(
        f"Could not bring the passage into {band.display} after "
        f"{rl_config.MAX_GENERATION_ATTEMPTS} attempts.",
        reason=failure_reason,
        score=last_result.final_score if last_result else None,
        band=band,
    )


if __name__ == "__main__":
    content, _ = generate_worksheet_content(
        grade="6",
        reading_level="1st",
        interests="my student is interested in watercolor paintings",
    )
    print(json.dumps(content, indent=2))
