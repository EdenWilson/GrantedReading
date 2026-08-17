import base64
import json
import os
import re

import requests
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

load_dotenv()

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


def create_image_prompt(title, story):
    return f"A children's educational illustration titled '{title}'. Scene: {story[:200]}"


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


def _get_openai_client():
    credential = DefaultAzureCredential()
    project_client = AIProjectClient(
        endpoint=LLM_ENDPOINT,
        credential=credential,
        allow_preview=True,
    )
    return project_client, project_client.get_openai_client(agent_name=LLM_AGENT_NAME)


def generate_text(prompt):
    ensure_ai_configured()
    project_client, openai_client = _get_openai_client()

    try:
        response = openai_client.responses.create(input=prompt)
        text = response.output_text
        if text is None or not str(text).strip():
            raise RuntimeError(f"Azure agent returned empty text output: {response}")
        return text
    finally:
        project_client.close()


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


def build_pdf_data(worksheet_content):
    pdf_json = normalize_model_output(worksheet_content)
    image_prompt = create_image_prompt(pdf_json["Title"], pdf_json["Story"])
    image_base64 = generate_image(image_prompt)
    pdf_json["image"] = f"data:image/jpeg;base64,{image_base64}"
    return pdf_json


if __name__ == "__main__":
    content, _ = generate_worksheet_content(
        grade="6",
        reading_level="1st",
        interests="my student is interested in watercolor paintings",
    )
    print(json.dumps(content, indent=2))
