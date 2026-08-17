import base64
import html
import os
import re
import tempfile

import pymupdf as fitz

PAGE_MARGIN = 40
STORY_COL_WIDTH = 280
IMAGE_WIDTH = 200
IMAGE_HEIGHT = 200
IMAGE_TOP = 125
IMAGE_RIGHT_MARGIN = 40

WORKSHEET_CSS = """
body {
  font-family: Helvetica, Arial, sans-serif;
  font-size: 11pt;
  line-height: 1.45;
  color: #1a1a1a;
}

.meta-table {
  width: 100%;
  border-collapse: collapse;
  margin-bottom: 12px;
}

.meta-table td {
  vertical-align: top;
}

.meta-label {
  margin-bottom: 4px;
}

.meta-line {
  border-bottom: 1px solid #1a1a1a;
  height: 14px;
  width: 95%;
}

.title {
  text-align: center;
  font-size: 18pt;
  font-weight: 700;
  margin: 6px 0 14px 0;
}

.story-table {
  width: 100%;
  border-collapse: collapse;
  margin-bottom: 18px;
}

.story-table td {
  vertical-align: top;
}

.story-col {
  width: 280px;
  padding-right: 16px;
  font-size: 11pt;
  line-height: 1.5;
}

.image-col {
  width: 200px;
}

.image-slot {
  width: 200px;
  height: 200px;
}

.question {
  margin-bottom: 14px;
}

.question-number {
  font-weight: 700;
}

.question-text {
  font-weight: 400;
}

.answer-line {
  border-bottom: 1px solid #9ca3af;
  height: 22px;
  margin-top: 6px;
}
"""

QUESTION_FIELDS = [
    ("Who", "Who"),
    ("What", "What"),
    ("When", "When"),
    ("Where", "Where"),
    ("Why", "Why"),
]


def decode_image_bytes(data):
    image = str(data.get("image", "")).strip()
    if not image:
        return None

    if image.startswith("data:"):
        match = re.match(r"data:image/[\w+.-]+;base64,(.+)", image, re.DOTALL)
        if not match:
            return None
        return base64.b64decode(match.group(1))

    try:
        return base64.b64decode(image)
    except (ValueError, TypeError):
        return None


def build_worksheet_html(data, include_image_slot=False):
    title = html.escape(str(data.get("Title", "")))
    story = html.escape(str(data.get("Story", ""))).replace("\n", "<br/>")

    image_cell = ""
    if include_image_slot:
        image_cell = """
          <td class="image-col">
            <div class="image-slot"></div>
          </td>
        """

    questions_html = []
    for index, (key, _label) in enumerate(QUESTION_FIELDS, start=1):
        question = html.escape(str(data.get(key, "")))
        questions_html.append(
            f"""
            <div class="question">
              <div>
                <span class="question-number">{index}.</span>
                <span class="question-text">{question}</span>
              </div>
              <div class="answer-line"></div>
            </div>
            """
        )

    return f"""
    <div class="worksheet">
      <table class="meta-table">
        <tr>
          <td width="50%">
            <div class="meta-label">Name</div>
            <div class="meta-line"></div>
          </td>
          <td width="50%">
            <div class="meta-label">Date</div>
            <div class="meta-line"></div>
          </td>
        </tr>
      </table>
      <h1 class="title">{title}</h1>
      <table class="story-table">
        <tr>
          <td class="story-col">{story}</td>
          {image_cell}
        </tr>
      </table>
      {''.join(questions_html)}
    </div>
    """


def _render_html_to_pdf(worksheet_html):
    story = fitz.Story(html=worksheet_html, user_css=WORKSHEET_CSS)

    fd, path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)

    try:
        writer = fitz.DocumentWriter(path)
        mediabox = fitz.paper_rect("a4")
        content_rect = mediabox + (
            PAGE_MARGIN,
            PAGE_MARGIN,
            -PAGE_MARGIN,
            -PAGE_MARGIN,
        )

        more = 1
        while more:
            device = writer.begin_page(mediabox)
            more, _ = story.place(content_rect)
            story.draw(device)
            writer.end_page()

        writer.close()

        with open(path, "rb") as pdf_file:
            return pdf_file.read()
    finally:
        if os.path.exists(path):
            os.unlink(path)


def _image_rect(page):
    x1 = page.rect.width - IMAGE_RIGHT_MARGIN
    x0 = x1 - IMAGE_WIDTH
    y0 = IMAGE_TOP
    return fitz.Rect(x0, y0, x1, y0 + IMAGE_HEIGHT)


def _insert_image_on_first_page(pdf_bytes, image_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[0]
    page.insert_image(_image_rect(page), stream=image_bytes, keep_proportion=True)
    return doc.tobytes()


def generate_worksheet_pdf(data):
    image_bytes = decode_image_bytes(data)
    worksheet_html = build_worksheet_html(data, include_image_slot=bool(image_bytes))
    pdf_bytes = _render_html_to_pdf(worksheet_html)

    if image_bytes:
        pdf_bytes = _insert_image_on_first_page(pdf_bytes, image_bytes)

    return pdf_bytes
