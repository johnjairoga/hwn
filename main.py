import io
import json
import logging
import os
import time

import anthropic
import pdfplumber
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="HireWithNear – AI Candidate Evaluator")
templates = Jinja2Templates(directory="templates")
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

MAX_CV_CHARS = 5000
MAX_JD_CHARS = 3000
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

SYSTEM_PROMPT = """You are an expert technical recruiter.

Evaluate how well the candidate CV matches the job description.

Scoring:
- 0–30: poor fit
- 31–60: partial fit
- 61–80: good fit
- 81–100: strong fit

Instructions:
- Be objective and concise
- Base your evaluation only on the provided information
- Return ONLY valid JSON matching the schema

Focus on:
- relevant skills
- experience alignment
- missing requirements"""


def extract_text(file_bytes: bytes) -> str:
    logger.info("Extracting text from PDF (%d bytes)", len(file_bytes))
    try:
        stream = io.BytesIO(file_bytes)
        with pdfplumber.open(stream) as pdf:
            text_parts = []
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text_parts.append(page_text)
            text = "\n".join(text_parts)
            text = text[: MAX_CV_CHARS]

            if not text.strip():
                raise ValueError("No text extracted from PDF")

            logger.info("Extracted %d chars from PDF", len(text))
            return text
    except Exception as e:
        logger.error("PDF extraction failed: %s", e)
        raise HTTPException(422, "Could not extract text from PDF. The file may be encrypted or scanned.")


def call_ai(job_description: str, cv_text: str) -> dict:
    logger.info("Calling Claude API for evaluation")
    t0 = time.perf_counter()

    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"Job Description:\n{job_description}\n\nCandidate CV:\n{cv_text}",
                }
            ],
            output_config={
                "format": {
                    "type": "json_schema",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "score": {"type": "integer"},
                            "fit": {"type": "string", "enum": ["poor", "partial", "good", "strong"]},
                            "match_label": {"type": "string"},
                            "emoji": {"type": "string"},
                            "summary": {"type": "string"},
                            "strengths": {"type": "array", "items": {"type": "string"}},
                            "weaknesses": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["score", "fit", "match_label", "emoji", "summary", "strengths", "weaknesses"],
                        "additionalProperties": False,
                    },
                }
            },
        )

        elapsed = time.perf_counter() - t0
        logger.info(
            "AI response received in %.2fs | input=%d output=%d",
            elapsed,
            response.usage.input_tokens,
            response.usage.output_tokens,
        )
        return parse_response(response)
    except anthropic.APIError as e:
        logger.error("Anthropic API error: %s", e)
        raise HTTPException(500, "AI service error. Please try again later.")


def parse_response(raw_response) -> dict:
    logger.info("Parsing AI response")
    try:
        text_block = next((b for b in raw_response.content if b.type == "text"), None)
        if not text_block:
            raise ValueError("No text block in response")

        data = json.loads(text_block.text)

        # Validate and sanitize
        if "score" not in data or not isinstance(data["score"], int):
            raise ValueError("Missing or invalid score")
        data["score"] = max(0, min(100, data["score"]))  # Clamp to 0-100

        if "fit" not in data:
            data["fit"] = "partial"

        if "match_label" not in data or not data["match_label"]:
            data["match_label"] = "Unrated"

        if "emoji" not in data or not data["emoji"]:
            data["emoji"] = "❓"

        if "summary" not in data or not data["summary"]:
            data["summary"] = "Evaluation complete."

        if "strengths" not in data or not isinstance(data["strengths"], list) or len(data["strengths"]) == 0:
            data["strengths"] = ["N/A"]
        else:
            data["strengths"] = [s for s in data["strengths"] if s][:3]  # Keep first 3, filter empty

        if "weaknesses" not in data or not isinstance(data["weaknesses"], list) or len(data["weaknesses"]) == 0:
            data["weaknesses"] = ["N/A"]
        else:
            data["weaknesses"] = [w for w in data["weaknesses"] if w][:3]  # Keep first 3, filter empty

        data["display_match"] = f"{data['emoji']} {data['match_label']} – {data['score']}%"
        logger.info("Evaluation complete: score=%d fit=%s", data["score"], data["fit"])
        return data
    except Exception as e:
        logger.error("Failed to parse AI response: %s", e)
        raise HTTPException(500, "AI response could not be parsed")


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "error": exc.detail},
        status_code=exc.status_code,
    )


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/analyze", response_class=HTMLResponse)
async def analyze(
    request: Request,
    job_description: str = Form(...),
    file: UploadFile = File(...),
):
    logger.info("Request received: filename=%s content_type=%s", file.filename, file.content_type)

    # Validation: PDF only
    if file.content_type not in ("application/pdf", "application/octet-stream") \
       or not (file.filename or "").lower().endswith(".pdf"):
        logger.error("Invalid file type: %s / %s", file.filename, file.content_type)
        raise HTTPException(400, "Only PDF files are accepted")

    # Read file into memory
    file_bytes = await file.read()

    # Validation: max 5MB
    if len(file_bytes) > MAX_FILE_SIZE:
        logger.error("File too large: %d bytes", len(file_bytes))
        raise HTTPException(413, "PDF must be smaller than 5MB")

    # Validation: non-empty job description
    job_description = job_description.strip()
    if not job_description:
        raise HTTPException(400, "Job description cannot be empty")

    # Truncate job description
    job_description = job_description[:MAX_JD_CHARS]

    # Extract → call AI → render
    cv_text = extract_text(file_bytes)
    result = call_ai(job_description, cv_text)

    return templates.TemplateResponse(
        "index.html",
        {"request": request, "result": result},
    )
