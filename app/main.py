"""
RecruitSheriff — FastAPI Backend
----------------------------------
POST /analyze : accepts a resume file (PDF or DOCX) + job description text,
                returns match score, strengths, gaps, and interview questions
                from the fine-tuned model.

Model is loaded ONCE at startup (not per-request) to avoid reload latency.
"""

import io
import re
from pathlib import Path

import pdfplumber
from docx import Document
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
ADAPTER_PATH = "training/output"
MAX_SEQ_LENGTH = 2048
SYSTEM_MESSAGE = "You are an expert HR recruiter and ATS system."
INSTRUCTION = "Analyze this resume against the job description."
MIN_TEXT_LENGTH = 50  # below this, we assume a scanned/image PDF with no extractable text

app = FastAPI(title="RecruitSheriff API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # fine for a portfolio demo; tighten if this ever needs real auth
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model handles, populated on startup
model = None
tokenizer = None


# ---------------------------------------------------------------------------
# MODEL LOADING (once, at startup)
# ---------------------------------------------------------------------------
@app.on_event("startup")
def load_model():
    global model, tokenizer
    from unsloth import FastLanguageModel

    print("Loading fine-tuned model + adapter...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=ADAPTER_PATH,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    print("Model loaded and ready.")


# ---------------------------------------------------------------------------
# TEXT EXTRACTION
# ---------------------------------------------------------------------------
def extract_text_from_pdf(file_bytes: bytes) -> str:
    text_parts = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n".join(text_parts).strip()


def extract_text_from_docx(file_bytes: bytes) -> str:
    doc = Document(io.BytesIO(file_bytes))
    return "\n".join(p.text for p in doc.paragraphs).strip()


def extract_resume_text(filename: str, file_bytes: bytes) -> str:
    ext = Path(filename).suffix.lower()

    if ext == ".pdf":
        text = extract_text_from_pdf(file_bytes)
    elif ext == ".docx":
        text = extract_text_from_docx(file_bytes)
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Please upload a PDF or DOCX file.",
        )

    if len(text) < MIN_TEXT_LENGTH:
        raise HTTPException(
            status_code=422,
            detail=(
                "This appears to be a scanned/image-based PDF with no extractable text. "
                "Please upload a text-based PDF or DOCX file. "
                "(OCR support for scanned resumes is planned for a future update.)"
            ),
        )

    return text


# ---------------------------------------------------------------------------
# MODEL INFERENCE
# ---------------------------------------------------------------------------
def extract_fields(text: str) -> dict:
    """Parses the model's trained output format into structured fields."""
    result = {}

    score_match = re.search(r"Match Score:\s*(\d+)\s*/\s*100", text)
    if not score_match:
        return {"error": "could_not_parse_model_output", "raw": text[:1000]}
    result["match_score"] = int(score_match.group(1))

    def extract_section(section_name, next_sections):
        pattern = rf"{section_name}:\s*(.*?)(?=(?:{'|'.join(next_sections)})|\Z)"
        m = re.search(pattern, text, re.DOTALL)
        if not m:
            return []
        items = re.findall(r"\d+\.\s*(.+)", m.group(1))
        return [i.strip() for i in items if i.strip().lower() not in ("none", "n/a")]

    result["strengths"] = extract_section("Strengths", ["Gaps", "Top Interview Questions"])
    result["gaps"] = extract_section("Gaps", ["Top Interview Questions"])
    result["interview_questions"] = extract_section("Top Interview Questions", [])
    return result


def run_model(resume_text: str, jd_text: str, max_retries: int = 2) -> dict:
    user_content = f"{INSTRUCTION}\n\nResume: {resume_text}\n\nJob Description: {jd_text}"
    chat = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{SYSTEM_MESSAGE}<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{user_content}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""
    inputs = tokenizer(chat, return_tensors="pt").to(model.device)

    for attempt in range(max_retries + 1):
        outputs = model.generate(**inputs, max_new_tokens=600, temperature=0.3, do_sample=False)
        generated_text = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        result = extract_fields(generated_text)
        if "error" not in result:
            return result

    return result


# ---------------------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------------------
@app.post("/analyze")
async def analyze(
    resume: UploadFile = File(...),
    job_description: str = Form(...),
):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model is still loading, please retry shortly.")

    if not job_description or len(job_description.strip()) < 20:
        raise HTTPException(status_code=400, detail="Job description is missing or too short.")

    file_bytes = await resume.read()
    resume_text = extract_resume_text(resume.filename, file_bytes)

    result = run_model(resume_text, job_description)

    if "error" in result:
        raise HTTPException(
            status_code=500,
            detail="Model output could not be parsed. Please try again.",
        )

    return result


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


# ---------------------------------------------------------------------------
# STATIC FRONTEND
# ---------------------------------------------------------------------------
app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/")
def serve_frontend():
    return FileResponse("app/static/index.html")