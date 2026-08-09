# RecruitSheriff

A fine-tuned LLaMA 3.2 3B model that scores how well a resume matches a job description. Upload a resume (PDF or DOCX) and paste a job description to get a match score, strengths, gaps, and suggested interview questions, generated entirely by a model trained specifically for this task, not a general-purpose LLM prompted on the fly.

**Model on Hugging Face Hub:** [huggingface.co/SomkeX/recruitsheriff](https://huggingface.co/SomkeX/recruitsheriff)

<table>
  <tr>
    <td><img width="480" alt="Resume upload and job description input" src="https://github.com/user-attachments/assets/8ccc4fb9-9921-46e8-b995-ae2a0bcf3b1e" /></td>
    <td><img width="480" alt="Match score, strengths, and gaps result" src="https://github.com/user-attachments/assets/23a1115a-14c3-4868-99b9-ec920bc18f84" /></td>
  </tr>
</table>

## Why this exists

Most resume-scoring tools are thin wrappers around a general cloud LLM. Send a resume, get whatever GPT or Gemini says back. The output format is inconsistent, the reasoning is generic, and there's no real model ownership behind it.

RecruitSheriff fine-tunes an open-source model specifically on resume-to-job-description scoring, so the output format and reasoning behavior are baked into the model's weights instead of held together by prompt engineering. Inference runs on the fine-tuned model directly, with no calls to a third-party LLM API at request time.

---

## What it actually does

1. You upload a resume (PDF or DOCX) and paste a job description.
2. The backend extracts the resume text (`pdfplumber` for PDF, `python-docx` for DOCX).
3. The fine-tuned model generates a structured analysis: a 0 to 100 match score, strengths grounded in the resume, gaps relative to the job description, and interview questions.
4. If a generation doesn't parse into the expected format, it's retried automatically (up to 2 times) before failing. The deployed model hit 100% format compliance across 15 held-out test cases in the most recent evaluation run.

---

## Tech stack

| Layer | Tool |
|---|---|
| Base model | LLaMA 3.2 3B Instruct |
| Fine-tuning | QLoRA via Unsloth (SFT) |
| Training data | 600 resume/JD pairs, generated locally via Ollama |
| Backend | FastAPI |
| PDF parsing | pdfplumber |
| DOCX parsing | python-docx |
| Frontend | HTML, CSS, vanilla JS |
| Model hosting | Hugging Face Hub |

---

## How it was trained

The model was fine-tuned with QLoRA (4-bit quantization plus LoRA adapters) on 600 resume/job-description examples, using [Unsloth](https://github.com/unslothai/unsloth) for faster, lower-memory training. Training data was generated locally with Ollama running LLaMA 3.2 3B, keeping the whole pipeline free of external API costs.

- **Trainable parameters:** 24.3M of 3.24B (0.75%)
- **Training run:** 3 epochs, 225 steps, about 7m43s on an RTX 4060 (8GB VRAM laptop GPU)
- **Final training loss:** 0.2444

Supervised fine-tuning was the focus for this version. Preference optimization (DPO) is a natural next step and is on the roadmap once more preference-labeled data is available to make it worthwhile.

---

## Evaluation

Run against 15 held-out resume/JD pairs not seen during training:

| Metric | Result |
|---|---|
| Format compliance (with retry) | 100% (15/15) |
| Gaps grounded in resume (no contradictions) | 100% (0/45 flagged) |
| Strengths grounded in resume | 86.4% (19/22, see note below) |
| Score consistency (stdev across sampled runs) | about 5.8 points |

**Note on the strengths metric:** the automated grounding checker flags a strength as "ungrounded" using word overlap against the resume text. Manual review of the flagged cases found most were checker limitations (for example, "leadership" not string-matching "led a team"), not actual model errors. One case was a genuine issue: the model stated a JD requirement ("cloud deployment experience") as a resume strength when the resume didn't mention it. That's kept in the numbers as an honest, documented limitation rather than smoothed over.

**Baseline comparison:** against Gemini on the same 15 cases, the fine-tuned model tracks closely on clear mismatches but is measurably more conservative on strong-fit candidates (for example, scoring 60 to 80 where Gemini scores 92 to 95). This likely reflects the training data's own scoring tendencies. Full comparison in [`results/baseline_comparison.json`](results/baseline_comparison.json).

---

## Project structure

```
RecruitSheriff/
├── app/
│   ├── main.py              # FastAPI backend, POST /analyze, GET /health
│   ├── static/index.html    # Frontend
│   └── requirements.txt     # Runtime dependencies only
├── data/
│   ├── generate_dataset.py  # Dataset generation via local Ollama
│   └── dataset.jsonl        # 600 training examples
├── training/
│   ├── finetune.py          # QLoRA fine-tuning via Unsloth
│   └── output/               # LoRA adapter (weights hosted on HF Hub, not in repo)
├── evaluation/
│   ├── evaluate.py           # Format compliance, grounding, consistency checks
│   └── baseline_compare.py   # Comparison against Gemini
├── results/                   # Evaluation output (JSON)
└── requirements-training.txt  # Full training and evaluation dependencies
```

---

## Running it locally

Requires Python 3.11 and an NVIDIA GPU with CUDA. The model loads in 4-bit via `bitsandbytes`, which requires CUDA, so CPU-only inference isn't currently supported.

```bash
git clone https://github.com/Swaraj-Mandre/RecruitSheriff.git
cd RecruitSheriff
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r app/requirements.txt

python -m uvicorn app.main:app --port 8000
```

Open `http://127.0.0.1:8000`, upload a resume, paste a job description, and click analyze.

The app loads the adapter from `training/output/` on disk by default. The same adapter is also published on [Hugging Face Hub](https://huggingface.co/SomkeX/recruitsheriff) if you want to load it independently of this repo.

To retrain or run the evaluation suite, install `requirements-training.txt` instead. It includes `trl`, `datasets`, and the other training-only dependencies that aren't needed just to run the app.

---

## Limitations

- **No live deployment yet.** Free CPU hosting (Hugging Face Spaces) can't run this model. It was tested and confirmed to run out of memory at both full and 8-bit precision on a 16GB-class machine. A GPU-backed deployment is planned; in the meantime, the app runs locally with the steps above.
- **No OCR support.** Scanned or image-based PDFs return a clear error instead of a result. The app checks for extractable text and fails honestly rather than guessing.
- **Score consistency:** sampled (non-greedy) generation shows a standard deviation of about 5.8 points across repeated runs on the same input, so scores are best read as an estimate rather than an exact number.