"""
Compares our fine-tuned model against Gemini on 15 resumes it has never seen
during training. Goal is to check whether fine-tuning actually buys us
anything over just prompting a general model.

Saves results to results/baseline_comparison.json and prints a table.
"""

import os
import json
import re
import time
from pathlib import Path

ADAPTER_PATH = "training/output"
MAX_SEQ_LENGTH = 2048
GEMINI_MODEL = "gemini-flash-latest"
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

SYSTEM_MESSAGE = "You are an expert HR recruiter and ATS system."
INSTRUCTION =  "Analyze this resume against the job description."

# Held-out test cases, not part of data/dataset.jsonl. Mix of easy fits,
# easy mismatches, and a few genuinely ambiguous ones so the comparison
# actually means something.
TEST_CASES = [
    {
        "id": 1,
        "resume": "Data analyst with 2 years experience in SQL, Excel, and Tableau. "
                  "Built dashboards for sales reporting. No Python or cloud experience.",
        "jd": "Looking for a Data Scientist with strong Python, SQL, and machine learning "
              "skills. AWS experience preferred. 3+ years experience required.",
    },
    {
        "id": 2,
        "resume": "Full-stack developer, 4 years experience. React, Node.js, PostgreSQL, "
                  "Docker. Led a team of 3 on an e-commerce platform rebuild.",
        "jd": "Senior Full-Stack Engineer needed. React, Node.js, and cloud deployment "
              "experience (AWS/GCP) required. Leadership experience a plus.",
    },
    {
        "id": 3,
        "resume": "Recent CS graduate. Coursework in data structures, algorithms, and "
                  "operating systems. One internship doing QA testing. No industry ML experience.",
        "jd": "Machine Learning Engineer, entry-level. Strong Python and ML fundamentals "
              "required. Production ML experience is a plus but not mandatory for the right candidate.",
    },
    {
        "id": 4,
        "resume": "DevOps engineer, 5 years. Kubernetes, Terraform, AWS, CI/CD pipelines. "
                  "Managed infrastructure for a 50-person engineering org.",
        "jd": "Backend Software Engineer position. Java, Spring Boot, microservices "
              "architecture. Kubernetes experience is a bonus.",
    },
    {
        "id": 5,
        "resume": "Marketing coordinator with 3 years experience in social media campaigns "
                  "and content strategy. Basic Excel skills. No technical background.",
        "jd": "Product Manager role. Requires technical background, experience working "
              "with engineering teams, and data-driven decision making skills.",
    },
    {
        "id": 6,
        "resume": "Senior backend engineer, 6 years. Python, Django, PostgreSQL, Redis, "
                  "AWS (EC2, S3, RDS). Built and scaled a payments API handling 2M requests/day.",
        "jd": "Backend Engineer, Python/Django, AWS experience required. 4+ years experience. "
              "Payments or fintech domain experience strongly preferred.",
    },
    {
        "id": 7,
        "resume": "Graphic designer, 5 years experience. Adobe Photoshop, Illustrator, "
                  "InDesign. Branding and print design for retail clients.",
        "jd": "Frontend Developer position. React, TypeScript, CSS required. "
              "UI/UX sensibility a plus.",
    },
    {
        "id": 8,
        "resume": "Mechanical engineer, 3 years, AutoCAD and SolidWorks. Worked on HVAC "
                  "system design. Some Python scripting for automating design calculations.",
        "jd": "Data Engineer role. Python, SQL, Airflow, and cloud data warehousing "
              "(Snowflake/BigQuery) required. 2+ years experience.",
    },
    {
        "id": 9,
        "resume": "ML researcher, PhD, 2 published papers on transformer architectures. "
                  "PyTorch, distributed training experience. No production deployment experience.",
        "jd": "Applied Scientist role. Strong research background in deep learning required. "
              "Ability to publish and prototype novel architectures valued over production experience.",
    },
    {
        "id": 10,
        "resume": "Customer support representative, 4 years. Zendesk, basic SQL for "
                  "pulling reports. No coding background.",
        "jd": "Software Engineer, backend focus. Requires strong programming fundamentals "
              "in at least one language (Python, Java, or Go) and CS degree or equivalent experience.",
    },
    {
        "id": 11,
        "resume": "DevOps/SRE, 7 years. Kubernetes, Terraform, Prometheus, Grafana, "
                  "on-call rotation experience, incident response lead for a 20-person team.",
        "jd": "Site Reliability Engineer. Kubernetes and observability tooling required. "
              "On-call and incident management experience essential. 5+ years.",
    },
    {
        "id": 12,
        "resume": "Junior developer, 1 year experience. JavaScript, basic React. Built "
                  "two small personal projects. No professional team experience.",
        "jd": "Frontend Engineer, mid-level. React, TypeScript, 3+ years professional "
              "experience, comfortable owning features independently.",
    },
    {
        "id": 13,
        "resume": "Business analyst, 5 years. SQL, Power BI, stakeholder management, "
                  "requirements gathering. Led migration of legacy reporting to Power BI.",
        "jd": "Data Analyst position. SQL and BI tool experience (Power BI or Tableau) "
              "required. Strong communication skills with non-technical stakeholders valued.",
    },
    {
        "id": 14,
        "resume": "iOS developer, 4 years. Swift, SwiftUI, published 3 apps on the App Store, "
                  "one with 100K+ downloads. No backend experience.",
        "jd": "Mobile Engineer (iOS). Swift/SwiftUI required, 3+ years, App Store publishing "
              "experience preferred. Backend/API integration experience a plus, not required.",
    },
    {
        "id": 15,
        "resume": "Sales executive, 6 years, enterprise SaaS sales, exceeded quota 4 years "
                  "running. No technical background whatsoever.",
        "jd": "Machine Learning Engineer. Strong Python and ML framework experience "
              "(PyTorch/TensorFlow) required. 3+ years hands-on ML experience.",
    },
]


def extract_fields(text):
    """Pulls score/strengths/gaps/questions out of the model's free-text output."""
    result = {}

    score_match = re.search(r"Match Score:\s*(\d+)\s*/\s*100", text)
    if not score_match:
        result["error"] = "no_score_found"
        result["raw"] = text[:1000]
        return result
    result["match_score"] = int(score_match.group(1))

    def extract_section(name, stop_at):
        pattern = rf"{name}:\s*(.*?)(?=(?:{'|'.join(stop_at)})|\Z)"
        m = re.search(pattern, text, re.DOTALL)
        if not m:
            return []
        items = re.findall(r"\d+\.\s*(.+)", m.group(1))
        return [i.strip() for i in items]

    result["strengths"] = extract_section("Strengths", ["Gaps", "Top Interview Questions"])
    result["gaps"] = extract_section("Gaps", ["Top Interview Questions"])
    result["interview_questions"] = extract_section("Top Interview Questions", [])
    result["raw_length"] = len(text)
    return result


def load_local_model():
    from unsloth import FastLanguageModel

    print("Loading fine-tuned model + adapter...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=ADAPTER_PATH,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def query_local_model(model, tokenizer, resume, jd):
    user_content = f"{INSTRUCTION}\n\nResume: {resume}\n\nJob Description: {jd}"
    chat = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{SYSTEM_MESSAGE}<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{user_content}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""
    inputs = tokenizer(chat, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=600, temperature=0.3, do_sample=False)
    text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return extract_fields(text)


GEMINI_PROMPT_TEMPLATE = """{system}

{instruction}

Respond in EXACTLY this format (no extra commentary, no markdown, no JSON):

Match Score: X/100
Strengths:
1. ...
2. ...
Gaps:
1. ...
2. ...
Top Interview Questions:
1. ...
2. ...

Resume: {resume}

Job Description: {jd}"""


def query_gemini(client, resume, jd, max_retries=5):
    # Free tier caps gemini-2.5-flash at 5 requests/minute, and it still
    # throws 503s under load sometimes. Backing off and retrying handles both.
    prompt = GEMINI_PROMPT_TEMPLATE.format(
        system=SYSTEM_MESSAGE, instruction=INSTRUCTION, resume=resume, jd=jd
    )

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
            return extract_fields(response.text)
        except Exception as e:
            if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e) or "UNAVAILABLE" in str(e) or "503" in str(e):
                wait = 20 * (attempt + 1)
                print(f"  rate limited, waiting {wait}s (retry {attempt + 1}/{max_retries})")
                time.sleep(wait)
            else:
                return {"error": "gemini_api_error", "raw": str(e)[:300]}

    return {"error": "rate_limited_after_retries"}


def main():
    from google import genai

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set.")

    gemini_client = genai.Client(api_key=api_key)
    local_model, tokenizer = load_local_model()

    out_path = RESULTS_DIR / "baseline_comparison.json"
    previous_results = {}
    if out_path.exists():
        with open(out_path) as f:
            for r in json.load(f):
                if "error" not in r.get("gemini", {"error": True}):
                    previous_results[r["id"]] = r

    results = []
    print(f"\nRunning {len(TEST_CASES)} test cases through both models...\n")

    for case in TEST_CASES:
        if case["id"] in previous_results:
            print(f"case {case['id']}: reusing previous result")
            results.append(previous_results[case["id"]])
            continue

        print(f"case {case['id']}...")
        local_result = query_local_model(local_model, tokenizer, case["resume"], case["jd"])

        time.sleep(20)
        gemini_result = query_gemini(gemini_client, case["resume"], case["jd"])

        results.append({
            "id": case["id"],
            "resume": case["resume"],
            "jd": case["jd"],
            "local_model": local_result,
            "gemini": gemini_result,
        })

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 90)
    print(f"{'ID':<4}{'Your Model Score':<20}{'Gemini Score':<15}{'Your Fmt OK':<14}{'Gemini Fmt OK':<14}")
    print("=" * 90)
    for r in results:
        local_score = r["local_model"].get("match_score", "ERR")
        gemini_score = r["gemini"].get("match_score", "ERR")
        local_ok = "error" not in r["local_model"]
        gemini_ok = "error" not in r["gemini"]
        print(f"{r['id']:<4}{str(local_score):<20}{str(gemini_score):<15}{str(local_ok):<14}{str(gemini_ok):<14}")
    print("=" * 90)
    print(f"\nfull results saved to {out_path}")


if __name__ == "__main__":
    main()