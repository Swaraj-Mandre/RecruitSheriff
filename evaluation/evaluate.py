"""
Numeric evaluation: format compliance, score consistency, and a
hallucination/grounding check on the fine-tuned model. Reuses the
15 test cases from evaluation/baseline_compare.py.

Output: results/evaluation_report.json + printed summary.
"""

import json
import re
import statistics
from pathlib import Path

from evaluation.baseline_compare import (
    TEST_CASES,
    SYSTEM_MESSAGE,
    INSTRUCTION,
    extract_fields,
)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
ADAPTER_PATH = "training/output"
MAX_SEQ_LENGTH = 2048
CONSISTENCY_RUNS = 3
MAX_GENERATION_RETRIES = 2

GROUNDING_STOPWORDS = {
    "the", "and", "for", "with", "not", "are", "was", "has", "have", "in",
    "to", "is", "of", "an", "as", "at", "on", "or", "but", "no",
    "experience", "background", "technical", "skills", "knowledge",
    "role", "job", "required", "requires", "missing", "lack", "lacking",
    "clear", "clearly", "specific", "specifically", "mentioned", "resume",
    "does", "any", "some", "none", "this", "that", "from", "into",
}

NEGATION_CUES = {"no", "not", "none", "lack", "lacking", "without", "n/a", "zero"}


def load_local_model():
    from unsloth import FastLanguageModel

    print(f"Loading model from {ADAPTER_PATH}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=ADAPTER_PATH,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def _run_chat(model, tokenizer, resume, jd, do_sample, temperature):
    user_content = f"{INSTRUCTION}\n\nResume: {resume}\n\nJob Description: {jd}"
    chat = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{SYSTEM_MESSAGE}<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{user_content}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""
    inputs = tokenizer(chat, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs, max_new_tokens=600, temperature=temperature, do_sample=do_sample
    )
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def generate_with_retry(model, tokenizer, resume, jd, do_sample=False, temperature=0.3):
    """Retries a couple of times if the model's output doesn't parse cleanly."""
    last_text = ""
    for attempt in range(MAX_GENERATION_RETRIES + 1):
        text = _run_chat(model, tokenizer, resume, jd, do_sample, temperature)
        last_text = text
        parsed = extract_fields(text)
        if "error" not in parsed:
            return text, parsed, attempt
    return last_text, extract_fields(last_text), MAX_GENERATION_RETRIES


def _clean_claim_words(text):
    """Turns a claim like 'Cloud deployment experience (matches JD)' into
    the meaningful words worth checking, dropping filler and punctuation."""
    words = [w.strip(".,()") for w in text.lower().split()]
    return [w for w in words if len(w) >= 3 and w not in GROUNDING_STOPWORDS]


def _word_in_resume(word, resume_lower):
    """Checks if a word appears in the resume. Also matches close variants
    like 'developer' vs 'development' by comparing the first 6 letters."""
    if word in resume_lower:
        return True
    if len(word) >= 6:
        return word[:6] in resume_lower
    return False


def _resume_confirms_gap(claim_words, resume_lower):
    """True if the resume itself states the gap (e.g. resume says 'No cloud
    experience' and the model's gap says 'Lacks cloud experience'). Plain
    word overlap can't tell 'resume mentions X' apart from 'resume says X is
    missing', so this checks each resume sentence for a negation word next
    to the claim's topic word."""
    for sentence in re.split(r"[.\n]", resume_lower):
        sentence_words = sentence.split()
        has_negation = any(cue in sentence_words for cue in NEGATION_CUES)
        has_topic = any(w in sentence for w in claim_words)
        if has_negation and has_topic:
            return True
    return False


def check_hallucination(resume, strengths):
    resume_lower = resume.lower()
    ungrounded = []
    real_claims = 0

    for s in strengths:
        if s.strip().lower() in ("none", "n/a", "none.", "none identified"):
            continue
        real_claims += 1
        words = _clean_claim_words(s)
        if len(words) < 2:
            continue
        matches = sum(1 for w in words if _word_in_resume(w, resume_lower))
        if matches / len(words) < 0.3:
            ungrounded.append(s)

    return {
        "total_strengths_claimed": real_claims,
        "ungrounded_count": len(ungrounded),
        "ungrounded_examples": ungrounded,
    }


def check_gaps_grounding(resume, gaps):
    resume_lower = resume.lower()
    contradicted = []
    real_claims = 0

    for g in gaps:
        if g.strip().lower() in ("none", "n/a", "none.", "none identified"):
            continue
        real_claims += 1
        words = _clean_claim_words(g)
        if len(words) < 2:
            continue
        matches = sum(1 for w in words if _word_in_resume(w, resume_lower))
        if matches / len(words) > 0.85 and not _resume_confirms_gap(words, resume_lower):
            contradicted.append(g)

    return {
        "total_gaps_claimed": real_claims,
        "contradicted_count": len(contradicted),
        "contradicted_examples": contradicted,
    }


def main():
    model, tokenizer = load_local_model()

    format_results = []
    hallucination_results = []
    gaps_results = []
    retry_counts = []

    print(f"\n--- Format Compliance + Grounding Check ({len(TEST_CASES)} cases) ---\n")
    for case in TEST_CASES:
        print(f"case {case['id']}...")
        text, parsed, retries_used = generate_with_retry(model, tokenizer, case["resume"], case["jd"])
        retry_counts.append(retries_used)

        format_ok = "error" not in parsed
        format_results.append({"id": case["id"], "format_ok": format_ok, "retries_used": retries_used})

        if format_ok:
            halluc = check_hallucination(case["resume"], parsed.get("strengths", []))
            halluc["id"] = case["id"]
            hallucination_results.append(halluc)

            gaps_check = check_gaps_grounding(case["resume"], parsed.get("gaps", []))
            gaps_check["id"] = case["id"]
            gaps_results.append(gaps_check)
        else:
            print(f"  still failed to parse after {retries_used} retries")

    consistency_results = []
    print(f"\n--- Score Consistency Check (first 5 cases, {CONSISTENCY_RUNS} sampled runs each) ---\n")
    for case in TEST_CASES[:5]:
        scores = []
        for run in range(CONSISTENCY_RUNS):
            _, parsed, _ = generate_with_retry(
                model, tokenizer, case["resume"], case["jd"], do_sample=True, temperature=0.7
            )
            if "match_score" in parsed:
                scores.append(parsed["match_score"])
        if len(scores) >= 2:
            consistency_results.append({
                "id": case["id"],
                "scores": scores,
                "stdev": round(statistics.stdev(scores), 2),
            })
        print(f"case {case['id']}: scores = {scores}")

    format_compliance_rate = sum(r["format_ok"] for r in format_results) / len(format_results) * 100
    retry_rate = sum(1 for r in retry_counts if r > 0) / len(retry_counts) * 100

    total_strengths = sum(h["total_strengths_claimed"] for h in hallucination_results)
    total_ungrounded = sum(h["ungrounded_count"] for h in hallucination_results)
    hallucination_rate = (total_ungrounded / total_strengths * 100) if total_strengths else 0

    total_gaps = sum(g["total_gaps_claimed"] for g in gaps_results)
    total_contradicted = sum(g["contradicted_count"] for g in gaps_results)
    gaps_contradiction_rate = (total_contradicted / total_gaps * 100) if total_gaps else 0

    avg_stdev = (
        statistics.mean(c["stdev"] for c in consistency_results) if consistency_results else 0
    )

    report = {
        "format_compliance_rate_pct": round(format_compliance_rate, 1),
        "first_try_retry_rate_pct": round(retry_rate, 1),
        "hallucination_rate_pct": round(hallucination_rate, 1),
        "gaps_contradiction_rate_pct": round(gaps_contradiction_rate, 1),
        "avg_score_stdev": round(avg_stdev, 2),
        "format_results": format_results,
        "consistency_results": consistency_results,
        "hallucination_results": hallucination_results,
        "gaps_results": gaps_results,
    }

    out_path = RESULTS_DIR / "evaluation_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Format compliance rate (with retry) : {format_compliance_rate:.1f}%  ({sum(r['format_ok'] for r in format_results)}/{len(format_results)} cases)")
    print(f"Cases that needed a retry            : {retry_rate:.1f}%")
    print(f"Hallucination rate (strengths)        : {hallucination_rate:.1f}%  ({total_ungrounded}/{total_strengths} claimed strengths ungrounded)")
    print(f"Gaps contradiction rate               : {gaps_contradiction_rate:.1f}%  ({total_contradicted}/{total_gaps} claimed gaps possibly wrong)")
    print(f"Avg score stdev (consistency)         : {avg_stdev:.2f}")
    print("=" * 60)
    print(f"\nFull report saved to: {out_path}")


if __name__ == "__main__":
    main()