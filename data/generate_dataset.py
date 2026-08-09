import json
import random #shuffle pairs
import requests #HTTP POST request to Ollama local API
import os

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2:3b"
OUTPUT_FILE = "data/dataset.jsonl"
NUM_EXAMPLES = 600

resumes = [
    "Python developer with 2 years experience in Flask, REST APIs, and PostgreSQL.",
    "Java backend engineer with 4 years experience in Spring Boot, Hibernate, and MySQL.",
    "Data analyst with 3 years experience in Python, SQL, Pandas, and Tableau.",
    "Frontend developer with 2 years experience in React, TypeScript, and CSS.",
    "ML engineer with 1 year experience in scikit-learn, Pandas, and model deployment.",
    "DevOps engineer with 5 years experience in Docker, Kubernetes, Jenkins, and AWS.",
    "Full stack developer with 3 years experience in Node.js, React, and MongoDB.",
    "Android developer with 2 years experience in Kotlin, Jetpack Compose, and REST APIs.",
    "Data scientist with 4 years experience in Python, TensorFlow, NLP, and Jupyter.",
    "Cloud engineer with 3 years experience in AWS, Terraform, and CI/CD pipelines.",
    "Cybersecurity analyst with 2 years experience in penetration testing and SIEM tools.",
    "Backend developer with 3 years experience in Django, Celery, and Redis.",
    "iOS developer with 2 years experience in Swift, UIKit, and CoreData.",
    "Data engineer with 4 years experience in Apache Spark, Airflow, and BigQuery.",
    "AI researcher with 2 years experience in PyTorch, transformers, and fine-tuning LLMs.",
    "Software engineer with 1 year experience in Go, microservices, and gRPC.",
    "QA engineer with 3 years experience in Selenium, pytest, and CI/CD testing.",
    "Database administrator with 5 years experience in Oracle, PostgreSQL, and query optimization.",
    "Computer vision engineer with 2 years experience in OpenCV, YOLO, and Python.",
    "NLP engineer with 3 years experience in HuggingFace, BERT, and text classification.",
]

job_descriptions = [
    "Backend Python engineer. Django or Flask required. PostgreSQL and Docker experience preferred.",
    "Java developer. Spring Boot mandatory. AWS and microservices experience is a plus.",
    "Data analyst role. SQL and Python required. Power BI or Tableau experience preferred.",
    "Frontend engineer. Strong React and TypeScript skills required. Next.js is a plus.",
    "Machine learning engineer. Python and scikit-learn required. MLflow and deployment experience preferred.",
    "DevOps engineer. Docker and Kubernetes required. AWS certified preferred.",
    "Full stack developer. Node.js backend and React frontend required. GraphQL is a plus.",
    "Android developer. Kotlin required. Experience with MVVM architecture and REST APIs needed.",
    "Data scientist. NLP and deep learning experience required. PyTorch preferred.",
    "Cloud engineer. AWS certified preferred. Terraform and infrastructure-as-code required.",
    "Security engineer. Penetration testing experience required. Knowledge of OWASP top 10 needed.",
    "Senior backend developer. Django and Celery required. Redis and message queues experience needed.",
    "iOS developer. Swift required. SwiftUI experience preferred. App Store deployment experience needed.",
    "Data engineer. Apache Spark and Airflow required. Experience with cloud data warehouses preferred.",
    "LLM engineer. Experience fine-tuning open-source models required. HuggingFace and PEFT knowledge needed.",
    "Backend Go engineer. Microservices and gRPC required. Kubernetes deployment experience preferred.",
    "QA automation engineer. Selenium and pytest required. CI/CD pipeline experience needed.",
    "Senior DBA. PostgreSQL and query optimization required. Experience with replication and backups needed.",
    "Computer vision engineer. YOLO and OpenCV required. Real-time inference optimization experience preferred.",
    "NLP engineer. HuggingFace transformers required. Experience with text classification and NER needed.",
]

PROMPT_TEMPLATE = """You are a senior HR recruiter and ATS system with 10 years of experience.

Analyze the resume against the job description below. Respond in EXACTLY this format with no extra text:

Match Score: [number]/100
Strengths:
1. [specific strength from resume that matches JD]
2. [specific strength from resume that matches JD]
3. [specific strength from resume that matches JD]
Gaps:
1. [specific skill or experience missing from resume but required in JD]
2. [specific skill or experience missing from resume but required in JD]
3. [specific skill or experience missing from resume but required in JD]
Top Interview Questions:
1. [question targeting a gap or testing a claimed skill]
2. [question targeting a gap or testing a claimed skill]
3. [question targeting a gap or testing a claimed skill]

Resume: {resume}

Job Description: {jd}"""


def generate_example(resume, jd):
    prompt = PROMPT_TEMPLATE.format(resume=resume, jd=jd)
    response = requests.post(OLLAMA_URL, json={
        "model": MODEL,
        "prompt": prompt,
        "stream": False
    })
    result = response.json()
    return result["response"].strip()


def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    examples = []
    print(f"Generating {NUM_EXAMPLES} examples...")

    for i in range(NUM_EXAMPLES):
        resume = random.choice(resumes)
        jd = random.choice(job_descriptions)
        output = generate_example(resume, jd)

        entry = {
            "instruction": "Analyze this resume against the job description.",
            "input": f"Resume: {resume}\n\nJob Description: {jd}",
            "output": output
        }
        examples.append(entry)

        if (i + 1) % 20 == 0:
            print(f"Generated {i + 1}/{NUM_EXAMPLES} examples")

    with open(OUTPUT_FILE, "w") as f:
        for entry in examples:
            f.write(json.dumps(entry) + "\n")

    print(f"Done. Dataset saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()