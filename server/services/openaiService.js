import fs from "fs";
import axios from "axios";
import dotenv from "dotenv";
dotenv.config();

const OPENAI_API_KEY = process.env.OPENAI_API_KEY;

export const parseAndScoreResume = async (resumePath, jobDescription) => {
  const resumeText = fs.readFileSync(resumePath, "utf8");

  const messages = [
    {
      role: "user",
      content: `
      Analyze this resume text and job description.
      1. Extract structured data (name, email, skills, education, experience).
      2. Score the candidate out of 100.
      3. Generate 5 interview questions tailored to the resume & job.

      Return a JSON like:
      {
        "structuredResume": {...},
        "score": {"overall_score": 78},
        "interviewQuestions": [{"question": "..."}, ...]
      }

      Resume:
      """${resumeText}"""

      Job Description:
      """${jobDescription}"""
      `,
    },
  ];

  const response = await axios.post(
    "https://api.openai.com/v1/chat/completions",
    {
      model: "gpt-4o-mini",
      messages,
    },
    {
      headers: { Authorization: `Bearer ${OPENAI_API_KEY}` },
    }
  );

  const output = response.data.choices[0].message.content;
  return JSON.parse(output);
};
