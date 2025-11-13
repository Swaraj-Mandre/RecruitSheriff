// server/index.js
import express from "express";
import multer from "multer";
import cors from "cors";
import fs from "fs";
import axios from "axios";
import dotenv from "dotenv";
import { createRequire } from "module";
const require = createRequire(import.meta.url);

const pdfParse = require("pdf-parse");
const mammoth = require("mammoth");

dotenv.config();

const app = express();
app.use(cors());
app.use(express.json());

const upload = multer({ dest: "uploads/" });

// ============= Multiple Resume Analyzer =============
app.post("/api/resume", upload.array("resumeFiles", 10), async (req, res) => {
  try {
    const jobDescription = req.body.jobDescription || "";
    const files = req.files || [];
    const results = [];

    console.log(`📄 Received ${files.length} resumes for analysis`);

    for (const file of files) {
      const fileName = file.originalname;
      const filePath = file.path;
      let resumeText = "";

      try {
        if (fileName.endsWith(".pdf")) {
          const pdfData = await pdfParse(fs.readFileSync(filePath));
          resumeText = pdfData.text;
        } else if (fileName.endsWith(".docx")) {
          const docxData = await mammoth.extractRawText({ path: filePath });
          resumeText = docxData.value;
        } else {
          resumeText = fs.readFileSync(filePath, "utf8");
        }
      } catch (err) {
        console.error(`❌ Error reading ${fileName}:`, err.message);
        resumeText = "Unable to read resume text.";
      }

      const prompt = `
You are an AI hiring assistant.
Given the RESUME and JOB DESCRIPTION, do the following:
1. Evaluate how well the resume matches the job.
2. Provide an overall ATS score (0–100).
3. Suggest 5 specific interview questions for this candidate.
Return ONLY valid JSON and nothing else in this format:
{
  "score": {"overall_score": <number>},
  "interviewQuestions": [
    {"question": "<Q1>"},
    {"question": "<Q2>"},
    {"question": "<Q3>"},
    {"question": "<Q4>"},
    {"question": "<Q5>"}
  ]
}
RESUME:
"""${resumeText.slice(0, 5000)}"""
JOB DESCRIPTION:
"""${jobDescription}"""
`;

      const groqResponse = await axios({
        method: "post",
        url: "https://api.groq.com/openai/v1/chat/completions",
        headers: {
          Authorization: `Bearer ${process.env.GROQ_API_KEY}`,
          "Content-Type": "application/json",
        },
        data: {
          model: "llama-3.1-8b-instant",
          messages: [
            {
              role: "system",
              content:
                "You are an expert ATS evaluator and interviewer AI. Always return valid JSON only.",
            },
            { role: "user", content: prompt },
          ],
          temperature: 0.6,
          max_tokens: 1200,
        },
      });

      const rawOutput = groqResponse.data.choices[0].message.content;
      let parsed;
      try {
        const cleanText = rawOutput
          .replace(/```json/g, "")
          .replace(/```/g, "")
          .trim();
        const match = cleanText.match(/\{[\s\S]*\}/);
        parsed = match ? JSON.parse(match[0]) : null;
      } catch {
        parsed = null;
      }

      results.push({
        file: fileName,
        score: parsed?.score?.overall_score || 0,
        interviewQuestions:
          parsed?.interviewQuestions?.map((q) => q.question) || [],
      });
    }

    res.json({ results });
  } catch (err) {
    console.error("❌ Error:", err.response?.data || err.message);
    res.status(500).json({ error: err.response?.data || err.message });
  }
});

app.listen(8080, () =>
  console.log("✅ Multi Resume Analyzer running on port 8080")
);
