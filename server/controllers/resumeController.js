import { parseAndScoreResume } from "../services/openaiService.js";

export const analyzeResume = async (req, res) => {
  try {
    const jobDesc = req.body.jobDescription;
    const resumePath = req.file.path;

    const result = await parseAndScoreResume(resumePath, jobDesc);
    res.json(result);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
};
