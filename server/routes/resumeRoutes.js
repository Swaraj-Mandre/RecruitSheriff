import express from "express";
import multer from "multer";
import { analyzeResume } from "../controllers/resumeController.js";

const router = express.Router();
const upload = multer({ dest: "uploads/" });

router.post("/", upload.single("resumeFile"), analyzeResume);

export default router;
