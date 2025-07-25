// backend/index.js
import express from "express";
import multer from "multer";
import fetch from "node-fetch";
import path from "path";

const app = express();
const upload = multer({ dest: "uploads/" });
const PORT = process.env.PORT || 4000;

// Assume Python microservice URL on Render
const PYTHON_MICROSERVICE_URL = "https://your-python-service.onrender.com";

app.post("/api/analyze", upload.single("media"), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: "Media file required" });
  }

  try {
    const formData = new FormData();
    // Readfile and append to formData
    const fs = await import("fs");
    const fileStream = fs.createReadStream(req.file.path);
    formData.append("media", fileStream, req.file.originalname);

    // Forward to python microservice
    const response = await fetch(`${PYTHON_MICROSERVICE_URL}/analyze`, {
      method: "POST",
      body: formData,
    });
    const data = await response.json();

    // Delete file after processing
    fs.unlink(req.file.path, () => {});

    res.json(data);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Internal Server Error" });
  }
});

// Start server
app.listen(PORT, () => {
  console.log(`Backend server running on port ${PORT}`);
});
