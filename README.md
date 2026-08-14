# RAG Resume Analyzer

A Streamlit application that compares a resume against a job description and highlights the relevant skills gap using a lightweight retrieval-augmented workflow.

## What the app does

1. Uploads a resume in PDF, DOCX, or TXT format.
2. Extracts the text from the uploaded document.
3. Splits the resume into chunks for focused matching.
4. Compares the job description with the resume chunks using TF-IDF and cosine similarity.
5. Calculates a job-readiness score based on the similarity results.
6. Sends the most relevant resume sections to Gemini for a structured analysis.
7. Provides:
   - required skills
   - current skills
   - missing skills
   - improvement suggestions
   - suitable job roles

## Current app flow

```text
Resume upload
   ↓
Text extraction
   ↓
Chunking
   ↓
TF-IDF vectorization
   ↓
Cosine similarity search
   ↓
Job readiness score
   ↓
Gemini prompt analysis
```

## Project structure

```text
rag_resume_analyzer/
├── app.py
├── gemini_client.py
├── resume_reader.py
├── resume_retriever.py
├── README.md
├── requirements.txt
├── .gitignore
├── sample_job_description.txt
├── sample_resumes/
│   └── sample_resume.txt
├── .streamlit/
│   └── secrets.toml
└── .venv/
```

## Setup

### 1. Open the project folder

```powershell
cd path\to\rag_resume_analyzer
```

### 2. Create and activate a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Add your Gemini API key

This app uses Streamlit secrets, so create the file below:

```text
.streamlit/secrets.toml
```

Then add:

```toml
GEMINI_API_KEY = "your_api_key_here"
```

Do not commit real API keys to GitHub.

### 5. Run the app

```powershell
streamlit run app.py
```

Open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## Notes

- The job-readiness score is a similarity-based estimate, not a professional hiring metric.
- The analysis is intended to help identify gaps and opportunities, not to make hiring decisions automatically.
- The app uses the most relevant resume chunks only to keep the prompt grounded and reduce repeated content.
