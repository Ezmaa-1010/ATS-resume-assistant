# 📄 ATS Resume Checker

A Streamlit app that scores a resume for ATS (Applicant Tracking System) friendliness
and gives specific, prioritized improvements, powered by Google's Gemini Flash model.

## Features
- Upload a resume as **PDF or DOCX**
- Optional **job description** input for targeted keyword matching
- **ATS score (0-100)** with a 5-category weighted breakdown
- Prioritized improvements (High / Medium / Low) with example rewrites
- Missing keywords, strengths, weaknesses, and section-by-section feedback
- Instant rule-based checks (email, phone, links, metrics, length)
- Shows the extracted text, i.e. what an ATS actually "sees"

## Project structure
```
├── app.py            # Streamlit app
├── requirements.txt  # Python dependencies
└── README.md
```

## Run locally
1. Get a free Gemini API key: https://aistudio.google.com/apikey
2. Install and run:
   ```bash
   python -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   streamlit run app.py
   ```
3. Provide the API key in **one** of these ways:
   - Paste it in the app sidebar, or
   - Set an environment variable: `export GEMINI_API_KEY="your-key"`, or
   - Create `.streamlit/secrets.toml` (never commit this file):
     ```toml
     GEMINI_API_KEY = "your-key"
     ```

## Deploy on Streamlit Community Cloud
1. Push this project to a GitHub repository.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app** → **Deploy a public app from GitHub**.
4. Choose your repo, branch `main`, and main file path `app.py`.
5. Open **Advanced settings → Secrets** and add:
   ```toml
   GEMINI_API_KEY = "your-key"
   ```
6. Click **Deploy**.

## How scoring works
Gemini rates five categories from 0-100; the final score is calculated in code so it is consistent:

| Category | Weight |
|---|---|
| Keywords & Relevance | 25% |
| Content & Impact | 25% |
| Formatting & Parseability | 20% |
| Structure & Sections | 15% |
| Language & Clarity | 15% |

## Limitations
- The score is an **AI estimate**, not the output of a real ATS (Workday, Greenhouse, Taleo, etc. each behave differently).
- Formatting is judged from extracted text, so visual layout problems may not be fully detected.
- Scanned/image-only resumes can't be read (which is also a problem for real ATS software).
- The resume text is sent to Google's Gemini API; don't upload anything you aren't comfortable sharing.
- To use a different model, change `MODEL_NAME` at the top of `app.py`.

## Troubleshooting
| Problem | Fix |
|---|---|
| "API key looks invalid" | Re-copy the key from Google AI Studio |
| "Rate limit / quota reached" | Wait a minute (free tier is limited) |
| "Almost no text could be extracted" | Export a text-based PDF/DOCX, not a scan |
| Model not found error | Update `MODEL_NAME` to a currently available Gemini Flash model |
