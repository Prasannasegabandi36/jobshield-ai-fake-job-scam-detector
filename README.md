# JobShield AI — Fake Job and Internship Scam Detection Copilot

JobShield AI analyzes job offers, recruiter emails, WhatsApp job messages, internship posters, and offer letters. It generates an evidence-based scam risk report with RAG, memory, few-shot prompting, guardrails, PII masking, prompt injection detection, LangSmith tracing hooks, evaluation matrix, and summarization.

## Features

- RAG over scam knowledge base + uploaded offer content
- Chunking strategy with configurable chunk size and overlap
- Groq LLM integration
- Optional LangSmith tracing
- Rule-based red-flag detection
- PII masking for email, phone, Aadhaar-like, PAN-like, OTP, and API key patterns
- Prompt injection detection
- Few-shot prompt examples
- Memory for user preferences and recent cases
- Evaluation matrix
- Downloadable risk report

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Secrets

Create `.streamlit/secrets.toml` locally or add these secrets in Streamlit Cloud.

```toml
GROQ_API_KEY = "your_groq_key"

# Optional
LANGSMITH_TRACING = "true"
LANGSMITH_API_KEY = "your_langsmith_key"
LANGSMITH_PROJECT = "JobShield-AI"
```

## Deployment

1. Push this folder to GitHub.
2. Open Streamlit Community Cloud.
3. Create new app and select `app.py`.
4. Add secrets in app settings.
5. Deploy.

## Important Safety Note

This app is a scam-risk assistant, not a legal authority. It should not directly accuse a company without evidence. It provides risk indicators and safe verification steps.
