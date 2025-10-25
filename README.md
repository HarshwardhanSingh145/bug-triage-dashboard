# GitHub Issue Triage Dashboard

An issue triage and deduplication dashboard built with Python, Streamlit, Google Gemini, and Sentence Transformers.

## Project Overview
Software engineering teams receive numerous bug reports every week. Manually reviewing issues, identifying duplicates, and routing them to the right team takes significant time.

This project automates issue triage into a single dashboard:
1. Duplicate Detection: Uses vector embeddings (`all-MiniLM-L6-v2`) to compare incoming issues against past issues.
2. Classification and Routing: Uses Gemini to classify issue category, priority, affected component, and team routing.
3. Live GitHub Ingestion: Fetches live issues directly from any public GitHub repository.

## Features
- Triage Metrics: Category, Priority, Assignee Team, and Performance Impact.
- Component and Action: Highlights the affected module and recommended next steps.
- Deduplication: Flags potential duplicate issues using cosine similarity.
- Quality Check: Assesses whether an issue description is actionable.
- GitHub Integration: Fetches issues via the GitHub REST API.

## Technologies Used
- Streamlit
- Google GenAI SDK (`gemini-3.5-flash-lite`)
- Sentence-Transformers (`all-MiniLM-L6-v2`)
- Scikit-Learn and NumPy
- Pydantic
- Requests

## Project Structure
```text
bug-triage-dashboard/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── .streamlit/
    ├── secrets.toml.example
    └── secrets.toml
```

## Setup and Installation

### 1. Clone the Repository
```bash
git clone https://github.com/HarshwardhanSingh145/bug-triage-dashboard.git
cd bug-triage-dashboard
```

### 2. Create Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Gemini API Key
Create `.streamlit/secrets.toml`:
```toml
GEMINI_API_KEY = "your_gemini_api_key_here"
```
Or set an environment variable:
```bash
export GEMINI_API_KEY="your_gemini_api_key_here"
```

## Running the Application
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

## Testing
Run the verification check:
```bash
python3 -c "
import app
key = app.get_gemini_api_key()
print('API Key:', bool(key))
score, matched = app.find_similar_issue('App crashes on upload')
print(f'Duplicate score: {score:.1f}%')
res = app.analyze_issue_with_gemini('Database connection timeout error', key)
print('Triage:', res['issue_type'], '->', res['assigned_team'])
"
```

## Limitations
- GitHub API Rate Limits: Unauthenticated requests to GitHub are limited to 60 requests per hour.
- Baseline Dataset: The demo uses an in-memory sample list for duplicates. Production use cases typically connect to a persistent database.
