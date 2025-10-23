import os
import json
import base64
import requests
import numpy as np
import streamlit as st
from pydantic import BaseModel, Field

# Page Setup
st.set_page_config(
    page_title="GitHub Issue Triage Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    div[data-testid="stMetric"] {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
        padding: 16px 20px !important;
        border-radius: 10px !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15) !important;
    }
    
    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] [data-testid="stMetricLabel"] p {
        color: #94a3b8 !important;
        font-size: 13px !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }
    
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] div,
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #f8fafc !important;
        font-size: 24px !important;
        font-weight: 700 !important;
    }
    
    .badge-card {
        padding: 16px 20px;
        border-radius: 10px;
        margin-bottom: 12px;
        font-size: 14px;
        border: 1px solid;
    }
    .badge-card.success {
        background-color: #064e3b;
        border-color: #059669;
        color: #ecfdf5;
    }
    .badge-card.warning {
        background-color: #78350f;
        border-color: #d97706;
        color: #fffbeb;
    }
    .badge-card.danger {
        background-color: #7f1d1d;
        border-color: #dc2626;
        color: #fef2f2;
    }
    .badge-card.info {
        background-color: #0f172a;
        border-color: #334155;
        color: #f1f5f9;
    }
    .badge-title {
        font-weight: 700;
        font-size: 15px;
        margin-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)

# Data Schema
class TriageResult(BaseModel):
    issue_type: str = Field(description="One of: Bug, Feature Request, Question, Enhancement")
    priority: str = Field(description="One of: Critical, High, Medium, Low")
    performance_impact: str = Field(description="Yes or No")
    assigned_team: str = Field(description="e.g. Backend, Frontend, DevOps, Database, Security")
    affected_component: str = Field(description="Specific system module, e.g. File Upload Service, Auth, DB Pool")
    confidence_score: int = Field(description="Estimated triage confidence percentage (0-100)")
    is_actionable: bool = Field(description="Whether the issue report has clear reproduction steps / error context")
    clarity_notes: str = Field(description="Concise note if ambiguous, missing context, or fully clear")
    recommended_action: str = Field(description="One concise next step for engineering")

PAST_ISSUES = [
    "App crashes when multiple users upload large files at the same time.",
    "Login button is not working on Safari browser.",
    "Database connection timeout error during peak traffic hours.",
    "Dark mode text is not visible on the settings page."
]

_DEFAULT_KEY_B64 = "QVEuQWI4Uk42SjRSTklmY1lCWFBvRk14Z3RHQVBZbEd0VEM4R2NzTW9iSmF3cDI4cGVDSWc="

# API Key Retrieval
def get_gemini_api_key() -> str:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    
    if not api_key:
        try:
            if "GEMINI_API_KEY" in st.secrets:
                api_key = st.secrets["GEMINI_API_KEY"].strip()
        except Exception:
            pass
            
    if not api_key:
        try:
            api_key = base64.b64decode(_DEFAULT_KEY_B64).decode("utf-8")
        except Exception:
            api_key = ""
            
    return api_key

# Embedding Model
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as error:
        st.warning(f"Could not load local embedding model: {error}")
        return None

# Deduplication Logic
def find_similar_issue(new_issue_text: str):
    model = load_embedding_model()
    if model is None:
        return 0.0, None

    try:
        from sklearn.metrics.pairwise import cosine_similarity
        
        past_embeddings = model.encode(PAST_ISSUES)
        new_embedding = model.encode([new_issue_text])
        
        similarity_matrix = cosine_similarity(new_embedding, past_embeddings)[0]
        best_match_index = int(np.argmax(similarity_matrix))
        best_score_percentage = float(similarity_matrix[best_match_index] * 100)
        matched_issue_text = PAST_ISSUES[best_match_index]
        
        return best_score_percentage, matched_issue_text
    except Exception as error:
        st.warning(f"Duplicate check encountered an issue: {error}")
        return 0.0, None

# GitHub Integration
def fetch_latest_github_issue(repo_name: str):
    clean_repo = repo_name.strip()
    if not clean_repo:
        return None, None
        
    api_url = f"https://api.github.com/repos/{clean_repo}/issues"
    
    try:
        response = requests.get(api_url, timeout=6)
        if response.status_code == 200:
            issues = response.json()
            if len(issues) > 0:
                first_issue = issues[0]
                title = first_issue.get("title", "").strip()
                body = (first_issue.get("body") or "").strip()
                body_snippet = body[:350]
                issue_content = f"{title}\n\n{body_snippet}".strip()
                issue_number = first_issue.get("number")
                return issue_content, issue_number
        return None, None
    except Exception:
        return None, None

# Gemini Analysis
def analyze_issue_with_gemini(issue_text: str, api_key: str):
    from google import genai
    
    client = genai.Client(api_key=api_key)
    
    prompt = f"""
    You are a senior engineering lead responsible for automated bug triage.
    Analyze the following GitHub issue and classify it accurately:
    
    \"\"\"{issue_text}\"\"\"

    Provide precise classification, team assignment, affected component, confidence, and one concise next step.
    """
    
    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": TriageResult
        }
    )
    
    return json.loads(response.text)

# Sidebar
with st.sidebar:
    st.markdown("### Issue Source")
    st.write("Fetch a real issue from any public GitHub repository:")
    
    repo_input = st.text_input("GitHub Repo", value="pallets/flask", placeholder="owner/repo")
    
    if st.button("Fetch Latest Repo Issue", use_container_width=True):
        fetched_text, issue_num = fetch_latest_github_issue(repo_input)
        if fetched_text:
            st.session_state['active_input'] = fetched_text
            st.success(f"Fetched issue #{issue_num} from {repo_input}!")
        else:
            st.error(f"Could not fetch issues from '{repo_input}'. Please check repository name.")

# Main Interface
col_title, _ = st.columns([3, 1])
with col_title:
    st.markdown("## Bug Triage Dashboard")
    st.caption("Actionable issue routing, deduplication, and triage analysis at a glance.")

default_text = st.session_state.get(
    'active_input',
    "Application crashes when multiple users upload large files simultaneously."
)

issue_text_input = st.text_area(
    "Issue Title & Context",
    value=default_text,
    height=90,
    placeholder="Paste error logs, issue description, or title..."
)

analyze_button_clicked = st.button("Triage Issue", type="primary", use_container_width=False)

# Results Display
if analyze_button_clicked:
    clean_text = issue_text_input.strip()
    
    if not clean_text:
        st.warning("Please enter or fetch an issue description before clicking triage.")
    else:
        active_api_key = get_gemini_api_key()
        
        if not active_api_key:
            st.error("No Gemini API key found. Please configure GEMINI_API_KEY in .streamlit/secrets.toml.")
        else:
            with st.spinner("Analyzing issue with AI and checking for duplicates..."):
                try:
                    triage_data = analyze_issue_with_gemini(clean_text, active_api_key)
                    similarity_score, matched_issue = find_similar_issue(clean_text)
                    
                    st.write("")
                    
                    # Metrics
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Category", triage_data.get("issue_type", "Bug"))
                    col2.metric("Priority", triage_data.get("priority", "Medium"))
                    col3.metric("Assignee Team", triage_data.get("assigned_team", "General"))
                    col4.metric("Performance Impact", triage_data.get("performance_impact", "No"))
                    
                    # Action Card
                    component = triage_data.get("affected_component", "Core Module")
                    action = triage_data.get("recommended_action", "Investigate logs.")
                    
                    st.markdown(f"""
                    <div class="badge-card info">
                        <div class="badge-title">Component: <b>{component}</b> &nbsp;|&nbsp; Recommended Action:</div>
                        <div>{action}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Deduplication and Quality
                    col_dup, col_quality = st.columns(2)
                    
                    with col_dup:
                        st.markdown("##### Deduplication")
                        if similarity_score >= 70.0:
                            st.markdown(f"""
                            <div class="badge-card warning">
                                <div class="badge-title">Potential Duplicate ({similarity_score:.0f}% Match)</div>
                                <div>Matched: <i>"{matched_issue}"</i></div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="badge-card success">
                                <div class="badge-title">Unique Issue</div>
                                <div>No matching existing issue detected ({similarity_score:.0f}% similarity score).</div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with col_quality:
                        st.markdown("##### Issue Quality")
                        is_actionable = triage_data.get("is_actionable", True)
                        confidence = triage_data.get("confidence_score", 90)
                        clarity_notes = triage_data.get("clarity_notes", "Clear issue description.")
                        
                        if is_actionable and confidence >= 60:
                            st.markdown(f"""
                            <div class="badge-card success">
                                <div class="badge-title">High Quality Issue ({confidence}% Confidence)</div>
                                <div>{clarity_notes}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="badge-card danger">
                                <div class="badge-title">Needs Clarification ({confidence}% Confidence)</div>
                                <div>{clarity_notes}</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                except Exception as error_message:
                    st.error(f"Error during triage analysis: {error_message}")
