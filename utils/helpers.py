import os, json, subprocess
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ----------------- App Folders -----------------
APP_DATA = "app_data"
UPLOAD_DIR = os.path.join(APP_DATA, "uploads")
CLEANED_DIR = os.path.join(APP_DATA, "cleaned")
SCRIPTS_DIR = os.path.join(APP_DATA, "scripts")
REPORTS_DIR = os.path.join(APP_DATA, "reports")

for d in [UPLOAD_DIR, CLEANED_DIR, SCRIPTS_DIR, REPORTS_DIR]:
    os.makedirs(d, exist_ok=True)


# ----------------- LLM Client -----------------
def get_client():
    """Return OpenAI client using env var OPENAI_API_KEY"""
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("Set OPENAI_API_KEY in environment or Streamlit sidebar")
    return OpenAI(api_key=key)


# ----------------- File Readers -----------------
def read_any(path: str):
    """Read CSV/XLSX into pandas DataFrame"""
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    return pd.read_csv(path)


# ----------------- Script Writing -----------------
def write_script(code: str, file_id: str):
    """Write cleaning Python script to scripts folder"""
    folder = os.path.join(SCRIPTS_DIR, file_id)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"clean_{file_id}.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)
    return path


# ----------------- Script Execution -----------------
def run_script(path: str, timeout: int = 300):
    """
    Run Python script safely with timeout.
    Returns: exit_code, stdout, stderr
    """
    try:
        proc = subprocess.run(
            ["python", path],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        return 1, "", f"Script timed out after {timeout} seconds"
    except Exception as e:
        return 1, "", f"Error running script: {e}"
