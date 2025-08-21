import os, json, subprocess
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import shutil
import glob

load_dotenv()

APP_DATA = "app_data"
UPLOAD_DIR = os.path.join(APP_DATA, "uploads")
CLEANED_DIR = os.path.join(APP_DATA, "cleaned")
SCRIPTS_DIR = os.path.join(APP_DATA, "scripts")
REPORTS_DIR = os.path.join(APP_DATA, "reports")

for d in [UPLOAD_DIR, CLEANED_DIR, SCRIPTS_DIR, REPORTS_DIR]:
    os.makedirs(d, exist_ok=True)


def get_client():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("Set OPENAI_API_KEY in environment or Streamlit sidebar")
    return OpenAI(api_key=key)


def read_any(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    return pd.read_csv(path)


def write_script(code: str, file_id: str):
    folder = os.path.join(SCRIPTS_DIR, file_id)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"clean_{file_id}.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)
    return path


def run_script(path: str, timeout: int = 300):
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



def cleanup_files(file_id: str):
    """
    Remove old uploaded files, scripts, and cleaned files related to a previous upload.
    """
    for file_path in glob.glob(os.path.join(UPLOAD_DIR, f"{file_id}*")):
        if os.path.isfile(file_path):
            os.remove(file_path)

    cleaned_file = os.path.join(CLEANED_DIR, f"cleaned_{file_id}.csv")
    if os.path.isfile(cleaned_file):
        os.remove(cleaned_file)

    script_folder = os.path.join(SCRIPTS_DIR, file_id)
    if os.path.isdir(script_folder):
        shutil.rmtree(script_folder, ignore_errors=True)

    report_folder = os.path.join(REPORTS_DIR, file_id)
    if os.path.isdir(report_folder):
        shutil.rmtree(report_folder, ignore_errors=True)

