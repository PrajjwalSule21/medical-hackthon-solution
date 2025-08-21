import os, json, re, ast, textwrap
import numpy as np
from .helpers import get_client, read_any, REPORTS_DIR, CLEANED_DIR, write_script
import pandas as pd
import subprocess

# ----------------- Utility: Extract & Validate Python Code -----------------
def _extract_code_from_text(text: str) -> str:
    m = re.search(r"```(?:python)?\s*([\s\S]+?)```", text)
    if m:
        return m.group(1).strip()
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if re.match(r"^\s*(import |from |def |class |#|@|with |if |print\(|open\()", line):
            return "\n".join(lines[i:]).strip()
    return text.strip()

def _is_valid_python(code: str):
    try:
        ast.parse(code)
        return True, ""
    except Exception as e:
        return False, str(e)


def agent1_analyze(df: pd.DataFrame):
    client = get_client()
    summary = {}
    for c in df.columns:
        col_data = df[c].astype(str)
        sample_values = col_data.head(10).tolist()

        # Detect dates
        is_date = any(re.search(r"\d{1,4}[-/]\d{1,2}[-/]\d{1,4}", str(s)) for s in sample_values)
        # Detect numeric
        numeric_ratio = col_data.map(lambda x: str(x).replace('.', '', 1).isdigit()).mean() if not col_data.empty else 0
        is_numeric = float(numeric_ratio) > 0.8

        summary[c] = {
            "dtype": str(df[c].dtype),
            "nulls": int(df[c].isna().sum()),
            "non_nulls": int(df[c].notna().sum()),
            "unique": int(df[c].nunique(dropna=True)),
            "sample_values": [str(v) for v in sample_values],
            "max_len": int(col_data.map(len).max()) if not col_data.empty else 0,
            "is_probably_date": bool(is_date),
            "is_probably_numeric": bool(is_numeric)
        }

    system_prompt = """
You are a senior data-quality analyst for medical datasets.
Detect:
- Spelling errors in categorical/medical terms
- Inconsistent terminology
- Missing values
- Duplicates
- Format inconsistencies (especially dates)
- Outliers
Return STRICT JSON:
{
 "issues":[{"type":"","column":"","suggestion":""}],
 "mapping":{"terminology":{column:{raw:canonical}}}
}
If a column contains datetime, suggest splitting into <col>_date and <col>_time if appropriate.
"""
    try:
        resp = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(summary, indent=2)}
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        print("Agent1 Error:", str(e))
        return {"issues": [], "mapping": {"terminology": {}}}



def agent2b_llm_clean(src_path, issues, mapping, file_id):
    """
    Agent 2B (LLM Master Cleaner):
    - Takes raw data + Agent 1 suggestions
    - Uses LLM to directly clean and engineer features
    - Returns cleaned DataFrame without running any local script
    """
    import pandas as pd
    client = get_client()
    
    # Read raw data
    df = read_any(src_path)
    
    # Prepare a small sample for context
    # sample_data = df.head(10).to_dict(orient="records")
    
    ctx = {
        "full_columns": df.columns.tolist(),
        "entire_data": df.to_dict(orient="records"),
        "issues": issues,
        "mapping": mapping,
        "instructions": {
            "preserve_all_rows": True,
            "handle_datetime": True,
            "feature_engineering": True,
            "fill_missing_logically": True
        }
    }
    
    system_prompt = """
You are an expert Python data engineer and feature engineer.

Task:
- Take the raw dataset and Agent 1 suggestions (issues + mapping)
- Clean the data, handle datetimes, apply feature engineering
- Preserve all rows; do NOT drop any data
- Apply terminology mapping safely
- Fill missing values only if logically correct
- Return the fully cleaned dataset as JSON array of records (list of dicts)
"""
    
    resp = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(ctx)}
        ]
    )
    
    # LLM returns JSON array of cleaned records
    cleaned_records = json.loads(resp.choices[0].message.content)
    
    # Convert back to DataFrame
    df_cleaned = pd.DataFrame(cleaned_records)
    
    return df_cleaned


def agent2_clean(src_path, issues, mapping, file_id):
    """
    Agent 2: Generates a cleaning script for preview/download and internally
    produces cleaned data using LLM (via agent2b_llm_clean) without executing
    the script externally.
    """
    client = get_client()
    out_path = os.path.join(CLEANED_DIR, f"cleaned_{file_id}.csv")
    
    # Context passed to the model
    ctx = {
        "src_path": src_path,
        "issues": issues,
        "mapping": mapping,
        "out_path": out_path
    }

    # System prompt instructs LLM to produce script only
    system_prompt = """
You are a senior Python data engineer and data-quality expert.

Task:
1. Generate a Python cleaning script that reads `src_path`, applies all
   cleaning suggestions in `issues` and terminology mappings in `mapping`,
   handles datetime columns, numeric/categorical columns, missing values,
   feature engineering, and writes cleaned CSV to `out_path`.
2. Return ONLY the Python script inside ```python ... ``` markdown.
3. DO NOT execute code externally; the LLM will handle the actual cleaning internally.
"""

    # Call the GPT-5-mini model to generate cleaning script
    resp = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(ctx)}
        ]
    )

    # Extract Python code from the response
    raw_code = _extract_code_from_text(resp.choices[0].message.content)
    ok, err = _is_valid_python(raw_code)
    if not ok:
        raise ValueError(f"Generated code is invalid: {err}")

    # Save the script for preview/download
    script_path = write_script(raw_code, file_id)

    # 🔹 Internally generate cleaned data using LLM (agent2b)
    try:
        df_cleaned = agent2b_llm_clean(src_path, issues, mapping, file_id)
        df_cleaned.to_csv(out_path, index=False)  # Save final cleaned CSV
    except Exception as e:
        raise RuntimeError(f"Error during LLM-powered cleaning: {e}")

    return raw_code, script_path, out_path


def agent3_qa(src_path, cleaned_path, file_id):
    client = get_client()
    if not os.path.exists(src_path) or not os.path.exists(cleaned_path):
        raise FileNotFoundError("Original or cleaned file missing.")

    src_df = read_any(src_path)
    clean_df = read_any(cleaned_path)
    before = {"rows": len(src_df), "cols": len(src_df.columns)}
    after = {"rows": len(clean_df), "cols": len(clean_df.columns)}
    delta = {"row_delta": after["rows"] - before["rows"], "col_delta": after["cols"] - before["cols"]}
    ctx = {"before": before, "after": after, "delta": delta}

    system_prompt = "You are a data-quality auditor. Write a clear Markdown report (5-8 bullets) summarizing improvements."
    resp = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(ctx)}
        ]
    )

    report = resp.choices[0].message.content
    rdir = os.path.join(REPORTS_DIR, file_id)
    os.makedirs(rdir, exist_ok=True)
    rpath = os.path.join(rdir, f"report_{file_id}.md")
    with open(rpath, "w") as f:
        f.write(report)

    return report, rpath
