import os, json, re, ast, textwrap
import numpy as np
from .helpers import get_client, read_any, REPORTS_DIR, CLEANED_DIR

# ----------------- Utility: Extract & Validate Python Code -----------------
def _extract_code_from_text(text: str) -> str:
    """Extract Python code from LLM response safely."""
    # 1) Look for code fences
    m = re.search(r"```(?:python)?\s*([\s\S]+?)```", text)
    if m:
        return m.group(1).strip()

    # 2) Fallback: look for first Python-like line
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if re.match(r"^\s*(import |from |def |class |#|@|with |if |print\(|open\()", line):
            return "\n".join(lines[i:]).strip()

    return text.strip()


def _is_valid_python(code: str):
    """Check Python syntax validity."""
    try:
        ast.parse(code)
        return True, ""
    except Exception as e:
        return False, str(e)


# ----------------- Agent 1: Analyze Dataset -----------------
def agent1_analyze(df):
    client = get_client()

    # Extended profiling with safe JSON serialization
    summary = {}
    for c in df.columns:
        col_data = df[c].dropna().astype(str)
        sample_values = col_data.head(10).tolist()

        # Safe boolean and numeric checks
        is_date = any(re.search(r"\d{1,4}[-/]\d{1,2}[-/]\d{1,4}", s) for s in sample_values)
        is_numeric = False
        if not col_data.empty:
            numeric_ratio = col_data.map(lambda x: x.replace('.', '', 1).isdigit()).mean()
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

    agent1_system_prompt = """
You are a senior data-quality analyst specializing in messy hospital datasets.
You will analyze the dataset summary and detect all quality issues.

Return STRICT JSON with this structure:
{
 "issues":[{"type":"","column":"","suggestion":""}],
 "mapping":{"terminology":{column:{raw:canonical}}}
}

Guidelines:
- Detect inconsistent date formats; if column mixes date+time, suggest splitting into separate date and time columns.
- Detect categorical columns with spelling mistakes, typos, inconsistent cases; suggest canonical values.
- Detect free-text columns with long text; suggest trimming, normalization if needed.
- Detect numeric columns stored as text; suggest conversion to numeric types.
- Detect IDs or keys with missing values; suggest removal if too sparse.
- Do NOT always fill missing values; only suggest fill if it makes logical sense.
- For duplicates, suggest removing only if truly identical rows exist.
- For timestamps, standardize to UTC format if possible.
- For diagnosis or categorical codes, suggest mapping if many variants exist.
"""


    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": agent1_system_prompt},
                {"role": "user", "content": json.dumps(summary, indent=2)}
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        print("Agent1 Error:", str(e))
        return {"issues": [], "mapping": {"terminology": {}}}



# ----------------- Agent 2: Cleaning Code Generation -----------------
def agent2_generate(src_path, issues, mapping, file_id):
    client = get_client()
    out_path = os.path.join(CLEANED_DIR, f"cleaned_{file_id}.csv")
    ctx = {
        "src_path": src_path,
        "issues": issues,
        "mapping": mapping,
        "out_path": out_path
    }

    agent2_system_prompt = """
You are a Python data engineer.
Return ONLY a complete Python script as plain text.
Do NOT include explanations, markdown, or triple backticks.

Rules for the script:
1. Must be valid Python code with no syntax errors.
2. Read CSV/XLSX from src_path.
3. mapping is a Python dictionary, not part of the DataFrame.  
   - Use mapping.get('terminology', {}) to access terminology mappings.
   - Rename columns only if they exist in the DataFrame.
4. Always check if a column exists before processing.
5. For datetime columns:
   - Convert using pd.to_datetime(col, errors='coerce', utc=True).
   - If valid dates exist, create two columns: <col>_date (YYYY-MM-DD) and <col>_time (HH:MM:SS).
   - Drop the original column only if instructed, else keep as string.
6. For mixed-type or object columns:
   - Safely cast to string before saving to CSV.
7. When filling null values, avoid chained assignment:
      df[col] = df[col].fillna('Unknown')
8. Avoid KeyErrors by always checking column existence first.
9. Print final row/col counts at the end.
10. Save cleaned CSV to out_path.
11. The script must run without warnings or errors even if some columns don't exist.
"""

    # First attempt
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": agent2_system_prompt},
            {"role": "user", "content": json.dumps(ctx)}
        ]
    )

    raw_code = _extract_code_from_text(resp.choices[0].message.content)

    # Validate Python code
    ok, err = _is_valid_python(raw_code)
    if ok:
        return textwrap.dedent(raw_code), out_path

    # Retry with stricter instructions if invalid
    retry_prompt = f"""
Previous code had syntax errors: {err}.
Return ONLY valid Python code as plain text. No markdown, no ``` fences.
Context: {json.dumps(ctx)}
"""
    resp2 = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": retry_prompt}
        ]
    )

    raw_code = _extract_code_from_text(resp2.choices[0].message.content)
    return textwrap.dedent(raw_code), out_path



# ----------------- Agent 3: QA Report -----------------
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

    system_prompt = """
You are a data-quality auditor.
Write a clear Markdown report (5-8 bullet points) summarizing:
- Changes in row/col counts
- Data quality improvements
- Potential risks or next steps
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(ctx)}
        ]
    )

    report = resp.choices[0].message.content

    # Save report
    rdir = os.path.join(REPORTS_DIR, file_id)
    os.makedirs(rdir, exist_ok=True)
    rpath = os.path.join(rdir, f"report_{file_id}.md")
    with open(rpath, "w") as f:
        f.write(report)

    return report, rpath




def agent4_master_clean(raw_path, cleaned_path, file_id):
    client = get_client()
    final_out = os.path.join(CLEANED_DIR, f"final_{file_id}.csv")
    ctx = {
        "raw_path": raw_path,
        "cleaned_path": cleaned_path,
        "out_path": final_out
    }

    master_agent_prompt = """
You are a Senior Data Quality Engineer.
You will receive:
1. Raw dataset path (raw_data)
2. Cleaned dataset path (cleaned_data)

Your task:
- Validate if cleaning was done properly by comparing raw and cleaned data.
- Detect any remaining data quality issues: inconsistent dates, missing categorical mappings, null handling errors, unexpected duplicates, mixed data types, typos in categories, improper column naming.
- For datetime columns: ensure they are split into <col>_date and <col>_time if applicable.
- Ensure no ambiguous or inconsistent data remains.
- Produce FINAL dataset with 0 known data quality issues.

Rules:
1. Always check if a column exists before processing.
2. Apply fixes carefully without deleting useful data.
3. Avoid chained assignment warnings.
4. Ensure all date columns use ISO8601 for dates and standard 24hr for time.
5. Convert mixed types into appropriate numeric, categorical, or string as needed.
6. Maintain consistent column names & casing.
7. Print final dataset shape at the end.
8. Save final dataset as CSV at out_path.
9. Script must run without errors even if columns are missing or formats are inconsistent.

Return ONLY the full Python script. No markdown, no explanation.
"""


    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": master_agent_prompt},
            {"role": "user", "content": json.dumps(ctx)}
        ]
    )

    code = resp.choices[0].message.content
    return code, final_out

