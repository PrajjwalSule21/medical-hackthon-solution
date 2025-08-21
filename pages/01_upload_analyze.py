import streamlit as st, uuid, os, traceback
from utils.helpers import UPLOAD_DIR, read_any, cleanup_files
from utils.agents import agent1_analyze

st.title("📂 Upload & Analyze")

for key in ["file_id", "src_path", "analysis", "preview_data"]:
    if key not in st.session_state:
        st.session_state[key] = None

up = st.file_uploader("Upload CSV or XLSX", type=["csv", "xlsx", "xls"])
if up:
    # Clean previous files if any
    if st.session_state.file_id:
        cleanup_files(st.session_state.file_id)

    fid = str(uuid.uuid4())
    st.session_state.file_id = fid
    ext = os.path.splitext(up.name)[1]
    path = os.path.join(UPLOAD_DIR, f"{fid}{ext}")
    with open(path, "wb") as f:
        f.write(up.read())
    st.session_state.src_path = path

    # Read and store preview data
    try:
        df_preview = read_any(path)
        st.session_state.preview_data = df_preview.head(10)
        st.success(f"✅ File uploaded: {path}")
    except Exception as e:
        st.error(f"Failed to read uploaded file: {e}")
        st.session_state.preview_data = None

if st.session_state.preview_data is not None:
    st.subheader("📄 Data Preview (First 10 Rows)")
    st.dataframe(st.session_state.preview_data)

if st.session_state.src_path and st.button("Run Analysis"):
    try:
        with st.spinner("Analyzing dataset with Agent 1..."):
            df = read_any(st.session_state.src_path)
            st.session_state.analysis = agent1_analyze(df)
        st.success("✅ Analysis complete!")
    except Exception as e:
        st.error(f"Analysis failed: {e}")
        st.text(traceback.format_exc())

if st.session_state.analysis:
    st.subheader("Detected Issues")
    issues = st.session_state.analysis.get("issues", [])
    if issues:
        for i, issue in enumerate(issues, 1):
            st.markdown(f"**{i}. {issue.get('type','Unknown')}** - {issue.get('column','')} → {issue.get('suggestion','')}")
    else:
        st.info("No major issues detected.")

    st.subheader("Terminology Mapping Suggestions")
    mapping = st.session_state.analysis.get("mapping", {}).get("terminology", {})
    if mapping:
        st.json(mapping)
    else:
        st.info("No terminology mappings suggested.")
