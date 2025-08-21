import streamlit as st, traceback, os
from utils.helpers import read_any
from utils.agents import agent2_clean

st.title("🧹 Clean Data & Download Script")

# ----------------- Preconditions -----------------
if "src_path" not in st.session_state or not st.session_state.src_path:
    st.warning("⚠️ Please upload and analyze a dataset first in the Upload & Analyze page.")
elif "analysis" not in st.session_state or not st.session_state.analysis:
    st.warning("⚠️ Please run Agent 1 analysis first.")
else:
    issues = st.session_state.analysis.get("issues", [])
    mapping = st.session_state.analysis.get("mapping", {})
    file_id = st.session_state.file_id
    src_path = st.session_state.src_path

    # ----------------- Generate Cleaning Script & Clean Data -----------------
    if st.button("Generate Cleaning Script & Clean Data"):
        try:
            with st.spinner("Generating cleaning script and cleaning data with Agent 2..."):
                # Agent 2 now generates script AND runs it to produce cleaned CSV
                code, script_path, cleaned_path = agent2_clean(src_path, issues, mapping, file_id)

                # Save paths in session_state
                st.session_state.script_path = script_path
                st.session_state.cleaned_out = cleaned_path

            st.success("✅ Cleaning script generated and cleaned data created!")

            # ----------------- Show Script -----------------
            st.subheader("📄 Cleaning Script Preview")
            st.code(code, language="python")

            # Download script
            if os.path.exists(script_path):
                with open(script_path, "rb") as f:
                    st.download_button(
                        "📥 Download Cleaning Script",
                        data=f,
                        file_name=f"clean_{file_id}.py",
                        mime="text/x-python"
                    )

            # ----------------- Show Cleaned Data -----------------
            if os.path.exists(cleaned_path):
                df_cleaned = read_any(cleaned_path)
                st.subheader("✅ Cleaned Data Preview (First 10 Rows)")
                st.dataframe(df_cleaned.head(10))

                with open(cleaned_path, "rb") as f:
                    st.download_button(
                        "📥 Download Cleaned CSV",
                        data=f,
                        file_name=f"cleaned_{file_id}.csv",
                        mime="text/csv"
                    )
            else:
                st.error("Cleaned file was not created successfully.")

        except Exception as e:
            st.error(f"Error generating cleaning script or cleaning data: {e}")
            st.text(traceback.format_exc())
