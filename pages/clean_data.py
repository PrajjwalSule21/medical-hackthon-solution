import streamlit as st, os, traceback
from utils.agents import agent2_generate
from utils.helpers import write_script, run_script, read_any  # added read_any to read CSV/XLSX

st.title("🧹 Clean Data")

# ----------------- Preconditions -----------------
if "analysis" not in st.session_state or not st.session_state.analysis:
    st.warning("⚠️ Please run the analysis step first before cleaning.")
else:
    # ----------------- Generate Cleaning Script -----------------
    if st.button("Generate Cleaning Script"):
        try:
            with st.spinner("Generating cleaning script with Agent 2..."):
                code, out_path = agent2_generate(
                    st.session_state.src_path,
                    st.session_state.analysis.get("issues", []),
                    st.session_state.analysis.get("mapping", {}),
                    st.session_state.file_id
                )
                st.session_state.clean_code = code
                st.session_state.clean_out = out_path
            st.success("✅ Cleaning script generated!")
            st.code(code, language="python")
        except Exception as e:
            st.error(f"Error generating cleaning script: {e}")
            st.text(traceback.format_exc())

    # ----------------- Run Cleaning Script -----------------
    if "clean_code" in st.session_state and st.button("Run Cleaning Script"):
        try:
            script_path = write_script(st.session_state.clean_code, st.session_state.file_id)
            with st.spinner("Running cleaning script..."):
                code, out, err = run_script(script_path)

            st.subheader("Execution Logs")
            if out:
                st.text(out)
            if err:
                st.error(err)

            if code == 0 and os.path.exists(st.session_state.clean_out):
                st.success(f"✅ Cleaned file saved: {st.session_state.clean_out}")
                
                # ----------------- Show Preview of Cleaned Data -----------------
                try:
                    cleaned_df = read_any(st.session_state.clean_out)
                    st.subheader("📄 Preview of Cleaned Data (First 10 Rows)")
                    st.dataframe(cleaned_df.head(10))
                except Exception as e:
                    st.warning(f"Could not read cleaned file for preview: {e}")

            else:
                st.error("Cleaning script failed. Check logs above.")

        except Exception as e:
            st.error(f"Error running cleaning script: {e}")
            st.text(traceback.format_exc())