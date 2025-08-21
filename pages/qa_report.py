import streamlit as st, os, traceback
from utils.agents import agent3_qa

st.title("📊 QA Report")

# ----------------- Preconditions -----------------
if "clean_out" not in st.session_state or not st.session_state.clean_out or not os.path.exists(st.session_state.clean_out):
    st.warning("⚠️ Please run the cleaning step first before generating QA report.")
else:
    # ----------------- Generate QA Report -----------------
    if st.button("Generate QA Report"):
        try:
            with st.spinner("Generating QA Report with Agent 3..."):
                report, rpath = agent3_qa(
                    st.session_state.src_path,
                    st.session_state.clean_out,
                    st.session_state.file_id
                )
            st.success("✅ QA Report generated successfully!")

            # Display report content
            st.markdown(report)

            # Download button for Markdown report
            with open(rpath, "rb") as f:
                st.download_button(
                    "📥 Download Report",
                    data=f,
                    file_name=os.path.basename(rpath),
                    mime="text/markdown"
                )

        except FileNotFoundError as e:
            st.error(f"File missing: {e}")
        except Exception as e:
            st.error(f"Error generating QA Report: {e}")
            st.text(traceback.format_exc())
