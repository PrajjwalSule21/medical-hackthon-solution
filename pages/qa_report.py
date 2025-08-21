import streamlit as st
import os, traceback
from utils.agents import agent3_qa

st.title("📊 QA Report")

# ----------------- Preconditions -----------------
cleaned_file = st.session_state.get("cleaned_out")
src_file = st.session_state.get("src_path")
file_id = st.session_state.get("file_id")

if not cleaned_file or not os.path.exists(cleaned_file):
    st.warning("⚠️ Please run the cleaning step first before generating QA report.")
elif not src_file or not os.path.exists(src_file):
    st.warning("⚠️ Original source file is missing. Please upload and analyze first.")
else:
    if st.button("Generate QA Report"):
        try:
            with st.spinner("Generating QA Report with Agent 3..."):
                report, rpath = agent3_qa(
                    src_file,
                    cleaned_file,
                    file_id
                )

            if os.path.exists(rpath):
                st.success("✅ QA Report generated successfully!")

                st.subheader("📄 QA Report Preview")
                st.markdown(report)

                with open(rpath, "rb") as f:
                    st.download_button(
                        "📥 Download QA Report",
                        data=f,
                        file_name=os.path.basename(rpath),
                        mime="text/markdown"
                    )
            else:
                st.error("QA report file was not created successfully.")

        except FileNotFoundError as e:
            st.error(f"File missing: {e}")
        except Exception as e:
            st.error(f"Error generating QA Report: {e}")
            st.text(traceback.format_exc())
