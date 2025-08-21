import streamlit as st, os, traceback
from utils.agents import agent4_master_clean
from utils.helpers import write_script, run_script

st.title("🧠 Master Clean Agent (Agent 4)")

if "clean_out" not in st.session_state or not os.path.exists(st.session_state.clean_out):
    st.warning("⚠️ Run Cleaning first!")
else:
    if st.button("Run Master Cleaning"):
        try:
            with st.spinner("Running final validation and cleanup..."):
                code, final_out = agent4_master_clean(
                    st.session_state.src_path,
                    st.session_state.clean_out,
                    st.session_state.file_id
                )
                st.session_state.final_out = final_out

                # Save and run script
                script_path = write_script(code, st.session_state.file_id + "_final")
                c, o, e = run_script(script_path)

            if c == 0 and os.path.exists(final_out):
                st.success(f"✅ Final cleaned file: {final_out}")
                st.subheader("Preview of Final Data")
                import pandas as pd
                df = pd.read_csv(final_out)
                st.dataframe(df.head(10))

                with open(final_out, "rb") as f:
                    st.download_button(
                        "📥 Download Final Cleaned File",
                        data=f,
                        file_name=os.path.basename(final_out),
                        mime="text/csv"
                    )
            else:
                st.error("Final cleaning script failed.")
                st.text(e)
        except Exception as e:
            st.error(f"Error: {e}")
            st.text(traceback.format_exc())
