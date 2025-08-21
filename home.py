import streamlit as st
st.set_page_config(page_title="Raw Data Preprocessor", page_icon="🧹")
st.title("🧹 Raw Data Preprocessor")

st.markdown("""
### Welcome!
This app cleans messy medical datasets using **3 LLM Agents**:
1. **Analyze** issues & suggest terminology mapping  
2. **Generate Cleaning Code** & run it  
3. **QA Report** with before/after metrics + AI summary  

Use the sidebar to navigate →  
""")
