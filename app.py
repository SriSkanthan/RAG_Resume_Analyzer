import streamlit as st

from gemini_client import configure_gemini, generate_response
from resume_reader import read_resume
from resume_retriever import create_chunks, retrieve_chunks

st.set_page_config(page_title="Resume Analyzer")
st.title("AI Resume Analyzer and Job Suggestion Tool")
st.write("Upload a Resume")

configure_gemini()

uploaded_file = st.file_uploader("Upload Resume", type=["pdf", "docx", "txt"])
job_description = st.text_area("Enter Job Description")

if st.button("Analyze Resume"):
    if uploaded_file is None:
        st.warning("Please upload a resume.")
    elif job_description == "":
        st.warning("Please enter a job description.")
    else:
        resume_text = read_resume(uploaded_file)
        chunks = create_chunks(resume_text)
        retrieved_chunks = retrieve_chunks(chunks, job_description)
        final_response = generate_response(job_description, retrieved_chunks)

        st.subheader("Resume Analysis")
        st.write(final_response)

        st.download_button(
            label="Download Report",
            data=final_response,
            file_name="resume_analysis_report.txt",
            mime="text/plain",
        )