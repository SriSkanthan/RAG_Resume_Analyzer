import streamlit as st

from gemini_client import configure_gemini, generate_response
from resume_reader import read_resume
from resume_retriever import calculate_job_fit_score, create_chunks, retrieve_chunks

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

        if not resume_text or not resume_text.strip():
            st.warning("The uploaded file could not be read or contains no text. Please upload a valid resume PDF, DOCX, or TXT file.")
            st.stop()

        chunks = create_chunks(resume_text)
        retrieved_chunks = retrieve_chunks(chunks, job_description)

        if not retrieved_chunks:
            st.warning("No meaningful resume content could be matched. Please check that the resume text is readable and contains relevant skills.")
            st.stop()

        job_fit_score = calculate_job_fit_score(resume_text, job_description)
        final_response = generate_response(job_description, retrieved_chunks)

        st.subheader("Resume Analysis")
        st.write(f"**Job Fit Score:** {job_fit_score:.2f}/100")
        st.write(final_response)

        st.download_button(
            label="Download Report",
            data=final_response,
            file_name="resume_analysis_report.txt",
            mime="text/plain",
        )