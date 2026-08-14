import google.generativeai as genai
import streamlit as st


def configure_gemini():
    """Configure the Gemini API key from Streamlit secrets."""
    if "GEMINI_API_KEY" not in st.secrets:
        raise KeyError("GEMINI_API_KEY not found in Streamlit secrets.")

    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])


def generate_response(job_description, retrieved_chunks):
    """Generate the final AI resume analysis using Gemini."""
    context = ""

    for item in retrieved_chunks:
        context += item["chunk"] + "\n\n"

    prompt = f"""
You are an AI Resume Analyzer.

This analysis compares the job description with the candidate's resume to identify the skills required by the role, the skills already present in the resume, and the skills missing from the candidate profile. The goal is to provide a clear skills gap summary for better readability and decision-making.

Job Description:

{job_description}

Relevant Resume Information:

{context}

Tasks:

1. List the required skills from the job description.
2. List the candidate's current skills based on the resume.
3. List the missing skills that the candidate does not yet have.
4. Suggest improvements to help the candidate close the skill gap.
5. Suggest suitable job roles for the candidate based on the resume and job requirements.

Important:
- Do not provide a resume match percentage instead provide the score from the retrieved chunks.
- Keep the output concise, clear, and readable.
- Avoid repeating the same content across sections.
- Present the information in a structured format with headings.
"""
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    return response.text
