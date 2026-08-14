import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


GENERIC_SKILLS = {
    "python": ["python", "python programming"],
    "sql": ["sql", "sqlite", "database", "postgres", "mysql"],
    "api": ["api", "rest api", "rest-api", "fastapi", "flask", "django"],
    "cloud": ["cloud", "aws", "azure", "gcp", "devops"],
    "git": ["git", "github", "version control"],
    "javascript": ["javascript", "js", "node", "nodejs", "react", "angular", "vue"],
    "data": ["data analysis", "pandas", "numpy", "excel", "power bi", "tableau"],
    "machine learning": ["machine learning", "ml", "ai", "scikit-learn", "tensorflow", "pytorch", "model training"],
    "communication": ["communication", "problem solving", "stakeholder management", "presentation", "teamwork"],
    "testing": ["testing", "qa", "automation", "unit testing", "pytest", "selenium"],
    "java": ["java", "spring", "spring boot"],
    "csharp": ["c#", "c sharp", ".net", "asp.net"],
    "design": ["ui", "ux", "figma", "design systems", "wireframes"],
    "sales": ["sales", "customer success", "business development", "account management"],
    "product": ["product management", "roadmap", "requirements", "agile", "scrum"],
}


GENERIC_SKILL_WEIGHTS = {
    "python": 3.0,
    "sql": 3.0,
    "api": 2.5,
    "cloud": 2.5,
    "git": 1.5,
    "javascript": 2.5,
    "data": 2.5,
    "machine learning": 3.0,
    "communication": 1.5,
    "testing": 2.0,
    "java": 2.5,
    "csharp": 2.5,
    "design": 2.0,
    "sales": 2.0,
    "product": 2.0,
}

MIN_SCORE_WEIGHT = 10.0


def normalize_text(text):
    """Clean text so skill matching is less sensitive to punctuation and formatting."""
    if not text:
        return ""
    text = text.lower()
    text = text.replace("-", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def extract_required_skills(job_description):
    """Extract likely required skills from a job description in a generic way."""
    if not job_description or not job_description.strip():
        return []

    normalized_job = normalize_text(job_description)
    matched_skills = []

    for skill_name, keywords in GENERIC_SKILLS.items():
        if any(keyword in normalized_job for keyword in keywords):
            matched_skills.append(skill_name)

    return matched_skills


def calculate_job_fit_score(resume_text, job_description):
    """Calculate a weighted skill match score out of 100 for a resume against a job description."""
    if not resume_text or not resume_text.strip():
        return 0.0
    if not job_description or not job_description.strip():
        return 0.0

    required_skills = extract_required_skills(job_description)
    if not required_skills:
        return 0.0

    normalized_resume = normalize_text(resume_text)
    total_weight = sum(GENERIC_SKILL_WEIGHTS.get(skill, 1.0) for skill in required_skills)
    effective_total_weight = max(total_weight, MIN_SCORE_WEIGHT)
    earned_weight = 0.0

    for skill_name in required_skills:
        weight = GENERIC_SKILL_WEIGHTS.get(skill_name, 1.0)
        keywords = GENERIC_SKILLS[skill_name]

        if any(keyword in normalized_resume for keyword in keywords):
            earned_weight += weight

    score = (earned_weight / effective_total_weight) * 100 if effective_total_weight else 0.0
    return round(score, 2)


def calculate_skill_match_score(resume_text, job_description):
    """Backward-compatible alias for the basic job-fit score."""
    return calculate_job_fit_score(resume_text, job_description)


def create_chunks(resume_text, chunk_size=50):
    """Split resume text into word-based chunks."""
    words = resume_text.split()
    chunks = []

    for start in range(0, len(words), chunk_size):
        end = start + chunk_size
        chunk = " ".join(words[start:end])

        if chunk.strip():
            chunks.append(chunk)

    return chunks


def retrieve_chunks(chunks, job_description, top_k=3):
    """Find the most relevant resume chunks for the job description."""
    if not chunks or not any(chunk.strip() for chunk in chunks):
        return []

    if not job_description or not job_description.strip():
        return []

    all_documents = [job_description] + chunks

    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform(all_documents)

    job_vector = vectors[0:1]
    resume_vectors = vectors[1:]

    similarity_scores = cosine_similarity(job_vector, resume_vectors)[0]
    best_indexes = similarity_scores.argsort()[-top_k:][::-1]
    retrieved_results = []

    for index in best_indexes:
        retrieved_results.append(
            {
                "chunk": chunks[index],
                "score": float(similarity_scores[index]),
            }
        )

    return retrieved_results
