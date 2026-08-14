from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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
