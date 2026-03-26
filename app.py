import streamlit as st
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# Function: Extract text from PDF
# -------------------------------
def extract_text(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            if page.extract_text():
                text += page.extract_text()
    return text

# -------------------------------
# Function: Match Score
# -------------------------------
def get_match_score(resume, job_desc):
    tfidf = TfidfVectorizer()
    vectors = tfidf.fit_transform([resume, job_desc])
    score = cosine_similarity(vectors[0], vectors[1])
    return round(score[0][0] * 100, 2)

# -------------------------------
# Function: Missing Skills
# -------------------------------
skills_list = [
    "python", "sql", "machine learning",
    "data analysis", "excel", "power bi",
    "communication", "deep learning"
]

def get_missing_skills(resume):
    resume = resume.lower()
    missing = []

    for skill in skills_list:
        if skill not in resume:
            missing.append(skill)

    return missing

# -------------------------------
# Function: Suggestions
# -------------------------------
def get_suggestions(missing_skills):
    suggestions = []
    for skill in missing_skills:
        suggestions.append(f"Add {skill} to improve your resume")
    return suggestions

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="AI Resume Analyzer", layout="centered")

st.title("📄 AI Resume Analyzer")
st.write("Upload your resume and compare with job description")

# Upload Resume
resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

# Job Description
job_desc = st.text_area("Paste Job Description")

# Button
if st.button("Analyze Resume"):

    if resume_file is not None and job_desc != "":
        # Extract text
        resume_text = extract_text(resume_file)

        # Score
        score = get_match_score(resume_text, job_desc)

        # Missing skills
        missing = get_missing_skills(resume_text)

        # Suggestions
        suggestions = get_suggestions(missing)

        # Output
        st.subheader(f"📊 Match Score: {score}%")

        # Progress bar
        st.progress(int(score))

        st.subheader("❌ Missing Skills:")
        if missing:
            for skill in missing:
                st.write(f"- {skill}")
        else:
            st.write("No missing skills 🎉")

        st.subheader("💡 Suggestions:")
        for s in suggestions:
            st.write(f"- {s}")

    else:
        st.warning("Please upload resume and enter job description")