import streamlit as st
import fitz  # PyMuPDF
import re
import os
import base64
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------- SETTINGS -------------------
RESUME_DIR = "uploaded_resume.pdf"
BG_IMAGE = "streamlit_bg.png"

# ------------------- BACKGROUND -------------------
def set_bg(png_file):
    with open(png_file, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    bg_css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{b64}");
        background-size: cover;
        background-attachment: fixed;
        font-family: 'Arial', sans-serif;
    }}
    .block-container {{
        background-color: rgba(255, 255, 255, 0.85);
        padding: 2rem;
        border-radius: 15px;
    }}
    </style>
    """
    st.markdown(bg_css, unsafe_allow_html=True)

# ------------------- RESUME PARSER -------------------
def extract_text_from_pdf(pdf_file):
    text = ""
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

# ------------------- SKILL MATCHING -------------------
def get_skills(text):
    keywords = ["python", "sql", "machine learning", "data analysis", "excel", "git", "power bi",
                "java", "c++", "tensorflow", "deep learning", "flask", "django", "aws"]
    found = [kw for kw in keywords if kw.lower() in text.lower()]
    return found

def calculate_score(user_skills, required_skills):
    vectorizer = CountVectorizer().fit_transform([" ".join(user_skills), " ".join(required_skills)])
    vectors = vectorizer.toarray()
    return round(cosine_similarity([vectors[0]], [vectors[1]])[0][0] * 100, 2)

# ------------------- FEEDBACK -------------------
def generate_feedback(matched, missing):
    feedback = ""
    if missing:
        feedback += "\n\n*Skill Improvement Tips:*\n"
        for skill in missing:
            feedback += f"- Consider learning {skill} via Coursera, Udemy, or YouTube.\n"
    if not matched:
        feedback += "\n\n*Formatting Tip:*\n- Highlight your skills in a separate 'Skills' section.\n"
    return feedback

# ------------------- MAIN APP -------------------
def main():
    set_bg(BG_IMAGE)
    st.title("📄 Smart Resume Matcher")
    st.markdown("Match your resume with company expectations and get instant feedback!")

    st.sidebar.header("💼 HR / CEO Input")
    required_skills_input = st.sidebar.text_area("Enter Required Skills (comma-separated)",
                                                "Python, SQL, Machine Learning, Power BI")
    required_skills = [x.strip().lower() for x in required_skills_input.split(",")]

    st.header("📤 Upload Resume")
    uploaded_file = st.file_uploader("Upload a PDF Resume", type=["pdf"])

    if uploaded_file:
        with open(RESUME_DIR, "wb") as f:
            f.write(uploaded_file.read())

        resume_text = extract_text_from_pdf(open(RESUME_DIR, "rb"))

        st.subheader("📋 Resume Content")
        st.text_area("Extracted Resume Text", resume_text, height=250)

        user_skills = get_skills(resume_text)
        matched = [s for s in user_skills if s in required_skills]
        missing = [s for s in required_skills if s not in user_skills]
        score = calculate_score(matched, required_skills)

        st.subheader("✅ Skill Match Results")
        st.success(f"🔢 Match Score: {score}%")
        st.markdown(f"*✅ Matched Skills:* {', '.join(matched) if matched else 'None'}")
        st.markdown(f"*❌ Missing Skills:* {', '.join(missing) if missing else 'None'}")

        feedback = generate_feedback(matched, missing)
        if feedback:
            st.subheader("🧠 Feedback & Learning Tips")
            st.markdown(feedback)

        # ------------------- ANALYTICS -------------------
        st.subheader("📊 Analytics Dashboard")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Skill Match Pie Chart")
            fig1, ax1 = plt.subplots()
            ax1.pie([len(matched), len(missing)],
                    labels=["Matched", "Missing"],
                    colors=["#00C49F", "#FF5C5C"],
                    autopct="%1.1f%%", startangle=140)
            ax1.axis("equal")
            st.pyplot(fig1)

        with col2:
            st.markdown("### Skills Summary Table")
            df = pd.DataFrame({
                "Skill": matched + missing,
                "Status": ["Matched"] * len(matched) + ["Missing"] * len(missing)
            })
            st.dataframe(df.style.applymap(
                lambda x: 'background-color: #d4edda' if x == "Matched" else 'background-color: #f8d7da',
                subset=["Status"]
            ))

if __name__ == "__main__":
    import os
    os.system("streamlit run app.py --server.address=0.0.0.0 --server.port=5000")
