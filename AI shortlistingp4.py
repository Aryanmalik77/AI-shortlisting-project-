import streamlit as st
import pdfplumber
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import re


st.title("AI Resume Application")
required_skills = ["c++", "python", "mysql", "java", "flask", "javascript", "c"]

uploaded_files = st.file_uploader("Upload your resume",accept_multiple_files=True, type=["pdf"])
def extract_resume_details(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text+= page_text + ""
    return text.lower()

def extract_name(text):

    lines = text.split("\n")
    for line in lines:
     if line.strip():
         return line.strip()
def extract_skills(text):
    skills_list = ["python", "java", "c++", "mysql", "flask", "javascript", "machine learning", "excel"]
    found = [skill for skill in skills_list if skill in text]
    return found if found else ["No skills found"]

def extract_experience(text):
    match = re.search(r'(\d+)\s*year[s]?', text, re.IGNORECASE)
    if match:
        years = int(match.group(1))
        if years <= 25:
            return f"{years} years"
        else:
            return "Above 25 years not allowed"
    return "FRESHER"

def extract_education(text):
    education= (  "b.tech", "bachelor of technology","Btech",
        "b.sc", "bachelor of science",
        "m.sc", "master of science",
        "m.tech", "master of technology",
        "mba", "bba",
        "phd", "doctorate",
        "10th", "12th", "high school", "senior secondary",
        "diploma","Mtech","MCA")
    for education in education:
        if education in text:
            return education
        return" other degree"
if uploaded_files:
    result=[]
    for file in uploaded_files:
        text = extract_resume_details(file)
        name = extract_name(text)
        skills = extract_skills(text)
        experience = extract_experience(text)
        education = extract_education(text)

        result.append([name, skills, experience, education])
    df = pd.DataFrame(result, columns=["Name", "Skills", "Experience", "Education"])
    st.write("data as follows")
    st.dataframe(df)
    min_score = st.selectbox("select Minimum Score",options = [1,2,3])
    st.markdown(f"### 🛠️ Select exactly {min_score} skill(s)")
    selected_skills = st.multiselect(
    "Choose Skills",
    options=required_skills,
    default=required_skills[:min_score],
    max_selections=min_score)


    def match_skills(candidate_skill, selected_skill):
        matched = []
        for skill in selected_skill:
            if skill in candidate_skill:
                matched.append(skill)

        return matched


    df["Skills"].apply(lambda x: match_skills(x, selected_skills))
    df["Matched_skills"] = df["Skills"].apply(lambda x: match_skills(x, selected_skills))
    df["Score"] = df["Matched_skills"].apply(len)
    df_sorted = df.sort_values(by="Score")
    shortlist = df[df["Score"] >= min_score]
    shortlisted = shortlist.sort_values(by=["Score", "Experience"])
    shortlisted["Rank"] = range(1, len(shortlisted) + 1)
    shortlisted[["Rank", "Name", "Matched_skills", "Score", "Experience"]]
    total_required_skill = len(required_skills)
    df_sorted = df.sort_values(by="Score")
    shortlist = df[df["Score"] >= min_score]
    shortlisted = shortlist.sort_values(by=["Score", "Experience"])
    shortlisted["Rank"] = range(1, len(shortlisted) + 1)
    shortlisted[["Rank", "Name", "Matched_skills", "Score", "Experience"]]
    total_required_skill = len(required_skills)


    def match_percentage(skills):
        matched = []
        for skill in required_skills:
            if skill in skills:
                matched.append(skill)
        return (len(matched) / total_required_skill) * 100

    df["match%"] = df["Skills"].apply(match_percentage)
    df["shortlisted"] = df["Score"].apply(lambda x: 1 if x >= min_score else 0)
    df = df.fillna({"match%": 0, "Score": 0})
    X = df[["match%", "Score"]]
    Y = df["shortlisted"]
    if len(df) > 1:
        if len(Y.unique()) >1:
            test_size = 0.2 if len(df) >= 5 else 0.5
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = test_size, random_state=42)
            le = LogisticRegression()
            le.fit(X_train,Y_train)
            y_pred = le.predict(X_test)
            df["prediction"] = le.predict(X)
            st.write("Accuracy:", accuracy_score(Y_test, y_pred))
            colors = df["prediction"].map({0: "red", 1: "green"})
            figure = plt.figure(figsize=(10, 6))
            plt.scatter(df["match%"], df["Score"], c=colors, s=100, edgecolor='k')  # scatter plot for better visualization
            plt.title("Resume Match % vs Score with Logistic Regression Prediction")
            st.pyplot(figure)
        else:
            st.warning("⚠ Logistic Regression cannot run because all resumes fall into the same class (all shortlisted or none).")



   




