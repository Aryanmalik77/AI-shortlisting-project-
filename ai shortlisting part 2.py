
import streamlit as st

import pandas as pd
import matplotlib.pyplot as plt
st.title("AI Resume Shortlisting App")
required_skills = ["c++", "python", "mysql", "java", "flask", "javascript", "c"]
st.write("sample_resumes_50.csv")
df = pd.read_csv("sample_resumes_50.csv")
df["Skills"] = df["Skills"].str.lower()
import string
exclude =" !#$%&\'!#$%&\'()*"
def removepuntautions(text):
    for char in exclude:
        text = text.replace(char, "")
    return text
df["Skills"]=df["Skills"].apply(removepuntautions)
df = df[["Name","Phone","Skills","Experience (Years)"]]
def match_skills(candidate_skill):
    matched = []
    for skill in required_skills:
        if skill in candidate_skill:
            matched.append(skill)
            score = len(matched)
    return matched
df["Skills"].apply(match_skills)
df["Matched_skills"] =df["Skills"].apply(match_skills)
def match_skills(candidate_skill ):
    matched = []
    for skill in required_skills:
        if skill in candidate_skill:
            matched.append(skill)
            score = len(matched)
    return score
df["Skills"].apply(match_skills)
df["Score"] = df["Skills"].apply(match_skills)
df_sorted = df.sort_values(by="Score")
st.markdown("## Select Minimum Score" )

min_score = st.selectbox("select Minimum Score",options = [1,2,3])
shortlist = df[df["Score"]>=min_score]
shortlisted = shortlist.sort_values(by = ["Score","Experience (Years)"])
shortlisted["Rank"] = range(1, len(shortlisted) + 1)
shortlisted[["Rank", "Name", "Matched_skills", "Score", "Experience (Years)"]]
total_required_skill = len(required_skills)
def match_percentage(skills):
  matched = []
  for skill in required_skills:
    if skill in skills:
      matched.append(skill)
  return (len(matched)/total_required_skill)*100
df["match%"]= df["Skills"].apply(match_percentage)
df.sort_values(by= "match%")
fig = plt.figure(figsize=(6,4))
plt.bar(df["match%"],df["Score"], color = "red")
plt.title("Match Percentage vs Score")
plt.show()
st.pyplot(fig)


