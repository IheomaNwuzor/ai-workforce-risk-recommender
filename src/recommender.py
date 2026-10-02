import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_tfidf_similarity(user_skills_str, target_skills_series):
    """
    Computes TF-IDF vector cosine similarity between user skills and dataset skills.
    """
    if not user_skills_str or str(user_skills_str).strip() == "":
        return np.zeros(len(target_skills_series))
        
    # Clean text inputs
    cleaned_user = str(user_skills_str).replace(",", " ")
    cleaned_targets = target_skills_series.fillna("").astype(str).str.replace(",", " ")
    
    corpus = [cleaned_user] + cleaned_targets.tolist()
    
    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    # Cosine similarity between user vector (idx 0) and target vectors (idx 1:)
    user_vector = tfidf_matrix[0]
    target_vectors = tfidf_matrix[1:]
    
    similarities = cosine_similarity(user_vector, target_vectors).flatten()
    return np.round(similarities * 100, 1)

def get_recommendations(user_profile, dataset, top_n=3, min_salary=0, max_risk=None):
    """
    Generates advanced career recommendations with TF-IDF similarity, salary filtering, and risk limits.
    """
    if dataset.empty:
        return pd.DataFrame()

    user_skills = user_profile.get("skills", "")
    current_job = user_profile.get("job_title", "")
    
    # Filter out current job title
    filtered_df = dataset[dataset["Job_Title"] != current_job].copy()
    
    # Apply user-defined filters
    if "Salary_USD" in filtered_df.columns and min_salary > 0:
        filtered_df = filtered_df[filtered_df["Salary_USD"] >= min_salary]
        
    if "Automation_Risk" in filtered_df.columns and max_risk:
        if max_risk != "All":
            filtered_df = filtered_df[filtered_df["Automation_Risk"] == max_risk]

    if filtered_df.empty:
        return pd.DataFrame()

    # Identify skills column dynamically
    skill_col = "Required_Skills" if "Required_Skills" in filtered_df.columns else "Skills"
    if skill_col not in filtered_df.columns:
        filtered_df["Skill_Match"] = 0.0
        return filtered_df.head(top_n)

    # Compute TF-IDF similarities
    filtered_df["Skill_Match"] = calculate_tfidf_similarity(user_skills, filtered_df[skill_col])
    
    # Sort by best similarity score
    recommendations = filtered_df.sort_values(by="Skill_Match", ascending=False)
    
    return recommendations.head(top_n)