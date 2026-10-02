import streamlit as st
import pandas as pd
import numpy as np
import joblib
from src.recommender import get_recommendations

st.set_page_config(page_title="StrataWork Global", layout="wide")

# Safe Model Loading
@st.cache_resource
def load_model():
    try:
        return joblib.load("models/automation_risk_model.pkl")
    except Exception as e:
        st.error(f"Failed to load risk model: {e}")
        return None

# Safe Data Loading
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data/ai_job_market_insights.csv")
        return df
    except Exception as e:
        st.error(f"Failed to load workforce dataset: {e}")
        return pd.DataFrame()

model_artifact = load_model()
df = load_data()

# App Header
st.title("💼 StrataWork Global: Workforce Transition Portal")
st.subheader("AI-Powered Occupational Displacement Risk & Career Reskilling Engine")

if df.empty:
    st.warning("Workforce dataset is empty or unreadable. Please check the data directory.")
    st.stop()

# Sidebar Controls
st.sidebar.header("Employee Profile Input")
job_titles = sorted(df["Job_Title"].dropna().unique()) if "Job_Title" in df.columns else []
industries = sorted(df["Industry"].dropna().unique()) if "Industry" in df.columns else []

job_title = st.sidebar.selectbox("Current Job Title", job_titles)
industry = st.sidebar.selectbox("Industry Sector", industries)
company_size = st.sidebar.selectbox("Company Size", ["Small", "Medium", "Large"])
ai_adoption = st.sidebar.select_slider("Company AI Adoption Level", options=["Low", "Medium", "High"], value="High")
remote_friendly = st.sidebar.radio("Remote Friendly", ["Yes", "No"])
user_skills = st.sidebar.text_area("Your Current Skills (comma-separated)", "SQL, Python, Data Visualization, Excel, Tableau")

# Dynamic Filters for Career Mapping
st.sidebar.markdown("---")
st.sidebar.header("Transition Filters")
min_salary_filter = st.sidebar.number_input("Minimum Target Salary ($)", min_value=0, value=0, step=5000)
max_risk_filter = st.sidebar.selectbox("Filter Target Risk", ["All", "Low", "Medium", "High"])

# Main Navigation Tabs
tab1, tab2, tab3 = st.tabs(["📊 Risk Assessment", "🔍 Career Mapping & Reskilling", "📈 Workforce Insights"])

# TAB 1: RISK ASSESSMENT
with tab1:
    st.header("Occupational Displacement Risk Prediction")
    
    if st.button("Evaluate Automation Risk", type="primary"):
        if model_artifact is None:
            st.error("Model artifact is unavailable.")
        else:
            try:
                if isinstance(model_artifact, dict):
                    model = model_artifact["model"]
                    expected_features = model_artifact.get("feature_names", [])
                    classes = model_artifact.get("target_names", getattr(model, "classes_", ["Low", "Medium", "High"]))
                else:
                    model = model_artifact
                    expected_features = getattr(model, "feature_names_in_", [])
                    classes = getattr(model, "classes_", ["Low", "Medium", "High"])

                # One-hot input map alignment
                input_dict = {feat: 0 for feat in expected_features}

                if f"AI_Adoption_Level_{ai_adoption}" in input_dict:
                    input_dict[f"AI_Adoption_Level_{ai_adoption}"] = 1
                if f"Company_Size_{company_size}" in input_dict:
                    input_dict[f"Company_Size_{company_size}"] = 1
                if f"Industry_{industry}" in input_dict:
                    input_dict[f"Industry_{industry}"] = 1
                if f"Job_Title_{job_title}" in input_dict:
                    input_dict[f"Job_Title_{job_title}"] = 1
                if "Remote_Friendly" in input_dict:
                    input_dict["Remote_Friendly"] = 1 if remote_friendly == "Yes" else 0
                if "Has_AI_ML" in input_dict:
                    user_skills_lower = user_skills.lower()
                    input_dict["Has_AI_ML"] = int(any(kw in user_skills_lower for kw in ["python", "machine learning", "ai", "sql", "deep learning"]))

                input_df = pd.DataFrame([input_dict])

                raw_prediction = model.predict(input_df)[0]
                label_map = {0: "Low", 1: "Medium", 2: "High"}
                
                if isinstance(raw_prediction, (int, np.integer)):
                    predicted_label = label_map.get(int(raw_prediction), str(raw_prediction))
                else:
                    predicted_label = str(raw_prediction)

                probabilities = model.predict_proba(input_df)[0] if hasattr(model, "predict_proba") else [1.0] * len(classes)

                col1, col2, col3 = st.columns(3)
                pred_str = str(predicted_label).strip().lower()
                
                if pred_str == "high":
                    col1.error(f"### Predicted Risk: {predicted_label}")
                elif pred_str == "medium":
                    col1.warning(f"### Predicted Risk: {predicted_label}")
                else:
                    col1.success(f"### Predicted Risk: {predicted_label}")

                col2.metric("Industry Sector", industry)
                col3.metric("AI Adoption Exposure", ai_adoption)

                st.markdown("#### Model Confidence Breakdown")
                prob_df = pd.DataFrame({
                    "Risk Level": ["Low", "Medium", "High"] if len(classes) == 3 else classes,
                    "Probability": [f"{p * 100:.1f}%" for p in probabilities]
                })
                st.dataframe(prob_df, use_container_width=True)

                st.info("💡 **Risk Factor Insights**: High exposure to routine syntax generation, manual dashboard updates, and transactional reporting increases automation vulnerability.")

            except Exception as e:
                st.error(f"Inference Error: {e}")

# TAB 2: CAREER MAPPING
with tab2:
    st.header("Vectorized Skill Adjacency & Career Reskilling Recommendations")
    
    user_profile = {
        "job_title": job_title,
        "skills": user_skills
    }
    
    rec_df = get_recommendations(
        user_profile, 
        df, 
        top_n=5, 
        min_salary=min_salary_filter, 
        max_risk=max_risk_filter
    )
    
    st.markdown(f"**Based on your skill profile:** '{user_skills}'")
    
    if rec_df.empty:
        st.warning("No career transition paths matched your selected filter criteria.")
    else:
        for idx, row in rec_df.iterrows():
            with st.expander(f"Recommended Path: {row['Job_Title']} ({row.get('Industry', 'N/A')})", expanded=True):
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Automation Risk", row.get("Automation_Risk", "N/A"))
                col2.metric("Target Salary (USD)", f"${row.get('Salary_USD', 0):,.0f}")
                col3.metric("Growth Projection", row.get("Growth_Projection", "Stable"))
                col4.metric("TF-IDF Match", f"{row['Skill_Match']}%")
        
        # Report Export Feature
        st.markdown("---")
        csv_data = rec_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Career Transition Plan (CSV)",
            data=csv_data,
            file_name=f"career_transition_{job_title.lower().replace(' ', '_')}.csv",
            mime="text/csv"
        )

# TAB 3: WORKFORCE INSIGHTS
with tab3:
    st.header("StrataWork Global Workforce Intelligence Insights")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Displacement Risk Distribution by Industry")
        if "Industry" in df.columns and "Automation_Risk" in df.columns:
            st.bar_chart(df.groupby(["Industry", "Automation_Risk"]).size().unstack().fillna(0))
        else:
            st.warning("Required schema columns missing for risk distribution.")
    
    with col2:
        st.subheader("Salary vs. Automation Risk Distribution")
        if "Salary_USD" in df.columns and "Automation_Risk" in df.columns:
            st.scatter_chart(data=df, x="Automation_Risk", y="Salary_USD")
        else:
            st.warning("Required schema columns missing for salary chart.")