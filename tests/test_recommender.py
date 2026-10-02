import pytest
import pandas as pd
from src.recommender import get_recommendations

def test_recommendation_output_structure():
    user_profile = {"skills": "Data Scientist", "role": "Data Scientist"}
    
    # Use 'Job_Title' with an underscore to match src/recommender.py
    df = pd.DataFrame({
        "Job_Title": ["Data Scientist", "ML Engineer"],
        "Risk Score": [0.2, 0.4]
    })
    
    results = get_recommendations(user_profile, df)
    assert results is not None