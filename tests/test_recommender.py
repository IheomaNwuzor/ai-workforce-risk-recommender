import pytest
from src.recommender import CareerRecommender

@pytest.fixture
def recommender():
    return CareerRecommender()

def test_recommendation_output_structure(recommender):
    results = recommender.recommend(
        current_job="AI Researcher",
        skills=["Python", "SQL"],
        min_salary=0,
        target_risk="All"
    )
    assert isinstance(results, list)
    if len(results) > 0:
        assert "target_role" in results[0]
        assert "match_score" in results[0]