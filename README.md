# StrataWork Global: Workforce Automation Risk Prediction & Reskilling Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io/)
[![ML Library](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end workforce analytics platform and Machine Learning recommendation system designed to evaluate occupational displacement risks driven by AI/automation and deliver dynamic, data-driven career reskilling pathways.

---

## 1. Executive Summary & Problem Statement

As artificial intelligence and workflow automation scale across global enterprise sectors, workforce planners and employees face mounting disruption. Traditional career guidance relies on static job classifications and lacks granular, data-backed insights into automation displacement probabilities and transferable skill alignment.

**StrataWork Global** resolves this problem through a dual-engine architecture:
1. **Supervised Multi-Class Classifier**: Evaluates worker profiles and assigns actionable displacement risk tiers (**Low**, **Medium**, **High**).
2. **TF-IDF NLP Career Recommender**: Maps displaced or at-risk worker skill profiles against high-growth target job roles using term frequency-inverse document frequency vectorization and cosine similarity matching.

---

## 2. Live Application & Portfolio Links

* **Live Interactive Web App**: [Click Here to Access the Streamlit App](https://YOUR-APP-NAME.streamlit.app)
* **GitHub Repository**: [https://github.com/IheomaNwuzor/ai-workforce-risk-recommender](https://github.com/IheomaNwuzor/ai-workforce-risk-recommender)

---

## 3. Exploratory Workforce Insights (EDA)

Exploratory analysis on the workforce dataset (`ai_job_market_insights.csv`) revealed key structural trends driving automation risk:

* **AI Adoption vs. Risk Correlation**: High enterprise AI adoption (>70%) strongly correlates with elevated automation risk in routine operational and administrative roles.
* **Salary Vulnerability Inversion**: Mid-tier salary bands ($45,000 - $85,000 USD) show high exposure to displacement, whereas low-skill manual roles and executive strategic roles exhibit lower immediate structural risk.
* **Sector Concentration**: Financial services, logistics, and back-office IT services show the highest proportion of high-risk worker classifications.

---

## 4. Automation Risk Prediction & Model Performance

### Predictive Risk Scoring Engine
The multi-class classification pipeline predicts an individual worker's vulnerability based on industry, company size, current role, salary band, and technology exposure score.

### Performance Evaluation Matrix

| Metric | Low Risk | Medium Risk | High Risk | Overall Weighted Avg |
| :--- | :--- | :--- | :--- | :--- |
| **Precision** | 0.88 | 0.84 | 0.89 | **0.87** |
| **Recall** | 0.86 | 0.85 | 0.90 | **0.87** |
| **F1-Score** | 0.87 | 0.84 | 0.89 | **0.87** |
| **Accuracy** | — | — | — | **87.2%** |

### Top Feature Importances
1. `AI_Adoption_Level` (28.4%)
2. `Automation_Risk_Score` (22.1%)
3. `Salary_USD` (16.8%)
4. `Company_Size` (12.3%)
5. `Industry_Sector` (10.5%)

---

## 5. Skill-Gap Reskilling & Career Mapping Engine

To map displaced workers into sustainable, low-risk career transitions, worker skills and target role descriptions are sanitized, tokenized, and transformed into high-dimensional vector space:

$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \log\left(\frac{\vert{}D\vert{}}{\vert{}\{d \in D : t \in d\}\vert{}}\right)$$

Match confidence is computed using Cosine Similarity between the user's skill vector $\mathbf{A}$ and target job database vector $\mathbf{B}$:

$$\text{Cosine Similarity} = \frac{\mathbf{A} \cdot \mathbf{B}}{\Vert{}\mathbf{A}\Vert{} \Vert{}\mathbf{B}\Vert{}}$$

---

## 6. Project Architecture & File Directory

```text
ai-workforce-risk-recommender/
├── .gitignore                      # Git exclusion patterns
├── Dockerfile                      # Container deployment specification
├── README.md                       # Comprehensive project documentation
├── app.py                          # Streamlit interactive application interface
├── requirements.txt                # Production dependency requirements
├── data/
│   ├── ai_job_market_insights.csv  # Raw workforce dataset
│   └── processed_ai_job_market.csv # Cleaned & feature-engineered dataset
├── models/
│   └── automation_risk_model.pkl   # Serialized Random Forest model artifact
└── src/
    ├── __init__.py
    ├── data_loader.py              # Data ingestion & schema checking
    ├── eda.py                      # Visualization & summary stats logic
    ├── model_trainer.py            # Model training & evaluation routines
    ├── preprocessor.py             # Feature encoding & transformation pipelines
    └── recommender.py              # TF-IDF vectorization & filtering engine




## 7. Author & License

* **Developer**: Iheoma Nwuzor
* **License**: MIT License — open-source software, free for evaluation and commercial use.