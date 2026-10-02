# StrataWork Global: Workforce Transition & Career Reskilling Engine

An end-to-end workforce analytics platform and machine learning recommendation system designed to evaluate occupational displacement risks driven by AI automation and provide data-driven career reskilling pathways.

---

## Executive Summary & Problem Statement

As artificial intelligence and workflow automation accelerate across industry sectors, workforce planners and employees face unprecedented disruption. Traditional career guidance lacks granular, data-backed insights regarding displacement probability and skill transferability. 

**StrataWork Global** solves this gap by providing:
1. **Displacement Risk Prediction**: A supervised machine learning classifier evaluating occupational risk based on industry, company scale, and AI adoption metrics.
2. **TF-IDF Career Match Engine**: An NLP-driven vector similarity model mapping user skill sets against target market opportunities.
3. **Interactive Workforce Analytics**: High-level market visualizations mapping salary distributions against displacement vulnerability.

---

## Key Features & System Architecture

### 1. Risk Evaluation Engine (`app.py`)
* Evaluates worker risk profiles into categorical output tiers: **Low**, **Medium**, or **High**.
* Provides feature probability breakdowns across model confidence scores.

### 2. Advanced TF-IDF Career Recommender (`src/recommender.py`)
* Replaces raw string matching with term frequency-inverse document frequency (TF-IDF) vectorization.
* Calculates cosine similarity scores between user skill sets and target role requirements.
* Supports multi-variable filtering by minimum target salary and maximum risk tolerance.
* Features automated CSV report generation for user export.

### 3. Exploratory Workforce Intelligence
* Aggregates industry displacement distributions using Streamlit charts.
* Visualizes salary variance across automation risk tiers.

---

## Repository Structure

```text
ai_job_market_analysis/
├── app.py                      # Main Streamlit web application interface
├── src/
│   └── recommender.py          # TF-IDF vectorization & filtering module
├── models/
│   └── automation_risk_model.pkl # Trained Random Forest artifact
├── data/
│   └── ai_job_market_insights.csv # Workforce dataset
├── requirements.txt            # Production dependencies
├── .gitignore                  # Git exclusions
└── README.md                   # Project documentation