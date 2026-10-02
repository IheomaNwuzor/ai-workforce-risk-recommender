"""
src/eda.py
----------
Performs Exploratory Data Analysis (EDA) on the dataset to reveal structure,
distributions, and relationships between features.
"""

import pandas as pd


class DataExplorer:
    """Class to run comprehensive exploratory data analysis."""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def display_overview(self):
        """Prints categorical unique values and distributions."""
        print("\n--- CATEGORICAL FEATURE DISTRIBUTIONS ---")
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            print(f"\nFeature: {col} ({self.df[col].nunique()} unique values)")
            print(self.df[col].value_counts().to_dict())

    def display_numerical_stats(self):
        """Prints summary statistics for numerical columns."""
        print("\n--- NUMERICAL FEATURE STATISTICS ---")
        print(self.df.describe().T)

    def analyze_cross_tabulations(self):
        """Analyzes cross-tabulations between key categorical features."""
        print("\n--- CROSS-TABULATION: Job Title vs Required Skills ---")
        job_skill = pd.crosstab(self.df['Job_Title'], self.df['Required_Skills'])
        print(job_skill)

        print("\n--- CROSS-TABULATION: Job Title vs Automation Risk ---")
        job_risk = pd.crosstab(self.df['Job_Title'], self.df['Automation_Risk'])
        print(job_risk)