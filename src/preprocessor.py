"""
src/preprocessor.py
-------------------
Handles encoding and feature engineering for Multi-Class Automation Risk Classification.
"""

from pathlib import Path
import pandas as pd


class DataPreprocessor:
    """Class to clean and encode features for Automation Risk Classification."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def encode_target(self) -> pd.DataFrame:
        """Encodes target variable Automation_Risk into numerical classes."""
        target_map = {"Low": 0, "Medium": 1, "High": 2}
        if "Automation_Risk" in self.df.columns:
            self.df["Automation_Risk_Encoded"] = self.df["Automation_Risk"].map(target_map)
        return self.df

    def engineer_features(self) -> pd.DataFrame:
        """Engineers non-leaky predictive features."""
        if "Required_Skills" in self.df.columns:
            # Skill complexity features
            self.df["Skill_Count"] = self.df["Required_Skills"].apply(
                lambda x: len([s.strip() for s in str(x).split(",") if s.strip()])
            )
            self.df["Has_AI_ML"] = self.df["Required_Skills"].apply(
                lambda x: 1 if any(k in str(x).lower() for k in ["ai", "machine learning", "deep learning", "python"]) else 0
            )

        # Salary tier relative to dataset
        if "Salary_USD" in self.df.columns:
            self.df["Salary_Per_Skill"] = self.df["Salary_USD"] / (self.df["Skill_Count"] + 1)

        return self.df

    def encode_categoricals(self) -> pd.DataFrame:
        """One-hot encodes categorical predictor features."""
        ignore_cols = ["Automation_Risk", "Automation_Risk_Encoded", "Required_Skills"]
        cat_cols = [c for c in self.df.select_dtypes(include=["object", "category"]).columns if c not in ignore_cols]
        
        self.df = pd.get_dummies(self.df, columns=cat_cols, drop_first=True)
        return self.df

    def run_preprocessing_pipeline(self, output_path: str = "data/processed_ai_job_market.csv") -> pd.DataFrame:
        """Executes full classification preprocessing sequence."""
        self.encode_target()
        self.engineer_features()
        self.encode_categoricals()

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(path, index=False)
        print(f"[SUCCESS] Classification Preprocessing complete. Target: Automation_Risk_Encoded")
        return self.df