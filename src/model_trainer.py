"""
src/model_trainer.py
--------------------
Handles training, cross-validation, evaluation, and serialization 
for Multi-Class Automation Risk Classification.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score


class ModelTrainer:
    """Class to train, evaluate, and serialize Classification models."""

    def __init__(self, df: pd.DataFrame, target_col: str = "Automation_Risk_Encoded"):
        self.df = df
        self.target_col = target_col
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}

    def prepare_data(self):
        """Prepares features and target, removing any potential target leakage."""
        self.y = self.df[self.target_col]

        # Explicitly drop target columns, raw target string, and leaked interaction variables
        drop_cols = [self.target_col, "Automation_Risk", "Risk_Salary_Index", "Salary_Tier"]
        features_df = self.df.drop(columns=[c for c in drop_cols if c in self.df.columns])

        # Keep strictly numerical/boolean features
        self.X = features_df.select_dtypes(include=["number", "bool"])

        # Stratified Train/Test Split (80/20)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        print(f"[INFO] Classification Features Used: {self.X.shape[1]}")
        print(f"[INFO] Train shape: {self.X_train.shape[0]} | Test shape: {self.X_test.shape[0]}")

    def train_classifiers(self):
        """Trains multi-class classification models."""
        print("\n--- Training Multi-Class Classification Models ---")

        # 1. Logistic Regression
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(self.X_train, self.y_train)
        self.models["Logistic Regression"] = lr

        # 2. Random Forest Classifier
        rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        rf.fit(self.X_train, self.y_train)
        self.models["Random Forest"] = rf

        # 3. Gradient Boosting Classifier
        gb = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
        gb.fit(self.X_train, self.y_train)
        self.models["Gradient Boosting"] = gb

    def evaluate_models(self):
        """Evaluates classification models using Accuracy, Multi-class ROC-AUC, and F1-Scores."""
        print("\n==================================================")
        print("     AUTOMATION RISK CLASSIFICATION RESULTS       ")
        print("==================================================")

        for name, model in self.models.items():
            y_pred = model.predict(self.X_test)
            y_proba = model.predict_proba(self.X_test)

            acc = accuracy_score(self.y_test, y_pred)
            auc = roc_auc_score(self.y_test, y_proba, multi_class="ovr")

            print(f"\nModel: {name}")
            print(f"  - Test Accuracy     : {acc * 100:.2f}%")
            print(f"  - Multi-class ROC-AUC: {auc:.4f}")
            print("  - Detailed Classification Report:")
            print(classification_report(self.y_test, y_pred, target_names=["Low", "Medium", "High"], zero_division=0))

    def save_best_model(self, model_name: str = "Random Forest", output_path: str = "models/automation_risk_model.pkl"):
        """Saves trained classification model artifact to disk for application deployment."""
        model = self.models.get(model_name)
        if model is None:
            print(f"[ERROR] Model '{model_name}' not found.")
            return

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        artifact = {
            "model": model,
            "feature_names": self.X.columns.tolist(),
            "target_names": ["Low", "Medium", "High"]
        }
        joblib.dump(artifact, path)
        print(f"\n[SUCCESS] Multi-Class Model ({model_name}) saved to: {path.resolve()}")