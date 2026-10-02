"""
main.py
-------
Pipeline execution engine for Multi-Class Automation Risk Classification.
"""

from src.data_loader import DataLoader
from src.preprocessor import DataPreprocessor
from src.model_trainer import ModelTrainer


def main():
    print("==================================================")
    print("  STRATAWORK GLOBAL - WORKFORCE TRANSITION PIPELINE ")
    print("==================================================")

    # 1. Ingestion
    loader = DataLoader("data/ai_job_market_insights.csv")
    df = loader.load_data()

    # 2. Preprocessing & Feature Engineering
    preprocessor = DataPreprocessor(df)
    processed_df = preprocessor.run_preprocessing_pipeline("data/processed_ai_job_market.csv")

    # 3. Model Training & Multi-Class Evaluation
    trainer = ModelTrainer(processed_df, target_col="Automation_Risk_Encoded")
    trainer.prepare_data()
    trainer.train_classifiers()
    trainer.evaluate_models()

    # 4. Save Model Artifact for Application Deployment
    trainer.save_best_model(model_name="Random Forest", output_path="models/automation_risk_model.pkl")


if __name__ == "__main__":
    main()