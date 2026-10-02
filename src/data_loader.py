"""
src/data_loader.py
------------------
Handles loading and schema validation for the AI Job Market dataset.
"""

from pathlib import Path
import pandas as pd


class DataLoader:
    """Class to manage data loading and baseline integrity checks."""

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.df = None

    def load_data(self) -> pd.DataFrame:
        """Loads dataset from CSV file."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"Dataset not found at path: {self.file_path.resolve()}")

        self.df = pd.read_csv(self.file_path)
        print(f"[INFO] Successfully loaded dataset: {self.df.shape[0]} rows, {self.df.shape[1]} columns.")
        return self.df

    def get_data_summary(self) -> dict:
        """Returns baseline health metrics of the raw data."""
        if self.df is None:
            raise ValueError("Data is not loaded. Call load_data() first.")

        return {
            "shape": self.df.shape,
            "missing_values": self.df.isnull().sum().to_dict(),
            "duplicate_rows": int(self.df.duplicated().sum()),
            "dtypes": {col: str(dtype) for col, dtype in self.df.dtypes.items()}
        }