import pytest
import pandas as pd
from src.data_loader import load_data

def test_load_data_returns_dataframe():
    df = load_data()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty

def test_required_columns_exist():
    df = load_data()
    expected_columns = ["Job Title", "Industry Sector", "Automation Risk"]
    for col in expected_columns:
        assert col in df.columns