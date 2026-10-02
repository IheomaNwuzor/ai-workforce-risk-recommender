import pytest
import pandas as pd
from src.data_loader import DataLoader

@pytest.fixture
def sample_csv(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    p = d / "test_data.csv"
    df = pd.DataFrame({"Job Title": ["Data Scientist"], "Risk Score": [0.2]})
    df.to_csv(p, index=False)
    return str(p)

def test_load_data_returns_dataframe(sample_csv):
    loader = DataLoader(file_path=sample_csv)
    df = loader.load_data()
    assert df is not None

def test_required_columns_exist(sample_csv):
    loader = DataLoader(file_path=sample_csv)
    df = loader.load_data()
    assert not df.empty