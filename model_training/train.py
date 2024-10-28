import argparse
from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LassoCV

def read_data(path: Path) -> pd.DataFrame:
    """Read training data into pandas."""
    df = pd.read_csv(path)
    return df






if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train power consumption models.")

