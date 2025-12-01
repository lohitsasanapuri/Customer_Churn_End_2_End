import os
import pandas as pd

import sys
sys.path.append(os.path.abspath("src"))

from data.load_data import load_data
from data.preprocess import preprocess_data
from features.build_features import build_features

# Configure 
DATA_PATH = r"data/raw/Telco-Customer-Churn.csv"
TARGET_COL = "Churn"

def main():
    print( " Testing Phase 1 : Load -> Preprocess -> Build Feature ")

    # 1. Load Data
    print(" Loading Data .. ")
    df = load_data(DATA_PATH)
    print(f" Data Loaded Shape : {df.shape}")
    print(df.head(3))

    # 2. Preprocess
    print(" Preprocessing Data ..")
    df_clean = preprocess_data(df,target_col = TARGET_COL)
    print(f" Data after preprocessing. shape {df_clean.shape}")
    print(df_clean.head(3))

    # 3. Build Featrures
    print(" Building Features .. ")
    df_features = build_features(df_clean,target_col=TARGET_COL)
    print(f"  Data after feature engineering. shape {df_features.shape}")
    print(df_features.head(3))

    print("Phase 1 Pipeline Completed")

if __name__ == "__main__":
    main()