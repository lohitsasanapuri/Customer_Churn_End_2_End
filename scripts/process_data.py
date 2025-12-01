import os
import sys

# Make Src Importable

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"..")))

from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.features.build_features import build_features

Raw = "data/raw/Telco-Customer-Churn.csv"
Out = "data/processed/Telco_Churn_Processed.csv"

# 1. Data Loading
df = load_data(Raw)
# 2. Data Preprocessing
df = preprocess_data(df)


if "Churn" in df.columns and df['Churn'].dtype =='object':
    df['Churn'] = df['Churn'].str.strip().map({"No":0, "Yes":1}).astype("int64")


# 3. sanity checks

assert df["Churn"].isna().sum() == 0, "Churn has NaNs afterr preprocessing"
assert set(df["Churn"].unique()) <= {0,1}, " Churn not 0/1 after preprocessing"

# 4. features
df_precessed = build_features(df)

# 5. Persist data
os.makedirs(os.path.dirname(Out), exist_ok=True)
df_precessed.to_csv(Out, index=False)
print(f" Processed dataset saved to {Out} | Shape : {df_precessed.shape}")