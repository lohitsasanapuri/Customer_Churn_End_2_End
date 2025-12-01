import os
import pandas as pd
import mlflow
import glob
from pathlib import Path

# 1. Setup Paths Dynamicallly
CURRENT_DIR = Path(__file__).resolve().parent
MODEL_FOLDER_NAME = "m-e2655f75ee9a490ab154aef6b4cfbe19"

# Construct the absolute path to the artifacts folder
MODEL_PATH = CURRENT_DIR.parent / "app" / "models" / MODEL_FOLDER_NAME / "artifacts"

# 2. Create two versions of the path
# For MLflow: Needs a URI (starts with file:///)
MODEL_URI = MODEL_PATH.as_uri()
# For OS operations: Needs a standard string path (starts with C:\)
MODEL_PATH_STR = str(MODEL_PATH)

print(f"Attempting to load model from URI: {MODEL_URI}")

# 3. Load Model
try:
    model = mlflow.pyfunc.load_model(MODEL_URI)
    print(f"Model Loaded Successfully From {MODEL_URI}")
    
    # Update MODEL_DIR to the successful path so subsequent steps use the correct one
    # We use the string path for file reading later
    ACTIVE_MODEL_DIR_STR = MODEL_PATH_STR

except Exception as e:
    print(f"Primary load failed: {e}")
    print("Attempting fallback to local mlruns...")

    try:
        # Fallback Logic
        # Note: This looks for mlruns relative to where you run the command
        local_model_paths = glob.glob("mlruns/*/*/models")
        
        if not local_model_paths:
            # Try looking one level up if running from src
            local_model_paths = glob.glob("../mlruns/*/*/models")

        if local_model_paths:
            latest_model = max(local_model_paths, key=os.path.getmtime)
            print(f"Fallback: Found latest model at {latest_model}")
            
            # Load the fallback model
            model = mlflow.pyfunc.load_model(latest_model)
            
            # Update the directory string to point to this new fallback location
            # assuming feature_columns.txt is inside the artifacts folder of the run
            ACTIVE_MODEL_DIR_STR = latest_model
            
            print(f"Fallback: Loaded model from {latest_model}")
        else:
            raise Exception("No model found in primary path OR local mlruns")

    except Exception as fallback_error:
        raise Exception(f"Failed to load Model. Primary error: {e}. Fallback error: {fallback_error}")

# 4. Feature Schema Loading
try:
    # FIX: Use the standard string path (ACTIVE_MODEL_DIR_STR), NOT the URI
    feature_file = os.path.join(ACTIVE_MODEL_DIR_STR, "feature_columns.txt")
    
    print(f"Loading features from: {feature_file}")
    
    with open(feature_file, "r") as f:
        FEATURE_COLS = [ln.strip() for ln in f if ln.strip()]
        
    print(f"Loaded {len(FEATURE_COLS)} feature columns from training")
    
except Exception as e:
    raise Exception(f"Failed to load feature columns: {e}")

# Deterministic binary feature mappings
BINARY_MAP = {
    "gender": {"Female": 0, "Male": 1},
    "Partner": {"No": 0, "Yes": 1},
    "Dependents": {"No": 0, "Yes": 1},
    "PhoneService": {"No": 0, "Yes": 1},
    "PaperlessBilling": {"No": 0, "Yes": 1},
}

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

def _serve_transform(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Clean Columns
    df.columns = df.columns.str.strip()
    
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df[c] = df[c].fillna(0)

    for c, mapping in BINARY_MAP.items():
        if c in df.columns:
            df[c] = (
                df[c]
                .astype(str)
                .str.strip()
                .map(mapping)
                .astype("Int64")
                .fillna(0)
                .astype(int)
            )
            
    # FIX: Corrected pandas syntax for selecting object columns
    # Old/Buggy: df.select_dtypes(include=["object"].columns)
    # New/Fixed: df.select_dtypes(include=["object"]).columns
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()

    if obj_cols:
        df = pd.get_dummies(df, columns=obj_cols, drop_first=True)

    bool_cols = df.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)

    # Reindex ensures we have exactly the columns the model expects, in the right order
    df = df.reindex(columns=FEATURE_COLS, fill_value=0)
    return df

def predict(input_dict: dict) -> str:
    df = pd.DataFrame([input_dict])
    df_enc = _serve_transform(df)
    
    try:
        preds = model.predict(df_enc)

        if hasattr(preds, "tolist"):
            preds = preds.tolist()
            
        if isinstance(preds, (list, tuple)) and len(preds) == 1:
            result = preds[0]
        else:
            result = preds
            
    except Exception as e:
        raise Exception(f"Model predictions failed: {e}")
    
    if result == 1:
        return "Likely to Churn"
    else:
        return "Not Likely to Churn"