"""
Runs Sequentially : Load -> Validate -> preprocess -> feature engineering
"""
import os
import sys
import time
import argparse
import pandas as pd
import mlflow
import json
import joblib
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,precision_score, recall_score, f1_score,roc_auc_score
from xgboost import XGBClassifier

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"..")))

from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.features.build_features import build_features
from src.utils.validate_data import validate_telco_data

def main(args):

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
    mlrun_path = args.mlflow_uri or '../mlruns'
    mlflow.set_tracking_uri(mlrun_path)
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run():
        mlflow.log_param("model","xgboost")
        mlflow.log_param("threshold", args.threshold)
        mlflow.log_param("test_size", args.test_size)

        print(" Loading Data ..")
        df = load_data(args.input)
        print(f" Data loaded : {df.shape[0]} rows, {df.shape[1]} columns")

        print(" Validating Data with great expectataions")
        is_valid, failed = validate_telco_data(df)
        mlflow.log_metric("data_quality_pass",int(is_valid))

        if not is_valid:
            mlflow.log_text(json.dumps(failed, indent = 2 ),artifact_file=" failed_expectataions.json" )
            raise ValueError(f"Data Quality check failed. issue : {failed}")
        else:
            print(" Data Validation Passed .. Logged to Mlflow ..")

        print("Preprocessing data ..")
        df = preprocess_data(df)

        process_data_path = os.path.join(project_root,"data","processed","Telco_Churn_Processed.csv")
        os.makedirs(os.path.dirname(process_data_path), exist_ok= True)
        df.to_csv(process_data_path,index=False)
        print(f" Processed date and save to {process_data_path} | Shape {df.shape}")

        print(" Building Features")
        target = args.target
        if target not in df.columns:
            raise ValueError(f"Target Column {target} column  not found in the data")
        
        df_enc = build_features(df, target_col=target)

        # conversting boolean into integers
        for c in df_enc.select_dtypes(include=["bool"]).columns:
            df_enc[c] = df_enc[c].astype(int)
        print(f" Feature engineering completed : {df_enc.shape[1]} features")

        #  Svaing Feature metadata
        artifacts_dir = os.path.join(project_root, "artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)

        feature_cols = list(df_enc.drop(columns=[target]).columns)

        # saving Locally for the deployment
        with open(os.path.join(artifacts_dir, "feature_columns.json"), 'w') as f:
            json.dump(feature_cols,f)

        with open(os.path.join(artifacts_dir, "feature_columns.txt"), 'w') as f:
            json.dump("\n".join(feature_cols),f)
        
        mlflow.log_text("\n".join(feature_cols),artifact_file="feature_columns.txt")

        preprocesing_artifact ={
            "feature_columns" : feature_cols,
            "target" : target
        }
        joblib.dump(preprocesing_artifact,os.path.join(artifacts_dir,"preprocessing.pkl"))
        mlflow.log_artifact(os.path.join(artifacts_dir,"preprocessing.pkl"))
        print(f"Saved {len(feature_cols)} for the serving consistancy")

        print(" Splitting Data")
        X = df_enc.drop(columns=[target])
        y = df_enc[target]
        X_train, X_test, y_train, y_test = train_test_split(
            X,y,
            test_size= args.test_size,
            stratify=y,
            random_state=41
        )
        print(f" Train : {X_train.shape[0]} samples | Test : {X_test.shape[0]} samples ")

        scale_pos_weight = (y_train == 0).sum()/(y_train == 1).sum()
        print(f" Class Imbalance ratio: {scale_pos_weight:.2f} -- applied to positive class")
        print("building XGboost Model")

        model = XGBClassifier(
            n_estimators = 300,
            learning_rate = 0.03,
            max_depth = 7,
            
            subsample = 0.95,
            colsample_bytree= 0.98,

            n_jobs=-1,
            random_state = 42,
            eval_metric ="logloss",

            scale_pos_weight= scale_pos_weight
        )

        train_start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - train_start
        mlflow.log_metric("train_time",train_time)
        print(f"model Trained in {train_time:.2f} seconds")

        print(" Evaluating the Model Performance ")

        eval_time = time.time()
        proba = model.predict_proba(X_test)[:,1]
        y_pred = (proba >= args.threshold).astype(int)
        pred_time = time.time()- eval_time
        mlflow.log_metric("pred_time", pred_time)

        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test,y_pred)
        f1 = f1_score(y_test,y_pred)
        roc_auc = roc_auc_score(y_test,proba)

        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("roc_auc",roc_auc)

        print(" Model Performance .. " )
        print(f" precision : {precision} | recall :  {recall}")
        print(f" F1 Score : {f1} | roc_auc { roc_auc}")

        print(" Saving Model to Mlflow ")
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model"
        )
        print(" Model Saved to mlflow ")

        print(f"   Performance Summary:")
        print(f"   Training time: {train_time:.2f}s")
        print(f"   Inference time: {pred_time:.4f}s")
        print(f"   Samples per second: {len(X_test)/pred_time:.0f}")
        
        print(f" Detailed Classification Report:")
        print(classification_report(y_test, y_pred, digits=3))

if __name__ == "__main__" :
    p = argparse.ArgumentParser(description= " Run Churn pipeline with XGBoost + Mlflow")
    p.add_argument("--input", type=str, required= True,
                   help=" Path to csv e.g: data/raw/Telco-Customer-Churn.csv ")
    p.add_argument("--target", type=str, default="Churn")
    p.add_argument("--threshold", type=float, default=0.35)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--experiment", type=str, default=" Telco Churn - XGBOOST")
    p.add_argument("--mlflow_uri", type=str, default=None,
                   help=" Override Mlflow tracking URI, else uses project_root/mlruns ")
    args = p.parse_args()
    main(args)