import os
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dagshub

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Dagshub
# dagshub.init(repo_owner='ChristianNathanielP', repo_name='covertype-msml', mlflow=True)
# mlflow.set_tracking_uri("https://dagshub.com/ChristianNathanielP/covertype-msml.mlflow")

def save_confusion_matrix(y_true, y_pred, labels, title, filename):
    os.makedirs("artifacts", exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()

    path = os.path.join("artifacts", filename)
    plt.savefig(path)
    plt.close()

    return path

if __name__ == '__main__':
    # Load Dataset
    df = pd.read_csv("preprossed_dataset/covertype_processed.csv")

    X = df.drop('Cover_Type', axis=1)
    y = df['Cover_Type']

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Aktifkan MLFlow
    # mlflow.set_tracking_uri("http://127.0.0.1:5000/")
    # mlflow.set_experiment("CoverType_Experiment")

    # Train Random Forest Model
    with mlflow.start_run(run_name="RandomForest_Baseline", nested=True):
        mlflow.sklearn.autolog()
        
        rf_model = RandomForestClassifier(
            n_estimators=200, 
            random_state=42, 
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)

        y_pred_rf = rf_model.predict(X_test)

        # Confusion matrix Random Forest
        rf_class_labels = sorted(y_test.unique())
        cm_path = save_confusion_matrix(
            y_test,
            y_pred_rf,
            labels=rf_class_labels,
            title="RF Confusion Matrix",
            filename="rf_confusion_matrix.png",
        )
        mlflow.log_artifact(cm_path)

        # Classification report Random Forest
        report = classification_report(y_test, y_pred_rf)
        report_path = os.path.join("artifacts", "rf_classification_report.txt")
        with open(report_path, "w") as f:
            f.write(report)
        mlflow.log_artifact(report_path)

        # Calculate additional metrics
        rf_acc = accuracy_score(y_test, y_pred_rf)
        rf_f1 = f1_score(y_test, y_pred_rf, average="weighted")
        
        # Log additional metrics manually
        mlflow.log_metrics({
            "test_accuracy": rf_acc,
            "test_f1_score": rf_f1
        })

        print("Accuracy Random Forest Model: ", rf_acc)
        print("F1 Score Random Forest Model: ", rf_f1)

    # Train XGBoost Model 
    with mlflow.start_run(run_name="XGBoost_Baseline", nested=True):
        mlflow.xgboost.autolog()
        
        y_train_xgb = y_train - 1
        y_test_xgb = y_test - 1

        xgb_model = XGBClassifier(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=10,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        xgb_model.fit(X_train, y_train_xgb)

        y_pred_xgb = xgb_model.predict(X_test)

        # Confusion matrix XGBoost
        xgb_class_labels = sorted(y_test_xgb.unique())
        cm_path = save_confusion_matrix(
            y_test_xgb,
            y_pred_xgb,
            labels=xgb_class_labels,
            title="XGB Confusion Matrix",
            filename="xgb_confusion_matrix.png",
        )
        mlflow.log_artifact(cm_path)

        # Classification report Random Forest
        report = classification_report(y_test, y_pred_xgb)
        report_path = os.path.join("artifacts", "xgb_classification_report.txt")
        with open(report_path, "w") as f:
            f.write(report)
        mlflow.log_artifact(report_path)

        # Calculate additional metrics 
        xgb_acc = accuracy_score(y_test_xgb, y_pred_xgb)
        xgb_f1 = f1_score(y_test_xgb, y_pred_xgb, average="weighted")
        
        # Log additional metrics
        mlflow.log_metrics({
            "test_accuracy": xgb_acc,
            "test_f1_score": xgb_f1
        })

        print("Accuracy XGBoost Model: ", xgb_acc)
        print("F1 Score XGBoost Model: ", xgb_f1)


print("\nTraining Model Run Successfully 🤓.")