"""
MODULO 4: MLOps Pipeline con MLflow e Docker (VERSIONE DEFINITIVA)
- Training, versioning, promozione con MLflow
- Deploy con Docker (caricamento del modello via joblib, senza dipendere da MLflow nel container)
"""

import subprocess
import sys
import importlib.util
import os
import time
import json

# =====================================================
# SEZIONE 1: AUTO-INSTALLAZIONE LIBRERIE
# =====================================================
def install_and_import(package_name, import_name=None):
    if import_name is None:
        import_name = package_name
    spec = importlib.util.find_spec(import_name)
    if spec is None:
        print(f"📦 Installazione di '{package_name}'...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"✅ '{package_name}' installato.")
    else:
        print(f"✅ '{package_name}' già presente.")

install_and_import("mlflow")
install_and_import("scikit-learn", "sklearn")
install_and_import("pandas")
install_and_import("numpy")
install_and_import("pyyaml", "yaml")
install_and_import("flask")
install_and_import("requests")
install_and_import("joblib")   # Per salvare il modello

# =====================================================
# SEZIONE 2: IMPORT LIBRERIE
# =====================================================
import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.datasets import make_classification
import joblib
import logging
import requests

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =====================================================
# SEZIONE 3: CLASSE MLOPS PIPELINE
# =====================================================
class MLOpsPipeline:
    def __init__(self, experiment_name="demo_mlops"):
        os.environ['MLFLOW_ALLOW_FILE_STORE'] = 'true'
        mlflow.set_tracking_uri("file:./mlruns")
        self.client = MlflowClient()
        exp = self.client.get_experiment_by_name(experiment_name)
        self.exp_id = exp.experiment_id if exp else self.client.create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)
        logger.info("MLOpsPipeline inizializzata")

    def run_training(self):
        with mlflow.start_run(experiment_id=self.exp_id) as run:
            X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, random_state=42)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            mlflow.log_param("n_samples", len(X))
            mlflow.log_param("n_features", X.shape[1])

            model = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1_score': f1_score(y_test, y_pred, average='weighted'),
            }
            mlflow.log_metrics(metrics)
            mlflow.log_params(model.get_params())
            mlflow.sklearn.log_model(model, "random_forest_model", registered_model_name="RandomForestModel")

            # Salva anche in joblib per il deploy
            joblib.dump(model, "model.joblib")
            mlflow.log_artifact("model.joblib")

            if metrics['accuracy'] > 0.8:
                self.promote_model("RandomForestModel", "Staging")
                self.promote_model("RandomForestModel", "Production")

            return {"run_id": run.info.run_id, "metrics": metrics, "status": "success"}

    def promote_model(self, model_name, stage):
        versions = self.client.get_latest_versions(model_name)
        if not versions:
            logger.warning(f"Nessuna versione per {model_name}")
            return
        latest = versions[0]
        self.client.transition_model_version_stage(
            name=model_name,
            version=latest.version,
            stage=stage
        )
        logger.info(f"Modello {model_name} v{latest.version} promosso a {stage}")

    def deploy_model(self, model_name="RandomForestModel", stage="Production"):
        try:
            subprocess.run(["docker", "--version"], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("❌ Docker non installato o non avviato.")
            return {"status": "error", "error": "Docker non disponibile"}

        try:
            # Crea serve.py (senza MLflow, carica direttamente il modello)
            serve_script = '''# -*- coding: utf-8 -*-
import joblib
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

model_path = "/app/model.joblib"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found at {model_path}")

model = joblib.load(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['features']
    prediction = model.predict([data])
    return jsonify({'prediction': int(prediction[0])})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
'''
            with open("serve.py", "w", encoding="utf-8") as f:
                f.write(serve_script)

            with open("requirements.txt", "w", encoding="utf-8") as f:
                f.write("flask\njoblib\nscikit-learn\n")

            dockerfile = '''
FROM python:3.9-slim
WORKDIR /app
ENV PYTHONIOENCODING=utf-8
ENV LANG=C.UTF-8

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY serve.py .
COPY model.joblib .

EXPOSE 5001
CMD ["python", "serve.py"]
'''
            with open("Dockerfile", "w", encoding="utf-8") as f:
                f.write(dockerfile)

            image_name = f"model_{model_name}_{stage}".lower()
            logger.info(f"Costruzione immagine Docker: {image_name}...")
            subprocess.run(["docker", "rmi", "-f", image_name], capture_output=True)
            build_result = subprocess.run(
                ["docker", "build", "-t", image_name, "."],
                capture_output=True, text=True
            )
            if build_result.returncode != 0:
                logger.error(f"❌ Docker build fallito:\n{build_result.stderr}")
                return {"status": "error", "error": build_result.stderr}

            container_name = f"{model_name}_{stage}_container".lower()
            logger.info(f"Avvio container: {container_name}...")
            subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)
            run_result = subprocess.run(
                ["docker", "run", "-d", "--name", container_name, "-p", "5001:5001", image_name],
                capture_output=True, text=True
            )
            if run_result.returncode != 0:
                logger.error(f"❌ Docker run fallito:\n{run_result.stderr}")
                return {"status": "error", "error": run_result.stderr}

            logger.info(f"✅ Container {container_name} avviato sulla porta 5001")
            time.sleep(3)
            try:
                test_data = {"features": [0.5] * 20}
                response = requests.post("http://localhost:5001/predict", json=test_data, timeout=5)
                if response.status_code == 200:
                    logger.info(f"✅ API test riuscita: {response.json()}")
                else:
                    logger.warning(f"⚠️ API test response: {response.status_code}")
            except Exception as e:
                logger.warning(f"⚠️ API test fallito: {e}")
                logs = subprocess.run(["docker", "logs", container_name], capture_output=True, text=True)
                logger.info(f"Log del container:\n{logs.stdout}\n{logs.stderr}")

            return {"status": "success", "container_name": container_name, "port": 5001}

        except Exception as e:
            logger.error(f"Errore durante il deploy: {str(e)}")
            return {"status": "error", "error": str(e)}

    def cleanup(self):
        try:
            subprocess.run(["docker", "rm", "-f", "randomforestmodel_production_container"], capture_output=True)
        except Exception:
            pass
        for f in ["serve.py", "Dockerfile", "requirements.txt", "model.joblib"]:
            if os.path.exists(f):
                os.remove(f)
        logger.info("Pulizia completata")

# =====================================================
# SEZIONE 4: MAIN
# =====================================================
def main():
    logger.info("=== Demo MLOps Pipeline (Versione con Deploy Docker) ===")
    pipeline = MLOpsPipeline(experiment_name="demo_mlops")

    # 1. Training e versioning
    train_result = pipeline.run_training()
    logger.info(f"Training result: {train_result['metrics']}")

    # 2. Deploy
    deploy_result = pipeline.deploy_model()
    logger.info(f"Deploy result: {deploy_result}")

    if deploy_result.get('status') == 'success':
        print("\n🔍 Per testare l'API, esegui in un altro terminale:\n")
        print("curl -X POST http://localhost:5001/predict -H 'Content-Type: application/json' -d '{\"features\": [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]}'\n")

    # pipeline.cleanup()
    print("\n✅ Programma completato con successo!")

if __name__ == "__main__":
    main()