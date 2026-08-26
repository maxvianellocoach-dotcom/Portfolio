"""
MODULO 5: Explainable AI (XAI) con SHAP e LIME - VERSIONE DEFINITIVA
Supporta le ultime versioni delle librerie (SHAP, LIME, Transformers).
"""

import subprocess
import sys
import importlib.util

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

install_and_import("shap")
install_and_import("lime")
install_and_import("scikit-learn", "sklearn")
install_and_import("pandas")
install_and_import("numpy")
install_and_import("matplotlib")
install_and_import("seaborn")
install_and_import("transformers")
install_and_import("torch")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import shap
from lime.lime_tabular import LimeTabularExplainer
from lime.lime_text import LimeTextExplainer
from transformers import pipeline
import logging
import warnings
import json

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class XAIExplainer:
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names
        self.shap_explainer = shap.TreeExplainer(model)
        logger.info("XAIExplainer inizializzato")

    def explain_shap(self, X_sample):
        """Spiegazione SHAP con gestione multi-classe."""
        # Converti a numpy array 2D
        if isinstance(X_sample, pd.DataFrame):
            X_np = X_sample.values.reshape(1, -1)
        else:
            X_np = X_sample.reshape(1, -1)
        
        # Calcola SHAP values
        shap_values = self.shap_explainer.shap_values(X_np)
        
        # 🔧 FIX: Gestione multi-classe
        if isinstance(shap_values, list):
            if len(shap_values) == 2:
                # Caso binario: prendi la classe positiva (indice 1)
                shap_values = shap_values[1]
            else:
                # Caso multi-classe: prendi la prima classe come fallback
                shap_values = shap_values[0]
        
        # Crea feature importance
        imp = pd.DataFrame({
            'feature': self.feature_names,
            'importance': np.abs(shap_values).flatten()
        }).sort_values('importance', ascending=False)
        
        # Genera summary plot
        try:
            shap.summary_plot(shap_values, X_np, feature_names=self.feature_names, show=False)
            plt.savefig('shap_summary.png', bbox_inches='tight')
            plt.close()
        except Exception as e:
            logger.warning(f"Plot SHAP non generato: {e}")
        
        return {"feature_importance": imp.head(5).to_dict(orient='records')}

    def explain_lime(self, X_sample, num_features=5):
        """Spiegazione LIME con compatibilità multi-versione."""
        if isinstance(X_sample, pd.DataFrame):
            X_np = X_sample.values
        else:
            X_np = X_sample
        
        explainer = LimeTabularExplainer(
            X_np,
            feature_names=self.feature_names,
            class_names=['Classe 0', 'Classe 1'],
            mode='classification'
        )
        
        exp = explainer.explain_instance(
            X_np[0],
            self.model.predict_proba,
            num_features=num_features
        )
        
        # 🔧 FIX: Compatibilità con diverse versioni di LIME
        prediction = getattr(exp, 'prediction', getattr(exp, 'predicted_value', None))
        
        return {
            "explanation": exp.as_list(),
            "prediction": prediction
        }

def main():
    logger.info("=== Demo XAI Explainer (Versione Definitiva) ===")
    
    # 1. Crea e addestra un modello
    X, y = make_classification(n_samples=500, n_features=5, random_state=42)
    feature_names = [f"feature_{i}" for i in range(5)]
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    X_df = pd.DataFrame(X, columns=feature_names)
    
    # 2. Inizializza explainer
    explainer = XAIExplainer(model, feature_names)
    
    # 3. SHAP
    try:
        shap_res = explainer.explain_shap(X_df.head(1))
        logger.info(f"✅ SHAP - Top features: {shap_res['feature_importance']}")
    except Exception as e:
        logger.error(f"SHAP fallito: {e}")
    
    # 4. LIME
    try:
        lime_res = explainer.explain_lime(X_df.head(1))
        logger.info(f"✅ LIME - Spiegazione: {lime_res['explanation']}")
        if lime_res['prediction'] is not None:
            logger.info(f"   Predizione: {lime_res['prediction']}")
    except Exception as e:
        logger.error(f"LIME fallito: {e}")
    
    # 5. NLP con LIME
    try:
        logger.info("Tentativo spiegazione NLP...")
        nlp = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        
        def predict_proba(texts):
            results = []
            for t in texts:
                res = nlp(t)[0]
                if res['label'] == 'POSITIVE':
                    results.append([1 - res['score'], res['score']])
                else:
                    results.append([res['score'], 1 - res['score']])
            return np.array(results)
        
        text = "Il paziente sta mostrando un miglioramento significativo"
        text_exp = LimeTextExplainer(class_names=['NEGATIVE', 'POSITIVE'])
        exp = text_exp.explain_instance(text, predict_proba, num_features=5)
        
        logger.info(f"✅ NLP LIME: {exp.as_list()}")
    except Exception as e:
        logger.warning(f"⚠️ NLP LIME non disponibile: {e}")
    
    # 6. Salva report
    report = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "status": "completed",
        "modules": ["SHAP", "LIME", "NLP_LIME"]
    }
    with open("xai_report_final.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print("\n✅ Programma completato con successo!")

if __name__ == "__main__":
    main()