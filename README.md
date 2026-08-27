🚀 ML Engineer Portfolio – Massimo Vianello


## 🗂️ Progetti Inclusi

### 1. NLP Classifier (BERT) – [`NLP_Classifier_Module_BERT.py`](NLP_Classifier_Module_BERT.py)

**Obiettivo:** Classificare testi clinici come "richiede attenzione" (1) o "stabile" (0) usando BERT fine-tuned.

**Tecnologie:** PyTorch, HuggingFace Transformers, Scikit-learn, tqdm

**Output reale:**

Epoch 1/2 - Average Loss: 0.6908
Epoch 2/2 - Average Loss: 0.6890
Predizione per 'Paziente dimesso, follow-up in 2 settimane' -> Classe: 1 (1=attenzione, 0=stabile)
text


---

### 2. Recommender System – [`Recommender_System_HuggingFace.py`](Recommender_System_HuggingFace.py)

**Obiettivo:** Generare raccomandazioni personalizzate usando Matrix Factorization con Stochastic Gradient Descent (SGD) e regolarizzazione.

**Tecnologie:** NumPy, Pandas, Scikit-learn, tqdm

**Output reale:**

Iterazione 30/30 - RMSE: 0.7127
RMSE su test: 1.0625
Raccomandazioni per utente 1: [(40, 3.32), (21, 3.32), (44, 3.30), (17, 3.20), (5, 3.20)]
Item simili a 10: [(6, 0.516), (43, 0.438), (13, 0.394), (48, 0.367), (18, 0.290)]
text


---

### 3. ETL Pipeline (PySpark) – [`ETL_Pipeline_PySpark.py`](ETL_Pipeline_PySpark.py)

**Obiettivo:** Pipeline ETL per dati su larga scala. **Versione in-memory, Windows-compatibile** (non richiede winutils).

**Tecnologie:** PySpark, Pandas

**Output reale:**

📊 Dati originali:
+---+----------+-----+------+
| id| timestamp|value|status|
+---+----------+-----+------+
| 1|2024-01-01| 10.5| OK|
| 2|2024-01-01| 15.2| ERROR|
| 3|2024-01-02| 12.1| OK|
| 4|2024-01-02| 8.7| OK|
| 5|2024-01-03| 22.3| ERROR|
+---+----------+-----+------+

📈 Dati aggregati per timestamp:
+----------+--------+-------------+
| timestamp|avg_value|count_records|
+----------+--------+-------------+
|2024-01-02| 10.4| 2|
|2024-01-01| 10.5| 1|
+----------+--------+-------------+
text


---

### 4. Data Pipeline Orchestrator (PySpark + Airflow Logic) – [`ZPF1_data_pipeline_orchestrator.py`](ZPF1_data_pipeline_orchestrator.py)

**Obiettivo:** Dimostrare una pipeline ETL orchestrata con logica stile Apache Airflow. Gestisce dipendenze tra task, retry automatici e fallback in caso di errori di scrittura.

**Tecnologie:** PySpark, Apache Airflow (concetto), Pandas

**Output reale:**

▶️ Esecuzione task 'etl_daily' (tentativo 1)...
🚀 Avvio task ETL per la data: 2026-08-27
📊 Dati grezzi per 2026-08-27:
+---+----------------+------+-----+
| id|data_riferimento|valore|stato|
+---+----------------+------+-----+
| 1| 2026-08-27| 10.5| OK|
| 2| 2026-08-27| 15.2|ERROR|
| 3| 2026-08-27| 12.1| OK|
+---+----------------+------+-----+

📈 Dati aggregati per 2026-08-27:
+----------------+------------+----------------+
|data_riferimento|valore_medio|conteggio_record|
+----------------+------------+----------------+
| 2026-08-27| 11.3| 2|
+----------------+------------+----------------+

✅ Task ETL per 2026-08-27 completato.
✅ Task 'etl_daily' completato con successo.

▶️ Esecuzione task 'etl_backfill' (tentativo 1)...
🚀 Avvio task ETL per la data: 2026-08-26
... (stessa logica) ...
✅ Task ETL per 2026-08-26 completato.
✅ Task 'etl_backfill' completato con successo.

📋 Report finale della pipeline:

    etl_daily: success

    etl_backfill: success

text


**Nota:** Il programma gestisce il fallimento di scrittura su disco (tipico di Windows) e mostra i dati trasformati come fallback, dimostrando robustezza.

---

### 5. MLOps Pipeline – [`MLOpsPipeline_MLflow_Docke.py`](MLOpsPipeline_MLflow_Docke.py)

**Obiettivo:** Ciclo di vita completo del modello: training, versioning con MLflow, promozione a Staging/Production, e deploy in container Docker con API Flask.

**Tecnologie:** MLflow, Docker, Flask, Scikit-learn, Joblib

**Output reale:**

Modello RandomForestModel v13 promosso a Staging
Modello RandomForestModel v13 promosso a Production
✅ Container randomforestmodel_production_container avviato sulla porta 5001
✅ API test riuscita: {'prediction': 1}
text


**📄 File API:** Il file [`serve.py`](serve.py) contiene il codice Flask che espone il modello come servizio REST.

**Test dell'API:**
```bash
curl -X POST http://localhost:5001/predict -H 'Content-Type: application/json' -d '{"features": [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]}'
# Risposta: {"prediction":1}

6. XAI Explainer – Explainable(XAI)_SHAP_LIME.py

Obiettivo: Explainability di modelli ML (SHAP) e modelli NLP (LIME). Supporta le ultime versioni delle librerie.

Tecnologie: SHAP, LIME, Transformers, Scikit-learn

Output reale:
text

✅ LIME - Spiegazione: [('feature_0 <= 1.39', 0.0), ('feature_1 <= -0.97', 0.0), ...]
✅ NLP LIME: [('paziente', -0.26), ('sta', -0.25), ('miglioramento', -0.21), ('Il', 0.15), ('significativo', -0.14)]

🛠️ Tecnologie Utilizzate
Area	Tecnologie
Machine Learning	PyTorch, HuggingFace Transformers, Scikit-learn
Deep Learning	BERT, Fine-tuning, Transformer-based NLP
Data Engineering	PySpark, Pandas, NumPy
Orchestrazione	Apache Airflow (concetto), gestione DAG, retry, dipendenze
MLOps	MLflow (tracking, versioning, registry), Docker, Flask
Explainable AI	SHAP, LIME
Linguaggi	Python 3.9+
🚀 Come Eseguire i Programmi

Ogni programma è auto-installante. Basta eseguire:
bash

# NLP Classifier
python NLP_Classifier_Module_BERT.py

# Recommender System
python Recommender_System_HuggingFace.py

# ETL Pipeline (Windows-compatibile)
python ETL_Pipeline_PySpark.py

# Data Pipeline Orchestrator (PySpark + Airflow Logic)
python ZPF1_data_pipeline_orchestrator.py

# MLOps Pipeline (richiede Docker)
python MLOpsPipeline_MLflow_Docke.py

# XAI Explainer
python Explainable\(XAI\)_SHAP_LIME.py

    Nota: I programmi installeranno automaticamente tutte le librerie necessarie. Per MLOps è richiesto Docker.

📬 Contatti

Massimo Vianello
📧 max.vianello69@libero.it
📱 347 000 6339